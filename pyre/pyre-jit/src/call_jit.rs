//! JIT-specific call infrastructure — force/bridge callbacks, callee
//! frame creation helpers, frame pool.
//!
//! Separated from pyre-interpreter/src/call.rs so pyre-interpreter stays JIT-free.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Once;

use pyre_interpreter::bytecode::{Instruction, OpArgState};
use pyre_interpreter::{
    PyResult, function_get_closure, function_get_globals, function_get_name, is_function,
    register_jit_function_caller,
};
use pyre_object::intobject::w_int_get_value;
use pyre_object::intobject::w_int_new;
use pyre_object::pyobject::is_int;
use pyre_object::{PY_NULL, PyObjectRef};

use pyre_interpreter::pyframe::PyFrame;
use pyre_jit_trace::trace::trace_bytecode;

// Force cache removed: CallAssemblerI + bridge handles recursion
// natively without memoization.

thread_local! {
    /// Stash Python exceptions from blackhole/force paths that cross
    /// FFI boundaries (compiled code → callback → exception).
    static LAST_CA_EXCEPTION: std::cell::RefCell<Option<pyre_interpreter::error::PyError>> =
        const { std::cell::RefCell::new(None) };
    static SELF_RECURSIVE_DISPATCH_CACHE: UnsafeCell<Option<(u64, Option<u64>)>> =
        const { UnsafeCell::new(None) };
}

/// Take stashed exception from blackhole/force FFI paths.
pub fn take_ca_exception() -> Option<pyre_interpreter::error::PyError> {
    LAST_CA_EXCEPTION.with(|c| c.borrow_mut().take())
}

/// Park a Python exception that needs to surface across an FFI boundary
/// (callback emitted by compiled code → here → eventually picked up by
/// `take_ca_exception` in the eval loop).
pub fn set_pending_ca_exception(err: pyre_interpreter::error::PyError) {
    LAST_CA_EXCEPTION.with(|c| {
        *c.borrow_mut() = Some(err);
    });
}

// warmspot.py:449 portal result_type == REF: FINISH always boxes via
// wrapint, so the force/resume paths always receive a boxed Ref.
// FinishProtocol and normalize_direct_finish_result removed — they
// were dead code since result_type is always Type::Ref.

#[inline]
pub(crate) fn recursive_force_cache_safe(callable: PyObjectRef) -> bool {
    unsafe {
        if pyre_interpreter::is_builtin_code(
            pyre_interpreter::function_get_code(callable) as pyre_object::PyObjectRef
        ) {
            return false;
        }
        if !function_get_closure(callable).is_null() {
            return false;
        }
        let code =
            &*(pyre_interpreter::get_pycode(callable) as *const pyre_interpreter::CodeObject);
        let func_name = function_get_name(callable);
        let mut arg_state = OpArgState::default();
        let mut saw_self_reference = false;

        for code_unit in code.instructions.iter().copied() {
            let (instruction, op_arg) = arg_state.get(code_unit);
            match instruction {
                Instruction::LoadName { namei } => {
                    let idx = namei.get(op_arg) as usize;
                    if code.names[idx].as_str() != func_name {
                        return false;
                    }
                    saw_self_reference = true;
                }
                Instruction::LoadGlobal { namei } => {
                    let raw = namei.get(op_arg) as usize;
                    let name_idx = raw >> 1;
                    if code.names[name_idx].as_str() != func_name {
                        return false;
                    }
                    saw_self_reference = true;
                }
                Instruction::StoreName { .. }
                | Instruction::StoreGlobal { .. }
                | Instruction::LoadDeref { .. }
                | Instruction::StoreDeref { .. }
                | Instruction::DeleteDeref { .. }
                | Instruction::ImportName { .. }
                | Instruction::ImportFrom { .. }
                | Instruction::DeleteName { .. }
                | Instruction::DeleteGlobal { .. }
                | Instruction::MakeCell { .. }
                | Instruction::CopyFreeVars { .. } => return false,
                _ => {}
            }
        }

        if !saw_self_reference {
            return false;
        }
    }

    true
}

fn self_recursive_dispatch(green_key: u64) -> Option<u64> {
    SELF_RECURSIVE_DISPATCH_CACHE.with(|cell| unsafe {
        let slot = &mut *cell.get();
        if let Some((cached_key, token_num)) = *slot {
            if cached_key == green_key && token_num.is_some() {
                return token_num;
            }
        }

        let (driver, _) = crate::eval::driver_pair();
        let token_num = driver.get_loop_token(green_key).map(|token| token.number);
        if token_num.is_some() {
            *slot = Some((green_key, token_num));
        }
        token_num
    })
}

// Force cache implementation removed — CallAssemblerI + bridge
// handles recursive dispatch natively.

// ── Callee frame arena (RPython nursery bump equivalent) ─────────
// ── Global arena pointers for Cranelift inline access ──────────────
//
// Single-threaded JIT invariant: only one thread executes compiled code
// at a time, so these globals need no synchronization.
static mut ARENA_BUF_BASE: *mut u8 = std::ptr::null_mut();
static mut ARENA_TOP: usize = 0;
static mut ARENA_INITIALIZED: usize = 0;

fn arena_jitframe_descrs() -> majit_gc::rewrite::JitFrameDescrs {
    use majit_metainterp::jitframe::*;
    majit_gc::rewrite::JitFrameDescrs {
        jitframe_tid: crate::jit::descr::JITFRAME_GC_TYPE_ID,
        jitframe_fixed_size: JITFRAME_FIXED_SIZE,
        jf_frame_info_ofs: JF_FRAME_INFO_OFS,
        jf_descr_ofs: JF_DESCR_OFS,
        jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
        jf_savedata_ofs: JF_SAVEDATA_OFS,
        jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
        jf_forward_ofs: JF_FORWARD_OFS,
        jf_frame_ofs: JF_FRAME_OFS,
        // RPython llmodel.py:385-395 + rewrite.py:680-684 consume
        // unpack_arraydescr()/lendescr offsets as jitframe-base-relative
        // addresses, not offsets relative to jf_frame itself.
        jf_frame_baseitemofs: FIRST_ITEM_OFFSET,
        jf_frame_lengthofs: JF_FRAME_OFS + LENGTHOFS,
        sign_size: SIGN_SIZE,
    }
}

#[cfg(test)]
mod tests {
    use super::arena_jitframe_descrs;
    use majit_metainterp::jitframe::{FIRST_ITEM_OFFSET, JF_FRAME_OFS};

    #[test]
    fn arena_jitframe_descrs_uses_frame_relative_offsets() {
        let descrs = arena_jitframe_descrs();
        assert_eq!(descrs.jf_frame_baseitemofs, FIRST_ITEM_OFFSET);
        assert_eq!(descrs.jf_frame_lengthofs, JF_FRAME_OFS);
    }
}

#[cfg(feature = "cranelift")]
pub fn arena_global_info() -> majit_backend_cranelift::JitFrameLayoutInfo {
    majit_backend_cranelift::JitFrameLayoutInfo {
        jitframe_descrs: Some(arena_jitframe_descrs()),
    }
}

#[cfg(feature = "dynasm")]
pub fn arena_global_info_dynasm() -> majit_backend_dynasm::JitFrameLayoutInfo {
    majit_backend_dynasm::JitFrameLayoutInfo {
        jitframe_descrs: Some(arena_jitframe_descrs()),
    }
}

//
// LIFO stack of pre-allocated PyFrame slots. Recursive call/return
// order is naturally LIFO, so arena_take/arena_put are O(1).
// Eliminates heap allocation for recursion depths up to ARENA_CAP.

const ARENA_CAP: usize = 64;

/// PyPy GcStruct layout: [GcHeader (8 bytes)] [struct fields].
/// Every GC object (including PyFrame / W_Root) is prepended by a
/// zeroed GcHeader. Arena slots and heap fallbacks match this layout.
const GC_HEADER_SIZE: usize = 8;

/// Arena slot with prepended GcHeader (zeroed, layout parity only).
#[repr(C)]
struct GcFrameSlot {
    gc_header: u64,
    frame: MaybeUninit<PyFrame>,
}

impl GcFrameSlot {
    const fn zeroed() -> Self {
        GcFrameSlot {
            gc_header: 0,
            frame: MaybeUninit::uninit(),
        }
    }
}

/// Heap-allocated frame with prepended GcHeader.
#[repr(C)]
struct GcPyFrame {
    gc_header: u64,
    frame: PyFrame,
}

fn heap_alloc_frame(frame: PyFrame) -> *mut PyFrame {
    let gc_frame = Box::into_raw(Box::new(GcPyFrame {
        gc_header: 0,
        frame,
    }));
    unsafe { &mut (*gc_frame).frame as *mut PyFrame }
}

fn heap_free_frame(ptr: *mut PyFrame) {
    let gc_frame = unsafe { (ptr as *mut u8).sub(GC_HEADER_SIZE) as *mut GcPyFrame };
    unsafe { drop(Box::from_raw(gc_frame)) };
}

struct FrameArena {
    buf: Box<[GcFrameSlot; ARENA_CAP]>,
    /// Number of frames currently in use (LIFO stack pointer).
    top: usize,
    /// Frames below this index have been initialized at least once.
    /// Reuse only needs reinit of changed fields, not full new_for_call.
    initialized: usize,
}

impl FrameArena {
    fn new() -> Self {
        let mut arena = Self {
            buf: Box::new([const { GcFrameSlot::zeroed() }; ARENA_CAP]),
            top: 0,
            initialized: 0,
        };
        // Publish stable pointers so Cranelift-generated code can
        // inline arena take/put without going through TLS.
        unsafe {
            ARENA_BUF_BASE = arena.buf.as_mut_ptr() as *mut u8;
            ARENA_TOP = 0;
            ARENA_INITIALIZED = 0;
        }
        arena
    }

    /// Take the next frame slot. Returns (ptr, was_previously_initialized).
    /// The returned pointer points to the PyFrame part (after the GcHeader).
    #[inline]
    fn take(&mut self) -> Option<(*mut PyFrame, bool)> {
        if self.top < ARENA_CAP {
            let idx = self.top;
            self.top += 1;
            unsafe {
                ARENA_TOP = self.top;
            }
            let ptr = self.buf[idx].frame.as_mut_ptr();
            let was_init = idx < self.initialized;
            Some((ptr, was_init))
        } else {
            None
        }
    }

    /// Return a frame to the arena. Must be the most recently taken frame (LIFO).
    #[inline]
    fn put(&mut self, ptr: *mut PyFrame) -> bool {
        if self.top > 0 && ptr == self.buf[self.top - 1].frame.as_mut_ptr() {
            self.top -= 1;
            unsafe {
                ARENA_TOP = self.top;
            }
            return true;
        }
        // Check if within arena range — don't free, but mark as non-LIFO.
        let base = self.buf[0].frame.as_mut_ptr() as usize;
        let end = unsafe { (self.buf.as_ptr() as *const GcFrameSlot).add(ARENA_CAP) as usize };
        let addr = ptr as usize;
        addr >= base && addr < end
    }

    /// Mark that frames up to `top` have been fully initialized.
    #[inline]
    fn mark_initialized(&mut self) {
        if self.top > self.initialized {
            self.initialized = self.top;
            unsafe {
                ARENA_INITIALIZED = self.top;
            }
        }
    }
}

thread_local! {
    static FRAME_ARENA: UnsafeCell<FrameArena> = UnsafeCell::new(FrameArena::new());
}

#[inline]
fn arena_ref() -> &'static mut FrameArena {
    FRAME_ARENA.with(|cell| unsafe { &mut *cell.get() })
}

// ── JIT call callbacks ───────────────────────────────────────────

extern "C" fn jit_call_user_function_from_frame(
    frame_ptr: i64,
    callable: i64,
    args_ptr: *const i64,
    nargs: i64,
) -> i64 {
    let frame = unsafe { &*(frame_ptr as *const PyFrame) };
    let args =
        unsafe { std::slice::from_raw_parts(args_ptr as *const PyObjectRef, nargs as usize) };
    // Depth tracked by pyre_interpreter::call::CALL_DEPTH (call_user_function path).
    match pyre_interpreter::call::call_user_function(frame, callable as PyObjectRef, args) {
        Ok(result) => result as i64,
        Err(err) => {
            // llmodel.py:194-199 _store_exception: write the exception
            // to the backend's `_exception_emulator` tp/val cells. The
            // matching GUARD_NO_EXCEPTION in the trace then reads
            // pos_exception()/pos_exc_value() and fails, and resume
            // data hands control to the except block. Do NOT stash the
            // PyError through a side channel — that would let the
            // interpreter-side eval loop surface it before the guard
            // machinery sees it, bypassing try/except.
            let exc_obj = err.exc_object;
            if exc_obj != pyre_object::PY_NULL {
                #[cfg(feature = "cranelift")]
                majit_backend_cranelift::jit_exc_raise(exc_obj as i64);
                #[cfg(feature = "dynasm")]
                majit_backend_dynasm::jit_exc_raise(exc_obj as i64);
            }
            0 // garbage — GUARD_NO_EXCEPTION will fire
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_force_callee_frame(frame_ptr: i64) -> i64 {
    #[cfg(feature = "cranelift")]
    let _ = majit_backend_cranelift::take_pending_frame_restore();
    #[cfg(feature = "cranelift")]
    let pending = majit_backend_cranelift::take_pending_force_local0();
    #[cfg(not(feature = "cranelift"))]
    let pending: Option<i64> = None;

    // Lazy frame (RPython parity): when CallR(create_frame) is elided,
    // frame_ptr is the CALLER frame. pending_force_local0 contains the
    // raw int arg. Create callee frame lazily and execute it.
    if let Some(raw_local0) = pending {
        return jit_force_self_recursive_call_raw_1(frame_ptr, raw_local0);
    }
    // Nursery-safe force: read code/namespace/exec_ctx via raw offsets
    // (valid for both arena PyFrame AND nursery-allocated raw blocks).
    // Then create a proper PyFrame for the interpreter.
    //
    // warmspot.py:1021 assembler_call_helper parity: the callee frame
    // (deadframe) may be a nursery-allocated JitFrame-like block. We
    // reconstruct a proper interpreter frame from its raw fields.
    let (code, namespace, exec_ctx) = unsafe {
        use pyre_interpreter::pyframe::*;
        let p = frame_ptr as *const u8;
        let code = *(p.add(PYFRAME_PYCODE_OFFSET) as *const *const ());
        let ns = *(p.add(std::mem::offset_of!(PyFrame, w_globals))
            as *const *mut pyre_interpreter::DictStorage);
        let ec = *(p.add(std::mem::offset_of!(PyFrame, execution_context))
            as *const *const pyre_interpreter::PyExecutionContext);
        (code, ns, ec)
    };

    let mut func_frame = PyFrame::new_for_call(code, &[], namespace, exec_ctx);
    func_frame.fix_array_ptrs();

    // warmspot.py:1021-1028 assembler_call_helper:
    //   fail_descr.handle_fail(deadframe, metainterp_sd, jd)
    //   except JitException as e: return handle_jitexception(e)
    //
    // handle_jitexception (warmspot.py:961) handles ContinueRunningNormally
    // by calling portal_ptr(*args) — the JIT-aware portal. RPython does
    // NOT prevent JIT re-entry here. The callee can enter compiled code
    // through maybe_compile_and_run in the portal runner.
    let result = crate::eval::portal_runner(&mut func_frame);

    // warmspot.py:449 result_type=REF: always boxed Ref
    result as i64
}

/// warmspot.py:1021-1028 — assembler_call_helper.
///
/// Called when CALL_ASSEMBLER guard-fails (not a finish exit).
/// Receives a JitFrame pointer, reconstructs a PyFrame from the
/// jf_frame slots, and resumes execution in the interpreter.
///
/// This is the JitFrame-aware counterpart to `jit_force_callee_frame`
/// (which operates on PyFrame directly). When the GC rewriter wires
/// nursery JitFrame allocation, this function replaces the force path.
#[allow(dead_code)]
#[majit_macros::jit_may_force]
pub extern "C" fn assembler_call_helper(jitframe_ptr: i64, _virtualizable_ref: i64) -> i64 {
    use majit_metainterp::jitframe::JitFrame;

    let jf = jitframe_ptr as *mut JitFrame;

    // warmspot.py:1022 — fail_descr = cpu.get_latest_descr(deadframe)
    // compile.py:701 handle_fail: dispatches on fail_descr to either
    // _trace_and_compile_from_bridge or resume_in_blackhole.
    // Bridge compilation is driven by must_compile() in jitdriver.
    // This force path always resumes in the interpreter (blackhole).
    let _descr = unsafe { majit_backend::llmodel::get_latest_descr(jf) };

    // For now, reconstruct a PyFrame and run it in the interpreter.
    // This is the "blackhole" path — RPython resume.py parity.
    //
    // Step 1: read the raw int arg from jf_frame[0]
    let raw_arg = unsafe { majit_backend::llmodel::get_int_value_direct(jf, 0) };

    // Step 2: get caller frame from the force context
    #[cfg(feature = "cranelift")]
    let pending = majit_backend_cranelift::take_pending_force_local0();
    #[cfg(not(feature = "cranelift"))]
    let pending: Option<i64> = None;
    let raw_local0 = pending.unwrap_or(raw_arg as i64);

    // Step 3: create a PyFrame and run it
    // The caller_frame is in inputs[0] which was the JitFrame's first
    // virtualizable input. For now, fall back to the existing force path.
    jit_force_self_recursive_call_raw_1(jitframe_ptr, raw_local0)
}

/// RPython: FieldDescr.offset is resolved at rtyper time. In pyre, Rust struct
/// layout determines field offsets. This resolver maps (owner_type, field_name)
/// to byte offsets for BhDescr::Field resolution in the blackhole.
/// Called by `bh.resolve_field_offsets()` after `setposition()`.
fn resolve_field_offset(owner: &str, field_name: &str) -> usize {
    use pyre_interpreter::pyframe::PyFrame;
    match field_name {
        "execution_context" => std::mem::offset_of!(PyFrame, execution_context),
        "code" | "pycode" => std::mem::offset_of!(PyFrame, pycode),
        "locals_cells_stack_w" => std::mem::offset_of!(PyFrame, locals_cells_stack_w),
        "valuestackdepth" => std::mem::offset_of!(PyFrame, valuestackdepth),
        "next_instr" | "f_lasti" | "last_instr" => std::mem::offset_of!(PyFrame, last_instr),
        "namespace" | "w_globals" => std::mem::offset_of!(PyFrame, w_globals),
        "vable_token" => std::mem::offset_of!(PyFrame, vable_token),
        _ => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][blackhole] WARNING: unresolved field offset owner={:?} name={:?}",
                    owner, field_name
                );
            }
            0
        }
    }
}

/// blackhole.py:1095 get_portal_runner / warmspot.py portal_runner parity:
/// Callback for bhimpl_recursive_call. Receives a frame pointer, executes
/// the frame through the JIT-enabled interpreter (eval_loop_jit), and
/// returns the result. This enables JIT re-entry at recursive portal depth.
/// warmspot.py:941-959 ll_portal_runner(*args) parity.
/// Portal runner with full portal arg ABI.
///
/// blackhole.py:1113-1116: called with merged arg lists:
///   all_i = greens_i + reds_i = [next_instr, is_being_profiled]
///   all_r = greens_r + reds_r = [pycode, frame, ec]
///   all_f = greens_f + reds_f = []
///
/// warmspot.py:972-975: portalfunc_ARGS extraction order:
///   (Int, 'green_int', 0) → next_instr = all_i[0]
///   (Int, 'green_int', 1) → is_being_profiled = all_i[1]
///   (Ref, 'green_ref', 0) → pycode = all_r[0]
///   (Ref, 'red_ref', 0)   → frame = all_r[1]
///   (Ref, 'red_ref', 1)   → ec = all_r[2]
pub(crate) fn bh_portal_runner(all_i: &[i64], all_r: &[i64], _all_f: &[i64]) -> i64 {
    // warmspot.py:972-975: extract portal args from merged lists.
    let next_instr = all_i.first().copied().unwrap_or(0) as usize;
    let _is_being_profiled = all_i.get(1).copied().unwrap_or(0);
    let pycode = all_r.first().copied().unwrap_or(0) as PyObjectRef;
    let frame_ptr = all_r.get(1).copied().unwrap_or(0) as *mut PyFrame;
    let ec = all_r.get(2).copied().unwrap_or(0) as *const pyre_interpreter::PyExecutionContext;

    if frame_ptr.is_null() {
        return pyre_object::PY_NULL as i64;
    }
    let frame = unsafe { &mut *frame_ptr };
    // warmspot.py:976: set portal args on frame before dispatch.
    if !pycode.is_null() {
        frame.pycode = pycode as *const ();
    }
    if !ec.is_null() {
        frame.execution_context = ec;
    }
    frame.set_last_instr_from_next_instr(next_instr);
    match crate::eval::portal_runner_result(frame) {
        Ok(result) => result as i64,
        Err(err) => {
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.exc_object as i64));
            pyre_object::PY_NULL as i64
        }
    }
}

/// jitexc.py JitException hierarchy — structural parity with RPython.
///
/// `_run_forever` must exit via exactly one of these variants.
/// Introduced in Phase 0 of the "portal locals → vable array" epic;
/// call sites still return `BlackholeResult` and will be migrated in
/// Phase 6 once `consume_vable_info` guarantees resume data validity.
#[allow(dead_code)] // populated progressively over the epic
pub enum JitException {
    /// jitexc.py:53 ContinueRunningNormally(gi, gr, gf, ri, rr, rf):
    /// blackhole reached the merge point → restart the portal. The six
    /// vectors carry the green/red int/ref/float live-in arguments used
    /// by `maybe_compile_and_run` to resume execution.
    ContinueRunningNormally {
        green_int: Vec<i64>,
        green_ref: Vec<PyObjectRef>,
        green_float: Vec<f64>,
        red_int: Vec<i64>,
        red_ref: Vec<PyObjectRef>,
        red_float: Vec<f64>,
    },
    /// jitexc.py:17 DoneWithThisFrameVoid.
    DoneWithThisFrameVoid,
    /// jitexc.py:21 DoneWithThisFrameInt(result).
    DoneWithThisFrameInt(i64),
    /// jitexc.py:29 DoneWithThisFrameRef(result).
    DoneWithThisFrameRef(PyObjectRef),
    /// jitexc.py:37 DoneWithThisFrameFloat(result).
    DoneWithThisFrameFloat(f64),
    /// jitexc.py:45 ExitFrameWithExceptionRef(value): blackhole ran out
    /// of caller frames while propagating a Python exception.
    ExitFrameWithExceptionRef(pyre_interpreter::error::PyError),
}

/// RPython jitexc.py parity: typed exception channels for blackhole result.
///
/// Each variant matches an RPython JitException subclass 1:1.
/// The value is carried in its native type — no boxing into PyObjectRef.
pub enum BlackholeResult {
    /// jitexc.py:53 ContinueRunningNormally(gi, gr, gf, ri, rr, rf)
    ContinueRunningNormally {
        green_int: Vec<i64>,
        green_ref: Vec<PyObjectRef>,
        green_float: Vec<f64>,
        red_int: Vec<i64>,
        red_ref: Vec<PyObjectRef>,
        red_float: Vec<f64>,
    },
    /// jitexc.py:16 DoneWithThisFrameVoid
    DoneWithThisFrameVoid,
    /// jitexc.py:20 DoneWithThisFrameInt(result: Signed)
    DoneWithThisFrameInt(i64),
    /// jitexc.py:28 DoneWithThisFrameRef(result: GCREF)
    DoneWithThisFrameRef(PyObjectRef),
    /// jitexc.py:36 DoneWithThisFrameFloat(result: FLOATSTORAGE)
    DoneWithThisFrameFloat(f64),
    /// jitexc.py:44 ExitFrameWithExceptionRef(value: GCREF)
    ExitFrameWithExceptionRef(pyre_interpreter::error::PyError),
    /// pyre-only: resume couldn't run (bad resume data, BC_ABORT, etc).
    Failed,
}

impl From<JitException> for BlackholeResult {
    /// jitexc.py parity: each JitException variant maps to BlackholeResult
    /// with its payload preserved — ContinueRunningNormally carries the
    /// 6 green/red lists through.
    fn from(exc: JitException) -> Self {
        match exc {
            JitException::ContinueRunningNormally {
                green_int,
                green_ref,
                green_float,
                red_int,
                red_ref,
                red_float,
            } => BlackholeResult::ContinueRunningNormally {
                green_int,
                green_ref,
                green_float,
                red_int,
                red_ref,
                red_float,
            },
            JitException::DoneWithThisFrameVoid => BlackholeResult::DoneWithThisFrameVoid,
            JitException::DoneWithThisFrameInt(v) => BlackholeResult::DoneWithThisFrameInt(v),
            JitException::DoneWithThisFrameRef(r) => BlackholeResult::DoneWithThisFrameRef(r),
            JitException::DoneWithThisFrameFloat(f) => BlackholeResult::DoneWithThisFrameFloat(f),
            JitException::ExitFrameWithExceptionRef(err) => {
                BlackholeResult::ExitFrameWithExceptionRef(err)
            }
        }
    }
}

impl BlackholeResult {
    /// warmspot.py:985-1005: convert typed DoneWithThisFrame* result to PyResult.
    ///
    /// This is the warmspot boundary where the typed JIT exception value
    /// is converted back into a Python-level result. RPython's warmspot
    /// does this implicitly via result_kind dispatch; pyre boxes here.
    pub fn to_pyresult(&self) -> Option<PyResult> {
        match self {
            BlackholeResult::DoneWithThisFrameVoid => Some(Ok(pyre_object::PY_NULL)),
            BlackholeResult::DoneWithThisFrameInt(v) => {
                Some(Ok(pyre_object::intobject::w_int_new(*v) as PyObjectRef))
            }
            BlackholeResult::DoneWithThisFrameRef(r) => Some(Ok(*r)),
            BlackholeResult::DoneWithThisFrameFloat(f) => {
                Some(Ok(pyre_object::floatobject::w_float_new(*f) as PyObjectRef))
            }
            // warmspot.py:998-1005: raise the exception
            BlackholeResult::ExitFrameWithExceptionRef(err) => Some(Err(err.clone())),
            _ => None,
        }
    }
}

/// resume.py:1042 rebuild_from_numbering / read_jitcode_pos_pc output.
/// Each decoded frame section from rd_numb.
pub struct ResumedFrame {
    /// resume.py:1050 jitcode_pos → jitcodes[jitcode_pos].
    /// W_CodeObject pointer — same level as frame.pycode / getcode(func).
    pub code: *const (),
    /// resume.py:1050 pc (Python bytecode PC for blackhole setposition).
    pub py_pc: usize,
    /// Raw frame.pc from rd_numb (= orgpc from snapshot).
    /// Some(pc): snapshot guard — orgpc known, liveness-based filling.
    ///   pc=0 is valid (function start / loop header at bytecode 0).
    /// None: no-snapshot guard (rd_numb pc=-1), positional fallback.
    pub rd_numb_pc: Option<usize>,
    /// CHAIN virtualizable pointer (same value on every section).
    /// RPython parity: there is ONE virtualizable per jitdriver_sd for the
    /// whole blackhole chain; inner sections do not own a separate PyFrame.
    /// Carried on every `ResumedFrame` only because pyre pre-decodes the
    /// rd_numb stream into a `Vec<ResumedFrame>` instead of streaming it
    /// like RPython's `blackhole_from_resumedata` — the value MUST be
    /// identical across sections (enforced by `build_resumed_frames`).
    pub frame_ptr: *mut PyFrame,
    /// valuestackdepth extracted from vable_values (snapshot).
    pub vsd: usize,
    /// interp_jit.py:31 w_globals — namespace pointer from vable_values.
    /// virtualizable.py:126-137 write_from_resume_data_partial:
    /// ALL static fields come from resume data, not from the heap.
    pub namespace: *const (),
    /// resume.py:928-931 consume_one_section: resolved values.
    /// Structure: [live_registers...] — no scalar inputarg header.
    /// RPython parity: vable values come from snapshot, not fail_args.
    pub values: Vec<majit_ir::Value>,
}

/// RPython resume.py:1312 blackhole_from_resumedata +
/// blackhole.py:1752 _run_forever parity.
///
/// Takes rd_numb-decoded per-frame data (ResumedFrame) instead of a
/// flat Value array. Frame boundaries come from rd_numb, not from
/// heuristic header detection.
///
/// Builds a blackhole chain (innermost first), then runs _run_forever:
/// callee blackhole → RETURN_VALUE → caller blackhole → merge point.
pub fn resume_in_blackhole(
    _caller_frame: &mut PyFrame,
    frames: &[ResumedFrame],
) -> BlackholeResult {
    let nbody_debug = std::env::var_os("PYRE_NBODY_DEBUG").is_some();
    if frames.is_empty() {
        if nbody_debug {
            eprintln!("[nbody-debug] resume_in_blackhole failed: empty frames");
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!("[jit][bh-fail] resume_in_blackhole: empty frames");
        }
        return BlackholeResult::Failed;
    }

    thread_local! {
        // RPython `blackhole.py:55-65` constructs a fully-initialized
        // `BlackholeInterpBuilder` (setup_insns + setup_descrs +
        // wire_bhimpl_handlers) at metainterp startup; pyre's strict
        // analogue is `build_default_bh_builder()`.  Earlier this slot
        // held a bare `BlackholeInterpBuilder::new()` whose
        // `dispatch_table` was empty, so every acquired interpreter
        // routed every byte through `dispatch_step_with_fallback` to
        // the legacy `dispatch_one`.
        static BH_BUILDER3: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(pyre_jit_trace::jitcode_runtime::build_pyre_production_bh_builder());
    }
    let sync_bh_builder_control_opcodes =
        |builder: &mut majit_metainterp::blackhole::BlackholeInterpBuilder| {
            let (op_live, op_catch_exception, op_rvmprof_code) =
                pyre_jit_trace::state::blackhole_control_opcodes();
            builder.setup_cached_control_opcodes(op_live, op_catch_exception, op_rvmprof_code);
        };
    // Helper closures that scope the &mut to a single call so that
    // bh.run() (which may re-enter resume_in_blackhole through
    // bh_call_fn_impl → eval_with_jit → guard failure) cannot create
    // overlapping &mut references to the same thread-local pool.
    let acquire_bh = || -> majit_metainterp::blackhole::BlackholeInterpreter {
        BH_BUILDER3.with(|cell| unsafe {
            let builder = &mut *cell.get();
            sync_bh_builder_control_opcodes(builder);
            builder.acquire_interp()
        })
    };
    let release_bh = |bh: majit_metainterp::blackhole::BlackholeInterpreter| {
        BH_BUILDER3.with(|cell| unsafe { (&mut *cell.get()).release_interp(bh) });
    };
    let release_chain_bh = |chain: Option<majit_metainterp::blackhole::BlackholeInterpreter>| {
        BH_BUILDER3.with(|cell| unsafe { (&mut *cell.get()).release_chain(chain) });
    };

    // resume.py:1333-1343 blackhole_from_resumedata:
    // RPython iterates the resume reader in stream order. After
    // `opencoder.py:217` `framestack.reverse()` parity the rd_numb
    // stream is outermost-first, so feeding `frames` forward produces
    // exactly the upstream chain — each new bh's `nextblackholeinterp`
    // points to the previously-built (more outer) caller, and the
    // final `prev_bh` is the innermost frame the runner executes.
    let mut prev_bh: Option<majit_metainterp::blackhole::BlackholeInterpreter> = None;

    // pyjitpl.py:2264: metainterp_sd.liveness_info — one shared pool for
    // every jitcode. Snapshot once per call so per-section enumerate_vars
    // borrows a stable slice.
    let all_liveness = pyre_jit_trace::state::liveness_info_snapshot();

    // resume.py:1333-1343 parity: virtualizable_ptr is chain-level (one
    // for the whole blackhole chain). RPython doesn't have a per-frame
    // frame_ptr — the `vable` argument is passed to each `bhimpl_*_vable_*`
    // bytecode explicitly. pyre carries the same virtualizable pointer on
    // every `ResumedFrame` (enforced by build_resumed_frames), so we read
    // it once from the first section for the whole chain.
    let chain_vable_ptr = frames
        .first()
        .map(|f| f.frame_ptr)
        .unwrap_or(std::ptr::null_mut());
    if chain_vable_ptr.is_null() {
        if nbody_debug {
            eprintln!("[nbody-debug] resume_in_blackhole failed: chain virtualizable is null");
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!("[jit][bh-fail] resume_in_blackhole: chain virtualizable is null",);
        }
        return BlackholeResult::Failed;
    }
    // Enforce the invariant: every section carries the same chain vable.
    debug_assert!(
        frames.iter().all(|f| f.frame_ptr == chain_vable_ptr),
        "ResumedFrame.frame_ptr must be identical across sections (chain virtualizable)"
    );

    for (sec_idx, section) in frames.iter().enumerate() {
        if section.code.is_null() {
            if nbody_debug {
                eprintln!(
                    "[nbody-debug] resume_in_blackhole failed: null code sec={} py_pc={}",
                    sec_idx, section.py_pc,
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bh-fail] resume_in_blackhole: null code at sec={} py_pc={}",
                    sec_idx, section.py_pc,
                );
            }
            release_chain_bh(prev_bh);
            return BlackholeResult::Failed;
        }
        let code = unsafe {
            &*(pyre_interpreter::w_code_get_ptr(section.code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject)
        };
        let nlocals = code.varnames.len();
        let frame_ptr = chain_vable_ptr;

        // resume.py:1340 curbh.setposition(jitcode, pc)
        let mut py_pc = section.py_pc;
        // Skip Cache/ExtendedArg/NotTaken (CPython 3.13 pseudo-instructions).
        while py_pc > 0 {
            match pyre_interpreter::decode_instruction_at(code, py_pc) {
                Some((pyre_interpreter::bytecode::Instruction::Cache, _))
                | Some((pyre_interpreter::bytecode::Instruction::ExtendedArg, _))
                | Some((pyre_interpreter::bytecode::Instruction::NotTaken, _)) => {
                    py_pc -= 1;
                }
                _ => break,
            }
        }
        if py_pc >= code.instructions.len() {
            if nbody_debug {
                eprintln!(
                    "[nbody-debug] resume_in_blackhole failed: py_pc out of bounds sec={} py_pc={} instr_len={}",
                    sec_idx,
                    py_pc,
                    code.instructions.len()
                );
            }
            release_chain_bh(prev_bh);
            return BlackholeResult::Failed;
        }

        // RPython parity: vsd from vable_values (snapshot), stored in
        // ResumedFrame.vsd by build_resumed_frames.
        let vsd = section.vsd;
        // call.py:148: jitcode via jitcodes dict lookup (jitdriver_sd
        // set on portal). virtualizable.py:126-137: code from resume
        // data, not heap. Lookup-only: trace setup already compiled.
        let pyjitcode = match pyre_jit_trace::state::pyjitcode_for_code(section.code) {
            Some(pjc) => pjc,
            None => {
                if nbody_debug {
                    eprintln!(
                        "[nbody-debug] resume_in_blackhole failed: find_jitcode miss sec={} code={:#x} py_pc={}",
                        sec_idx, section.code as usize, py_pc
                    );
                }
                release_chain_bh(prev_bh);
                return BlackholeResult::Failed;
            }
        };
        let jitcode_pc = if let Some(jitcode_pc) = pyjitcode.metadata.pc_map.get(py_pc).copied() {
            jitcode_pc
        } else {
            if nbody_debug {
                eprintln!(
                    "[nbody-debug] resume_in_blackhole failed: pc_map miss sec={} py_pc={} pc_map_len={}",
                    sec_idx,
                    py_pc,
                    pyjitcode.metadata.pc_map.len()
                );
            }
            release_chain_bh(prev_bh);
            return BlackholeResult::Failed;
        };

        let mut bh = acquire_bh();
        bh.setposition(pyjitcode.jitcode.clone(), jitcode_pc);
        // blackhole.py:1095-1099 get_portal_runner parity:
        //   jitdriver_sd = self.builder.metainterp_sd.jitdrivers_sd[jdindex]
        //   fnptr        = adr2int(jitdriver_sd.portal_runner_adr)
        //   calldescr    = jitdriver_sd.mainjitcode.calldescr
        // pyre publishes the single Pyre jitdriver at jdindex 0 with
        // result_type Ref (the portal returns a PyObject Ref).
        bh.jitdrivers_sd = vec![majit_metainterp::blackhole::BhJitDriverSd {
            result_type: majit_metainterp::blackhole::BhReturnType::Ref,
            portal_runner_ptr: Some(bh_portal_runner),
            mainjitcode_calldescr: pyjitcode.jitcode.calldescr.clone(),
        }];

        // RPython warmspot.py: jitcode.fnaddr = getfunctionptr(graph).
        // pyre: all Python functions go through the single portal runner.
        // Set per-jitcode fnaddr on both the jitcode itself and its descrs.
        // After Phase A's `Arc<JitCode>` migration `bh.jitcode` is shared
        // with the originating `PyJitCode`; `Arc::make_mut` clones-on-write
        // so the fnaddr override does not leak back into the cached entry.
        std::sync::Arc::make_mut(&mut bh.jitcode).fnaddr =
            bh_portal_runner as *const () as usize as i64;
        // RPython: descrs carry FieldDescr.offset (byte offset from rtyper).
        // pyre: field offsets are resolved from Rust struct layout at runtime.
        bh.resolve_field_offsets(resolve_field_offset);
        bh.resolve_jitcode_fnaddrs(|_jitcode_index| {
            // RPython: each callee has its own fnaddr from getfunctionptr().
            // pyre: single-portal architecture — all callees share the same
            // portal_runner address. The blackhole's inline_call reads fnaddr
            // from BhDescr::JitCode and calls bh_call_*(fnaddr, ...).
            bh_portal_runner as *const () as usize as i64
        });

        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][blackhole-section] idx={} frame={:#x} py_pc={} vsd={} nvals={} vals={:?}",
                sec_idx,
                frame_ptr as usize,
                py_pc,
                vsd,
                section.values.len(),
                section.values.iter().take(5).collect::<Vec<_>>(),
            );
        }

        // resume.py:1381-1384 consume_one_section:
        //     info = blackholeinterp.get_current_position_info()
        //     self._prepare_next_section(info)
        //
        // _prepare_next_section → enumerate_vars(info, all_liveness,
        //     callback_i, callback_r, callback_f)
        //
        // Each callback: value = self.next_TYPE(); write_a_TYPE(reg, value)
        // Values are consumed in order: all ints, then all refs, then floats.
        assert!(
            !all_liveness.is_empty(),
            "resume_in_blackhole: missing liveness_info for jitcode at py_pc={} jit_pc={}",
            section.py_pc,
            bh.position
        );
        // jitcode.py:82 get_live_vars_info(pc, op_live)
        let liveness_offset = bh.get_current_position_info();
        // jitcode.py:146-167 enumerate_vars: collect live register indices
        let mut live_i: Vec<u32> = Vec::new();
        let mut live_r: Vec<u32> = Vec::new();
        let mut live_f: Vec<u32> = Vec::new();
        majit_metainterp::jitcode::enumerate_vars(
            liveness_offset,
            &all_liveness,
            |idx| live_i.push(idx),
            |idx| live_r.push(idx),
            |idx| live_f.push(idx),
        );
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][consume_one_section] py_pc={} jit_pc={} live_i={:?} live_r={:?} live_f={:?} vals={}",
                section.py_pc,
                bh.position,
                live_i,
                live_r,
                live_f,
                section.values.len(),
            );
        }
        // Phase 0 probe (Tasks #158/#159/#122 epic, plan
        // ~/.claude/plans/staged-sauteeing-koala.md): when
        // MAJIT_PROBE_LIVENESS env is set, log the per-ref-bank
        // (reg_idx → section.values[k]) mapping plus null/concrete
        // status. Goal P0-Q1: distinguish "trace export missing this
        // value" (section.values short / NULL entry) from "BH dispatch
        // can't find it" (later read-side issue). Default: off.
        let probe_liveness = std::env::var_os("MAJIT_PROBE_LIVENESS").is_some();
        if probe_liveness {
            eprintln!(
                "[probe-A][consume_one_section] jitcode={} py_pc={} jit_pc={} live_i={:?} live_r={:?} live_f={:?} section.values.len={}",
                pyjitcode.jitcode.name,
                section.py_pc,
                bh.position,
                live_i,
                live_r,
                live_f,
                section.values.len(),
            );
        }
        // resume.py:1017-1026 _prepare_next_section callbacks:
        // _callback_i → next_int() → write_an_int(register_index, value)
        // _callback_r → next_ref() → write_a_ref(register_index, value)
        // _callback_f → next_float() → write_a_float(register_index, value)
        let mut val_idx = 0;
        for &reg_idx in &live_i {
            if let Some(val) = section.values.get(val_idx) {
                bh.setarg_i(reg_idx as usize, materialize_virtual_int(val));
            }
            val_idx += 1;
        }
        let probe_bh_startup = std::env::var("PYRE_PROBE_BH_STARTUP").ok().as_deref() == Some("1");
        for &reg_idx in &live_r {
            if let Some(val) = section.values.get(val_idx) {
                let materialized = materialize_virtual(val);
                if probe_liveness {
                    eprintln!(
                        "[probe-A][ref] reg_idx={} val_idx={} raw={:?} materialized=0x{:x} is_null={}",
                        reg_idx,
                        val_idx,
                        val,
                        materialized as usize,
                        materialized == 0,
                    );
                }
                if probe_bh_startup && (reg_idx as usize) < nlocals {
                    eprintln!(
                        "[PROBE-BH-START][setarg_r] sec={} py_pc={} reg_idx={} (LOCAL) val_idx={} materialized=0x{:x}",
                        sec_idx, py_pc, reg_idx, val_idx, materialized as usize
                    );
                }
                bh.setarg_r(reg_idx as usize, materialized);
            } else if probe_liveness {
                eprintln!(
                    "[probe-A][ref] reg_idx={} val_idx={} OUT-OF-BOUNDS section.values.len={}",
                    reg_idx,
                    val_idx,
                    section.values.len(),
                );
            }
            val_idx += 1;
        }
        for &reg_idx in &live_f {
            if let Some(val) = section.values.get(val_idx) {
                bh.setarg_f(reg_idx as usize, materialize_virtual_float(val));
            }
            val_idx += 1;
        }
        // blackhole.py bhimpl_getfield_vable_*: set virtualizable pointer.
        bh.virtualizable_ptr = frame_ptr as i64;
        bh.virtualizable_info = crate::eval::get_virtualizable_info();
        bh.virtualizable_stack_base = nlocals + pyre_interpreter::pyframe::ncells(code);
        // Portal red-arg registers must be filled AFTER virtualizable_ptr.
        let ec = unsafe { (*frame_ptr).execution_context as i64 };
        bh.fill_portal_registers(
            pyjitcode.metadata.portal_frame_reg,
            pyjitcode.metadata.portal_ec_reg,
            ec,
        );

        // resume.py:1342 `curbh.handle_rvmprof_enter()` — runs the rvmprof
        // `entering=0` hook immediately after consume_one_section. For the
        // generic `blackhole_from_resumedata` path majit-metainterp already
        // invokes this at resume.rs:5824; the pyre-local chain builder sits
        // on a parallel code path and must replay the same step.
        bh.handle_rvmprof_enter();

        // PHASE 1.4 candidate D probe (BH startup): immediately after
        // bh.registers_r is populated and before any BH op runs, compare
        // per-local `bh.registers_r[i]` to heap PyFrame.locals_w[i] (==
        // `vable.values[i]`). The two SHOULD be identical — they both
        // represent the same logical local at the guard PC. Divergence
        // here means the JIT-compiled trace's dual-write (heap +
        // register-file) was broken by the optimizer/backend, OR the
        // build_resumed_frames materialization picked a stale dead_frame
        // slot for a register that the heap had moved past.
        if std::env::var("PYRE_PROBE_BH_STARTUP").ok().as_deref() == Some("1") {
            let vinfo_ptr = bh.virtualizable_info;
            if !vinfo_ptr.is_null() && bh.virtualizable_ptr != 0 {
                let vinfo = unsafe { &*vinfo_ptr };
                if !vinfo.array_fields.is_empty() {
                    let ainfo = &vinfo.array_fields[0];
                    let scan_len = nlocals.min(bh.registers_r.len());
                    for i in 0..scan_len {
                        let reg_val = bh.registers_r[i];
                        let vable_val = unsafe {
                            majit_metainterp::virtualizable::vable_read_array_item(
                                bh.virtualizable_ptr as *const u8,
                                ainfo,
                                i,
                            )
                        };
                        if reg_val != vable_val {
                            eprintln!(
                                "[PROBE-BH-START] sec={} py_pc={} local={} reg=0x{:x} vable=0x{:x} MISMATCH",
                                sec_idx, py_pc, i, reg_val, vable_val
                            );
                        }
                    }
                }
            }
        }

        // RPython: nextbh.nextblackholeinterp = curbh
        bh.nextblackholeinterp = prev_bh.map(Box::new);
        prev_bh = Some(bh);
    }

    let Some(mut bh) = prev_bh else {
        if nbody_debug {
            eprintln!("[nbody-debug] resume_in_blackhole failed: empty blackhole chain");
        }
        return BlackholeResult::Failed;
    };

    if majit_metainterp::majit_log_enabled() {
        eprintln!("[jit][blackhole-resume] chain_len={}", frames.len(),);
    }

    // RPython blackhole.py:1752 _run_forever parity:
    // Run the innermost blackhole. On RETURN_VALUE (LeaveFrame),
    // pop to caller blackhole and continue.
    loop {
        if let Some(args) = bh.run() {
            // blackhole.py:1068: raise ContinueRunningNormally(*args)
            // Propagated from run() as RPython's JitException equivalent.
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;

            let mut red_ref: Vec<PyObjectRef> =
                args.red_ref.iter().map(|&v| v as PyObjectRef).collect();
            if red_ref.is_empty() {
                red_ref.push(frame_ptr as PyObjectRef);
            }
            return BlackholeResult::ContinueRunningNormally {
                green_int: args.green_int,
                green_ref: args.green_ref.iter().map(|&v| v as PyObjectRef).collect(),
                green_float: args
                    .green_float
                    .iter()
                    .map(|&v| f64::from_bits(v as u64))
                    .collect(),
                red_int: args.red_int,
                red_ref,
                red_float: args
                    .red_float
                    .iter()
                    .map(|&v| f64::from_bits(v as u64))
                    .collect(),
            };
        }

        // BC_ABORT: unsupported bytecode hit during execution.
        if bh.aborted {
            if nbody_debug {
                eprintln!(
                    "[nbody-debug] resume_in_blackhole failed: bh.aborted position={} last_opcode_position={}",
                    bh.position, bh.last_opcode_position
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][blackhole] ABORT at jitcode_pc={} last_opcode_pos={}",
                    bh.position, bh.last_opcode_position
                );
            }
            release_bh(bh);
            return BlackholeResult::Failed;
        }

        if bh.got_exception {
            if majit_metainterp::majit_log_enabled() {
                let jitcode_bytes = &bh.jitcode.code;
                let opcode_at = if bh.last_opcode_position < jitcode_bytes.len() {
                    jitcode_bytes[bh.last_opcode_position]
                } else {
                    255
                };
                // Try to get fn_ptr info for call opcodes
                let fn_info = if bh.last_opcode_position + 2 < jitcode_bytes.len() {
                    let fn_idx = u16::from_le_bytes([
                        jitcode_bytes[bh.last_opcode_position + 1],
                        jitcode_bytes[bh.last_opcode_position + 2],
                    ]) as usize;
                    match bh.jitcode.exec.descrs.get(fn_idx) {
                        Some(majit_metainterp::jitcode::RuntimeBhDescr::Call(target)) => {
                            format!("fn_ptr={:#x}", target.concrete_ptr as usize)
                        }
                        Some(other) => format!("descrs[{fn_idx}]={other:?}"),
                        None => format!("fn_idx={fn_idx} (out of range)"),
                    }
                } else {
                    String::new()
                };
                eprintln!(
                    "[jit][blackhole] EXCEPTION at jitcode_pc={} last_opcode_pos={} opcode={} {}",
                    bh.position, bh.last_opcode_position, opcode_at, fn_info
                );
            }
        }

        // blackhole.py:1752 _run_forever exception propagation:
        // Exception not handled in this frame (no handler found by
        // dispatch_one's handle_exception_in_frame). Propagate to caller.
        if bh.got_exception {
            let exc_value = bh.exception_last_value;
            let next = bh.nextblackholeinterp.take();
            release_bh(bh);

            let Some(mut caller_bh) = next.map(|b| *b) else {
                if nbody_debug || majit_metainterp::majit_log_enabled() {
                    let kind_dbg = if exc_value != 0 {
                        let raw = unsafe {
                            pyre_object::excobject::w_exception_get_kind(
                                exc_value as pyre_object::PyObjectRef,
                            )
                        };
                        format!("{:?}", raw)
                    } else {
                        "<null>".to_string()
                    };
                    eprintln!(
                        "[bh-fail-1079] uncaught exception at outermost exc_value={:#x} kind={}",
                        exc_value as usize, kind_dbg
                    );
                }
                // blackhole.py:1679-1682 _exit_frame_with_exception:
                //   e = cast_opaque_ptr(GCREF, e)
                //   raise ExitFrameWithExceptionRef(e)
                //
                // Known parity gap. Verified on 2026-04-20: converting
                // exc_value with PyError::from_exc_object and returning
                // ExitFrameWithExceptionRef breaks raise_catch_loop /
                // nbody_50k / spectral_norm / fannkuch with a spurious
                // "call on non-function callable" TypeError. The
                // exc_value pointer survives from an earlier
                // bh_call_fn_impl setting BH_LAST_EXC_VALUE that the
                // blackhole never cleared even though the residual
                // call's exception was handled in-frame. Returning
                // Failed triggers eval.rs:1594 rollback which
                // reinterprets the bytecode correctly. The real fix
                // lives in the BH side: clear exception_last_value on
                // handle_exception_in_frame + reset the caller
                // BH_LAST_EXC_VALUE thread-local once the exception is
                // consumed. Tracked by Task #122 (rd_numb resume
                // unification), blocked by Task #158 (register-layout
                // refactor) + Task #159 (liveness pipeline rework).
                return BlackholeResult::Failed;
            };

            // blackhole.py:396 handle_exception_in_frame in caller.
            // Ensure last_opcode_position reflects the caller's call-site PC.
            // The caller may not have run any opcodes in this blackhole session
            // (it was suspended), so last_opcode_position must match position.
            caller_bh.last_opcode_position = caller_bh.position;
            if caller_bh.handle_exception_in_frame(exc_value) {
                bh = caller_bh;
                continue;
            }

            // No handler in caller: propagate further up.
            caller_bh.exception_last_value = exc_value;
            caller_bh.got_exception = true;
            bh = caller_bh;
            continue;
        }

        // blackhole.py:1632-1644: pass return value to caller by _return_type.
        use majit_metainterp::blackhole::BhReturnType;
        let rt = bh.return_type;
        let next = bh.nextblackholeinterp.take();

        let Some(mut caller_bh) = next.map(|b| *b) else {
            // blackhole.py:1664-1677 _done_with_this_frame
            let result = match rt {
                BhReturnType::Void => BlackholeResult::DoneWithThisFrameVoid,
                BhReturnType::Int => BlackholeResult::DoneWithThisFrameInt(bh.get_tmpreg_i()),
                BhReturnType::Ref => {
                    BlackholeResult::DoneWithThisFrameRef(bh.get_tmpreg_r() as PyObjectRef)
                }
                BhReturnType::Float => BlackholeResult::DoneWithThisFrameFloat(f64::from_bits(
                    bh.get_tmpreg_f() as u64,
                )),
            };
            return result;
        };

        // blackhole.py:1637-1644: dispatch by _return_type
        match rt {
            BhReturnType::Int => caller_bh.setup_return_value_i(bh.get_tmpreg_i()),
            BhReturnType::Ref => caller_bh.setup_return_value_r(bh.get_tmpreg_r()),
            BhReturnType::Float => caller_bh.setup_return_value_f(bh.get_tmpreg_f()),
            BhReturnType::Void => {}
        }

        bh = caller_bh;
    }
}

/// resume.py:945-956 decode_ref / getvirtual_ptr parity.
///
/// Re-box optimizer-unboxed values back to PyObjectRef for the
/// blackhole's ref register file. RPython's decode_ref dispatches
/// on TAGVIRTUAL/TAGCONST/TAGBOX/TAGSMALLINT; pyre's deadframe
/// already contains typed Values, so we just box Int/Float to
/// W_IntObject/W_FloatObject.
fn materialize_virtual(val: &majit_ir::Value) -> i64 {
    use majit_ir::Value;
    match val {
        Value::Ref(r) => r.as_usize() as i64,
        Value::Int(v) => pyre_object::intobject::w_int_new(*v) as i64,
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v) as i64,
        Value::Void => 0i64,
    }
}

/// resume.py:1028 _callback_i → next_int() → write_an_int.
/// RPython trusts type discipline — no cross-type coercion.
fn materialize_virtual_int(val: &majit_ir::Value) -> i64 {
    match val {
        majit_ir::Value::Int(v) => *v,
        other => panic!("materialize_virtual_int: expected Int, got {:?}", other),
    }
}

/// resume.py:1036 _callback_f → next_float() → write_a_float.
/// RPython trusts type discipline — no cross-type coercion.
fn materialize_virtual_float(val: &majit_ir::Value) -> i64 {
    match val {
        majit_ir::Value::Float(v) => v.to_bits() as i64,
        other => panic!("materialize_virtual_float: expected Float, got {:?}", other),
    }
}

/// Fused recursive call with boxed arg.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_1(
    caller_frame: i64,
    callable: i64,
    boxed_arg: i64,
) -> i64 {
    let boxed_arg_ref = boxed_arg as PyObjectRef;
    // result_type=REF: no RawInt unbox needed — arg is already boxed Ref
    if majit_metainterp::majit_log_enabled() {
        let caller = unsafe { &*(caller_frame as *const PyFrame) };
        let caller_arg0 = if caller.locals_w().len() > 0
            && !caller.locals_w()[0].is_null()
            && unsafe { is_int(caller.locals_w()[0]) }
        {
            Some(unsafe { w_int_get_value(caller.locals_w()[0]) })
        } else {
            None
        };
        let boxed = boxed_arg as PyObjectRef;
        let callee_arg0 = if !boxed.is_null() && unsafe { is_int(boxed) } {
            Some(unsafe { w_int_get_value(boxed) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-recursive-boxed] enter caller_arg0={:?} callee_arg0={:?}",
            caller_arg0, callee_arg0
        );
    }
    let frame_ptr = create_callee_frame_impl(caller_frame, callable, &[boxed_arg_ref]);
    let result = jit_force_callee_frame(frame_ptr);
    jit_drop_callee_frame(frame_ptr);
    if majit_metainterp::majit_log_enabled() {
        let caller = unsafe { &*(caller_frame as *const PyFrame) };
        let caller_arg0 = if caller.locals_w().len() > 0
            && !caller.locals_w()[0].is_null()
            && unsafe { is_int(caller.locals_w()[0]) }
        {
            Some(unsafe { w_int_get_value(caller.locals_w()[0]) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-recursive-boxed] exit caller_arg0={:?}",
            caller_arg0
        );
    }
    result
}

/// Fused recursive call with RAW INT arg, boxed result.
///
/// This keeps the trace-side argument in raw-int form even before the callee
/// has stabilized on a raw-int finish protocol. It is a closer match to
/// RPython's recursive portal argument flow than boxing the argument in the
/// trace before every helper-boundary call.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_argraw_boxed_1(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    // result_type=REF: box the int arg, dispatch as boxed Ref
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    jit_force_recursive_call_1(caller_frame, callable, boxed as i64)
}

/// Self-recursive single-arg boxed helper.
///
/// Keeps the boxed helper path off the generic callable redispatch and
/// blackhole fallback route. This mirrors the specialized raw helper:
/// the callee frame is created directly from the caller's code/globals.
/// RPython warmspot.py:941 portal_runner parity.
///
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_1(caller_frame: i64, boxed_arg: i64) -> i64 {
    let boxed_arg_ref = boxed_arg as PyObjectRef;
    if caller_frame == 0 {
        return boxed_arg;
    }
    // result_type=REF: arg is already boxed Ref
    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed_arg_ref);
    // blackhole.py:1101-1132 bhimpl_recursive_call_r: calls
    // cpu.bh_call_r(portal_runner_adr, ...) which re-enters JIT.
    // warmspot.py:941 ll_portal_runner: maybe_compile_and_run + portal_ptr.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        crate::eval::portal_runner(frame) as i64
    };
    jit_drop_callee_frame(frame_ptr);
    result
}

/// Self-recursive single-arg helper with raw-int arg and boxed result.
///
/// Mirrors `jit_force_self_recursive_call_1`, but keeps the trace-side
/// argument unboxed so recursive helper-boundary calls do not allocate a
/// temporary `W_Int` in the trace.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_argraw_boxed_1(
    caller_frame: i64,
    raw_int_arg: i64,
) -> i64 {
    // result_type=REF: box the int arg, dispatch as boxed Ref
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    jit_force_self_recursive_call_1(caller_frame, boxed as i64)
}

/// Fully fused recursive call with RAW INT arg — no boxing in trace at all.
///
/// Eliminates ALL per-recursive-call overhead from trace:
///   Before: CallI(box) + CallMayForce(force_1, frame, callable, boxed)
///   After:  CallMayForce(force_raw_1, frame, callable, raw_int)
///
/// Boxing happens inside this function, not in the trace.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_raw_1(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    let callable_ref = callable as PyObjectRef;

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_callee_frame_impl_1_boxed(caller_frame, callable_ref, boxed);
    // blackhole.py:1101-1116 bhimpl_recursive_call_r: a recursive call
    // from compiled assembler is `cpu.bh_call_r(portal_runner_adr, ...)`
    // — i.e. it always re-enters through the portal runner. The portal
    // runner (warmspot.py:944-953) calls `maybe_compile_and_run` and
    // then `portal_ptr(*args)`, so the JIT-vs-interpreter decision is
    // made there. There is no "try blackhole first, then fallback to
    // portal_runner" path in RPython.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        // warmspot.py:449 result_type=REF: always boxed Ref
        crate::eval::portal_runner(frame) as i64
    };
    jit_drop_callee_frame(frame_ptr);
    result
}

/// Specialized raw-int recursive helper for closure-free self-recursion.
///
/// Unlike `jit_force_recursive_call_raw_1`, this does not need to rediscover
/// the callee's code/globals from a function object on every call. The caller
/// frame already carries the exact recursive target:
/// - `caller.pycode` is the callee code object
/// - `caller.w_globals` is the module globals
/// - `caller.execution_context` is the shared execution context
///
/// Trace-time recursive CALL_ASSEMBLER handles the optimized path. The
/// concrete helper should mirror RPython's force_fn behavior: execute the
/// callee's own frame without JIT on that frame, but let nested portal
/// calls re-enter compiled code through the normal portal runner path.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_raw_1(caller_frame: i64, raw_int_arg: i64) -> i64 {
    if majit_metainterp::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!("[jit][force-self-recursive] enter arg={}", raw_int_arg);
    }
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let w_code = caller.pycode;
    let green_key = crate::eval::make_green_key(w_code, 0);
    let _token_num = self_recursive_dispatch(green_key);

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed);
    // blackhole.py:1110-1116 bhimpl_recursive_call_r: calls
    // cpu.bh_call_r(portal_runner_adr, ...) which invokes
    // warmspot.py:941 ll_portal_runner. portal_runner re-enters
    // the JIT through maybe_compile_and_run + portal_ptr.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        // warmspot.py:449 result_type=REF: always boxed Ref
        crate::eval::portal_runner(frame) as i64
    };
    jit_drop_callee_frame(frame_ptr);
    if majit_metainterp::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!(
            "[jit][force-self-recursive] exit arg={} result={}",
            raw_int_arg, result
        );
    }
    result
}

/// Dynasm x86/assembler.py:347-390 `_build_stack_check_slowpath` parity.
///
/// The interpreter-level slowpath stores the RecursionError in
/// `stack_check::JIT_PENDING_EXCEPTION` for non-dynasm backend glue. Dynasm's
/// prologue, however, mirrors RPython's `pos_exception()` path: after this
/// call returns non-zero, emitted code transfers `majit_backend_dynasm`'s
/// `JIT_EXC_VALUE` into `jf_guard_exc` and stamps `propagate_exception_descr`
/// into `jf_descr`. Bridge the two exception slots here.
#[cfg(feature = "dynasm")]
extern "C" fn dynasm_stack_check_slowpath_for_backend(current: usize) -> u8 {
    let result = pyre_interpreter::stack_check::pyre_stack_check_slowpath_for_backend(current);
    if result != 0 {
        if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            majit_backend_dynasm::jit_exc_raise(exc.to_exc_object() as i64);
        }
    }
    result
}

/// Unbox a Ref (PyObjectRef to boxed int) to a raw i64 value.
/// Used by call_assembler_guard_failure's FALLBACK path when the first
/// local is a Ref type (boxed int) instead of raw Int.
fn unbox_int_for_force(raw: i64) -> i64 {
    let obj = raw as pyre_object::PyObjectRef;
    if !obj.is_null() && unsafe { is_int(obj) } {
        unsafe { w_int_get_value(obj) }
    } else {
        raw
    }
}

/// resume.py:763-779 VStrPlainInfo.allocate / resume.py:817-829
/// VUniPlainInfo.allocate parity — materialize a Plain string/unicode
/// virtual via the frontend backend's bh_newstr / bh_strsetitem (and
/// unicode variants). Registered into Cranelift's guard-exit recovery
/// path so `rebuild_state_after_failure` hands bridge-input refs a real
/// string pointer instead of NULL (compiler.rs:1323).
fn materialize_str_plain_for_cranelift(is_unicode: bool, chars: &[i64]) -> i64 {
    use majit_backend::Backend;
    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let length = chars.len() as i64;
    let string = if is_unicode {
        backend.bh_newunicode(length)
    } else {
        backend.bh_newstr(length)
    };
    for (i, c) in chars.iter().enumerate() {
        if is_unicode {
            backend.bh_unicodesetitem(string, i as i64, *c);
        } else {
            backend.bh_strsetitem(string, i as i64, *c);
        }
    }
    string
}

/// resume.py:1143-1188 string_concat / slice_string and the unicode
/// counterparts — materialize Concat / Slice string virtuals via
/// cpu.bh_call_r(funcptr, args_i, args_r, args_f, calldescr).
fn materialize_str_call_for_cranelift(
    _is_unicode: bool,
    func: i64,
    calldescr: &majit_ir::DescrRef,
    args_i: &[i64],
    args_r: &[i64],
) -> i64 {
    use majit_backend::Backend;
    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let cd = calldescr
        .as_call_descr()
        .expect("materialize_str_call: calldescr must downcast to CallDescr");
    let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
    let result = backend.bh_call_r(
        func,
        if args_i.is_empty() {
            None
        } else {
            Some(args_i)
        },
        if args_r.is_empty() {
            None
        } else {
            Some(args_r)
        },
        None,
        &bh_calldescr,
    );
    result.0 as i64
}

pub fn install_jit_call_bridge() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        register_jit_function_caller(jit_call_user_function_from_frame);
        // compile.py:1090 `memory_error = MemoryError()` parity — give
        // the backend malloc helpers a way to set `JIT_EXC_VALUE` to
        // pyre's lazy `W_ExceptionObject(MemoryError, "")` singleton
        // before propagating NULL on OOM.  Backend-shared (mirrors
        // RPython where the same `memory_error` instance is reachable
        // from both the x86 and aarch64 backends).
        majit_backend::register_memory_error_provider(|| {
            pyre_object::excobject::memory_error_singleton() as i64
        });
        // rpython/translator/c/src/stack.h:42-43 LL_stack_criticalcode_start
        // /stop hooks — wrap blackhole_from_resumedata,
        // handle_async_forcing, and handle_guard_failure_in_trace so
        // StackOverflow doesn't interrupt those critical sections.
        // The pyre helpers are `extern "C" fn()`; thin wrappers adapt
        // them to the Rust `fn()` signature register_criticalcode_hooks
        // expects.
        fn criticalcode_start_adapter() {
            pyre_interpreter::stack_check::pyre_stack_criticalcode_start();
        }
        fn criticalcode_stop_adapter() {
            pyre_interpreter::stack_check::pyre_stack_criticalcode_stop();
        }
        majit_metainterp::register_criticalcode_hooks(
            criticalcode_start_adapter,
            criticalcode_stop_adapter,
        );
        // rpython/rlib/rstack.py:75-90 stack_almost_full hook — lets
        // compile.py:702-703 and warmstate.py:430 query the recursion-
        // limit-driven PYRE_STACKTOOBIG budget instead of the OS thread
        // stack.
        fn stack_almost_full_adapter() -> bool {
            pyre_interpreter::stack_check::stack_almost_full()
        }
        majit_metainterp::register_stack_almost_full_hook(stack_almost_full_adapter);
        #[cfg(feature = "cranelift")]
        {
            majit_backend_cranelift::register_call_assembler_force(jit_force_callee_frame);
            majit_backend_cranelift::register_call_assembler_bridge(jit_ca_handle_guard_failure);
            majit_backend_cranelift::register_call_assembler_blackhole(
                jit_blackhole_resume_from_guard,
            );
            majit_backend_cranelift::register_jitframe_layout(arena_global_info());
            majit_backend_cranelift::register_call_assembler_unbox_int(unbox_int_for_force);
            // resume.py:763-870 VStr/VUni.allocate parity — Cranelift
            // backend's materialize_virtual_recursive invokes these
            // callbacks so that bridge-input refs (compiler.rs:2477/2837)
            // and call_assembler blackhole inputs (compiler.rs:3007)
            // receive materialized string pointers, not NULL.
            majit_backend_cranelift::register_materialize_str_plain(
                materialize_str_plain_for_cranelift,
            );
            majit_backend_cranelift::register_materialize_str_call(
                materialize_str_call_for_cranelift,
            );
            // rpython/jit/backend/llsupport/llmodel.py:229-234 insert_stack_check
            // parity. Cranelift consumes these addresses to emit the same
            // load/sub/cmp fast path as dynasm, using a stack-slot address as
            // its current-stack approximation and calling slowpath only on miss.
            majit_backend_cranelift::register_stack_check_addresses(
                pyre_interpreter::stack_check::pyre_stack_get_end_adr(),
                pyre_interpreter::stack_check::pyre_stack_get_length_adr(),
                pyre_interpreter::stack_check::pyre_stack_check_slowpath_for_backend as *const ()
                    as usize,
            );
            majit_backend_cranelift::register_prologue_probe_addr(
                pyre_interpreter::stack_check::pyre_stack_check_for_jit_prologue as *const ()
                    as usize,
            );
        }
        #[cfg(feature = "dynasm")]
        {
            majit_backend_dynasm::register_call_assembler_force(jit_force_callee_frame);
            majit_backend_dynasm::register_call_assembler_bridge(jit_ca_handle_guard_failure);
            majit_backend_dynasm::register_call_assembler_blackhole(
                jit_blackhole_resume_from_guard,
            );
            majit_backend_dynasm::register_jitframe_layout(arena_global_info_dynasm());
            majit_backend_dynasm::register_call_assembler_unbox_int(unbox_int_for_force);
            // rpython/jit/backend/llsupport/llmodel.py:229-234 insert_stack_check
            // parity. The backend inlines MOV [endaddr]; SUB rsp; CMP [lengthaddr]
            // in every JIT prologue and calls slowpath_addr on miss.
            majit_backend_dynasm::register_stack_check_addresses(
                pyre_interpreter::stack_check::pyre_stack_get_end_adr(),
                pyre_interpreter::stack_check::pyre_stack_get_length_adr(),
                dynasm_stack_check_slowpath_for_backend as *const () as usize,
            );
        }
    });
}

/// compile.py:701-716 handle_fail → resume_in_blackhole parity.
///
/// RPython: guard failure always resumes via jitcode-level blackhole
/// (blackhole_from_resumedata → _run_forever). There is no IR-level
/// blackhole in RPython.
///
/// When rd_numb is available, uses ResumeDataDirectReader for exact
/// frame decoding (resume.py:1312 parity).
fn jit_blackhole_resume_from_guard(
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    fail_values_ptr: *const i64,
    num_fail_values: usize,
    raw_deadframe_ptr: *const i64,
    num_raw_deadframe: usize,
) -> Option<i64> {
    // rstack.stack_check_slowpath → _StackOverflow parity: drain the
    // pending JIT-prologue overflow exception when the backend probe
    // tripped. The blackhole resume path is one of the three
    // boundaries the user listed (compiled entry / call_assembler /
    // blackhole resume), so we surface RecursionError here as well as
    // in eval.rs. We do this BEFORE setting up resume state so deep
    // recursion through the blackhole interpreter cannot accumulate
    // further damage.
    if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
        // Stash for the eval loop to surface — same channel the
        // blackhole/force callbacks already use for cross-FFI errors.
        crate::call_jit::set_pending_ca_exception(exc);
        return None;
    }

    if fail_values_ptr.is_null() || num_fail_values == 0 {
        return None;
    }
    let fail_values = unsafe { std::slice::from_raw_parts(fail_values_ptr, num_fail_values) };
    let raw_deadframe = if !raw_deadframe_ptr.is_null() && num_raw_deadframe > 0 {
        unsafe { std::slice::from_raw_parts(raw_deadframe_ptr, num_raw_deadframe) }
    } else {
        fail_values
    };
    // `green_key == 0` is the giveup sentinel passed by the
    // CALL_ASSEMBLER guard-failure helpers (compiler.rs:2674,
    // dynasm/lib.rs:579) when `descr_owning_jct` returned `None`
    // (memmgr-evicted weakref — pyjitpl.py:2898 should-be-rare path).
    //
    // PRE-EXISTING-ADAPTATION (Pyre-only, Python-portal-specific):
    // pyre's resume storage is keyed by `(green_key, trace_id, fail_index)`,
    // so we MUST recover a green_key to look up the storage.  PyPy's
    // `resume_in_blackhole` uses descr identity directly (descr.rd_data),
    // so it has no such recovery problem.
    //
    // The recovery exploits pyre's CALL_ASSEMBLER virtualizable layout
    // `vable_boxes = [frame, ni, code, vsd, ns, locals..., stack...]`
    // (call_jit.rs:2471-2472) — `fail_values[0]` IS the callee's
    // `PyFrame*`, so `frame.pycode` plus `pc=0` reconstructs the
    // entry green_key.  This contract is Python-portal-specific and
    // would NOT hold for a non-virtualizable JIT or a portal whose
    // first fail arg is a scalar.  Convergence: Phase E.3+ unification
    // (Task #235) replaces `(green_key, trace_id, fail_index)` keying
    // with descr identity — at which point this whole recovery block
    // collapses.
    let actual_green_key = if green_key == 0 && num_fail_values >= 1 {
        let frame_ptr = fail_values[0] as *const pyre_interpreter::pyframe::PyFrame;
        if !frame_ptr.is_null() {
            let code = unsafe { (*frame_ptr).pycode };
            crate::eval::make_green_key(code, 0)
        } else {
            green_key
        }
    } else {
        green_key
    };
    let (driver, _) = crate::eval::driver_pair();

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[blackhole-resume] gk={} trace={} fail_idx={} nvals={}",
            actual_green_key, trace_id, fail_index, num_fail_values,
        );
    }

    // --- Path 1: rd_numb-based resume (resume.py:1312 exact parity) ---
    // When rd_numb is present, use ResumeDataDirectReader to decode
    // frame sections precisely, matching RPython blackhole_from_resumedata.
    //
    // compile.py:853 guard-owned `ResumeGuardDescr` storage — share the
    // pool through `Arc<ResumeStorage>` so blackhole resume reads the
    // same `rd_consts` the GC root walker updates. No owned-Vec copy.
    if let Some(storage) = driver.get_resume_storage(actual_green_key, trace_id, fail_index) {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[blackhole-resume] rd_numb len={} rd_consts len={} raw_deadframe len={}",
                storage.rd_numb.len(),
                storage.rd_consts().len(),
                raw_deadframe.len(),
            );
        }
        // resume.py parity: deadframe_types tells decode_ref() whether a
        // TAGBOX slot holds a raw int (needs boxing) or a GcRef (use as-is).
        // Without this, unboxed ints are treated as pointers → SIGSEGV.
        let deadframe_types =
            driver.get_recovery_slot_types(actual_green_key, trace_id, fail_index);
        // resume.py:922 storage.rd_consts: the decoder borrows the shared
        // pool; TAGCONST Ref entries stay visible to `walk_rd_consts_refs`.
        // resume.py:924 _prepare_pendingfields(storage.rd_pendingfields):
        // deferred field writes must be replayed before consume_vref_and_vable.
        let result = blackhole_resume_via_rd_numb(
            &storage.rd_numb,
            storage.rd_consts(),
            raw_deadframe,
            Some(&storage.rd_pendingfields),
            Some(&storage.rd_virtuals),
            deadframe_types.as_deref(),
        );
        return handle_blackhole_result(result, actual_green_key);
    }

    // RPython compile.py:701-716 parity: every guard must have rd_numb
    // from capture_resumedata + store_final_boxes_in_guard (resume.py:397).
    // Hitting this path means a guard was compiled without snapshot data.
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[blackhole-resume] no rd_numb for key={} trace={} fail={} (force_fn fallback)",
            actual_green_key, trace_id, fail_index,
        );
    }
    None
}

/// resume.py:1312 blackhole_from_resumedata parity:
/// Decode rd_numb via ResumeDataDirectReader, build blackhole chain,
/// run _run_forever.
pub fn blackhole_resume_via_rd_numb(
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    deadframe: &[i64],
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    deadframe_types: Option<&[majit_ir::Type]>,
) -> BlackholeResult {
    let nbody_debug = std::env::var_os("PYRE_NBODY_DEBUG").is_some();
    use majit_metainterp::resume;

    // Thread-local BH pool (RPython BlackholeInterpBuilder). Each access
    // is scoped to a single call so that bh.run() (which may re-enter
    // blackhole_resume_via_rd_numb) cannot create overlapping &mut refs.
    thread_local! {
        // See the parallel comment on `BH_BUILDER3` in `resume_in_blackhole`.
        static BH_BUILDER_RD: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(pyre_jit_trace::jitcode_runtime::build_pyre_production_bh_builder());
    }
    let sync_bh_builder_control_opcodes =
        |builder: &mut majit_metainterp::blackhole::BlackholeInterpBuilder| {
            let (op_live, op_catch_exception, op_rvmprof_code) =
                pyre_jit_trace::state::blackhole_control_opcodes();
            builder.setup_cached_control_opcodes(op_live, op_catch_exception, op_rvmprof_code);
        };
    let release_bh_rd = |bh: majit_metainterp::blackhole::BlackholeInterpreter| {
        BH_BUILDER_RD.with(|cell| unsafe { (&mut *cell.get()).release_interp(bh) });
    };

    // resume.py:1339 jitcodes[jitcode_pos]: resolve jitcode_index + pc
    // through the trace-side MetaInterpStaticData.jitcodes store.
    let resolve_jitcode = |jitcode_index: i32, pc: i32| -> Option<resume::ResolvedJitCode> {
        if pc < 0 {
            return None;
        }
        let pyjitcode = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)?;
        if pyjitcode.has_abort_opcode() {
            return None;
        }
        let jitcode_pc = pyjitcode.metadata.pc_map.get(pc as usize).copied()?;
        // resume.py:1339 reads from one `jitcodes[]` store.  pyre's
        // `state::code_for_jitcode_index` indices name the runtime
        // `MetaInterpStaticData.jitcodes` table keyed by CodeObject; they
        // are not the same index space as `jitcode_runtime::ALL_JITCODES`
        // (build-time opcode-dispatch artifacts).  Do not cross-lookup the
        // canonical store by `jitcode_index` until pyre actually shares a
        // single JitCode object graph end-to-end.
        Some(
            resume::ResolvedJitCode::new(pyjitcode.jitcode.clone(), jitcode_pc)
                .with_virtualizable_stack_base(pyjitcode.metadata.stack_base),
        )
    };

    // resume.py:983-991 _prepare_virtuals: convert RdVirtualInfo → VirtualInfo
    // for lazy materialization in getvirtual_ptr/getvirtual_int.
    let count = deadframe.len() as i32;
    let rd_virtuals_converted: Option<Vec<resume::VirtualInfo>> = rd_virtuals.map(|rd_virts| {
        rd_virts
            .iter()
            .map(|rd| resume::rd_virtual_to_virtual_info(rd, rd_consts, count))
            .collect()
    });
    let rd_virtuals_slice = rd_virtuals_converted.as_deref();

    // resume.py:1312-1343 blackhole_from_resumedata:
    // ResumeDataDirectReader decodes rd_numb, builds BH chain.
    // compile.py:990 parity: vinfo = self.jitdriver_sd.virtualizable_info —
    // read the active driver's cached Arc instead of rebuilding a fresh
    // VirtualizableInfo, so a single VirtualizableInfo identity is shared
    // with tracing, setup_bridge_sym, and the guard-failure recovery
    // consumers. resume.py:1314 vrefinfo = metainterp_sd.virtualref_info —
    // hand the metainterp's own VRefInfo through so consume_virtualref_info
    // can decode JIT_VIRTUAL_REF handles. resume.py:1316 ginfo is currently
    // unused in pyre (no greenfield_info installed on the driver).
    let (driver, driver_vinfo) = crate::eval::driver_pair();
    let vinfo_dyn: &dyn resume::VirtualizableInfo = driver_vinfo.as_ref();
    let vrefinfo_dyn: &dyn resume::VRefInfo = driver.meta_interp().virtualref_info();
    let allocator = crate::eval::PyreBlackholeAllocator;
    // pyjitpl.py:2264: metainterp_sd.liveness_info — single shared pool.
    // Snapshot once per call so the slice outlives ResumeDataDirectReader.
    let all_liveness = pyre_jit_trace::state::liveness_info_snapshot();
    // Scope the &mut to chain construction; the run() loop below uses
    // release_bh_rd to drop and re-acquire the borrow.
    let bh = BH_BUILDER_RD.with(|cell| unsafe {
        let builder = &mut *cell.get();
        sync_bh_builder_control_opcodes(builder);
        resume::blackhole_from_resumedata(
            builder,
            &resolve_jitcode,
            rd_numb,
            rd_consts,
            &all_liveness,
            deadframe,
            deadframe_types,        // deadframe_types: decode_ref boxes TAGBOX ints
            rd_virtuals_slice,      // rd_virtuals
            rd_guard_pendingfields, // rd_guard_pendingfields
            Some(vrefinfo_dyn),     // resume.py:1314 metainterp_sd.virtualref_info
            Some(vinfo_dyn),        // resume.py:1312 self.jitdriver_sd.virtualizable_info
            None,                   // resume.py:1316 greenfield_info unused in pyre
            &allocator,
        )
    });

    let Some((mut bh, virtualizable_ptr)) = bh else {
        if nbody_debug {
            eprintln!("[nbody-debug] blackhole_resume_via_rd_numb failed: builder returned None");
        }
        return BlackholeResult::Failed;
    };

    // resume.py:1404: virtualizable_ptr was read by consume_vable_info
    // from the vable section. Set on the blackhole for vable bytecodes.
    if virtualizable_ptr != 0 {
        bh.virtualizable_ptr = virtualizable_ptr;
    } else if !deadframe.is_empty() {
        // Fallback for guards without vable section.
        bh.virtualizable_ptr = deadframe[0];
    }
    bh.virtualizable_info = crate::eval::get_virtualizable_info();
    // blackhole.py:1095-1099 get_portal_runner parity:
    //   jitdriver_sd = self.builder.metainterp_sd.jitdrivers_sd[jdindex]
    //   fnptr        = adr2int(jitdriver_sd.portal_runner_adr)
    //   calldescr    = jitdriver_sd.mainjitcode.calldescr
    bh.jitdrivers_sd = vec![majit_metainterp::blackhole::BhJitDriverSd {
        result_type: majit_metainterp::blackhole::BhReturnType::Ref,
        portal_runner_ptr: Some(bh_portal_runner),
        mainjitcode_calldescr: bh.jitcode.calldescr.clone(),
    }];

    // interp_jit.py:64 parity (pyre-adaptation): fill dedicated portal
    // red-arg registers (frame_ptr, execution_context) for each jitcode
    // in the chain. RPython encodes these as regular inputargs of the
    // portal jitcode, so `_prepare_next_section` fills them during
    // `consume_one_section` without a side channel. Pyre's codewriter
    // assigns portal registers separately (PyJitCodeMetadata
    // .portal_frame_reg / .portal_ec_reg) and omits them from liveness,
    // so the orthodox `blackhole_from_resumedata` path leaves them zero.
    // Walk the chain and fill them explicitly until the codewriter
    // change lands; chains are short (typically 1–3 frames) so the
    // O(jitcodes) scan per frame is inexpensive.
    let callcontrol = crate::jit::codewriter::CodeWriter::instance().callcontrol();
    let mut cur: Option<&mut majit_metainterp::blackhole::BlackholeInterpreter> = Some(&mut bh);
    while let Some(bh_ref) = cur {
        let jitcode_ptr = std::sync::Arc::as_ptr(&bh_ref.jitcode);
        if let Some(pyjit) = callcontrol.find_pyjitcode_by_jitcode_ptr(jitcode_ptr) {
            let vable_ptr = bh_ref.virtualizable_ptr;
            let ec = if vable_ptr != 0 {
                unsafe { (*(vable_ptr as *const PyFrame)).execution_context as i64 }
            } else {
                0
            };
            bh_ref.fill_portal_registers(
                pyjit.metadata.portal_frame_reg,
                pyjit.metadata.portal_ec_reg,
                ec,
            );
        }
        cur = bh_ref.nextblackholeinterp.as_deref_mut();
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!("[blackhole-resume] rd_numb path, chain built, running _run_forever",);
    }

    // blackhole.py:1752 _run_forever parity.
    loop {
        if let Some(args) = bh.run() {
            // blackhole.py:1068: raise ContinueRunningNormally(*args)
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;

            let mut red_ref: Vec<PyObjectRef> =
                args.red_ref.iter().map(|&v| v as PyObjectRef).collect();
            if red_ref.is_empty() {
                red_ref.push(frame_ptr as PyObjectRef);
            }
            return BlackholeResult::ContinueRunningNormally {
                green_int: args.green_int,
                green_ref: args.green_ref.iter().map(|&v| v as PyObjectRef).collect(),
                green_float: args
                    .green_float
                    .iter()
                    .map(|&v| f64::from_bits(v as u64))
                    .collect(),
                red_int: args.red_int,
                red_ref,
                red_float: args
                    .red_float
                    .iter()
                    .map(|&v| f64::from_bits(v as u64))
                    .collect(),
            };
        }
        if bh.aborted {
            if nbody_debug {
                eprintln!(
                    "[nbody-debug] blackhole_resume_via_rd_numb failed: bh.aborted position={} last_opcode_position={}",
                    bh.position, bh.last_opcode_position
                );
            }
            release_bh_rd(bh);
            return BlackholeResult::Failed;
        }
        if bh.got_exception {
            let exc_value = bh.exception_last_value;
            let next = bh.nextblackholeinterp.take();
            release_bh_rd(bh);
            let Some(mut caller_bh) = next.map(|b| *b) else {
                // blackhole.py:1679-1682 _exit_frame_with_exception:
                //   e = cast_opaque_ptr(GCREF, e)
                //   raise ExitFrameWithExceptionRef(e)
                let err = if exc_value != 0 {
                    unsafe {
                        pyre_interpreter::PyError::from_exc_object(
                            exc_value as pyre_object::PyObjectRef,
                        )
                    }
                } else {
                    pyre_interpreter::PyError::new(
                        pyre_interpreter::PyErrorKind::RuntimeError,
                        "blackhole exception (null exc_value)",
                    )
                };
                return BlackholeResult::ExitFrameWithExceptionRef(err);
            };
            caller_bh.last_opcode_position = caller_bh.position;
            if caller_bh.handle_exception_in_frame(exc_value) {
                bh = caller_bh;
                continue;
            }
            caller_bh.exception_last_value = exc_value;
            caller_bh.got_exception = true;
            bh = caller_bh;
            continue;
        }

        // blackhole.py:1632-1644: pass return value to caller by _return_type.
        use majit_metainterp::blackhole::BhReturnType;
        let rt = bh.return_type;
        let next = bh.nextblackholeinterp.take();
        let caller = next.map(|b| *b);
        if caller.is_none() {
            // blackhole.py:1664-1677 _done_with_this_frame
            let result = match rt {
                BhReturnType::Void => BlackholeResult::DoneWithThisFrameVoid,
                BhReturnType::Int => BlackholeResult::DoneWithThisFrameInt(bh.get_tmpreg_i()),
                BhReturnType::Ref => {
                    BlackholeResult::DoneWithThisFrameRef(bh.get_tmpreg_r() as PyObjectRef)
                }
                BhReturnType::Float => BlackholeResult::DoneWithThisFrameFloat(f64::from_bits(
                    bh.get_tmpreg_f() as u64,
                )),
            };
            return result;
        }
        let mut caller_bh = caller.unwrap();
        // blackhole.py:1637-1644: dispatch by _return_type
        match rt {
            BhReturnType::Int => caller_bh.setup_return_value_i(bh.get_tmpreg_i()),
            BhReturnType::Ref => caller_bh.setup_return_value_r(bh.get_tmpreg_r()),
            BhReturnType::Float => caller_bh.setup_return_value_f(bh.get_tmpreg_f()),
            BhReturnType::Void => {}
        }
        release_bh_rd(bh);
        bh = caller_bh;
    }
}

/// warmspot.py:961-1007 handle_jitexception parity.
///
/// RPython captures result_kind in closure (warmspot.py:913). For pyre,
/// portal result_type == REF (warmspot.py:449), so ALL CALL_ASSEMBLER
/// ops use _R. The result is always a Ref (PyObjectRef).
fn handle_blackhole_result(bh_result: BlackholeResult, _green_key: u64) -> Option<i64> {
    match bh_result {
        // warmspot.py:985-987: DoneWithThisFrameVoid → return None
        BlackholeResult::DoneWithThisFrameVoid => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameVoid");
            }
            Some(0)
        }
        // warmspot.py:988-990: DoneWithThisFrameInt → box to Ref.
        // Portal result_type == REF, so blackhole should normally raise
        // DoneWithThisFrameRef. This path handles edge cases.
        BlackholeResult::DoneWithThisFrameInt(v) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[blackhole-resume] DoneWithThisFrameInt({}) → box to Ref",
                    v
                );
            }
            Some(w_int_new(v) as i64)
        }
        // warmspot.py:991-993: DoneWithThisFrameRef → return e.result
        BlackholeResult::DoneWithThisFrameRef(r) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameRef({:?})", r);
            }
            Some(r as i64)
        }
        // warmspot.py:994-996: DoneWithThisFrameFloat → return e.result
        BlackholeResult::DoneWithThisFrameFloat(f) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameFloat({})", f);
            }
            Some(f.to_bits() as i64)
        }
        // warmspot.py:998-1005: ExitFrameWithExceptionRef → raise value.
        BlackholeResult::ExitFrameWithExceptionRef(err) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] ExitFrameWithExceptionRef → raise");
            }
            let exc_obj = err.exc_object;
            if exc_obj != pyre_object::PY_NULL {
                // Symmetric with the regular-exception fall-through
                // below (line 2120-2122) and with `lib.rs::jit_exc_raise`
                // — every backend's blackhole resume publishes the
                // pending exception, not just cranelift.
                #[cfg(feature = "cranelift")]
                majit_backend_cranelift::jit_exc_raise(exc_obj as i64);
                #[cfg(feature = "dynasm")]
                majit_backend_dynasm::jit_exc_raise(exc_obj as i64);
            }
            Some(0) // garbage return — GUARD_NO_EXCEPTION will fire
        }
        // warmspot.py:970-983: ContinueRunningNormally → portal_ptr(*args).
        BlackholeResult::ContinueRunningNormally {
            green_int,
            green_ref,
            green_float,
            red_int,
            red_ref,
            red_float,
        } => {
            // warmspot.py:972-975: portalfunc_ARGS extraction.
            // Build merged arg lists: all_i = gi + ri, all_r = gr + rr, all_f = gf + rf.
            // warmstate.py:41 unspecialize_value: Ref→GCREF(i64), Float→FLOATSTORAGE(i64).
            let mut all_i = green_int;
            all_i.extend(&red_int);
            let mut all_r: Vec<i64> = green_ref.iter().map(|r| *r as i64).collect();
            all_r.extend(red_ref.iter().map(|r| *r as i64));
            let mut all_f: Vec<i64> = green_float.iter().map(|f| f.to_bits() as i64).collect();
            all_f.extend(red_float.iter().map(|f| f.to_bits() as i64));
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[blackhole-resume] ContinueRunningNormally → portal_ptr(all_i={:?}, all_r=[{:#x?}])",
                    all_i, all_r,
                );
            }
            // warmspot.py:976-1005: portal_ptr(*args), and if it raises a
            // regular exception propagate it like ExitFrameWithExceptionRef
            // instead of collapsing it to a null Ref.
            let next_instr = all_i.first().copied().unwrap_or(0) as usize;
            let pycode = all_r.first().copied().unwrap_or(0) as PyObjectRef;
            let frame_ptr = all_r.get(1).copied().unwrap_or(0) as *mut PyFrame;
            let ec =
                all_r.get(2).copied().unwrap_or(0) as *const pyre_interpreter::PyExecutionContext;
            if frame_ptr.is_null() {
                return Some(pyre_object::PY_NULL as i64);
            }
            let frame = unsafe { &mut *frame_ptr };
            if !pycode.is_null() {
                frame.pycode = pycode as *const ();
            }
            if !ec.is_null() {
                frame.execution_context = ec;
            }
            frame.set_last_instr_from_next_instr(next_instr);
            match crate::eval::portal_runner_result(frame) {
                Ok(result) => Some(result as i64),
                Err(err) => {
                    let exc_obj = err.exc_object;
                    if exc_obj != pyre_object::PY_NULL {
                        majit_metainterp::blackhole::BH_LAST_EXC_VALUE
                            .with(|c| c.set(exc_obj as i64));
                        #[cfg(feature = "cranelift")]
                        majit_backend_cranelift::jit_exc_raise(exc_obj as i64);
                        #[cfg(feature = "dynasm")]
                        majit_backend_dynasm::jit_exc_raise(exc_obj as i64);
                    }
                    Some(0)
                }
            }
        }
        BlackholeResult::Failed => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] Failed");
            }
            None
        }
    }
}

/// compile.py:714 (_trace_and_compile_from_bridge):
/// Called when a guard failure reaches the trace_eagerness threshold.
/// Traces the alternative path from the guard failure point and compiles
/// a bridge.
///
/// pyjitpl.py:2884 handle_guard_failure:
///   initialize_state_from_guard_failure(resumedescr, deadframe)
///   prepare_resume_from_failure(deadframe, inputargs, resumedescr, excdata)
///   self.interpret()
///
/// The tracing loop mirrors pyjitpl.py interpret(): execute bytecodes
/// from the guard failure PC until a Finish (return) or CloseLoop
/// (back-edge to loop header) is reached.
/// compile.py:714 _trace_and_compile_from_bridge parity.
///
/// Returns true if the bridge was successfully compiled and attached.
/// On failure (trace abort, start failure), returns false so the caller
/// falls through to resume_in_blackhole (RPython pyjitpl.py:2906-2907
/// SwitchToBlackhole → run_blackhole_interp_to_cancel_tracing).
pub fn trace_and_compile_from_bridge(
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    frame: &mut PyFrame,
    raw_values: &[i64],
    exit_layout: &majit_metainterp::CompiledExitLayout,
) -> bool {
    use crate::eval::build_jit_state;
    use crate::jit::state::PyreEnv;

    let info = {
        let (_, info) = crate::eval::driver_pair();
        info
    };

    // pyjitpl.py:2890-2911 handle_guard_failure parity:
    // RPython creates a fresh MetaInterp and calls
    // initialize_state_from_guard_failure(resumedescr, deadframe)
    // which internally calls rebuild_from_resumedata (resume.py:1042).
    // This restores the complete frame stack INSIDE the bridge function.
    let meta = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp().get_compiled_meta(green_key).cloned()
    };
    let mut jit_state_local = build_jit_state(frame, info);
    let resume_pc = if let Some(ref meta) = meta {
        if let Some((_, pc)) = crate::eval::decode_and_restore_guard_failure(
            &mut jit_state_local,
            meta,
            raw_values,
            exit_layout,
        ) {
            pc
        } else {
            0
        }
    } else {
        0
    };
    if resume_pc == 0 {
        return false;
    }
    frame.set_last_instr_from_next_instr(resume_pc);
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);

    // NOTE: guard resume_pc pointing to LOAD_CONST + RETURN_VALUE does NOT
    // mean the guard is a loop-exit guard. It means the blackhole resume
    // path leads to function return. RPython handles this correctly via
    // blackhole resume → interpreter runs remaining code → natural return.
    // Direct FINISH bridges are WRONG here — they skip the remaining loop
    // body that the blackhole should execute.
    // RPython rebuild_from_resumedata (pyjitpl.py:2901,3400)
    // restores the complete frame stack before bridge tracing.
    // Bridge tracing sees the full frame layout — no truncation.
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][bridge-trace] start key={} trace={} fail={} resume_pc={}",
            green_key, trace_id, fail_index, resume_pc
        );
        if trace_id == 2 && fail_index == 2 && resume_pc == 153 {
            let debug_values: Vec<String> = raw_values
                .iter()
                .zip(exit_layout.exit_types.iter())
                .enumerate()
                .map(|(idx, (&raw, &tp))| match tp {
                    majit_ir::Type::Ref => {
                        let obj = raw as pyre_object::PyObjectRef;
                        let detail = unsafe {
                            if obj.is_null() {
                                "null".to_string()
                            } else if pyre_object::is_float(obj) {
                                format!("float({})", pyre_object::w_float_get_value(obj))
                            } else if pyre_object::is_int(obj) {
                                format!("int({})", pyre_object::w_int_get_value(obj))
                            } else if pyre_object::is_list(obj) {
                                "list".to_string()
                            } else {
                                format!("ref({:#x})", obj as usize)
                            }
                        };
                        format!("#{idx}:Ref {detail}")
                    }
                    majit_ir::Type::Int => format!("#{idx}:Int {}", raw),
                    majit_ir::Type::Float => format!("#{idx}:Float {}", f64::from_bits(raw as u64)),
                    majit_ir::Type::Void => format!("#{idx}:Void"),
                })
                .collect();
            eprintln!("[jit][bridge-raw] {}", debug_values.join(", "));
        }
    }

    // compile.py:714: start_retrace_from_guard + set bridge_info.
    let started = {
        let (driver, _) = crate::eval::driver_pair();
        driver.start_bridge_tracing(
            green_key,
            trace_id,
            fail_index,
            &mut jit_state,
            &env,
            raw_values,
            resume_pc,
        )
    };
    if !started {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] start_bridge_tracing failed key={} trace={} fail={}",
                green_key, trace_id, fail_index
            );
        }
        return false;
    }

    // RPython pyjitpl.py:3101 _prepare_exception_resumption +
    // pyjitpl.py:3132 prepare_resume_from_failure parity:
    // For exception guard bridges (GUARD_EXCEPTION / GUARD_NO_EXCEPTION),
    // emit SAVE_EXC_CLASS + SAVE_EXCEPTION at trace start, then
    // RESTORE_EXCEPTION before the guard. The exception class/value
    // are read from the TLS exception state set by Cranelift codegen.
    let last_bridge_is_exception_guard = {
        let (driver, _) = crate::eval::driver_pair();
        driver.last_bridge_is_exception_guard
    };
    if last_bridge_is_exception_guard {
        #[cfg(feature = "cranelift")]
        let exc_class = majit_backend_cranelift::jit_exc_class_raw();
        #[cfg(not(feature = "cranelift"))]
        let exc_class: i64 = 0;
        #[cfg(feature = "cranelift")]
        let exc_value = majit_backend_cranelift::jit_exc_value_raw();
        #[cfg(not(feature = "cranelift"))]
        let exc_value: i64 = 0;
        if exc_class != 0 {
            // RPython pyjitpl.py:3125-3126 + 3138:
            // SAVE_EXC_CLASS, SAVE_EXCEPTION, RESTORE_EXCEPTION
            {
                let (driver, _) = crate::eval::driver_pair();
                driver
                    .meta_interp_mut()
                    .emit_exception_bridge_prologue(exc_class, exc_value);
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-exc] exception guard bridge: class={:#x} value={:#x}",
                    exc_class, exc_value
                );
            }
        }
        let (driver, _) = crate::eval::driver_pair();
        driver.last_bridge_is_exception_guard = false;
    }

    // pyjitpl.py:2841 interpret(): after start_retrace_from_guard, RPython
    // runs a single interpret() over the resumed frame state until the
    // bridge closes or aborts. `trace_bytecode` is the pyre equivalent of
    // that whole-loop interpreter; calling it once preserves the concrete PC
    // updates across branches/back-edges. Re-invoking it in a synthetic
    // `pc + 1` loop diverges from RPython and corrupts nested-loop bridges.
    let trace_frame = frame.snapshot_for_tracing();
    let outcome = {
        let (driver, _) = crate::eval::driver_pair();
        driver.jit_merge_point_keyed(
            green_key,
            resume_pc,
            &mut jit_state,
            &env,
            || {},
            |ctx, sym| {
                let (action, _executed) = trace_bytecode(ctx, sym, code, resume_pc, trace_frame);
                action
            },
        )
    };

    // merge_point handles Finish/CloseLoop via bridge_info.
    if outcome.is_some() {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] compiled at resume_pc={} key={}",
                resume_pc, green_key
            );
        }
        return true;
    }

    // pyjitpl.py:2982-2983 / 3095-3099 parity:
    // compile_trace() "raises in case it works". In pyre the bridge can
    // already be attached during this step even if jit_merge_point_keyed()
    // did not surface DetailedDriverRunOutcome::Jump yet. Stop tracing as
    // soon as the backend metadata shows that the bridge is attached.
    let compiled = {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp()
            .bridge_was_compiled(green_key, trace_id, fail_index)
    };
    if compiled {
        let (driver, _) = crate::eval::driver_pair();
        if driver.is_tracing() {
            driver.meta_interp_mut().abort_trace(false);
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] compiled at resume_pc={} key={} (attached)",
                resume_pc, green_key
            );
        }
        return true;
    }

    // If the driver is no longer tracing, the bridge was compiled
    // (or aborted) inside merge_point. Check whether a bridge was
    // actually attached to distinguish success from abort.
    let tracing_active = {
        let (driver, _) = crate::eval::driver_pair();
        driver.is_tracing()
    };
    if !tracing_active {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] trace ended at resume_pc={} key={} compiled={}",
                resume_pc, green_key, compiled
            );
        }
        return compiled;
    }

    // Trace did not converge into a bridge. Abort like RPython's
    // run_blackhole_interp_to_cancel_tracing fallback path.
    if tracing_active {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] abort: no-bridge key={} trace={} fail={} resume_pc={}",
                green_key, trace_id, fail_index, resume_pc
            );
        }
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().abort_trace(false);
    }
    false
}

/// compile.py:701-717 handle_fail for call_assembler guard failures.
/// Checks must_compile (jitcounter.tick), and if threshold reached,
/// traces the alternate path via trace_and_compile_from_bridge.
fn jit_ca_handle_guard_failure(
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    raw_values_ptr: *const i64,
    num_values: usize,
    descr_addr: usize,
) -> bool {
    if raw_values_ptr.is_null() || num_values == 0 {
        return false;
    }
    let raw_values = unsafe { std::slice::from_raw_parts(raw_values_ptr, num_values) };

    // compile.py:738-784 must_compile: jitcounter.tick(guard_hash, increment)
    let (must_compile, owning_key) = {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp_mut()
            .must_compile_with_values(green_key, trace_id, fail_index, raw_values, descr_addr)
    };
    // compile.py:702-703: must_compile() and not stack_almost_full()
    if !must_compile || majit_metainterp::MetaInterp::<()>::stack_almost_full() {
        return false;
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][ca-bridge] must_compile fired: key={} trace={} fail={}",
            green_key, trace_id, fail_index,
        );
    }

    // compile.py:719-726: get exit_layout from the compiled trace.
    // Use owning_key (not green_key) — after retrace the descriptor
    // may belong to a different compiled entry than green_key.
    let exit_layout = {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp()
            .get_compiled_exit_layout_in_trace(owning_key, trace_id, fail_index)
    };
    let Some(exit_layout) = exit_layout else {
        return false;
    };

    // Obtain callee frame from deadframe vable header.
    // pyre vable_boxes = [frame, ni, code, vsd, ns, locals..., stack...],
    // so raw_values[0] is the callee's PyFrame pointer.
    let frame_ptr = raw_values[0] as *mut PyFrame;
    if frame_ptr.is_null() {
        return false;
    }
    let frame = unsafe { &mut *frame_ptr };

    // compile.py:786-788 self.start_compiling(): set ST_BUSY_FLAG
    {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().start_guard_compiling(descr_addr);
    }

    // compile.py:706-708 _trace_and_compile_from_bridge
    let compiled = trace_and_compile_from_bridge(
        owning_key,
        trace_id,
        fail_index,
        frame,
        raw_values,
        &exit_layout,
    );

    // compile.py:790-795 self.done_compiling(): clear ST_BUSY_FLAG
    {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().done_guard_compiling(descr_addr);
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][ca-bridge] compiled={} key={} trace={} fail={}",
            compiled, green_key, trace_id, fail_index,
        );
    }

    compiled
}

// ── Callee frame creation for call_assembler ─────────────────────

/// Public wrapper for trace-through inlining.
pub fn create_callee_frame_impl_pub(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    create_callee_frame_impl(caller_frame, callable, args)
}

#[inline]
fn reset_reused_call_frame(frame: &mut PyFrame, args: &[PyObjectRef]) {
    frame.locals_w_mut().as_mut_slice().fill(PY_NULL);
    let nargs = args.len().min(frame.nlocals());
    for (idx, value) in args.iter().take(nargs).enumerate() {
        frame.locals_w_mut()[idx] = *value;
    }
    frame.valuestackdepth = frame.stack_base();
    frame.set_last_instr_from_next_instr(0);
    frame.vable_token = 0;
    frame.frame_finished_execution = false;
    frame.f_generator_nowref = PY_NULL;
    frame.w_yielding_from = PY_NULL;
    frame.f_backref = std::ptr::null_mut();
    // pyframe.py:78-86: reused arena frames must look like new frames.
    // debugdata and lastblock are GC-managed refs — release references only,
    // never manually free (JIT snapshots may still hold these pointers).
    frame.debugdata = std::ptr::null_mut();
    frame.escaped = false;
    frame.set_blocklist(&[]);
}

fn create_callee_frame_impl_1_boxed(
    caller_frame: i64,
    callable: PyObjectRef,
    boxed_arg: PyObjectRef,
) -> i64 {
    let w_code = unsafe { pyre_interpreter::getcode(callable) };
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let globals = unsafe { function_get_globals(callable) };

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        if was_init {
            let f = unsafe { &mut *ptr };
            if f.pycode == w_code
                && f.w_globals == globals
                && f.execution_context == caller.execution_context
            {
                reset_reused_call_frame(f, &[boxed_arg]);
            } else {
                unsafe {
                    // Different function: drop the previous frame before
                    // overwriting, so PyFrame::drop releases the old
                    // locals_cells_stack_w (pyframe.rs:150).
                    std::ptr::drop_in_place(ptr);
                    std::ptr::write(
                        ptr,
                        PyFrame::new_for_call(
                            w_code,
                            &[boxed_arg],
                            globals,
                            caller.execution_context,
                        ),
                    );
                    (&mut *ptr).fix_array_ptrs();
                }
            }
        } else {
            unsafe {
                std::ptr::write(
                    ptr,
                    PyFrame::new_for_call(w_code, &[boxed_arg], globals, caller.execution_context),
                );
                (&mut *ptr).fix_array_ptrs();
            }
            arena.mark_initialized();
        }
        return ptr as i64;
    }

    let frame_ptr = heap_alloc_frame(PyFrame::new_for_call(
        w_code,
        &[boxed_arg],
        globals,
        caller.execution_context,
    ));
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    frame_ptr as i64
}

fn create_self_recursive_callee_frame_impl_1_boxed(
    caller_frame: i64,
    boxed_arg: PyObjectRef,
) -> i64 {
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let func_code = caller.pycode;
    let globals = caller.w_globals;
    let execution_context = caller.execution_context;

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        if was_init {
            let f = unsafe { &mut *ptr };
            if f.pycode == func_code
                && f.w_globals == globals
                && f.execution_context == execution_context
            {
                // Reuse: same code/globals/ec — full reset matching
                // new_for_call_with_closure() semantics. No partial
                // shortcuts: blackhole/force paths must see a clean frame.
                reset_reused_call_frame(f, &[boxed_arg]);
            } else {
                unsafe {
                    std::ptr::drop_in_place(ptr);
                    std::ptr::write(
                        ptr,
                        PyFrame::new_for_call(func_code, &[boxed_arg], globals, execution_context),
                    );
                    (&mut *ptr).fix_array_ptrs();
                }
            }
        } else {
            unsafe {
                std::ptr::write(
                    ptr,
                    PyFrame::new_for_call(func_code, &[boxed_arg], globals, execution_context),
                );
                (&mut *ptr).fix_array_ptrs();
            }
            arena.mark_initialized();
        }
        if majit_metainterp::majit_log_enabled() {
            let f = unsafe { &*ptr };
            eprintln!(
                "[jit][ca-frame] ptr={ptr:p} locals=0x{:x} vsd={} reused={} boxed_arg=0x{:x}",
                f.locals_cells_stack_w as usize, f.valuestackdepth, was_init, boxed_arg as usize,
            );
        }
        return ptr as i64;
    }

    let frame_ptr = heap_alloc_frame(PyFrame::new_for_call(
        func_code,
        &[boxed_arg],
        globals,
        execution_context,
    ));
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    if majit_metainterp::majit_log_enabled() {
        let f = unsafe { &*frame_ptr };
        eprintln!(
            "[jit][ca-frame] ptr={frame_ptr:p} locals=0x{:x} vsd={} reused=false boxed_arg=0x{:x}",
            f.locals_cells_stack_w as usize, f.valuestackdepth, boxed_arg as usize,
        );
    }
    frame_ptr as i64
}

fn create_callee_frame_impl(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    let callable = callable as PyObjectRef;
    let w_code = unsafe { pyre_interpreter::getcode(callable) };
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let globals = unsafe { function_get_globals(callable) };

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        if was_init {
            // Fast reinit: only update fields that change between calls.
            // code, execution_context, namespace, locals_cells_stack_w.ptr
            // are stable for self-recursion (same function, same module).
            let f = unsafe { &mut *ptr };
            if f.pycode == w_code
                && f.w_globals == globals
                && f.execution_context == caller.execution_context
            {
                reset_reused_call_frame(f, args);
            } else {
                // Different function: full reinit (rare for fib)
                unsafe {
                    std::ptr::drop_in_place(ptr);
                    std::ptr::write(
                        ptr,
                        PyFrame::new_for_call(w_code, args, globals, caller.execution_context),
                    );
                    (&mut *ptr).fix_array_ptrs();
                }
            }
        } else {
            // First-time init for this arena slot
            unsafe {
                std::ptr::write(
                    ptr,
                    PyFrame::new_for_call(w_code, args, globals, caller.execution_context),
                );
                (&mut *ptr).fix_array_ptrs();
            }
            arena.mark_initialized();
        }
        return ptr as i64;
    }

    // Arena full: heap fallback (should not happen for recursion < 64)
    let frame_ptr = heap_alloc_frame(PyFrame::new_for_call(
        w_code,
        args,
        globals,
        caller.execution_context,
    ));
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    frame_ptr as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_0(caller_frame: i64, callable: i64) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[])
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_1(caller_frame: i64, callable: i64, arg0: i64) -> i64 {
    create_callee_frame_impl_1_boxed(caller_frame, callable as PyObjectRef, arg0 as PyObjectRef)
}

/// Self-recursive single-arg variant.
///
/// This skips rediscovering code/globals from a function object and reuses the
/// caller frame's code/namespace/execution_context directly, which matches the
/// existing self-recursive raw helper path more closely.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_self_recursive_callee_frame_1(caller_frame: i64, arg0: i64) -> i64 {
    debug_assert!(
        caller_frame != 0,
        "jit_create_self_recursive_callee_frame_1: caller_frame is null"
    );
    if caller_frame == 0 {
        // Invariant violation: fall back to heap allocation with a
        // minimal frame. RPython never aborts the process for JIT
        // invariant failures — it falls back to tracing abort or
        // blackhole resume.
        return 0;
    }
    create_self_recursive_callee_frame_impl_1_boxed(caller_frame, arg0 as PyObjectRef)
}

/// Self-recursive raw-int variant: creates the frame WITHOUT boxing
/// the argument. The raw int is passed directly to compiled code via
/// CallAssemblerI inputargs. Boxing only happens on guard failure
/// (in force_fn / jit_force_self_recursive_call_raw_1).
///
/// RPython parity: compiled code uses jitframe slots, not PyFrame
/// locals. Frame locals are only needed for interpreter fallback.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_self_recursive_callee_frame_1_raw_int(
    caller_frame: i64,
    raw_int_arg: i64,
) -> i64 {
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let func_code = caller.pycode;
    let globals = caller.w_globals;
    let execution_context = caller.execution_context;

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        let f = unsafe { &mut *ptr };
        if was_init
            && f.pycode == func_code
            && f.w_globals == globals
            && f.execution_context == execution_context
        {
            // Reuse: full reset matching new_for_call semantics.
            reset_reused_call_frame(f, &[boxed]);
        } else {
            unsafe {
                if was_init {
                    std::ptr::drop_in_place(ptr);
                }
                std::ptr::write(
                    ptr,
                    PyFrame::new_for_call(func_code, &[boxed], globals, execution_context),
                );
                (&mut *ptr).fix_array_ptrs();
            }
            if !was_init {
                arena.mark_initialized();
            }
        }
        if majit_metainterp::majit_log_enabled() {
            let f = unsafe { &*ptr };
            eprintln!(
                "[jit][ca-frame-raw] ptr={ptr:p} locals=0x{:x} local0=0x{:x} vsd={} reused={} raw_arg={}",
                f.locals_cells_stack_w as usize,
                f.locals_w()[0] as usize,
                f.valuestackdepth,
                was_init,
                raw_int_arg,
            );
        }
        return ptr as i64;
    }

    let frame_ptr = heap_alloc_frame(PyFrame::new_for_call(
        func_code,
        &[boxed],
        globals,
        execution_context,
    ));
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    if majit_metainterp::majit_log_enabled() {
        let f = unsafe { &*frame_ptr };
        eprintln!(
            "[jit][ca-frame-raw] ptr={frame_ptr:p} locals=0x{:x} local0=0x{:x} vsd={} reused=false raw_arg={}",
            f.locals_cells_stack_w as usize,
            f.locals_w()[0] as usize,
            f.valuestackdepth,
            raw_int_arg,
        );
    }
    frame_ptr as i64
}

/// Raw-int variant: accepts a raw int and boxes it internally.
/// Eliminates trace_box_int CallI from the trace (boxing folded into frame creation).
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_1_raw_int(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    create_callee_frame_impl_1_boxed(caller_frame, callable as PyObjectRef, boxed)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_2(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[arg0 as PyObjectRef, arg1 as PyObjectRef],
    )
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_3(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
    arg2: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[
            arg0 as PyObjectRef,
            arg1 as PyObjectRef,
            arg2 as PyObjectRef,
        ],
    )
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_4(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
    arg2: i64,
    arg3: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[
            arg0 as PyObjectRef,
            arg1 as PyObjectRef,
            arg2 as PyObjectRef,
            arg3 as PyObjectRef,
        ],
    )
}

pub fn callee_frame_helper(nargs: usize) -> Option<*const ()> {
    match nargs {
        0 => Some(jit_create_callee_frame_0 as *const ()),
        1 => Some(jit_create_callee_frame_1 as *const ()),
        2 => Some(jit_create_callee_frame_2 as *const ()),
        3 => Some(jit_create_callee_frame_3 as *const ()),
        4 => Some(jit_create_callee_frame_4 as *const ()),
        _ => None,
    }
}

/// Force callee and return BOXED result (for inline_function_call).
/// warmspot.py:449 result_type=REF: jit_force_callee_frame already
/// returns boxed Ref, so this is just a pass-through.
#[majit_macros::jit_may_force]
pub extern "C" fn jit_force_callee_frame_boxed(frame_ptr: i64) -> i64 {
    jit_force_callee_frame(frame_ptr)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_drop_callee_frame(frame_ptr: i64) {
    if frame_ptr & 1 != 0 {
        return;
    }
    let ptr = frame_ptr as *mut PyFrame;
    if majit_metainterp::majit_log_enabled() {
        eprintln!("[jit][ca-drop] ptr={ptr:p}");
    }
    let arena = arena_ref();
    let reused = arena.put(ptr);
    if majit_metainterp::majit_log_enabled() {
        eprintln!("[jit][ca-drop] ptr={ptr:p} arena_reused={reused}");
    }
    if !reused {
        // Not an arena frame (heap fallback) — free GcPyFrame allocation.
        heap_free_frame(ptr);
    }
}

// ===========================================================================
// Blackhole helper functions
//
// RPython blackhole.py: bhimpl_recursive_call_i, bhimpl_residual_call_*
//
// These are called by the BlackholeInterpreter through JitCode.fn_ptrs.
// Residual calls execute without accidental JIT re-entry; recursive portal
// calls are routed explicitly through the jitdriver's portal runner.
// ===========================================================================

fn bh_call_self_recursive_portal(
    parent_frame_ptr: *const PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Option<i64> {
    if parent_frame_ptr.is_null() {
        return None;
    }
    let parent_frame = unsafe { &*parent_frame_ptr };
    let callable_code = unsafe { pyre_interpreter::getcode(callable) };
    if parent_frame.pycode != callable_code {
        return None;
    }
    if !recursive_force_cache_safe(callable) {
        return None;
    }

    // blackhole.py:1095-1116 bhimpl_recursive_call_* reaches the
    // jitdriver's portal runner.  This branch narrows pyre's generic
    // Python CALL helper back to that shape for self-recursive portal
    // calls; non-recursive residual calls below remain opaque plain calls.
    let frame_ptr = create_callee_frame_impl(parent_frame_ptr as i64, callable as i64, args);
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        crate::eval::portal_runner_result(frame)
    };
    jit_drop_callee_frame(frame_ptr);
    Some(match result {
        Ok(result) => result as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    })
}

/// RPython: bhimpl_recursive_call_i — call a Python function in blackhole mode.
///
/// The blackhole pops callable and args into registers before calling this.
/// blackhole.py bhimpl_residual_call parity: variable-arity call helper.
///
/// Convention: call_int_function dispatches with args=[callable, arg0, ..., argN, frame_ptr].
/// frame_ptr is always the LAST argument. The number of Python args = total_args - 2
/// (subtract callable and frame_ptr).
///
/// For nargs=0: fn(callable, frame_ptr) → 2 args
/// For nargs=1: fn(callable, arg0, frame_ptr) → 3 args
/// For nargs=2: fn(callable, arg0, arg1, frame_ptr) → 4 args
/// For nargs=3: fn(callable, arg0, arg1, arg2, frame_ptr) → 5 args
/// etc.
///
/// bhimpl_residual_call: parent frame via BH_VABLE_PTR.
/// call_int_function in machine.rs transmutes to the correct arity.
pub extern "C" fn bh_call_fn(callable: i64, arg0: i64) -> i64 {
    bh_call_fn_impl(callable as PyObjectRef, &[arg0 as PyObjectRef])
}

pub extern "C" fn bh_call_fn_0(callable: i64) -> i64 {
    bh_call_fn_impl(callable as PyObjectRef, &[])
}

pub extern "C" fn bh_call_fn_2(callable: i64, arg0: i64, arg1: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[arg0 as PyObjectRef, arg1 as PyObjectRef],
    )
}

pub extern "C" fn bh_call_fn_3(callable: i64, a0: i64, a1: i64, a2: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[a0 as PyObjectRef, a1 as PyObjectRef, a2 as PyObjectRef],
    )
}

pub extern "C" fn bh_call_fn_4(callable: i64, a0: i64, a1: i64, a2: i64, a3: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_5(callable: i64, a0: i64, a1: i64, a2: i64, a3: i64, a4: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_6(
    callable: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_7(
    callable: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
            a6 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_8(
    callable: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    a7: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
            a6 as PyObjectRef,
            a7 as PyObjectRef,
        ],
    )
}

/// blackhole.py:1224 bhimpl_residual_call: cpu.bh_call_r.
/// RPython: cpu.bh_call_r (llmodel.py:816) invokes calldescr.call_stub_r
/// directly — a plain function-pointer call, no portal_runner indirection.
/// Only bhimpl_recursive_call_* (blackhole.py:1095) uses the portal
/// runner to re-enter JIT.
fn bh_call_fn_impl(callable: PyObjectRef, args: &[PyObjectRef]) -> i64 {
    if callable.is_null() {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "call on null callable".to_string(),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    }
    // llmodel.py:822 bh_call_r — calldescr.call_stub_r is callable-type-agnostic.
    // Hot path: Function callables dispatched directly here (builtin or user
    // code), matching call_user_function_plain so eval_frame_plain is used
    // and JIT is not re-entered from the blackhole.
    // Cold path: type/method/staticmethod/classmethod/callable-instance are
    // delegated to call_function_impl_result under ForcePlainEvalGuard, which
    // mirrors baseobjspace.py:1155 dispatch without re-entering the JIT.
    if unsafe { is_function(callable) } {
        let code = unsafe { pyre_interpreter::getcode(callable) };
        if unsafe { pyre_interpreter::is_builtin_code(code as pyre_object::PyObjectRef) } {
            let func =
                unsafe { pyre_interpreter::builtin_code_get(code as pyre_object::PyObjectRef) };
            return match func(args) {
                Ok(result) if !result.is_null() => result as i64,
                Ok(_) => 0,
                Err(err) => {
                    let exc_obj = err.to_exc_object();
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
                    0
                }
            };
        }
        let parent_frame_ptr =
            majit_metainterp::blackhole::BH_VABLE_PTR.with(|c| c.get()) as *const PyFrame;
        if let Some(result) = bh_call_self_recursive_portal(parent_frame_ptr, callable, args) {
            return result;
        }
        let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
        if !parent_frame_ptr.is_null() {
            unsafe {
                pyre_interpreter::call::set_last_exec_ctx((*parent_frame_ptr).execution_context);
            }
        }
        let parent_frame = unsafe { &*parent_frame_ptr };
        let result = {
            // blackhole.py:1225 bhimpl_residual_call_* is an opaque CPU
            // call.  Only blackhole.py:1095 bhimpl_recursive_call_* reaches
            // the portal runner, so nested Python CALLs from this residual
            // path must stay on eval_frame_plain as well.
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            pyre_interpreter::call::call_user_function_plain(parent_frame, callable, args)
        };
        pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
        return match result {
            Ok(result) => result as i64,
            Err(err) => {
                let exc_obj = err.to_exc_object();
                majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
                0
            }
        };
    }
    // Cold path: type/method/staticmethod/classmethod/callable-instance.
    // Ensure LAST_EXEC_CTX reflects the caller frame pinned in BH_VABLE_PTR
    // before delegating to `call_function_impl_result`. `type_descr_call_impl`
    // → `call_user_function_with_args` reads LAST_EXEC_CTX as the fallback
    // execution context for `__new__`/`__init__` (call.rs:1104-1106); without
    // this pin it would use whatever frame last entered `eval_frame_*`,
    // which is not guaranteed to be the blackhole caller.
    let parent_frame_ptr =
        majit_metainterp::blackhole::BH_VABLE_PTR.with(|c| c.get()) as *const PyFrame;
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !parent_frame_ptr.is_null() {
        unsafe {
            pyre_interpreter::call::set_last_exec_ctx((*parent_frame_ptr).execution_context);
        }
    }
    let _plain_guard = pyre_interpreter::call::force_plain_eval();
    let result = pyre_interpreter::call::call_function_impl_result(callable, args);
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => result as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// jtransform.py parity: namespace and code come from getfield_vable_r.
/// namespace = getfield_vable_r(frame, w_globals), code = getfield_vable_r(frame, pycode).
/// namei is the raw oparg from LOAD_GLOBAL: name_idx = namei >> 1.
pub extern "C" fn bh_load_global_fn(namespace_ptr: i64, w_code_ptr: i64, namei: i64) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    let raw = namei as usize;
    let idx = raw >> 1;

    if idx >= code.names.len() {
        return 0;
    }

    let name = code.names[idx].as_ref();
    let ns = unsafe { &*(namespace_ptr as *const pyre_interpreter::DictStorage) };
    match ns.get(name) {
        Some(&value) => value as i64,
        None => {
            // NameError: set exception object in TLS.
            let err = pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::NameError,
                format!("name '{}' is not defined", name),
            );
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// Load a constant from the code object.
/// jtransform.py parity: code comes from getfield_vable_r(frame, pycode).
pub extern "C" fn bh_load_const_fn(w_code_ptr: i64, consti: i64) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    pyre_interpreter::pyframe::load_const_from_code(code, consti as usize) as i64
}

/// Box a raw integer into a PyObject (w_int_new wrapper).
pub extern "C" fn bh_box_int_fn(value: i64) -> i64 {
    w_int_new(value) as i64
}

/// `eval.rs:1049-1128 RAISE_VARARGS` normalization for blackhole/JitCode.
///
/// JitCode's `raise/r` bytecode carries only the final exception object, so
/// callers normalize `raise Type` and `raise X from Y` through this helper
/// before emitting `raise/r`.
pub extern "C" fn bh_normalize_raise_varargs_fn(exc: i64, cause: i64) -> i64 {
    let exc = exc as PyObjectRef;
    let raw_cause = cause as PyObjectRef;

    // pyopcode.py:704-722 — cause and exc normalization share
    // `self.space` / `frame.execution_context`. Pin the blackhole's
    // `BH_VABLE_PTR` frame context for the whole body so the
    // cause-class-call and exc-class-call observe the same namespace.
    let parent_frame_ptr =
        majit_metainterp::blackhole::BH_VABLE_PTR.with(|c| c.get()) as *const PyFrame;
    let frame_ctx = if parent_frame_ptr.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*parent_frame_ptr).execution_context }
    };
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !frame_ctx.is_null() {
        pyre_interpreter::call::set_last_exec_ctx(frame_ctx);
    }

    let cause = if raw_cause.is_null() {
        None
    } else {
        // pyopcode.py:706-707 — cause class-call must mirror the exc
        // class-call (pyopcode.py:711-713) on blackhole re-entry.
        // Force both onto the plain interpreter path so the constructor
        // cannot re-enter the tracer.
        let result = {
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            pyre_interpreter::eval::normalize_raise_cause(raw_cause)
        };
        match result {
            Ok(cause) => Some(cause),
            Err(err) => {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                return err.to_exc_object() as i64;
            }
        }
    };

    let mut final_exc: PyObjectRef = unsafe {
        if pyre_object::is_exception(exc) {
            exc
        } else if pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(exc) {
            if frame_ctx.is_null() {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                return pyre_interpreter::PyError::runtime_error(
                    "raise helper missing current frame",
                )
                .to_exc_object() as i64;
            }
            let result = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(exc, &[])
            };
            match result {
                Ok(obj) if pyre_object::is_exception(obj) => obj,
                Ok(_) => pyre_interpreter::PyError::type_error(
                    "exceptions must derive from BaseException",
                )
                .to_exc_object(),
                Err(err) => err.to_exc_object(),
            }
        } else {
            pyre_interpreter::PyError::type_error("exceptions must derive from BaseException")
                .to_exc_object()
        }
    };

    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);

    if let Err(err) = pyre_interpreter::eval::attach_raise_cause(final_exc, cause) {
        final_exc = err.to_exc_object();
    }
    final_exc as i64
}

/// Truthiness check: PyObjectRef → raw 0 or 1.
pub extern "C" fn bh_truth_fn(value: i64) -> i64 {
    let obj = value as PyObjectRef;
    if obj.is_null() {
        return 0;
    }
    pyre_interpreter::opcode_ops::truth_value(obj) as i64
}

/// RPython: bhimpl_int_lt, bhimpl_int_eq, etc. — comparison helper.
///
/// Performs a Python-level comparison and returns a boolean PyObject.
/// op_code encodes the CompareOp tag from CPython 3.13 COMPARE_OP.
pub extern "C" fn bh_compare_fn(lhs: i64, rhs: i64, op_code: i64) -> i64 {
    let lhs = lhs as PyObjectRef;
    let rhs = rhs as PyObjectRef;
    if lhs.is_null() || rhs.is_null() {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "comparison on null operand".to_string(),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    }

    // op_code 10 = CHECK_EXC_MATCH isinstance check (from codewriter CheckExcMatch).
    // lhs = exception value, rhs = exception type to match.
    if op_code == 10 {
        let matched = unsafe {
            if !pyre_object::is_exception(lhs) {
                true
            } else {
                let kind = pyre_object::w_exception_get_kind(lhs);
                if pyre_object::is_str(rhs) {
                    let type_name = pyre_object::w_str_get_value(rhs);
                    pyre_object::exc_kind_matches(kind, type_name)
                } else if pyre_interpreter::is_function(rhs)
                    && pyre_interpreter::is_builtin_code(
                        pyre_interpreter::function_get_code(rhs) as pyre_object::PyObjectRef
                    )
                {
                    let type_name = pyre_interpreter::function_get_name(rhs);
                    pyre_object::exc_kind_matches(kind, type_name)
                } else {
                    true
                }
            }
        };
        return pyre_object::w_bool_from(matched) as i64;
    }

    // op_code is the compact tag from compare_op_tag (0-5), NOT the raw
    // ComparisonOperator discriminant. Reverse the mapping to get the enum.
    let Some(op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_code) else {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            format!("unknown compare op tag {op_code}"),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    };
    match pyre_interpreter::opcode_ops::compare_value(lhs, rhs, op) {
        Ok(result) => result as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// RPython: bhimpl_int_add, bhimpl_int_sub, etc. — binary op helper.
///
/// Performs a Python-level binary operation.
/// op_code is the BinaryOperator tag from CPython 3.13 BINARY_OP.
pub extern "C" fn bh_binary_op_fn(lhs: i64, rhs: i64, op_code: i64) -> i64 {
    let lhs = lhs as PyObjectRef;
    let rhs = rhs as PyObjectRef;
    if lhs.is_null() || rhs.is_null() {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "binary op on null operand".to_string(),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    }

    // op_code is the compact tag from binary_op_tag (0-12), NOT the raw
    // BinaryOperator discriminant. Reverse the mapping to get the enum.
    let Some(op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_code) else {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            format!("unknown binary op tag {op_code}"),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    };
    match pyre_interpreter::opcode_ops::binary_value(lhs, rhs, op) {
        Ok(result) => result as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// STORE_SUBSCR: obj[key] = value.
/// RPython bhimpl_newlist: create a list from N items.
/// argc is a raw integer count; items follow as PyObjectRef args.
/// Blackhole's call_may_force_ref passes args from registers.
pub extern "C" fn bh_build_list_fn(argc: i64, item0: i64, item1: i64) -> i64 {
    let n = argc as usize;
    let items: Vec<pyre_object::PyObjectRef> = match n {
        0 => vec![],
        1 => vec![item0 as pyre_object::PyObjectRef],
        2 => vec![
            item0 as pyre_object::PyObjectRef,
            item1 as pyre_object::PyObjectRef,
        ],
        _ => vec![], // argc > 2 not supported via this helper
    };
    pyre_interpreter::runtime_ops::build_list_from_refs(&items) as i64
}

pub extern "C" fn bh_store_subscr_fn(obj: i64, key: i64, value: i64) -> i64 {
    let obj = obj as pyre_object::PyObjectRef;
    let key = key as pyre_object::PyObjectRef;
    let value = value as pyre_object::PyObjectRef;
    if obj.is_null() || key.is_null() {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "store subscript on null operand".to_string(),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    }
    if let Err(err) = pyre_interpreter::baseobjspace::setitem(obj, key, value) {
        let exc_obj = err.to_exc_object();
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
        return 0;
    }
    1 // success (non-zero)
}

/// Read the current (per-thread) exception saved in
/// `pyre_interpreter::eval::CURRENT_EXCEPTION`. Matches the read at
/// `pyopcode.py:786 PUSH_EXC_INFO` (implicit via `executioncontext.sys_exc_info`).
pub extern "C" fn bh_get_current_exception() -> i64 {
    pyre_interpreter::eval::get_current_exception() as i64
}

/// Store `exc` into the per-thread `CURRENT_EXCEPTION` slot. Matches
/// the write at `pyopcode.py:778 POP_EXCEPT` (restore of saved
/// sys_exc_info) and at `pyopcode.py:786 PUSH_EXC_INFO` (new raised
/// exception becomes current).
pub extern "C" fn bh_set_current_exception(exc: i64) {
    pyre_interpreter::eval::set_current_exception(exc as pyre_object::PyObjectRef);
}

#[cfg(test)]
mod tests_bh_normalize_raise {
    use super::*;
    use majit_metainterp::jitframe::{FIRST_ITEM_OFFSET, JF_FRAME_OFS};
    use pyre_interpreter::{PyErrorKind, compile_exec};

    #[test]
    fn arena_jitframe_descrs_uses_frame_relative_offsets() {
        let descrs = arena_jitframe_descrs();
        assert_eq!(descrs.jf_frame_baseitemofs, FIRST_ITEM_OFFSET);
        assert_eq!(descrs.jf_frame_lengthofs, JF_FRAME_OFS);
    }

    #[test]
    fn bh_normalize_raise_varargs_rejects_builtin_callables_that_are_not_exception_classes() {
        let code = compile_exec("x = len\n").expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code);
        frame
            .execute_frame(None, None)
            .expect("module body should execute");
        let callable = unsafe {
            (*frame.fget_w_globals())
                .get("x")
                .copied()
                .expect("namespace should contain x")
        };

        let result = bh_normalize_raise_varargs_fn(callable as i64, pyre_object::PY_NULL as i64);
        let err = unsafe { pyre_interpreter::PyError::from_exc_object(result as PyObjectRef) };
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
    }
}
