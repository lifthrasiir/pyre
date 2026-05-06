use std::cell::{Cell, RefCell, UnsafeCell};
/// Cranelift-based JIT code generation backend.
///
/// Translates majit IR traces into native code via Cranelift, then
/// executes them as ordinary function pointers.
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{
    AbiParam, BlockArg, Function, InstBuilder, MemFlags, Signature, StackSlotData, StackSlotKind,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use cranelift_codegen::ir::Value as CValue;

use majit_backend::{
    AsmInfo, BackendError, CompiledLoopToken, CompiledTraceInfo, CpuDescrAttachments,
    CpuDescrHandle, DeadFrame, ExitFrameLayout, ExitRecoveryLayout, ExitValueSourceLayout,
    ExitVirtualLayout, FailDescrLayout, JitCellToken, TerminalExitLayout,
};
use majit_gc::header::{GcHeader, TYPE_ID_MASK};
use majit_gc::rewrite::GcRewriterImpl;
use majit_gc::{GcAllocator, GcRewriter, WriteBarrierDescr};
use majit_ir::{
    AccumInfo, CallDescr, Descr, EffectInfo, FailDescr, GcRef, InputArg, OopSpecIndex, Op, OpCode,
    OpRef, OpTypeIndex, Type, Value,
};

use crate::guard::{BridgeData, CraneliftFailDescr, JitFrameDeadFrame};

// ── compile.py:665-674 done_with_this_frame singletons ──────────────
//
// Per-result-type `DoneWithThisFrame*` singletons.  Two kinds of
// instances coexist:
//
//   - the cranelift-side `CraneliftFailDescr` `LazyLock`s below, used
//     as fallback identity when the metainterp has not attached yet
//     (backend-only integration tests) and as the runtime-classifier
//     companion that carries the result_type signature downstream
//     `bridge_ref()` / `fail_arg_types` code needs; and
//
//   - `majit_ir::DescrRef`s the metainterp attaches through
//     `Backend::set_done_with_this_frame_descr_*` into the per-cpu
//     `CpuDescrAttachments` (`majit_backend::CpuDescrHandle`).  Their
//     `Arc::as_ptr` address is what gets stamped into `jf_descr` so
//     `_call_assembler_check_descr` (assembler.py:2274) observes the
//     same identity the metainterp `handle_fail` does.
//
// Fully merging them (dropping `CraneliftFailDescr` singletons and
// threading a `DescrRef`-based classifier) is tracked as a follow-up
// because it restructures the `direct_descr` / `run_compiled_code`
// type flow.

static DONE_WITH_THIS_FRAME_DESCR_INT: std::sync::LazyLock<Arc<CraneliftFailDescr>> =
    std::sync::LazyLock::new(|| {
        Arc::new(
            CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
                u32::MAX,
                0,
                vec![Type::Int],
                true,
                vec![],
                None,
            ),
        )
    });

static DONE_WITH_THIS_FRAME_DESCR_FLOAT: std::sync::LazyLock<Arc<CraneliftFailDescr>> =
    std::sync::LazyLock::new(|| {
        Arc::new(
            CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
                u32::MAX,
                0,
                vec![Type::Float],
                true,
                vec![],
                None,
            ),
        )
    });

static DONE_WITH_THIS_FRAME_DESCR_REF: std::sync::LazyLock<Arc<CraneliftFailDescr>> =
    std::sync::LazyLock::new(|| {
        Arc::new(
            CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
                u32::MAX,
                0,
                vec![Type::Ref],
                true,
                vec![],
                None,
            ),
        )
    });

/// compile.py:658-662 ExitFrameWithExceptionDescrRef singleton.
/// Paired with the metainterp-side `exit_frame_with_exception_descr_ref`
/// Arc for the runtime classifier: when `jf_descr_raw` matches the
/// metainterp Arc address, `match_metainterp_finish_descr` returns
/// this singleton so the existing `CraneliftFailDescr`-typed codepaths
/// keep working while the exit is routed into
/// `jitexc.ExitFrameWithExceptionRef`.
static EXIT_FRAME_WITH_EXCEPTION_DESCR_REF_CL: std::sync::LazyLock<Arc<CraneliftFailDescr>> =
    std::sync::LazyLock::new(|| {
        let mut d = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            u32::MAX,
            0,
            vec![Type::Ref],
            true,
            vec![],
            None,
        );
        d.is_exit_frame_with_exception = true;
        Arc::new(d)
    });

static DONE_WITH_THIS_FRAME_DESCR_VOID: std::sync::LazyLock<Arc<CraneliftFailDescr>> =
    std::sync::LazyLock::new(|| {
        Arc::new(
            CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
                u32::MAX,
                0,
                vec![],
                true,
                vec![],
                None,
            ),
        )
    });

/// compile.py:665-674 parity: return the singleton FailDescr for a
/// given Finish result type.  Used by the `direct_descr` classifier
/// path that needs a `CraneliftFailDescr` for `bridge_ref()` /
/// `increment_fail_count()` access.  FINISH descrs never actually
/// carry bridges, so the returned singleton functions purely as a
/// "finish marker" there.
fn done_with_this_frame_descr(result_types: &[Type]) -> &'static Arc<CraneliftFailDescr> {
    match result_types {
        [Type::Float] => &DONE_WITH_THIS_FRAME_DESCR_FLOAT,
        [Type::Ref] => &DONE_WITH_THIS_FRAME_DESCR_REF,
        [] => &DONE_WITH_THIS_FRAME_DESCR_VOID,
        _ => &DONE_WITH_THIS_FRAME_DESCR_INT,
    }
}

fn attached_descr_ptrs_with_fallbacks(
    attached: majit_backend::AttachedDescrPtrs,
) -> majit_backend::AttachedDescrPtrs {
    fn or_fallback(attached: usize, fallback: i64) -> usize {
        if attached != 0 {
            attached
        } else {
            fallback as usize
        }
    }
    majit_backend::AttachedDescrPtrs {
        done_with_this_frame_descr_void: or_fallback(
            attached.done_with_this_frame_descr_void,
            Arc::as_ptr(done_with_this_frame_descr(&[])) as i64,
        ),
        done_with_this_frame_descr_int: or_fallback(
            attached.done_with_this_frame_descr_int,
            Arc::as_ptr(done_with_this_frame_descr(&[Type::Int])) as i64,
        ),
        done_with_this_frame_descr_ref: or_fallback(
            attached.done_with_this_frame_descr_ref,
            Arc::as_ptr(done_with_this_frame_descr(&[Type::Ref])) as i64,
        ),
        done_with_this_frame_descr_float: or_fallback(
            attached.done_with_this_frame_descr_float,
            Arc::as_ptr(done_with_this_frame_descr(&[Type::Float])) as i64,
        ),
        exit_frame_with_exception_descr_ref: or_fallback(
            attached.exit_frame_with_exception_descr_ref,
            Arc::as_ptr(&*EXIT_FRAME_WITH_EXCEPTION_DESCR_REF_CL) as i64,
        ),
        propagate_exception_descr: attached.propagate_exception_descr,
    }
}

/// Match `jf_descr_raw` against the metainterp-attached
/// `DoneWithThisFrameDescr*` set on a specific cpu instance.  Returns
/// `Some((metainterp_arc, cranelift_singleton))` when the address is
/// one of the six attached finish descrs; the paired cranelift
/// singleton carries the result_type signature that runtime
/// classifiers (`bridge_ref`, fail_arg_types) expect.
///
/// Parameter-threaded rather than thread-local because
/// `compile.py:665 setattr(cpu, name, descr)` binds the attachment to
/// a specific `cpu` instance — every caller must identify *which* cpu
/// the descr should be resolved against, not consult an ambient slot.
fn match_metainterp_finish_descr(
    jf_descr_raw: i64,
    attachments: &CpuDescrAttachments,
) -> Option<(majit_ir::DescrRef, Arc<CraneliftFailDescr>)> {
    let ptr = jf_descr_raw as usize;
    fn check(slot: &Option<majit_ir::DescrRef>, expected: usize) -> Option<majit_ir::DescrRef> {
        slot.as_ref().and_then(|arc| {
            if Arc::as_ptr(arc) as *const () as usize == expected {
                Some(arc.clone())
            } else {
                None
            }
        })
    }
    for (slot, cl_singleton) in [
        (
            &attachments.done_with_this_frame_descr_int,
            &*DONE_WITH_THIS_FRAME_DESCR_INT,
        ),
        (
            &attachments.done_with_this_frame_descr_float,
            &*DONE_WITH_THIS_FRAME_DESCR_FLOAT,
        ),
        (
            &attachments.done_with_this_frame_descr_ref,
            &*DONE_WITH_THIS_FRAME_DESCR_REF,
        ),
        (
            &attachments.done_with_this_frame_descr_void,
            &*DONE_WITH_THIS_FRAME_DESCR_VOID,
        ),
        (
            &attachments.exit_frame_with_exception_descr_ref,
            &*EXIT_FRAME_WITH_EXCEPTION_DESCR_REF_CL,
        ),
    ] {
        if let Some(arc) = check(slot, ptr) {
            return Some((arc, cl_singleton.clone()));
        }
    }
    None
}

// ── JitFrame layout constants (jitframe.py:61-83) ───────────────────
//
// The canonical layout lives in `majit_backend::jitframe`; re-export the
// byte offsets here so uses inside this file stay terse.
use majit_backend::jitframe::{
    BASEITEMOFS, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FORWARD_OFS, JF_FRAME_OFS, JF_GCMAP_OFS,
    JF_GUARD_EXC_OFS, JF_SAVEDATA_OFS,
};
/// Byte offset of `jf_frame_length` from JitFrame start
/// (`jitframe.py:84` — `jf_frame`'s length word sits at the array base).
const JF_FRAME_LENGTH_OFS: i32 = JF_FRAME_OFS as i32;
/// Byte offset of `jf_frame[0]` from JitFrame start
/// (`jitframe.py:99` — `BASEITEMOFS`: first item follows the length word).
const JF_FRAME_ITEM0_OFS: i32 = JF_FRAME_OFS as i32 + BASEITEMOFS as i32;
/// GC type id for JitFrame objects, assigned at runtime by the frontend
/// that owns the GC type registry (pyre-jit registers OBJECT / W_INT /
/// W_FLOAT before JITFRAME, so the JITFRAME id depends on that ordering
/// and cannot be hard-coded here).
///
/// `u32::MAX` means "not yet registered"; the backend uses this in
/// `ensure_jitframe_type_registered` to set it up lazily with its own
/// custom trace hook, and in `alloc_nursery_no_collect_typed` to pass
/// the correct id to the allocator.
static JITFRAME_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);
static JITFRAME_GC_TYPE_ID_IS_EXPLICIT: AtomicBool = AtomicBool::new(false);

/// Override the JITFRAME type id explicitly. Called from the frontend
/// after it has registered its own types so that our nursery allocations
/// (gc.alloc_nursery_no_collect_typed) use the same id that the frontend
/// assigned to the JITFRAME TypeInfo (jitframe.py:48-52).
pub fn set_jitframe_gc_type_id(id: u32) {
    JITFRAME_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
    JITFRAME_GC_TYPE_ID_IS_EXPLICIT.store(true, std::sync::atomic::Ordering::Release);
}

fn jitframe_gc_type_id() -> u32 {
    JITFRAME_GC_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}

fn set_jitframe_gc_type_id_lazy(id: u32) {
    JITFRAME_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
    JITFRAME_GC_TYPE_ID_IS_EXPLICIT.store(false, std::sync::atomic::Ordering::Release);
}

fn jitframe_gc_type_id_is_explicit() -> bool {
    JITFRAME_GC_TYPE_ID_IS_EXPLICIT.load(std::sync::atomic::Ordering::Acquire)
}

/// Ensure the JITFRAME GC type is registered, and that
/// `JITFRAME_GC_TYPE_ID` reflects the id the GC assigned to it.
///
/// RPython: rgc.register_custom_trace_hook(JITFRAME, jitframe_trace)
/// called from jitframe_allocate (jitframe.py:49).
///
/// Lazy path (no frontend registration): register JITFRAME directly on
/// each allocator and return the runtime-local type id. The `TypeInfo`
/// comes from `majit_backend::jitframe::jitframe_type_info()` so the
/// layout, item size, and custom-trace hook stay in sync with every
/// other consumer of JITFRAME.
fn ensure_jitframe_type_registered(gc: &mut dyn GcAllocator) -> Option<u32> {
    if gc.type_count() == 0 {
        return None; // Stub GC (TrackingGc), no type registry
    }
    if jitframe_gc_type_id_is_explicit() {
        let id = jitframe_gc_type_id();
        assert_ne!(id, u32::MAX, "explicit JITFRAME GC type id missing");
        return Some(id);
    }
    let id = gc.register_type(majit_backend::jitframe::jitframe_type_info());
    set_jitframe_gc_type_id_lazy(id);
    Some(id)
}

/// Follow jf_forward chain to get the final jitframe address.
///
/// RPython jitframe.py:54-57 jitframe_resolve:
///   while frame.jf_forward:
///       frame = frame.jf_forward
///   return frame
fn jitframe_resolve(jf_ptr: *mut i64) -> *mut i64 {
    let mut ptr = jf_ptr;
    loop {
        let forward_addr =
            unsafe { *((ptr as *const u8).add(JF_FORWARD_OFS as usize) as *const usize) };
        if forward_addr == 0 {
            return ptr;
        }
        ptr = forward_addr as *mut i64;
    }
}

#[derive(Debug)]
struct BuiltinFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
}

impl majit_ir::Descr for BuiltinFieldDescr {
    fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self)
    }
}

impl majit_ir::FieldDescr for BuiltinFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }

    fn field_size(&self) -> usize {
        self.field_size
    }

    fn field_type(&self) -> Type {
        self.field_type
    }

    fn is_field_signed(&self) -> bool {
        self.signed
    }
}

#[derive(Debug)]
struct BuiltinArrayDescr {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_descr: Arc<BuiltinFieldDescr>,
}

impl majit_ir::Descr for BuiltinArrayDescr {
    fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
        Some(self)
    }
}

impl majit_ir::ArrayDescr for BuiltinArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }

    fn item_size(&self) -> usize {
        self.item_size
    }

    fn type_id(&self) -> u32 {
        self.type_id
    }

    fn item_type(&self) -> Type {
        self.item_type
    }

    fn is_item_signed(&self) -> bool {
        self.signed
    }

    fn len_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self.len_descr.as_ref())
    }
}

#[derive(Debug)]
pub struct CallAssemblerDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: u64,
    extra_info: EffectInfo,
}

impl CallAssemblerDescr {
    pub fn new(target_token: u64, arg_types: Vec<Type>, result_type: Type) -> Self {
        CallAssemblerDescr {
            arg_types,
            result_type,
            target_token,
            extra_info: EffectInfo {
                extraeffect: majit_ir::ExtraEffect::CanRaise,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            },
        }
    }
}

impl majit_ir::Descr for CallAssemblerDescr {
    fn as_call_descr(&self) -> Option<&dyn majit_ir::CallDescr> {
        Some(self)
    }

    fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::LoopTokenDescr> {
        Some(self)
    }
}

impl majit_ir::LoopTokenDescr for CallAssemblerDescr {
    fn loop_token_number(&self) -> u64 {
        self.target_token
    }
}

impl majit_ir::CallDescr for CallAssemblerDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }

    fn result_type(&self) -> Type {
        self.result_type
    }

    fn result_size(&self) -> usize {
        8
    }

    fn call_target_token(&self) -> Option<u64> {
        Some(self.target_token)
    }

    fn get_extra_info(&self) -> &EffectInfo {
        &self.extra_info
    }
}

#[derive(Clone)]
struct RegisteredLoopTarget {
    trace_id: u64,
    /// RPython greenkey[1]: bytecode PC of the loop header.
    /// 0 for FINISH traces (no JUMP/Label).
    header_pc: u64,
    /// RPython greenkey hash: function identifier for guard lookup.
    green_key: u64,
    caller_prefix_layout: Option<ExitRecoveryLayout>,
    code_ptr: *const u8,
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py:183-203 record_loop_or_bridge).  Position
    /// equals `descr.fail_index` by an invariant asserted at construction.
    fail_descrs: Box<[Arc<CraneliftFailDescr>]>,
    num_inputs: usize,
    num_ref_roots: usize,
    max_output_slots: usize,
    inputarg_types: Vec<Type>,
    /// virtualizable.py:86 read_boxes: number of scalar inputargs
    /// (frame + static fields). First local is at this index.
    num_scalar_inputargs: usize,
    /// pyjitpl.py:3605 — outermost_jitdriver_sd.index_of_virtualizable.
    /// Read from JitCellToken at registration time; -1 when the driver
    /// has no virtualizable.
    index_of_virtualizable: i32,
    /// `rpython/jit/backend/model.py:292-338` `CompiledLoopToken` — the
    /// per-loop metadata RPython `handle_call_assembler` (rewrite.py:665-
    /// 695) reads as `loop_token.compiled_loop_token`. Sources
    /// `_ll_initial_locs` (`regalloc.py:861-871`) and `frame_info`
    /// (`jitframe.py:30-40`) for `call_assembler_callee_locs`. Arc
    /// continuity across pending → real registration is preserved by
    /// adopting the pending Arc onto `token.compiled_loop_token` in
    /// `register_call_assembler_target`.
    compiled_loop_token: Arc<CompiledLoopToken>,
    /// `compile.py:665 setattr(cpu, name, descr)` — heap-stable clone of
    /// the owning `CraneliftBackend`'s per-cpu descr attachments.  Kept
    /// here so `execute_registered_loop_target` (a free fn reached from
    /// extern-C trampolines) can pass a snapshot into
    /// `match_metainterp_finish_descr` without a `&self` receiver.
    cpu_attachments: CpuDescrHandle,
}

unsafe impl Send for RegisteredLoopTarget {}
unsafe impl Sync for RegisteredLoopTarget {}

// history.py:470-499 TargetToken._ll_loop_code parity.
// Maps TargetToken descriptor identity (Arc::as_ptr) to the loop entry
// point needed to re-enter the target from an external JUMP exit.
// assembler.py:2456-2462 closing_jump reads `target_token._ll_loop_code`
// directly for cross-loop JMPs; since Cranelift can't emit raw inter-
// function JMPs, the dispatcher reads this entry and invokes
// run_compiled_code on the target's code_ptr.
#[derive(Clone)]
struct LoopTargetEntry {
    code_ptr: *const u8,
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py:183-203 record_loop_or_bridge).
    fail_descrs: Box<[Arc<CraneliftFailDescr>]>,
    num_inputs: usize,
    num_ref_roots: usize,
    max_output_slots: usize,
}

unsafe impl Send for LoopTargetEntry {}
unsafe impl Sync for LoopTargetEntry {}

thread_local! {
    /// Per-thread `TargetToken._ll_loop_code` registry. Same PRE-EXISTING-
    /// ADAPTATION rationale as `CALL_ASSEMBLER_TARGETS` — pyre's u64-keyed
    /// trampoline dispatch needs a runtime lookup that RPython avoids by
    /// reading `target_token._ll_loop_code` off the descr directly.
    /// Thread-local matches pyre's single-threaded JIT execution and
    /// aligns with the adjacent `CALL_ASSEMBLER_TARGETS` migration.
    static LOOP_TARGET_REGISTRY: RefCell<HashMap<usize, LoopTargetEntry>> =
        RefCell::new(HashMap::new());
}

/// history.py:470 TargetToken identity key: Arc allocation address.
/// Mirrors PyPy's `target_tokens_currently_compiling[descr] = None`
/// dict keyed by descriptor object identity.
fn register_loop_target(descr: &majit_ir::DescrRef, entry: LoopTargetEntry) {
    let key = majit_ir::descr_identity(descr);
    LOOP_TARGET_REGISTRY.with(|r| r.borrow_mut().insert(key, entry));
}

fn lookup_loop_target(descr: &majit_ir::DescrRef) -> Option<LoopTargetEntry> {
    let key = majit_ir::descr_identity(descr);
    LOOP_TARGET_REGISTRY.with(|r| r.borrow().get(&key).cloned())
}

fn deadframe_layout(frame: &DeadFrame) -> Option<FailDescrLayout> {
    frame
        .data
        .downcast_ref::<JitFrameDeadFrame>()
        .map(|jf| jf.fail_descr.layout())
}

fn overlay_deadframe_fail_descr(
    base_layout: &FailDescrLayout,
    recovery_layout: ExitRecoveryLayout,
) -> Arc<CraneliftFailDescr> {
    let mut descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
        base_layout.fail_index,
        base_layout.trace_id,
        base_layout.fail_arg_types.clone(),
        base_layout.is_finish,
        base_layout.force_token_slots.clone(),
        Some(recovery_layout),
    );
    if let Some(source_op_index) = base_layout.source_op_index {
        descr.set_source_op_index(source_op_index);
    }
    if let Some(trace_info) = base_layout.trace_info.clone() {
        descr.set_trace_info(trace_info);
    }
    Arc::new(descr)
}

fn deadframe_recovery_layout_for_call_assembler(
    layout: &FailDescrLayout,
) -> Option<ExitRecoveryLayout> {
    layout.recovery_layout.clone().or_else(|| {
        layout.trace_info.as_ref().map(|trace_info| {
            identity_recovery_layout(
                layout.trace_id,
                trace_info.header_pc,
                None, // no per-guard resume_pc available here
                trace_info.source_guard,
                &layout.fail_arg_types,
                None,
            )
        })
    })
}

fn caller_prefix_recovery_layout(
    trace_id: u64,
    header_pc: u64,
    source_guard: Option<(u64, u32)>,
    slot_types: &[Type],
    inputs: &[i64],
    caller_prefix_layout: Option<&ExitRecoveryLayout>,
) -> ExitRecoveryLayout {
    ExitRecoveryLayout {
        vable_array: Vec::new(),
        vref_array: Vec::new(),
        frames: vec![ExitFrameLayout {
            trace_id: Some(trace_id),
            header_pc: Some(header_pc),
            source_guard,
            pc: header_pc,
            // Placeholder: this caller-prefix identity layout is replaced by
            // the resume_layout-derived layout via
            // `compile::patch_backend_guard_recovery_layouts_for_trace`
            // (compile.rs:1596) before any guard can fire. The
            // resume_layout carries the authoritative jitcode_index
            // threaded from `Snapshot::single_frame`.
            jitcode_index: 0,
            slots: slot_types
                .iter()
                .enumerate()
                .map(|(slot, _)| {
                    inputs
                        .get(slot)
                        .copied()
                        .map(ExitValueSourceLayout::Constant)
                        .unwrap_or(ExitValueSourceLayout::Unavailable)
                })
                .collect(),
            slot_types: Some(slot_types.to_vec()),
        }],
        virtual_layouts: Vec::new(),
        pending_field_layouts: Vec::new(),
    }
    .prefixed_by(caller_prefix_layout)
}

fn wrap_call_assembler_deadframe_with_caller_prefix(
    mut frame: DeadFrame,
    trace_id: u64,
    header_pc: u64,
    source_guard: Option<(u64, u32)>,
    input_types: &[Type],
    inputs: &[i64],
    caller_prefix_layout: Option<&ExitRecoveryLayout>,
) -> DeadFrame {
    let Some(layout) = deadframe_layout(&frame) else {
        return frame;
    };
    let Some(inner_recovery_layout) = deadframe_recovery_layout_for_call_assembler(&layout) else {
        return frame;
    };

    let caller_layout = caller_prefix_recovery_layout(
        trace_id,
        header_pc,
        source_guard,
        input_types,
        inputs,
        caller_prefix_layout,
    );
    let recovery_layout = inner_recovery_layout.prefixed_by(Some(&caller_layout));

    // Replace fail_descr on the inner JitFrameDeadFrame to carry the
    // caller-prefixed recovery layout.  RPython has no overlay wrapper —
    // the jitframe IS the deadframe with the correct descr.
    // Write to both the Rust wrapper AND the actual jf_frame header so
    // get_latest_descr (reading wrapper) and raw jf_descr consumers agree.
    let overlay_descr = overlay_deadframe_fail_descr(&layout, recovery_layout);
    if let Some(jf) = frame.data.downcast_mut::<JitFrameDeadFrame>() {
        // llmodel.py:270 parity: frame.jf_descr = descr  (writes to frame header)
        let descr_ptr = Arc::as_ptr(&overlay_descr) as usize;
        unsafe { *((jf.jf_gcref.0 + JF_DESCR_OFS as usize) as *mut usize) = descr_ptr };
        jf.fail_descr = overlay_descr;
        return frame;
    }
    // Fallback: return as-is (should not happen after FrameData removal).
    frame
}

/// Global exception state for JIT-compiled code.
/// pyre is no-GIL single-threaded, so global statics are safe and allow
/// GUARD_NO_EXCEPTION to emit a direct memory load instead of a TLS call.
static JIT_EXC_VALUE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
static JIT_EXC_TYPE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// Return the address of JIT_EXC_VALUE for direct memory load in JIT code.
pub fn jit_exc_value_addr() -> usize {
    &JIT_EXC_VALUE as *const _ as usize
}

/// Return the address of JIT_EXC_TYPE for direct memory store in JIT code.
fn jit_exc_type_addr() -> usize {
    &JIT_EXC_TYPE as *const _ as usize
}

thread_local! {
    /// Thread-local reference storage for JIT-compiled code.
    ///
    /// Mirrors RPython's `rpy_threadlocalref` — an array of integer-sized slots
    /// indexed by offset. The interpreter stores per-thread state here (e.g.,
    /// the current action flag, signal handler data, etc.).
    ///
    /// Slots are accessed via THREADLOCALREF_GET opcode.
    static JIT_THREADLOCAL_SLOTS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

// ── Exception state shims called from JIT-compiled code ──

/// Read the current exception value (returns 0 if no exception pending).
extern "C" fn jit_exc_get_value() -> i64 {
    JIT_EXC_VALUE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Read the current exception class (returns 0 if no exception pending).
extern "C" fn jit_exc_get_type() -> i64 {
    JIT_EXC_TYPE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Clear the current exception state and return the value that was pending.
extern "C" fn jit_exc_clear_and_get_value() -> i64 {
    let val = JIT_EXC_VALUE.swap(0, std::sync::atomic::Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, std::sync::atomic::Ordering::Relaxed);
    val
}

/// Clear the current exception state.
extern "C" fn jit_exc_clear() {
    JIT_EXC_VALUE.store(0, std::sync::atomic::Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Return 1 if the current exception class exactly matches `expected_type`.
extern "C" fn jit_exc_type_matches(expected_type: i64) -> i64 {
    let exc_value = JIT_EXC_VALUE.load(std::sync::atomic::Ordering::Relaxed);
    let exc_type = JIT_EXC_TYPE.load(std::sync::atomic::Ordering::Relaxed);
    i64::from(exc_value != 0 && exc_type == expected_type)
}

/// Set the current exception state (value, class).
extern "C" fn jit_exc_restore(value: i64, exc_type: i64) {
    JIT_EXC_VALUE.store(value, std::sync::atomic::Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, std::sync::atomic::Ordering::Relaxed);
}

/// llmodel.py:194-199 _store_exception parity: set exception state.
///
/// `value` must be a valid OBJECTPTR (or 0). The exception class is
/// derived from `value.typeptr` (offset 0), matching RPython's invariant
/// that pos_exception() == ptr2int(pos_exc_value().typeptr).
pub fn jit_exc_raise(value: i64) {
    let exc_type = if value == 0 {
        0
    } else {
        unsafe { *(value as *const i64) }
    };
    JIT_EXC_VALUE.store(value, std::sync::atomic::Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, std::sync::atomic::Ordering::Relaxed);
}

/// Check if an exception is currently pending.
pub fn jit_exc_is_pending() -> bool {
    JIT_EXC_VALUE.load(std::sync::atomic::Ordering::Relaxed) != 0
}

/// RPython cpu.grab_exc_value parity: read exception class from TLS.
pub fn jit_exc_class_raw() -> i64 {
    JIT_EXC_TYPE.load(std::sync::atomic::Ordering::Relaxed)
}

/// RPython cpu.grab_exc_value parity: read exception value from TLS.
pub fn jit_exc_value_raw() -> i64 {
    JIT_EXC_VALUE.load(std::sync::atomic::Ordering::Relaxed)
}

// ── Thread-local reference access shims ──

/// Read a thread-local slot at the given offset.
///
/// Called from JIT-compiled code for THREADLOCALREF_GET operations.
/// The offset is in bytes (divided by 8 to get the slot index).
extern "C" fn jit_threadlocalref_get(offset: i64) -> i64 {
    JIT_THREADLOCAL_SLOTS.with(|slots| {
        let slots = slots.borrow();
        let idx = (offset / 8) as usize;
        slots.get(idx).copied().unwrap_or(0)
    })
}

/// Write a value to a thread-local slot at the given offset.
///
/// Called by the interpreter to set up thread-local state before entering
/// JIT-compiled code.
pub fn jit_threadlocalref_set(offset: i64, value: i64) {
    JIT_THREADLOCAL_SLOTS.with(|slots| {
        let mut slots = slots.borrow_mut();
        let idx = (offset / 8) as usize;
        if idx >= slots.len() {
            slots.resize(idx + 1, 0);
        }
        slots[idx] = value;
    });
}

/// Get the base pointer for thread-local slots.
/// Returns 0 since we use callback-based access.
pub fn jit_threadlocalref_base() -> *const i64 {
    JIT_THREADLOCAL_SLOTS.with(|slots| {
        let slots = slots.borrow();
        if slots.is_empty() {
            std::ptr::null()
        } else {
            slots.as_ptr()
        }
    })
}

// ── GIL release/reacquire shims ──

/// User-installable hook for GIL release before CallReleaseGil.
static GIL_RELEASE_HOOK: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();
/// User-installable hook for GIL reacquire after CallReleaseGil.
static GIL_REACQUIRE_HOOK: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();

/// Install custom GIL release/reacquire hooks.
pub fn set_gil_hooks(
    release: impl Fn() + Send + Sync + 'static,
    reacquire: impl Fn() + Send + Sync + 'static,
) {
    let _ = GIL_RELEASE_HOOK.set(Box::new(release));
    let _ = GIL_REACQUIRE_HOOK.set(Box::new(reacquire));
}

extern "C" fn jit_release_gil_shim() {
    if let Some(hook) = GIL_RELEASE_HOOK.get() {
        hook();
    }
}

extern "C" fn jit_reacquire_gil_shim() {
    if let Some(hook) = GIL_REACQUIRE_HOOK.get() {
        hook();
    }
}

// ── rewrite.py:489 parity: inject str_descr/unicode_descr ──
//
// Token semantics come from `symbolic.get_array_token(rstr.STR/UNICODE, ...)`
// and `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...)` (see
// `rpython/jit/backend/llsupport/symbolic.py:7,29`). See the matching
// comment in `majit-backend-dynasm/src/runner.rs` for full layout
// rationale — both backends mirror the upstream rstr layout.

/// `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...).offset`.
const BUILTIN_STRING_HASH_OFFSET: usize = 0;
/// `symbolic.get_field_token(..., 'hash', ...).size` — assert == WORD at
/// rewrite.py:286,292.
const BUILTIN_STRING_HASH_SIZE: usize = std::mem::size_of::<usize>();
/// `symbolic.get_array_token(...).ofs_length`.
const BUILTIN_STRING_LEN_OFFSET: usize = std::mem::size_of::<usize>();
/// STR token `basesize` — header + extra_item_after_alloc.
const BUILTIN_STR_TOKEN_BASE_SIZE: usize = 2 * std::mem::size_of::<usize>() + 1;
/// UNICODE token `basesize` — header only.
const BUILTIN_UNICODE_TOKEN_BASE_SIZE: usize = 2 * std::mem::size_of::<usize>();
/// Byte offset of the first `chars[]` slot at runtime, for backend tests
/// (`BUILTIN_STR_TOKEN_BASE_SIZE - 1`: subtract the extra_item_after_alloc
/// since tests address char[0] directly).
#[cfg(test)]
const BUILTIN_STRING_CHARS_OFFSET: usize = BUILTIN_STR_TOKEN_BASE_SIZE - 1;

// ---------------------------------------------------------------------------
// Helpers (free functions to avoid borrow conflicts)
// ---------------------------------------------------------------------------

thread_local! {
    /// regalloc.py:140-181 RegisterManager.{reg_bindings, longevity} parity:
    /// RPython tracks an explicit Box → register/spill-slot dict so the
    /// allocator can hand out fresh slots for each Box independently of any
    /// underlying numbering. majit's `OpRef` is a position rather than an
    /// identity, so when Phase 2 OpRefs end up sparse (Box identity put
    /// Phase 2 starting at `phase1_high_water`) the previous
    /// `Variable::from_u32(opref.raw())` mapping inflated Cranelift's Variable
    /// space with dummy declarations to fill the gaps. The thread-local
    /// map below replaces that with a sparse-OpRef → dense-Variable lookup
    /// populated during `do_compile`'s declaration loop.
    static OPREF_VAR_MAP: std::cell::RefCell<Option<std::collections::HashMap<u32, Variable>>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard that restores `OPREF_VAR_MAP` on Drop, so nested compiles
/// (bridge compilation re-entry) don't leak the inner compile's mapping
/// past its scope. The early-return branches in do_compile then don't
/// need explicit cleanup hooks.
struct OprefVarMapGuard {
    saved: Option<std::collections::HashMap<u32, Variable>>,
}

impl Drop for OprefVarMapGuard {
    fn drop(&mut self) {
        let saved = self.saved.take();
        OPREF_VAR_MAP.with(|cell| {
            *cell.borrow_mut() = saved;
        });
    }
}

fn var(idx: u32) -> Variable {
    OPREF_VAR_MAP.with(|cell| {
        if let Some(map) = cell.borrow().as_ref() {
            if let Some(&v) = map.get(&idx) {
                return v;
            }
        }
        // Pre-declaration fallback: callers that materialise a Variable
        // before do_compile populates the map (e.g. test fixtures, helper
        // utilities) keep the legacy dense mapping. This branch is also
        // taken in the do_compile pre-declaration window before the loop
        // below installs OPREF_VAR_MAP.
        Variable::from_u32(idx)
    })
}

/// Convert a slice of Values to a Vec of BlockArgs for Cranelift 0.130 branch instructions.
fn block_args(vals: &[CValue]) -> Vec<BlockArg> {
    vals.iter().copied().map(BlockArg::from).collect()
}

/// Whether to use native SIMD (I64X2/F64X2) for Vec* codegen.
/// When false, falls back to scalar emulation.
const USE_NATIVE_SIMD: bool = true;

/// Returns true if `opcode` is a Vec* opcode that produces a vector-typed value
/// (I64X2 or F64X2). VecUnpack* opcodes produce scalars, so they return false.
/// Guard/void opcodes also return false.
fn is_vec_producing_opcode(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::VecIntAdd
            | OpCode::VecIntSub
            | OpCode::VecIntMul
            | OpCode::VecIntAnd
            | OpCode::VecIntOr
            | OpCode::VecIntXor
            | OpCode::VecFloatAdd
            | OpCode::VecFloatSub
            | OpCode::VecFloatMul
            | OpCode::VecFloatTrueDiv
            | OpCode::VecFloatNeg
            | OpCode::VecFloatAbs
            | OpCode::VecFloatXor
            | OpCode::VecI
            | OpCode::VecF
            | OpCode::VecPackI
            | OpCode::VecPackF
            | OpCode::VecExpandI
            | OpCode::VecExpandF
            | OpCode::VecLoadI
            | OpCode::VecLoadF
    )
}

/// Returns true if `opcode` is a Vec* float opcode producing F64X2.
fn is_vec_float_producing(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::VecFloatAdd
            | OpCode::VecFloatSub
            | OpCode::VecFloatMul
            | OpCode::VecFloatTrueDiv
            | OpCode::VecFloatNeg
            | OpCode::VecFloatAbs
            | OpCode::VecFloatXor
            | OpCode::VecF
            | OpCode::VecPackF
            | OpCode::VecExpandF
            | OpCode::VecLoadF
    )
}

/// Pre-scan ops to find positions that produce vector values.
fn build_vec_oprefs(ops: &[Op], num_inputs: usize) -> HashSet<u32> {
    if !USE_NATIVE_SIMD {
        return HashSet::new();
    }
    let mut set = HashSet::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if is_vec_producing_opcode(op.opcode) {
            let vi = op_var_index(op, op_idx, num_inputs) as u32;
            set.insert(vi);
        }
    }
    set
}

/// Pre-scan to find which vector-producing positions have float type (F64X2).
fn build_vec_float_oprefs(ops: &[Op], num_inputs: usize) -> HashSet<u32> {
    if !USE_NATIVE_SIMD {
        return HashSet::new();
    }
    let mut set = HashSet::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if is_vec_float_producing(op.opcode) {
            let vi = op_var_index(op, op_idx, num_inputs) as u32;
            set.insert(vi);
        }
    }
    set
}

/// Resolve an OpRef as a vector value (I64X2). If the OpRef refers to a
/// vector-producing op, uses the variable directly (bitcasting F64X2 to
/// I64X2 if needed). If it's a constant, splats it to fill all lanes.
fn resolve_opref_vec_int(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    vec_oprefs: &HashSet<u32>,
    vec_float_oprefs: &HashSet<u32>,
    opref: OpRef,
) -> CValue {
    if let Some(&c) = constants.get(&opref.raw()) {
        let scalar = builder.ins().iconst(cl_types::I64, c);
        return builder.ins().splat(cl_types::I64X2, scalar);
    }
    if vec_oprefs.contains(&opref.raw()) {
        let v = builder.use_var(var(opref.raw()));
        if vec_float_oprefs.contains(&opref.raw()) {
            // Variable is F64X2, bitcast to I64X2
            return builder.ins().bitcast(cl_types::I64X2, MemFlags::new(), v);
        }
        return v;
    }
    // Scalar variable referenced in a vector context: splat it
    let scalar = builder.use_var(var(opref.raw()));
    builder.ins().splat(cl_types::I64X2, scalar)
}

/// Resolve an OpRef as a vector float value (F64X2). If it's a constant,
/// reinterpret the i64 bits as f64 and splat. If it's a vector var,
/// bitcast to F64X2 if needed. If scalar, bitcast to f64 then splat.
fn resolve_opref_vec_float(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    vec_oprefs: &HashSet<u32>,
    vec_float_oprefs: &HashSet<u32>,
    opref: OpRef,
) -> CValue {
    if let Some(&c) = constants.get(&opref.raw()) {
        let scalar_i = builder.ins().iconst(cl_types::I64, c);
        let scalar_f = builder
            .ins()
            .bitcast(cl_types::F64, MemFlags::new(), scalar_i);
        return builder.ins().splat(cl_types::F64X2, scalar_f);
    }
    if vec_oprefs.contains(&opref.raw()) {
        let v = builder.use_var(var(opref.raw()));
        if vec_float_oprefs.contains(&opref.raw()) {
            // Already F64X2
            return v;
        }
        // I64X2 -> F64X2 bitcast
        return builder.ins().bitcast(cl_types::F64X2, MemFlags::new(), v);
    }
    // Scalar variable: bitcast i64 -> f64 then splat
    let scalar = builder.use_var(var(opref.raw()));
    let scalar_f = builder
        .ins()
        .bitcast(cl_types::F64, MemFlags::new(), scalar);
    builder.ins().splat(cl_types::F64X2, scalar_f)
}

/// Map a majit Type to the corresponding Cranelift IR type for call signatures.
fn cranelift_type_for(tp: &Type) -> cranelift_codegen::ir::Type {
    match tp {
        Type::Int | Type::Ref => cl_types::I64,
        Type::Float => cl_types::F64,
        Type::Void => cl_types::I64,
    }
}

thread_local! {
    /// `llmodel.py:58` `self.gc_ll_descr = get_ll_description(...)` — owned
    /// by the active cranelift backend on this thread. Stored as a
    /// thread-local so the backend-agnostic `majit_gc::ActiveGcGuardHooks`
    /// shims and trampoline-baked C addresses can reach the live allocator
    /// without taking a cranelift dependency. Mirrors
    /// `majit-backend-dynasm/src/runner.rs:46 DYNASM_ACTIVE_GC`.
    static CRANELIFT_ACTIVE_GC: RefCell<Option<Box<dyn GcAllocator>>> =
        const { RefCell::new(None) };
    /// JITFRAME `Type` id of the active GC runtime. Lazy-registered when
    /// the GC sees its first JITFRAME, cleared when the active GC is
    /// replaced or torn down.
    static CRANELIFT_JITFRAME_TYPE_ID: Cell<Option<u32>> = const { Cell::new(None) };
}

fn with_cranelift_gc<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> Option<R> {
    CRANELIFT_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        match guard.as_deref_mut() {
            Some(gc) => Some(f(gc)),
            None => None,
        }
    })
}

fn with_cranelift_gc_required<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> R {
    CRANELIFT_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        let gc = guard.as_deref_mut().expect("missing active GC runtime");
        f(gc)
    })
}

fn set_cranelift_active_gc(gc: Option<Box<dyn GcAllocator>>) {
    CRANELIFT_ACTIVE_GC.with(|cell| *cell.borrow_mut() = gc);
}

fn cranelift_jitframe_type_id() -> Option<u32> {
    CRANELIFT_JITFRAME_TYPE_ID.with(|c| c.get())
}

fn set_cranelift_jitframe_type_id(type_id: Option<u32>) {
    CRANELIFT_JITFRAME_TYPE_ID.with(|c| c.set(type_id));
}

fn cranelift_gc_active() -> bool {
    CRANELIFT_ACTIVE_GC.with(|cell| cell.borrow().is_some())
}

/// `majit_gc::CheckIsObjectFn` installed by `set_gc_allocator`. Dispatches
/// `gc.py:631-642 check_is_object` through the thread-local
/// `CRANELIFT_ACTIVE_GC` so backend-agnostic callers (optimizer) can
/// reach the live GC allocator without taking a cranelift dependency.
fn check_is_object_via_active_runtime(gcref: GcRef) -> bool {
    with_cranelift_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
}

/// `majit_gc::GetActualTypeidFn` installed by `set_gc_allocator`.
/// Mirrors `gc.py:624-629 get_actual_typeid`: extracts the managed GC
/// header half-word typeid, falling back to the
/// `vtable_to_type_id` table populated via `register_vtable_for_type`
/// for pyre's foreign PyObject layout.
fn get_actual_typeid_via_active_runtime(gcref: GcRef) -> Option<u32> {
    with_cranelift_gc(|gc| gc.get_actual_typeid(gcref)).flatten()
}

/// `majit_gc::SubclassRangeFn` installed by `set_gc_allocator`.
/// Resolves the codegen-time `rclass.CLASSTYPE.subclassrange_{min,max}`
/// lookup from x86/assembler.py:1971-1974 through the GC's
/// vtable→typeid table.
fn subclass_range_via_active_runtime(classptr: usize) -> Option<(i64, i64)> {
    with_cranelift_gc(|gc| gc.subclass_range(classptr)).flatten()
}

/// `majit_gc::TypeidSubclassRangeFn` installed by `set_gc_allocator`.
/// Companion to `subclass_range` keyed by typeid — the executor uses
/// it to recover `value.typeptr.subclassrange_min` after extracting
/// the typeid via `get_actual_typeid`.
fn typeid_subclass_range_via_active_runtime(typeid: u32) -> Option<(i64, i64)> {
    with_cranelift_gc(|gc| gc.typeid_subclass_range(typeid)).flatten()
}

/// `majit_gc::TypeidIsObjectFn` installed by `set_gc_allocator`.
/// Answers "does this typeid have `T_IS_RPYTHON_INSTANCE` set in its
/// TYPE_INFO entry" for the executor's `GuardIsObject` arm.
fn typeid_is_object_via_active_runtime(typeid: u32) -> Option<bool> {
    with_cranelift_gc(|gc| gc.typeid_is_object(typeid)).flatten()
}

/// `majit_gc::AllocNurseryTypedFn` installed by `set_gc_allocator`.
/// Routes host-side allocations (pyre-object `w_int_new`, …) through
/// the active cranelift-owned GC. Returns `GcRef(0)` when no active
/// runtime is set on this thread.
///
/// Uses `alloc_nursery_no_collect_typed`: host callers hold a raw
/// `*mut u8` on the Rust stack that is NOT a registered GC root.
/// Triggering a minor collection here would move the nursery object
/// we just returned, leaving the caller dangling. `no_collect` falls
/// back to old-gen on nursery full instead.
fn alloc_nursery_typed_via_active_runtime(type_id: u32, size: usize) -> GcRef {
    with_cranelift_gc(|gc| gc.alloc_nursery_no_collect_typed(type_id, size)).unwrap_or(GcRef(0))
}

/// `majit_gc::AllocOldgenTypedFn` installed by `set_gc_allocator`.
/// Routes host-side allocations that need a stable (non-moving)
/// pointer through the active cranelift-owned GC's old-gen. Used by
/// pyre-object `w_int_new` / `w_float_new` non-cached paths (Task
/// #141) where the caller holds the returned pointer on the Rust
/// stack across subsequent allocations. Returns `GcRef(0)` when no
/// active runtime is set on this thread.
fn alloc_oldgen_typed_via_active_runtime(type_id: u32, size: usize) -> GcRef {
    with_cranelift_gc(|gc| gc.alloc_oldgen_typed(type_id, size)).unwrap_or(GcRef(0))
}

/// Host-side root-register trampoline (Task #141 option a). Bridges
/// `majit_gc::gc_add_root` to the active cranelift-owned GC's
/// `RootSet`.
///
/// # Safety
/// Caller must keep `slot` valid until [`gc_remove_root_via_active_runtime`]
/// is called with the same pointer.
unsafe fn gc_add_root_via_active_runtime(slot: *mut GcRef) {
    with_cranelift_gc(|gc| unsafe { gc.add_root(slot) });
}

/// Companion to [`gc_add_root_via_active_runtime`].
fn gc_remove_root_via_active_runtime(slot: *mut GcRef) {
    with_cranelift_gc(|gc| gc.remove_root(slot));
}

/// Host-side `is_managed_heap_object` trampoline. Lets host-side
/// allocators (`pyre_object::dealloc_items_block`) discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks during the L1/L2 stepping-stone window.
fn gc_owns_object_via_active_runtime(addr: usize) -> bool {
    with_cranelift_gc(|gc| gc.is_managed_heap_object(addr)).unwrap_or(false)
}

/// Returns true when the active GC was present and roots were
/// registered; false when no GC is active and registration was a
/// no-op. Callers pair the bool with `unregister_gc_roots` on Drop.
pub(crate) fn register_gc_roots(roots: &mut [GcRef]) -> bool {
    if roots.is_empty() {
        return false;
    }
    with_cranelift_gc(|gc| {
        for root in roots.iter_mut() {
            unsafe {
                gc.add_root(root as *mut GcRef);
            }
        }
    })
    .is_some()
}

pub(crate) fn unregister_gc_roots(roots: &mut [GcRef]) {
    if roots.is_empty() {
        return;
    }
    with_cranelift_gc(|gc| {
        for root in roots.iter_mut() {
            gc.remove_root(root as *mut GcRef);
        }
    });
}

struct GcRootSlotGuard {
    slots: Vec<*mut GcRef>,
    registered: bool,
}

impl GcRootSlotGuard {
    fn new(slots: Vec<*mut GcRef>) -> Self {
        if slots.is_empty() {
            return Self {
                slots,
                registered: false,
            };
        }
        let registered = with_cranelift_gc(|gc| {
            for &slot in &slots {
                unsafe {
                    gc.add_root(slot);
                }
            }
        })
        .is_some();
        Self { slots, registered }
    }
}

impl Drop for GcRootSlotGuard {
    fn drop(&mut self) {
        if !self.registered {
            return;
        }
        with_cranelift_gc(|gc| {
            for &slot in &self.slots {
                gc.remove_root(slot);
            }
        });
    }
}

fn write_barrier_if_managed(obj: GcRef) {
    if obj.is_null() {
        return;
    }
    with_cranelift_gc(|gc| {
        if gc.is_managed_heap_object(obj.0) {
            gc.write_barrier(obj);
        }
    });
}

/// RPython rebuild_state_after_failure parity: materialize virtual objects
/// in raw fail_args before bridge/blackhole dispatch, using recovery_layout.
///
/// resume.py:1042-1057 rebuild_from_resumedata processes ALL frames in the
/// recovery layout (outermost first). call_assembler prefixes caller frames
/// via prefixed_by() at compiler.rs:512, so frames is [caller..., callee].
/// The blackhole consumer at call_jit.rs:806 expects a full section chain.
///
/// After frame reconstruction, pending_field_layouts are replayed
/// (resume.py:1003-1007 setfield/setarrayitem parity).
fn rebuild_state_after_failure(
    outputs: &mut Vec<i64>,
    types: &[majit_ir::Type],
    recovery: Option<&majit_backend::ExitRecoveryLayout>,
    _bridge_num_inputs: usize,
) {
    // Phase 1: recovery_layout-based materialization (rd_virtuals parity).
    // Process ALL frames, not just frames[0].
    if let Some(recovery) = recovery {
        if !recovery.frames.is_empty() {
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[rebuild] outputs.len={} virtual_layouts.len={} pending={}",
                    outputs.len(),
                    recovery.virtual_layouts.len(),
                    recovery.pending_field_layouts.len()
                );
                for (vi, vl) in recovery.virtual_layouts.iter().enumerate() {
                    eprintln!("[rebuild] virtual_layout[{}]={:?}", vi, vl);
                }
            }
            // Step 1: materialize all virtuals referenced by any frame.
            // RPython liveboxes are GC-visible during resume decoding.  Pyre's
            // backend holds them in raw i64 vectors, so register Ref-typed
            // output slots and materialized virtual slots while allocation can
            // trigger a nursery collection.
            let mut materialized_roots: Vec<GcRef> =
                vec![GcRef::NULL; recovery.virtual_layouts.len()];
            let mut materialized_done: Vec<bool> = vec![false; recovery.virtual_layouts.len()];
            let mut root_slots: Vec<*mut GcRef> = Vec::new();
            for (i, tp) in types.iter().enumerate().take(outputs.len()) {
                if *tp == majit_ir::Type::Ref {
                    root_slots.push((&mut outputs[i]) as *mut i64 as *mut GcRef);
                }
            }
            for root in materialized_roots.iter_mut() {
                root_slots.push(root as *mut GcRef);
            }
            let _root_guard = GcRootSlotGuard::new(root_slots);

            // Step 2: rebuild ALL frames' slots, concatenated in
            // callee-first order. ExitRecoveryLayout stores frames
            // outermost-first (caller, callee) per the
            // `opencoder.py:217` `framestack.reverse()` parity, while
            // the deadframe layout that downstream consumers
            // (`restore_guard_failure_values`) read expects innermost-
            // first concatenation, so reverse on read.
            let mut rebuilt = Vec::new();
            for frame_layout in recovery.frames.iter().rev() {
                for slot in &frame_layout.slots {
                    match slot {
                        majit_backend::ExitValueSourceLayout::ExitValue(idx) => {
                            rebuilt.push(outputs.get(*idx).copied().unwrap_or(0));
                        }
                        majit_backend::ExitValueSourceLayout::Constant(c) => {
                            rebuilt.push(*c);
                        }
                        majit_backend::ExitValueSourceLayout::Virtual(vidx) => {
                            rebuilt.push(materialize_virtual_recursive(
                                *vidx,
                                &recovery.virtual_layouts,
                                outputs,
                                &mut materialized_roots,
                                &mut materialized_done,
                            ));
                        }
                        _ => rebuilt.push(0),
                    }
                }
            }

            // Step 3: replay pending field writes (resume.py:1003-1007).
            // Must happen AFTER materialization so target/value virtuals
            // resolve to concrete pointers.
            let resolve_pending = |src: &majit_backend::ExitValueSourceLayout,
                                   materialized: &[GcRef]|
             -> Option<i64> {
                match src {
                    majit_backend::ExitValueSourceLayout::ExitValue(idx) => {
                        outputs.get(*idx).copied()
                    }
                    majit_backend::ExitValueSourceLayout::Constant(c) => Some(*c),
                    majit_backend::ExitValueSourceLayout::Virtual(vidx) => materialized
                        .get(*vidx)
                        .map(|root| root.0 as i64)
                        .filter(|value| *value != 0),
                    _ => None,
                }
            };
            for pf in &recovery.pending_field_layouts {
                let Some(target_ptr) = resolve_pending(&pf.target, &materialized_roots) else {
                    continue;
                };
                let Some(value_raw) = resolve_pending(&pf.value, &materialized_roots) else {
                    continue;
                };
                if target_ptr == 0 {
                    continue;
                }
                // resume.py:1003-1007 _prepare_pendingfields parity:
                //   if itemindex < 0: setfield → base + offset
                //   else:             setarrayitem → base + offset + idx * size
                let addr = if pf.is_array_item {
                    let item_index = pf.item_index.unwrap_or(0);
                    target_ptr as usize + pf.field_offset + item_index * pf.field_size
                } else {
                    target_ptr as usize + pf.field_offset
                };
                if pf.field_type == majit_ir::Type::Ref {
                    write_barrier_if_managed(GcRef(target_ptr as usize));
                }
                unsafe {
                    match pf.field_size {
                        8 => std::ptr::write(addr as *mut i64, value_raw),
                        4 => std::ptr::write(addr as *mut i32, value_raw as i32),
                        2 => std::ptr::write(addr as *mut i16, value_raw as i16),
                        1 => std::ptr::write(addr as *mut u8, value_raw as u8),
                        _ => {}
                    }
                }
            }

            *outputs = rebuilt;
            return;
        }
    }
    // resume.py:945/993 parity: virtual materialization is handled by
    // rd_virtuals (Phase 1 above). No virtual pair compaction needed.
    REBUILD_STATE_AFTER_FAILURE.with(|c| {
        if let Some(f) = c.get() {
            f(outputs, types);
        }
    });
}

/// Materialize a single virtual from its layout and raw output values.
/// Uses the REBUILD_STATE_AFTER_FAILURE callback for actual object allocation.
/// resume.py:617-621 VirtualInfo.allocate parity: cycle-safe materialization.
///
/// Phase 1: allocate struct (allocate_with_vtable / allocate_struct)
/// Phase 2: set_ptr(index, struct) — cache BEFORE setfields
/// Phase 3: setfields — fill fields (may reference self via cycle)
fn materialize_virtual_recursive(
    vidx: usize,
    virtual_layouts: &[majit_backend::ExitVirtualLayout],
    outputs: &[i64],
    materialized: &mut [GcRef],
    materialized_done: &mut [bool],
) -> i64 {
    if materialized_done.get(vidx).copied().unwrap_or(false) {
        return materialized.get(vidx).copied().unwrap_or(GcRef::NULL).0 as i64;
    }
    let Some(vl) = virtual_layouts.get(vidx) else {
        return 0;
    };

    // Phase 1: allocate (without setfields)
    let ptr = match vl {
        // resume.py:617-621 VirtualInfo.allocate → allocate_with_vtable(descr).
        majit_backend::ExitVirtualLayout::Object {
            descr,
            known_class,
            fields,
            fielddescrs,
            descr_size,
            type_id,
            ..
        } => {
            let ob_type = known_class.unwrap_or(0);

            // Fast path: W_IntObject/W_FloatObject (1 data field + known_class)
            if fields.len() == 1 && ob_type != 0 {
                let data_val = fields
                    .first()
                    .and_then(|(_, src)| resolve_virtual_field_value(src, outputs))
                    .unwrap_or(0);
                let mut temp = vec![0i64, ob_type, data_val];
                let temp_types = vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ];
                {
                    let _temp_root_guard =
                        GcRootSlotGuard::new(vec![(&mut temp[0]) as *mut i64 as *mut GcRef]);
                    REBUILD_STATE_AFTER_FAILURE.with(|c| {
                        if let Some(f) = c.get() {
                            f(&mut temp, &temp_types);
                        }
                    });
                }
                if temp[0] != 0 {
                    if vidx < materialized.len() {
                        materialized[vidx] = GcRef(temp[0] as usize);
                    }
                    if vidx < materialized_done.len() {
                        materialized_done[vidx] = true;
                    }
                    return temp[0];
                }
            }

            // resume.py:617 bh_new_with_vtable(descr): allocate via live SizeDescr.
            let (alloc_size, alloc_tid) = descr
                .as_ref()
                .and_then(|d| d.as_size_descr())
                .map(|sd| (sd.size(), sd.type_id()))
                .unwrap_or((*descr_size, *type_id));
            if fielddescrs.is_empty() && alloc_size == 0 && ob_type == 0 {
                return 0;
            }
            let size = if alloc_size > 0 { alloc_size } else { 16 };
            let ptr = if ob_type != 0 {
                if let Some(p) =
                    with_cranelift_gc(|gc| gc.alloc_nursery_typed(alloc_tid, size).0 as i64)
                {
                    if p != 0 {
                        unsafe { *(p as *mut i64) = ob_type };
                    }
                    p
                } else {
                    let layout = match std::alloc::Layout::from_size_align(size, 8) {
                        Ok(l) => l,
                        Err(_) => return 0,
                    };
                    let p = unsafe { std::alloc::alloc_zeroed(layout) as *mut u8 };
                    if p.is_null() {
                        return 0;
                    }
                    unsafe { *(p as *mut i64) = ob_type };
                    p as i64
                }
            } else {
                let layout = match std::alloc::Layout::from_size_align(size, 8) {
                    Ok(l) => l,
                    Err(_) => return 0,
                };
                let p = unsafe { std::alloc::alloc_zeroed(layout) as *mut u8 };
                if p.is_null() {
                    return 0;
                }
                p as i64
            };
            ptr
        }
        // resume.py:634-637 VStructInfo.allocate → allocate_struct(typedescr).
        majit_backend::ExitVirtualLayout::Struct {
            typedescr,
            type_id,
            fielddescrs,
            descr_size,
            ..
        } => {
            // resume.py:635 allocate_struct(self.typedescr): use live SizeDescr.
            let (alloc_size, alloc_tid) = typedescr
                .as_ref()
                .and_then(|d| d.as_size_descr())
                .map(|sd| (sd.size(), sd.type_id()))
                .unwrap_or((*descr_size, *type_id));
            if fielddescrs.is_empty() && alloc_size == 0 {
                return 0;
            }
            let size = if alloc_size > 0 { alloc_size } else { 16 };
            if let Some(p) =
                with_cranelift_gc(|gc| gc.alloc_nursery_typed(alloc_tid, size).0 as i64)
            {
                p
            } else {
                let layout = match std::alloc::Layout::from_size_align(size, 8) {
                    Ok(l) => l,
                    Err(_) => return 0,
                };
                unsafe { std::alloc::alloc_zeroed(layout) as i64 }
            }
        }
        // resume.py:763-779 VStrPlainInfo.allocate /
        // resume.py:817-829 VUniPlainInfo.allocate — bh_newstr/bh_newunicode
        // + per-char bh_strsetitem/bh_unicodesetitem. Frontend-provided
        // callback because the primitives live on pyre-jit's Backend impl.
        majit_backend::ExitVirtualLayout::StrPlain { is_unicode, chars } => {
            let resolved_chars: Vec<i64> = chars
                .iter()
                .map(|src| {
                    resolve_virtual_field_value_with_materialized(src, outputs, materialized)
                        .unwrap_or(0)
                })
                .collect();
            let Some(f) = MATERIALIZE_STR_PLAIN_FN.with(|c| c.get()) else {
                return 0;
            };
            f(*is_unicode, &resolved_chars)
        }
        // resume.py:781-796 VStrConcatInfo.allocate /
        // resume.py:830-848 VUniConcatInfo.allocate —
        //     left = decoder.decode_ref(fieldnums[0])
        //     right = decoder.decode_ref(fieldnums[1])
        //     return string_concat(metainterp_sd, left, right)
        // string_concat: bh_call_r(func, [], [left, right], [], calldescr).
        majit_backend::ExitVirtualLayout::StrConcat {
            is_unicode,
            func,
            calldescr,
            left,
            right,
        } => {
            let left_val =
                resolve_virtual_field_value_with_materialized(left, outputs, materialized)
                    .unwrap_or(0);
            let right_val =
                resolve_virtual_field_value_with_materialized(right, outputs, materialized)
                    .unwrap_or(0);
            let Some(f) = MATERIALIZE_STR_CALL_FN.with(|c| c.get()) else {
                return 0;
            };
            f(*is_unicode, *func, calldescr, &[], &[left_val, right_val])
        }
        // resume.py:799-813 VStrSliceInfo.allocate /
        // resume.py:854-868 VUniSliceInfo.allocate —
        //     largerstr = decoder.decode_ref(fieldnums[0])
        //     start = decoder.decode_int(fieldnums[1])
        //     length = decoder.decode_int(fieldnums[2])
        //     return slice_string(metainterp_sd, largerstr, start, start + length)
        // slice_string: bh_call_r(func, [start, stop], [str], [], calldescr).
        majit_backend::ExitVirtualLayout::StrSlice {
            is_unicode,
            func,
            calldescr,
            str_src,
            start,
            length,
        } => {
            let str_val =
                resolve_virtual_field_value_with_materialized(str_src, outputs, materialized)
                    .unwrap_or(0);
            let start_val =
                resolve_virtual_field_value_with_materialized(start, outputs, materialized)
                    .unwrap_or(0);
            let length_val =
                resolve_virtual_field_value_with_materialized(length, outputs, materialized)
                    .unwrap_or(0);
            // resume.py:1474 / 1501 — slice_string(str, start, start + length)
            // passes the stop index, not the length.
            let stop_val = start_val + length_val;
            let Some(f) = MATERIALIZE_STR_CALL_FN.with(|c| c.get()) else {
                return 0;
            };
            f(
                *is_unicode,
                *func,
                calldescr,
                &[start_val, stop_val],
                &[str_val],
            )
        }
        // Remaining variants — Array, ArrayStruct, RawBuffer, RawSlice
        // — are still frontend-ceded because the host allocator
        // (allocate_array / new_raw_buffer) is not yet exposed through
        // a backend callback. The frontend's rebuild pass overwrites
        // the null pointer produced here with the real allocation.
        _ => return 0,
    };

    // Phase 2: set_ptr — cache BEFORE setfields (cycle-safe)
    if vidx < materialized.len() {
        materialized[vidx] = GcRef(ptr as usize);
    }
    if vidx < materialized_done.len() {
        materialized_done[vidx] = true;
    }

    // Phase 3: setfields — may recurse into nested virtuals
    match vl {
        majit_backend::ExitVirtualLayout::Object {
            fields,
            fielddescrs,
            ..
        }
        | majit_backend::ExitVirtualLayout::Struct {
            fields,
            fielddescrs,
            ..
        } => {
            // ob_type is set from known_class during allocation (Phase 1).
            // fields only contain data fields (ob_type excluded).
            for (i, (_, src)) in fields.iter().enumerate() {
                if let Some(fd) = fielddescrs.get(i) {
                    {
                        // Resolve with materialized (nested virtuals now available)
                        let val = resolve_virtual_field_value_with_materialized(
                            src,
                            outputs,
                            materialized,
                        )
                        .unwrap_or_else(|| {
                            // Nested virtual not yet materialized — recurse
                            if let majit_backend::ExitValueSourceLayout::Virtual(nv) = src {
                                materialize_virtual_recursive(
                                    *nv,
                                    virtual_layouts,
                                    outputs,
                                    materialized,
                                    materialized_done,
                                )
                            } else {
                                0
                            }
                        });
                        unsafe {
                            let addr = (ptr as *mut u8).add(fd.offset);
                            match fd.field_type {
                                majit_ir::Type::Ref => {
                                    write_barrier_if_managed(GcRef(ptr as usize));
                                    std::ptr::write(addr as *mut i64, val)
                                }
                                majit_ir::Type::Float => {
                                    std::ptr::write(addr as *mut u64, val as u64)
                                }
                                _ => match fd.field_size {
                                    1 => std::ptr::write(addr, val as u8),
                                    2 => std::ptr::write(addr as *mut u16, val as u16),
                                    4 => std::ptr::write(addr as *mut u32, val as u32),
                                    _ => std::ptr::write(addr as *mut i64, val),
                                },
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    ptr
}

fn resolve_virtual_field_value(
    source: &majit_backend::ExitValueSourceLayout,
    outputs: &[i64],
) -> Option<i64> {
    match source {
        majit_backend::ExitValueSourceLayout::ExitValue(idx) => outputs.get(*idx).copied(),
        majit_backend::ExitValueSourceLayout::Constant(c) => Some(*c),
        _ => None,
    }
}

/// resolve_virtual_field_value with nested virtual support.
/// resume.py:1552-1573: TAGVIRTUAL → getvirtual_ptr/int.
fn resolve_virtual_field_value_with_materialized(
    source: &majit_backend::ExitValueSourceLayout,
    outputs: &[i64],
    materialized: &[GcRef],
) -> Option<i64> {
    match source {
        majit_backend::ExitValueSourceLayout::ExitValue(idx) => outputs.get(*idx).copied(),
        majit_backend::ExitValueSourceLayout::Constant(c) => Some(*c),
        majit_backend::ExitValueSourceLayout::Virtual(vidx) => materialized
            .get(*vidx)
            .map(|root| root.0 as i64)
            .filter(|value| *value != 0),
        _ => None,
    }
}

thread_local! {
    /// Per-thread CALL_ASSEMBLER target registry.
    ///
    /// RPython `CallAssemblerDescr.loop_token` carries a direct
    /// `JitCellToken` reference, so `handle_call_assembler`
    /// (rewrite.py:665-695) reads metadata via
    /// `loop_token.compiled_loop_token` with no indirection. pyre's
    /// backends emit u64-keyed trampolines, so a `token_number ->
    /// RegisteredLoopTarget` map is a PRE-EXISTING-ADAPTATION required
    /// for the u64 dispatch. Storing the map thread-locally (matching
    /// `CALL_ASSEMBLER_DEADFRAMES` below) keeps cranelift's per-thread
    /// CA state separated and avoids a process-global registry
    /// (`plan:staged-sauteeing-koala.md` Phase 2 goal).
    static CALL_ASSEMBLER_TARGETS: RefCell<HashMap<u64, RegisteredLoopTarget>> =
        RefCell::new(HashMap::new());
    /// Per-thread caller->callee result-kind expectations. Same
    /// PRE-EXISTING-ADAPTATION rationale as `CALL_ASSEMBLER_TARGETS`.
    static CALL_ASSEMBLER_EXPECTATIONS: RefCell<HashMap<u64, HashMap<CallAssemblerCallerId, u64>>> =
        RefCell::new(HashMap::new());
    /// Thread-local deadframe storage for call_assembler results.
    /// Each test thread gets its own isolated registry, preventing
    /// non-deterministic failures from shared global state.
    static CALL_ASSEMBLER_DEADFRAMES: RefCell<HashMap<u64, DeadFrame>> = RefCell::new(HashMap::new());
    static NEXT_CALL_ASSEMBLER_DEADFRAME_HANDLE: Cell<u64> = const { Cell::new(1) };
}

const CALL_ASSEMBLER_OUTCOME_FINISH: i64 = 0;
const CALL_ASSEMBLER_OUTCOME_DEADFRAME: i64 = 1;

/// Callback to "force" a callee frame through the interpreter when
/// call_assembler hits a guard failure.  The callback takes the
/// callee frame pointer (from fail_args[0]) and runs the interpreter
/// to completion, returning the result as an i64.
static CALL_ASSEMBLER_FORCE_FN: OnceLock<extern "C" fn(i64) -> i64> = OnceLock::new();

/// RPython resume_in_blackhole parity: callback to resume execution
/// from the guard failure point using the blackhole interpreter.
/// Args: (green_key, trace_id, fail_index, fail_values_ptr, num_fail_values) → result i64.
/// This reads the guard's resume data, restores state from fail_values (deadframe),
/// and executes the remaining IR ops from the guard point to Finish.
/// bh_fn(green_key, trace_id, fail_index, rebuilt_values, num_rebuilt, raw_deadframe, num_raw)
/// Unbox a Ref (boxed int pointer) to a raw i64 int value.
static CALL_ASSEMBLER_BLACKHOLE_FN: OnceLock<
    fn(u64, u64, u32, *const i64, usize, *const i64, usize) -> Option<i64>,
> = OnceLock::new();

/// Register a blackhole callback for call_assembler guard failure resume.
pub fn register_call_assembler_blackhole(
    f: fn(u64, u64, u32, *const i64, usize, *const i64, usize) -> Option<i64>,
) {
    let _ = CALL_ASSEMBLER_BLACKHOLE_FN.set(f);
}

/// compile.py:701-717 handle_fail callback for call_assembler guard failures.
/// (green_key, trace_id, fail_index, raw_values_ptr, num_values, descr_addr) -> bridge_compiled.
static CALL_ASSEMBLER_BRIDGE_FN: OnceLock<fn(u64, u64, u32, *const i64, usize, usize) -> bool> =
    OnceLock::new();

// Thread-local raw local0 value from CallAssemblerI inputs,
// for force_fn to re-box before interpreter execution.
thread_local! {
    static PENDING_FORCE_LOCAL0: std::cell::Cell<Option<i64>> =
        const { std::cell::Cell::new(None) };
}

/// Take the pending raw local0 value (if any).
pub fn take_pending_force_local0() -> Option<i64> {
    PENDING_FORCE_LOCAL0.with(|c| c.take())
}

/// RPython rebuild_state_after_failure parity: callback to materialize
/// virtual objects in fail_args before bridge dispatch. The Cranelift
/// backend cannot depend on pyre-object, so the interpreter registers
/// a callback that creates concrete objects from (type_ptr, raw_value) pairs.
///
/// Signature: fn(outputs: &mut [i64], types: &[Type]) — modifies outputs
/// in-place, replacing null Ref slots with materialized object pointers.
type RebuildStateAfterFailureFn = fn(&mut [i64], &[majit_ir::Type]);

thread_local! {
    static REBUILD_STATE_AFTER_FAILURE: std::cell::Cell<Option<RebuildStateAfterFailureFn>> =
        const { std::cell::Cell::new(None) };
}

/// Register the virtual materialization callback. Called from pyre-jit init.
pub fn register_rebuild_state_after_failure(f: RebuildStateAfterFailureFn) {
    REBUILD_STATE_AFTER_FAILURE.with(|c| c.set(Some(f)));
}

/// resume.py:763-779 VStrPlainInfo.allocate / resume.py:817-829
/// VUniPlainInfo.allocate parity — materialize a Plain string/unicode
/// virtual from per-character fieldnums. Frontend implements via
/// `backend.bh_newstr` + `backend.bh_strsetitem` (or the unicode
/// variants). Returns the materialized string pointer as i64.
pub type MaterializeStrPlainFn = fn(is_unicode: bool, chars: &[i64]) -> i64;

/// resume.py:1143-1188 parity — materialize a Concat / Slice string
/// virtual by invoking `cpu.bh_call_r(func, args_i, args_r, descr)`.
/// Concat: args_i=[], args_r=[left, right] (resume.py:1462-1470 /
/// resume.py:1489-1497). Slice: args_i=[start, stop], args_r=[str]
/// (resume.py:1472-1480 / resume.py:1499-1507, stop = start + length).
pub type MaterializeStrCallFn = fn(
    is_unicode: bool,
    func: i64,
    calldescr: &majit_ir::DescrRef,
    args_i: &[i64],
    args_r: &[i64],
) -> i64;

thread_local! {
    static MATERIALIZE_STR_PLAIN_FN: std::cell::Cell<Option<MaterializeStrPlainFn>> =
        const { std::cell::Cell::new(None) };
    static MATERIALIZE_STR_CALL_FN: std::cell::Cell<Option<MaterializeStrCallFn>> =
        const { std::cell::Cell::new(None) };
}

/// Register the VStrPlain/VUniPlain materialization callback.
/// Called from pyre-jit init.
pub fn register_materialize_str_plain(f: MaterializeStrPlainFn) {
    MATERIALIZE_STR_PLAIN_FN.with(|c| c.set(Some(f)));
}

/// Register the VStrConcat/VStrSlice/VUniConcat/VUniSlice materialization
/// callback. Called from pyre-jit init.
pub fn register_materialize_str_call(f: MaterializeStrCallFn) {
    MATERIALIZE_STR_CALL_FN.with(|c| c.set(Some(f)));
}

/// Frame state to restore from guard failure fail_args.
/// RPython resume_in_blackhole parity: the force_fn reads the frame state
/// from the deadframe (outputs buffer) rather than using the corrupted frame.
pub struct FrameRestore {
    pub next_instr: usize,
    pub valuestackdepth: usize,
    /// (type, raw_value) pairs for each fail_arg slot.
    pub slots: Vec<(majit_ir::Type, i64)>,
}

thread_local! {
    static PENDING_FRAME_RESTORE: std::cell::Cell<Option<FrameRestore>> =
        const { std::cell::Cell::new(None) };
}

/// Take the pending frame restore data (if any).
pub fn take_pending_frame_restore() -> Option<FrameRestore> {
    PENDING_FRAME_RESTORE.with(|c| c.take())
}

// ── JitFrame layout registration for the GC rewriter ────────────────

/// JitFrame field descriptors supplied by the interpreter crate so the
/// GC rewriter's `handle_call_assembler` pass (rewrite.py:665-695) can
/// emit the correct GC_LOAD / GC_STORE sequence for callee jitframes.
#[derive(Clone)]
pub struct JitFrameLayoutInfo {
    pub jitframe_descrs: Option<majit_gc::rewrite::JitFrameDescrs>,
}

static JITFRAME_LAYOUT: OnceLock<JitFrameLayoutInfo> = OnceLock::new();

pub fn register_jitframe_layout(info: JitFrameLayoutInfo) {
    let _ = JITFRAME_LAYOUT.set(info);
}

// ── Call-assembler dispatch table ──
// Each token gets a stable dispatch entry holding code_ptr + finish_index.
// Compiled code loads both from the entry via memory reads.
// redirect_call_assembler / register_call_assembler_target update atomically.
use std::sync::atomic::AtomicPtr;

/// Dispatch entry: code_ptr at offset 0, finish_descr_ptr at offset 8.
/// Laid out as two 8-byte words so Cranelift can load them directly.
/// RPython parity: finish check compares jf_descr pointer with the
/// finish FailDescr pointer (not integer index).
#[repr(C)]
struct CaDispatchEntry {
    code_ptr: AtomicPtr<u8>,
    finish_descr_ptr: AtomicU64,
    /// RPython redirect_call_assembler parity: compiled bridge code pointer
    /// for the guard at finish_index. When set, Cranelift codegen can
    /// dispatch guard failures directly without extern "C" call overhead.
    guard_bridge_ptr: AtomicPtr<u8>,
}

/// Sentinel: finish_index not yet known (self-recursion before compile).
const CA_FINISH_INDEX_UNKNOWN: u64 = u64::MAX;

thread_local! {
    /// Per-thread CALL_ASSEMBLER dispatch table. Entries are boxed so their
    /// addresses are stable across HashMap resizes — the JIT-emitted call
    /// site holds a raw `*const CaDispatchEntry` obtained via
    /// `ca_dispatch_slot`. Thread-local is safe because the JIT code and
    /// the entries live on the same thread; teardown is concurrent.
    static CA_DISPATCH_TABLE: RefCell<HashMap<u64, Box<CaDispatchEntry>>> =
        RefCell::new(HashMap::new());
}

/// Get or create the stable dispatch entry for a token.
/// Returns the address of the entry (stable for this thread's lifetime).
fn ca_dispatch_slot(token_number: u64, code_ptr: *const u8) -> *const CaDispatchEntry {
    CA_DISPATCH_TABLE.with(|c| {
        let mut table = c.borrow_mut();
        let entry = table.entry(token_number).or_insert_with(|| {
            Box::new(CaDispatchEntry {
                code_ptr: AtomicPtr::new(code_ptr as *mut u8),
                finish_descr_ptr: AtomicU64::new(CA_FINISH_INDEX_UNKNOWN),
                guard_bridge_ptr: AtomicPtr::new(std::ptr::null_mut()),
            })
        });
        // Update code_ptr if changed (e.g., recompilation)
        entry.code_ptr.store(code_ptr as *mut u8, Ordering::Release);
        &**entry as *const CaDispatchEntry
    })
}

/// Update the finish_descr_ptr for a token's dispatch entry.
/// Stores the FailDescr pointer (not index) for direct finish comparison.
fn ca_dispatch_set_finish_descr_ptr(token_number: u64, finish_descr_ptr: i64) {
    CA_DISPATCH_TABLE.with(|c| {
        if let Some(entry) = c.borrow().get(&token_number) {
            entry
                .finish_descr_ptr
                .store(finish_descr_ptr as u64, Ordering::Release);
        }
    });
}

/// Update dispatch slot for redirect — updates both code_ptr and finish_descr_ptr.
fn ca_dispatch_redirect(old_token: u64, new_code_ptr: *const u8, new_finish_descr_ptr: i64) {
    CA_DISPATCH_TABLE.with(|c| {
        if let Some(entry) = c.borrow().get(&old_token) {
            entry
                .code_ptr
                .store(new_code_ptr as *mut u8, Ordering::Release);
            entry
                .finish_descr_ptr
                .store(new_finish_descr_ptr as u64, Ordering::Release);
        }
    });
}

fn ca_dispatch_remove(token_number: u64) {
    // `try_with`: `CraneliftBackend::drop` calls `unregister_call_assembler_target`
    // which reaches here during thread TLS teardown. Silently no-op if TLS
    // has already been torn down.
    let _ = CA_DISPATCH_TABLE.try_with(|c| c.borrow_mut().remove(&token_number));
}

thread_local! {
    /// Thread-local cache for call_assembler target lookups.
    /// The HashMap holds Arc ownership; CA_FAST_PTR caches a raw pointer
    /// for the hot path to avoid atomic refcount ops on every call.
    static CA_TARGET_CACHE: RefCell<HashMap<u64, Arc<RegisteredLoopTarget>>> =
        RefCell::new(HashMap::new());

    /// Single-entry raw pointer cache: (token_number, ptr_as_usize).
    /// Valid because the Arc in CA_TARGET_CACHE keeps the data alive.
    static CA_FAST_PTR: Cell<(u64, usize)> = const { Cell::new((0, 0)) };
}

/// Look up a call_assembler target without atomic refcount operations.
///
/// # Safety
/// The returned pointer is valid for the duration of the current call
/// (the Arc in CA_TARGET_CACHE keeps the data alive, and invalidation
/// clears CA_FAST_PTR before removing from CA_TARGET_CACHE).
unsafe fn fast_lookup_ca_target(token_number: u64) -> *const RegisteredLoopTarget {
    // Hot path: single Cell read, no borrow, no atomic ops
    if let Ok((t, p)) = CA_FAST_PTR.try_with(|c| c.get()) {
        if t == token_number && p != 0 {
            return p as *const RegisteredLoopTarget;
        }
    }
    // Warm path: HashMap lookup, extract pointer without Arc clone
    if let Ok(Some(ptr)) = CA_TARGET_CACHE.try_with(|c| {
        c.borrow()
            .get(&token_number)
            .map(|arc| Arc::as_ptr(arc) as usize)
    }) {
        let _ = CA_FAST_PTR.try_with(|c| c.set((token_number, ptr)));
        return ptr as *const RegisteredLoopTarget;
    }
    // Cold path: thread-local registry, then cache
    let target = with_call_assembler_registry(|m| m.get(&token_number).cloned())
        .map(Arc::new)
        .unwrap_or_else(|| panic!("missing call_assembler target token {token_number}"));
    let ptr = Arc::as_ptr(&target) as usize;
    let _ = CA_TARGET_CACHE.try_with(|c| c.borrow_mut().insert(token_number, target));
    let _ = CA_FAST_PTR.try_with(|c| c.set((token_number, ptr)));
    ptr as *const RegisteredLoopTarget
}

fn invalidate_ca_thread_cache(token_number: u64) {
    // Clear fast pointer cache first (before removing Arc ownership)
    let _ = CA_FAST_PTR.try_with(|c| {
        let (t, _) = c.get();
        if t == token_number {
            c.set((0, 0));
        }
    });
    let _ = CA_TARGET_CACHE.try_with(|c| c.borrow_mut().remove(&token_number));
}

/// Register a force callback for call_assembler guard failures.
pub fn register_call_assembler_force(f: extern "C" fn(i64) -> i64) {
    let _ = CALL_ASSEMBLER_FORCE_FN.set(f);
}

static CALL_ASSEMBLER_UNBOX_INT_FN: OnceLock<fn(i64) -> i64> = OnceLock::new();

pub fn register_call_assembler_unbox_int(f: fn(i64) -> i64) {
    let _ = CALL_ASSEMBLER_UNBOX_INT_FN.set(f);
}

/// rpython/jit/backend/llsupport/llmodel.py:229-234 `insert_stack_check`
/// parity. Mirrors the dynasm-side address registration API so the
/// interpreter can install the three RPython-style probe addresses
/// uniformly across backends.
///
/// Cranelift does not emit an inline SP probe today; the prologue
/// calls [`register_prologue_probe_addr`]'s registered function (which
/// combines fast path + slowpath in one call) because Cranelift IR
/// does not expose a "read current SP" intrinsic. When such an
/// intrinsic lands, the prologue emitter will switch to the
/// `MOV [endaddr]; SUB sp; CMP [lengthaddr]; BLR slowpath` sequence
/// matching `assembler.py:1085-1091`, consuming the three RPython-
/// style addresses below directly.
#[derive(Copy, Clone, Debug)]
pub struct StackCheckAddresses {
    pub end_adr: usize,
    pub length_adr: usize,
    pub slowpath_addr: usize,
}

static STACK_CHECK_ADDRS: OnceLock<StackCheckAddresses> = OnceLock::new();

pub fn register_stack_check_addresses(end_adr: usize, length_adr: usize, slowpath_addr: usize) {
    let _ = STACK_CHECK_ADDRS.set(StackCheckAddresses {
        end_adr,
        length_adr,
        slowpath_addr,
    });
}

pub fn stack_check_addresses() -> Option<StackCheckAddresses> {
    STACK_CHECK_ADDRS.get().copied()
}

/// Address of the combined fast-path + slowpath stack check fallback.
/// Normal Cranelift prologues use [`stack_check_addresses`] to emit the
/// same inline fast path as dynasm and call the slowpath only on a miss.
/// This fallback remains for backend-only tests that register just the
/// combined probe.
static PROLOGUE_PROBE_ADDR: OnceLock<usize> = OnceLock::new();

pub fn register_prologue_probe_addr(addr: usize) {
    let _ = PROLOGUE_PROBE_ADDR.set(addr);
}

pub fn prologue_probe_addr() -> Option<usize> {
    PROLOGUE_PROBE_ADDR.get().copied()
}

/// Execute compiled code for a token directly, bypassing the JitDriver chain.
///
/// PyPy assembler_call_helper equivalent: dispatches through compiled code
/// for the given token_number. Returns Some(raw_result) on finish,
/// None if no compiled code or guard failure (caller should use interpreter).
///
/// This is used by jit_force_callee_frame to avoid the eval_with_jit →
/// try_function_entry_jit → run_compiled → execute_token indirection chain.
pub fn execute_call_assembler_direct(
    token_number: u64,
    inputs: &[i64],
    force_fn: extern "C" fn(i64) -> i64,
) -> Option<i64> {
    let target_ptr = unsafe { fast_lookup_ca_target(token_number) };
    if target_ptr.is_null() {
        return None;
    }
    let target = unsafe { &*target_ptr };
    if target.code_ptr.is_null() {
        return None;
    }

    let mut outcome = [0i64; 2];
    let result = call_assembler_fast_path(target, inputs, outcome.as_mut_ptr(), force_fn);

    if outcome[0] == CALL_ASSEMBLER_OUTCOME_FINISH {
        Some(result as i64)
    } else {
        if outcome[0] == CALL_ASSEMBLER_OUTCOME_DEADFRAME {
            // RPython assembler_call_helper consumes the guard-failure
            // deadframe immediately. This direct helper cannot propagate a
            // deadframe through its Option<i64> API, so drop the stored
            // JitFrameDeadFrame before falling back to the interpreter;
            // otherwise its registered GC root keeps stale fail-args alive.
            let handle = outcome[1] as u64;
            if handle != 0 {
                let _ = take_call_assembler_deadframe(handle);
            }
        }
        None // guard failure — caller should fallback
    }
}

pub fn register_call_assembler_bridge(f: fn(u64, u64, u32, *const i64, usize, usize) -> bool) {
    let _ = CALL_ASSEMBLER_BRIDGE_FN.set(f);
}

const CALL_ASSEMBLER_DEADFRAME_SENTINEL: u32 = u32::MAX - 1;
const CALL_ASSEMBLER_RESULT_VOID: u64 = 0;
const CALL_ASSEMBLER_RESULT_INT: u64 = 1;
const CALL_ASSEMBLER_RESULT_FLOAT: u64 = 2;
const CALL_ASSEMBLER_RESULT_REF: u64 = 3;
const CALL_ASSEMBLER_RESULT_FORCE_TOKEN_REF: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CallAssemblerCallerId {
    RootLoop(u64),
    BridgeTrace(u64),
}

fn with_call_assembler_registry<R>(
    f: impl FnOnce(&mut HashMap<u64, RegisteredLoopTarget>) -> R,
) -> R {
    // `try_with` + `expect`: normal call paths run long before the thread's
    // TLS destructor fires. The `Drop for CraneliftBackend` path uses
    // `try_call_assembler_registry_drop` below, which silently no-ops when
    // TLS has already been torn down.
    CALL_ASSEMBLER_TARGETS
        .try_with(|r| f(&mut r.borrow_mut()))
        .expect("CALL_ASSEMBLER_TARGETS TLS not available")
}

fn with_call_assembler_expectations<R>(
    f: impl FnOnce(&mut HashMap<u64, HashMap<CallAssemblerCallerId, u64>>) -> R,
) -> R {
    CALL_ASSEMBLER_EXPECTATIONS
        .try_with(|r| f(&mut r.borrow_mut()))
        .expect("CALL_ASSEMBLER_EXPECTATIONS TLS not available")
}

/// Drop-path variant: `CraneliftBackend::drop` runs during thread shutdown
/// and may reach these helpers after Rust has destroyed the `RefCell`
/// thread_local (see `std::thread::LocalKey::try_with`). Since the drop
/// path is discarding state anyway, silently no-op when TLS is gone.
fn try_call_assembler_registry_drop(f: impl FnOnce(&mut HashMap<u64, RegisteredLoopTarget>)) {
    let _ = CALL_ASSEMBLER_TARGETS.try_with(|r| f(&mut r.borrow_mut()));
}

fn try_call_assembler_expectations_drop(
    f: impl FnOnce(&mut HashMap<u64, HashMap<CallAssemblerCallerId, u64>>),
) {
    let _ = CALL_ASSEMBLER_EXPECTATIONS.try_with(|r| f(&mut r.borrow_mut()));
}

fn call_assembler_result_kind_name(kind: u64) -> &'static str {
    match kind {
        CALL_ASSEMBLER_RESULT_VOID => "void",
        CALL_ASSEMBLER_RESULT_INT => "int",
        CALL_ASSEMBLER_RESULT_FLOAT => "float",
        CALL_ASSEMBLER_RESULT_REF => "ref",
        CALL_ASSEMBLER_RESULT_FORCE_TOKEN_REF => "force-token-ref",
        _ => "unknown",
    }
}

fn actual_call_assembler_target_result_kind(
    fail_descrs: &[Arc<CraneliftFailDescr>],
) -> Result<u64, BackendError> {
    let finish_descr = fail_descrs
        .iter()
        .find(|descr| descr.is_finish())
        .ok_or_else(|| {
            unsupported_semantics(
                OpCode::CallAssemblerN,
                "call-assembler target must expose at least one finish exit",
            )
        })?;
    actual_call_assembler_result_kind(finish_descr.as_ref())
}

fn validate_call_assembler_target_result_kind(
    target_token: u64,
    expected_result_kind: u64,
    actual_result_kind: u64,
    context: &str,
) -> Result<(), BackendError> {
    if expected_result_kind == actual_result_kind {
        return Ok(());
    }
    Err(BackendError::Unsupported(format!(
        "call-assembler target {target_token} has incompatible {context}: expected {}, got {}",
        call_assembler_result_kind_name(expected_result_kind),
        call_assembler_result_kind_name(actual_result_kind),
    )))
}

fn validate_registered_target_against_call_assembler_expectations(
    target_token: u64,
    target: &RegisteredLoopTarget,
) -> Result<(), BackendError> {
    let expected_result_kinds: Vec<u64> = with_call_assembler_expectations(|m| {
        m.get(&target_token)
            .map(|target_expectations| target_expectations.values().copied().collect())
            .unwrap_or_default()
    });
    if expected_result_kinds.is_empty() {
        return Ok(());
    }

    let actual_result_kind = actual_call_assembler_target_result_kind(&target.fail_descrs)?;
    for expected_result_kind in expected_result_kinds {
        validate_call_assembler_target_result_kind(
            target_token,
            expected_result_kind,
            actual_result_kind,
            "callee finish result kind",
        )?;
    }
    Ok(())
}

fn remove_call_assembler_expectations_locked(
    expectations: &mut HashMap<u64, HashMap<CallAssemblerCallerId, u64>>,
    caller_id: CallAssemblerCallerId,
) {
    expectations.retain(|_, callers| {
        callers.remove(&caller_id);
        !callers.is_empty()
    });
}

fn unregister_call_assembler_expectations(caller_id: CallAssemblerCallerId) {
    // `try_*_drop`: this helper is called from `CraneliftBackend::drop`,
    // which may run during thread TLS teardown. See the drop-path comment
    // on `try_call_assembler_expectations_drop`.
    try_call_assembler_expectations_drop(|m| {
        remove_call_assembler_expectations_locked(m, caller_id)
    });
}

fn unregister_bridge_call_assembler_expectations(bridge: &BridgeData) {
    unregister_call_assembler_expectations(CallAssemblerCallerId::BridgeTrace(bridge.trace_id));
    for descr in &bridge.fail_descrs {
        let attached = descr.bridge_ref();
        if let Some(ref child_bridge) = *attached {
            unregister_bridge_call_assembler_expectations(child_bridge);
        }
    }
}

fn unregister_call_assembler_bridge_tree(fail_descrs: &[Arc<CraneliftFailDescr>]) {
    for descr in fail_descrs {
        let attached = descr.bridge_ref();
        if let Some(ref bridge) = *attached {
            unregister_bridge_call_assembler_expectations(bridge);
        }
    }
}

fn collect_call_assembler_expectations(ops: &[Op]) -> Result<HashMap<u64, u64>, BackendError> {
    let mut expectations = HashMap::new();
    for op in ops {
        let opcode = op.opcode;
        if !matches!(
            opcode,
            OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN
        ) {
            continue;
        }
        let descr = op.descr.as_ref().ok_or_else(|| {
            unsupported_semantics(opcode, "call-assembler op must have a descriptor")
        })?;
        let call_descr = descr.as_call_descr().ok_or_else(|| {
            unsupported_semantics(opcode, "call-assembler descriptor must be a CallDescr")
        })?;
        let target_token = call_descr.call_target_token().ok_or_else(|| {
            unsupported_semantics(
                opcode,
                "call-assembler descriptor must provide a compiled target token",
            )
        })?;
        let resolved_target = resolve_call_assembler_target(opcode, call_descr)?;
        let expected_result_kind =
            expected_call_assembler_result_kind(call_descr, resolved_target.as_ref())?;
        if let Some(previous) = expectations.insert(target_token, expected_result_kind) {
            validate_call_assembler_target_result_kind(
                target_token,
                previous,
                expected_result_kind,
                "caller result expectation",
            )?;
        }
    }
    Ok(expectations)
}

fn install_call_assembler_expectations(
    caller_id: CallAssemblerCallerId,
    ops: &[Op],
) -> Result<(), BackendError> {
    let expectations = collect_call_assembler_expectations(ops)?;

    for (&target_token, &expected_result_kind) in &expectations {
        if let Some(target) = lookup_call_assembler_target(target_token) {
            let actual_result_kind = actual_call_assembler_target_result_kind(&target.fail_descrs)?;
            validate_call_assembler_target_result_kind(
                target_token,
                expected_result_kind,
                actual_result_kind,
                "callee finish result kind",
            )?;
        }
    }

    with_call_assembler_expectations(|registry| -> Result<(), BackendError> {
        for (&target_token, &expected_result_kind) in &expectations {
            if let Some(callers) = registry.get(&target_token) {
                for (&other_caller, &other_expected_kind) in callers {
                    if other_caller != caller_id {
                        validate_call_assembler_target_result_kind(
                            target_token,
                            expected_result_kind,
                            other_expected_kind,
                            "caller result expectation",
                        )?;
                    }
                }
            }
        }

        remove_call_assembler_expectations_locked(registry, caller_id);
        for (&target_token, &expected_result_kind) in &expectations {
            registry
                .entry(target_token)
                .or_default()
                .insert(caller_id, expected_result_kind);
        }
        Ok(())
    })
}

fn register_call_assembler_target(
    token: &mut JitCellToken,
    compiled: &CompiledLoop,
    attached_descrs: majit_backend::AttachedDescrPtrs,
) -> Result<(), BackendError> {
    invalidate_ca_thread_cache(token.number);
    let depth = (compiled.max_output_slots + compiled.num_ref_roots) as i64;
    let base_ofs = JF_FRAME_ITEM0_OFS as i64;
    let num_scalar_inputargs = if token.num_scalar_inputargs > 0 {
        token.num_scalar_inputargs
    } else {
        // Derive from types: first N header entries.
        token.inputarg_types.len().min(compiled.num_inputs)
    };
    // P2.2 — CLT Arc continuity across pending → real registration.
    //
    // `compile_tmp_callback` (RPython `compile.py`) creates a placeholder
    // target before the real `JitCellToken` exists; caller traces
    // rewritten during the pending window bake in the pending CLT's
    // `frame_info` address. If we let `JitCellToken::new` replace that
    // Arc with a fresh one, the baked pointer dangles.
    //
    // Fix: adopt the pending Arc onto `token.compiled_loop_token` so the
    // same allocation survives registration. `CompiledLoopToken::new`
    // zero-initialises the fields we're about to populate, so swapping
    // the fresh CLT the token already owns is safe (no state lost).
    if let Some(existing_clt) = with_call_assembler_registry(|m| {
        m.get(&token.number).map(|t| t.compiled_loop_token.clone())
    }) {
        token.compiled_loop_token = Some(existing_clt);
    }
    let clt = token
        .compiled_loop_token
        .as_ref()
        .expect("JitCellToken missing compiled_loop_token")
        .clone();
    // `regalloc.py:861-871` `_set_initial_bindings`: contiguous layout
    // → `locs[i] = i * SIZEOFSIGNED`.
    *clt._ll_initial_locs.lock() = (0..compiled.num_inputs).map(|i| (i as i32) * 8).collect();
    // `jitframe.py:18-22` `jitframeinfo_update_depth`. The base_ofs is
    // shifted by `GcHeader::SIZE` because pyre's nursery allocator
    // returns the JITFRAME pointer past an 8-byte GcHeader; the depth
    // field counts payload slots after the header.
    clt.frame_info
        .lock()
        .update_frame_depth(base_ofs + GcHeader::SIZE as i64, depth);
    let target = RegisteredLoopTarget {
        trace_id: compiled.trace_id,
        header_pc: compiled.header_pc,
        green_key: token.green_key,
        caller_prefix_layout: compiled.caller_prefix_layout.clone(),
        code_ptr: compiled.code_ptr,
        fail_descrs: compiled.fail_descrs.clone(),
        num_inputs: compiled.num_inputs,
        num_ref_roots: compiled.num_ref_roots,
        max_output_slots: compiled.max_output_slots,
        inputarg_types: token.inputarg_types.clone(),
        // virtualizable.py:86 read_boxes: header = frame + static fields.
        num_scalar_inputargs,
        index_of_virtualizable: token
            .virtualizable_arg_index
            .map(|i| i as i32)
            .unwrap_or(-1),
        compiled_loop_token: clt,
        cpu_attachments: Arc::clone(&compiled.cpu_attachments),
    };
    validate_registered_target_against_call_assembler_expectations(token.number, &target)?;
    // Invalidate thread-local cache in case a pending placeholder was cached.
    invalidate_ca_thread_cache(token.number);
    // Create/update dispatch slot for direct call
    ca_dispatch_slot(token.number, compiled.code_ptr);
    // `compile.py:665-674` parity: the direct CA call path embeds the
    // finish descr pointer as a compile-time constant in the caller,
    // and the callee writes the same pointer into `jf_descr` at FINISH
    // emission (see `fail_descr_ptr` block in this file).  Both sides
    // resolve through `done_with_this_frame_descr_ptr` so they always
    // pick the metainterp-attached `Arc<DoneWithThisFrameDescr*>` when
    // available and fall back to the cranelift-side singleton in its
    // absence (backend-only tests).
    if let Some(finish_descr) = compiled
        .fail_descrs
        .iter()
        .find(|d| d.is_finish() && !d.is_exit_frame_with_exception)
    {
        let result_type = finish_descr
            .fail_arg_types
            .first()
            .copied()
            .unwrap_or(Type::Void);
        let ptr = attached_descrs.done_with_this_frame_descr_ptr_for_type(result_type) as i64;
        ca_dispatch_set_finish_descr_ptr(token.number, ptr);
    }
    with_call_assembler_registry(|m| m.insert(token.number, target));
    Ok(())
}

fn unregister_call_assembler_target(token_number: u64) {
    invalidate_ca_thread_cache(token_number);
    unregister_call_assembler_expectations(CallAssemblerCallerId::RootLoop(token_number));
    ca_dispatch_remove(token_number);
    // `try_*_drop`: called from `CraneliftBackend::drop` during thread
    // shutdown — use TLS teardown-safe variant.
    let mut removed = None;
    try_call_assembler_registry_drop(|m| removed = m.remove(&token_number));
    if let Some(target) = removed {
        unregister_call_assembler_bridge_tree(&target.fail_descrs);
    }
}

/// RPython compile_tmp_callback parity: register a placeholder target
/// with null code_ptr for a pending token. call_assembler_fast_path
/// detects null code_ptr and falls back to force_fn (interpreter).
/// When compilation completes, the placeholder is replaced by the real target.
pub(crate) fn register_pending_call_assembler_target(
    token_number: u64,
    inputarg_types: Vec<Type>,
    num_inputs: usize,
    num_scalar_inputargs: usize,
    index_of_virtualizable: i32,
    cpu_attachments: CpuDescrHandle,
) {
    // compile.py: compile_tmp_callback installs a placeholder target that must
    // not retain dispatch metadata from a previous token incarnation.
    ca_dispatch_remove(token_number);
    ca_dispatch_slot(token_number, std::ptr::null());
    ca_dispatch_set_finish_descr_ptr(token_number, CA_FINISH_INDEX_UNKNOWN as i64);
    // `rpython/jit/backend/model.py:292` — fresh placeholder
    // `CompiledLoopToken`. `register_call_assembler_target` adopts this
    // same Arc onto `token.compiled_loop_token` so the allocation (and
    // therefore the `frame_info` address baked into already-rewritten
    // caller traces) stays valid across the pending → real transition.
    let pending_clt = Arc::new(CompiledLoopToken::new(token_number));
    *pending_clt._ll_initial_locs.lock() = (0..num_inputs).map(|i| (i as i32) * 8).collect();
    let target = RegisteredLoopTarget {
        trace_id: 0,
        header_pc: 0,
        green_key: 0,
        caller_prefix_layout: None,
        code_ptr: std::ptr::null(),
        fail_descrs: Box::new([]),
        num_inputs,
        num_ref_roots: 0,
        max_output_slots: 1,
        inputarg_types,
        num_scalar_inputargs,
        index_of_virtualizable,
        compiled_loop_token: pending_clt,
        cpu_attachments,
    };
    with_call_assembler_registry(|m| m.insert(token_number, target));
}

fn lookup_call_assembler_target(token_number: u64) -> Option<RegisteredLoopTarget> {
    with_call_assembler_registry(|m| m.get(&token_number).cloned())
}

fn redirect_call_assembler_target(old_number: u64, new_number: u64) -> Result<(), BackendError> {
    invalidate_ca_thread_cache(old_number);
    let Some(new_target) = lookup_call_assembler_target(new_number) else {
        return Ok(());
    };
    if let Some(old_target) = lookup_call_assembler_target(old_number) {
        if old_target.inputarg_types != new_target.inputarg_types {
            return Err(BackendError::Unsupported(format!(
                "call-assembler redirect from token {old_number} to {new_number} changed input types"
            )));
        }
    }
    validate_registered_target_against_call_assembler_expectations(old_number, &new_target)?;
    // Update dispatch slot so existing compiled code sees the new target.
    // Must update both code_ptr AND finish_descr_ptr (the new target has
    // different FailDescr pointers for its finish exit).
    let attached_descrs = {
        let attached = new_target.cpu_attachments.read().unwrap().descr_ptrs();
        attached_descr_ptrs_with_fallbacks(attached)
    };
    let new_finish_descr_ptr = new_target
        .fail_descrs
        .iter()
        .find(|d| d.is_finish() && !d.is_exit_frame_with_exception)
        .map(|d| {
            let result_type = d.fail_arg_types.first().copied().unwrap_or(Type::Void);
            attached_descrs.done_with_this_frame_descr_ptr_for_type(result_type) as i64
        })
        .unwrap_or(CA_FINISH_INDEX_UNKNOWN as i64);
    ca_dispatch_redirect(old_number, new_target.code_ptr, new_finish_descr_ptr);
    with_call_assembler_registry(|m| m.insert(old_number, new_target));
    Ok(())
}

fn store_call_assembler_deadframe(frame: DeadFrame) -> u64 {
    let handle = NEXT_CALL_ASSEMBLER_DEADFRAME_HANDLE.with(|cell| {
        let h = cell.get();
        cell.set(h + 1);
        h
    });
    assert!(handle != 0, "call_assembler deadframe handle overflowed");
    CALL_ASSEMBLER_DEADFRAMES.with(|map| {
        map.borrow_mut().insert(handle, frame);
    });
    handle
}

fn take_call_assembler_deadframe(handle: u64) -> Option<DeadFrame> {
    CALL_ASSEMBLER_DEADFRAMES.with(|map| map.borrow_mut().remove(&handle))
}

/// Read typed values from a deadframe, dispatching on per-slot type.
/// RPython: cpu.get_{int,ref,float}_value(deadframe, i).
fn raw_values_from_deadframe_typed(
    frame: &DeadFrame,
    types: &[Type],
) -> Result<Vec<i64>, BackendError> {
    types
        .iter()
        .enumerate()
        .map(|(i, tp)| match tp {
            Type::Int => get_int_from_deadframe(frame, i),
            Type::Ref => get_ref_from_deadframe(frame, i).map(|value| value.0 as i64),
            Type::Float => get_float_from_deadframe(frame, i).map(|value| value.to_bits() as i64),
            Type::Void => Ok(0),
        })
        .collect()
}

fn finish_result_from_deadframe(frame: &mut DeadFrame) -> Result<i64, BackendError> {
    let fail_arg_types = {
        let descr = get_latest_descr_from_deadframe(frame)?;
        assert!(descr.is_finish(), "expected finish deadframe");
        descr.fail_arg_types().to_vec()
    };
    match fail_arg_types.as_slice() {
        [] => Ok(0),
        [Type::Int] => get_int_from_deadframe(frame, 0),
        [Type::Ref] => {
            if let Some(jf) = frame.data.downcast_mut::<JitFrameDeadFrame>() {
                return Ok(jf.take_ref_for_call_result(0).as_usize() as i64);
            }
            Err(BackendError::Unsupported(
                "unsupported dead frame type for Ref finish result".to_string(),
            ))
        }
        [Type::Float] => Ok(get_float_from_deadframe(frame, 0)?.to_bits() as i64),
        [Type::Void] => Ok(0),
        other => Err(BackendError::Unsupported(format!(
            "unsupported call_assembler finish result layout: {other:?}"
        ))),
    }
}

fn call_assembler_finish_or_blackhole_deadframe(mut frame: DeadFrame) -> Option<i64> {
    let (is_normal_finish, green_key, trace_id, fail_index, fail_arg_types) = {
        let jf = frame.data.downcast_ref::<JitFrameDeadFrame>()?;
        let fail_descr = &jf.fail_descr;
        // pyjitpl.py:2897-2899 parity: `rd_loop_token.loop_token_wref()` may be
        // dead (memmgr-evicted JCT) — `compile.giveup()` raises
        // `SwitchToBlackhole(ABORT_BRIDGE)` (compile.py:27-29) and falls
        // through to blackhole resume.  Pass `green_key=0` to the blackhole
        // callback so it derives the key from the deadframe's pyframe pointer
        // (call_jit.rs:1694-1704); never substitute the parent CALL_ASSEMBLER
        // target's green_key here — that would mis-route resume storage.
        (
            fail_descr.is_finish() && !fail_descr.is_exit_frame_with_exception,
            majit_backend::descr_owning_jct(fail_descr.as_ref())
                .map(|jct| jct.green_key)
                .unwrap_or(0),
            fail_descr.trace_id,
            fail_descr.fail_index,
            fail_descr.fail_arg_types.clone(),
        )
    };
    if is_normal_finish {
        return finish_result_from_deadframe(&mut frame).ok();
    }

    let raw_values = raw_values_from_deadframe_typed(&frame, &fail_arg_types).ok()?;
    let blackhole = CALL_ASSEMBLER_BLACKHOLE_FN.get()?;
    blackhole(
        green_key,
        trace_id,
        fail_index,
        raw_values.as_ptr(),
        raw_values.len(),
        raw_values.as_ptr(),
        raw_values.len(),
    )
}

fn take_call_assembler_deadframe_from_outputs(outputs: &[i64]) -> DeadFrame {
    let handle = outputs
        .first()
        .copied()
        .unwrap_or_else(|| panic!("missing call_assembler deadframe handle slot"))
        as u64;
    assert!(handle != 0, "missing call_assembler deadframe handle");
    take_call_assembler_deadframe(handle)
        .unwrap_or_else(|| panic!("unknown call_assembler deadframe handle {handle}"))
}

fn maybe_take_call_assembler_deadframe(fail_index: u32, outputs: &[i64]) -> Option<DeadFrame> {
    if fail_index != CALL_ASSEMBLER_DEADFRAME_SENTINEL {
        return None;
    }

    Some(take_call_assembler_deadframe_from_outputs(outputs))
}

fn actual_call_assembler_result_kind(descr: &dyn FailDescr) -> Result<u64, BackendError> {
    match descr.fail_arg_types() {
        [] | [Type::Void] => Ok(CALL_ASSEMBLER_RESULT_VOID),
        [Type::Int] => Ok(CALL_ASSEMBLER_RESULT_INT),
        [Type::Float] => Ok(CALL_ASSEMBLER_RESULT_FLOAT),
        [Type::Ref] if descr.force_token_slots() == [0] => {
            Ok(CALL_ASSEMBLER_RESULT_FORCE_TOKEN_REF)
        }
        [Type::Ref] => Ok(CALL_ASSEMBLER_RESULT_REF),
        other => Err(BackendError::Unsupported(format!(
            "call-assembler target exposes unsupported finish result layout: {other:?}"
        ))),
    }
}

/// llmodel.py:270-274 force() as a free function.
/// force_token is the JitFrame pointer (FORCE_TOKEN returns jf_ptr).
pub fn force_token_to_dead_frame(force_token: GcRef) -> DeadFrame {
    assert!(force_token.0 != 0, "force_token_to_dead_frame: null token");
    let jf_ptr = force_token.0 as *mut i64;
    let jf_ptr = jitframe_resolve(jf_ptr);
    // frame.jf_descr = frame.jf_force_descr
    let jf_force_descr = unsafe { *jf_ptr.add(JF_FORCE_DESCR_OFS as usize / 8) };
    unsafe { *jf_ptr.add(JF_DESCR_OFS as usize / 8) = jf_force_descr };
    let jf_gcref = GcRef(jf_ptr as usize);
    let descr_ptr = jf_force_descr as usize as *const CraneliftFailDescr;
    assert!(
        !descr_ptr.is_null(),
        "force_token_to_dead_frame: jf_force_descr is null"
    );
    let fail_descr = unsafe {
        Arc::increment_strong_count(descr_ptr);
        Arc::from_raw(descr_ptr)
    };
    // Register jf_gcref as GC root so moving GC can update the
    // pointer. Shadow stack keeps the object alive, but does NOT
    // update this separate copy of jf_gcref.
    deadframe_from_jitframe(jf_gcref, fail_descr, None)
}

fn deadframe_from_jitframe(
    jf_gcref: GcRef,
    fail_descr: Arc<CraneliftFailDescr>,
    heap_owner: Option<Vec<i64>>,
) -> DeadFrame {
    let mut frame = Box::new(JitFrameDeadFrame::new(
        jf_gcref, fail_descr, None, heap_owner,
    ));
    frame.register_roots();
    DeadFrame { data: frame }
}

pub fn set_savedata_ref_on_deadframe(
    frame: &mut DeadFrame,
    data: GcRef,
) -> Result<(), BackendError> {
    let jf = frame
        .data
        .downcast_mut::<JitFrameDeadFrame>()
        .ok_or_else(|| BackendError::Unsupported("expected JitFrameDeadFrame".to_string()))?;
    jf.set_savedata_ref(data);
    Ok(())
}

pub fn get_latest_descr_from_deadframe(frame: &DeadFrame) -> Result<&dyn FailDescr, BackendError> {
    // llmodel.py:411-419 get_latest_descr: cast deadframe → JITFRAMEPTR,
    // read jf_descr, show() → AbstractFailDescr.
    let jf = frame
        .data
        .downcast_ref::<JitFrameDeadFrame>()
        .ok_or_else(|| BackendError::Unsupported("expected JitFrameDeadFrame".to_string()))?;
    Ok(jf.fail_descr.as_ref())
}

pub fn get_int_from_deadframe(frame: &DeadFrame, index: usize) -> Result<i64, BackendError> {
    let jf = frame
        .data
        .downcast_ref::<JitFrameDeadFrame>()
        .ok_or_else(|| BackendError::Unsupported("expected JitFrameDeadFrame".to_string()))?;
    Ok(jf.get_int(index))
}

pub fn get_float_from_deadframe(frame: &DeadFrame, index: usize) -> Result<f64, BackendError> {
    let jf = frame
        .data
        .downcast_ref::<JitFrameDeadFrame>()
        .ok_or_else(|| BackendError::Unsupported("expected JitFrameDeadFrame".to_string()))?;
    Ok(jf.get_float(index))
}

pub fn get_ref_from_deadframe(frame: &DeadFrame, index: usize) -> Result<GcRef, BackendError> {
    let jf = frame
        .data
        .downcast_ref::<JitFrameDeadFrame>()
        .ok_or_else(|| BackendError::Unsupported("expected JitFrameDeadFrame".to_string()))?;
    Ok(jf.get_ref(index))
}

pub fn get_savedata_ref_from_deadframe(frame: &DeadFrame) -> Result<GcRef, BackendError> {
    let jf = frame
        .data
        .downcast_ref::<JitFrameDeadFrame>()
        .ok_or_else(|| BackendError::Unsupported("expected JitFrameDeadFrame".to_string()))?;
    Ok(jf.get_savedata_ref())
}

pub fn grab_exc_value_from_deadframe(frame: &DeadFrame) -> Result<GcRef, BackendError> {
    let jf = frame
        .data
        .downcast_ref::<JitFrameDeadFrame>()
        .ok_or_else(|| BackendError::Unsupported("expected JitFrameDeadFrame".to_string()))?;
    Ok(jf.grab_exc_value())
}

fn execute_registered_loop_target(target: &RegisteredLoopTarget, inputs: &[i64]) -> DeadFrame {
    let mut cur_code_ptr = target.code_ptr;
    let mut cur_fail_descrs = target.fail_descrs.clone();
    let mut cur_num_ref_roots = target.num_ref_roots;
    let mut cur_max_output_slots = target.max_output_slots;
    let mut current_inputs = inputs.to_vec();
    // `compile.py:665 setattr(cpu, name, descr)` — borrow the owning
    // cpu's descr set for the whole dispatch loop.  External-JUMP
    // bridge dispatch swaps the code entry but stays within the same
    // cpu, so the attachments are invariant across the loop.  Holding
    // the read lock is safe: `Backend::set_done_with_this_frame_descr_*`
    // only fires during `MetaInterpStaticData.finish_setup`, long
    // before compiled code runs.
    let attachments_guard = target.cpu_attachments.read().unwrap();
    let attachments: &CpuDescrAttachments = &*attachments_guard;
    loop {
        let exec = run_compiled_code(
            cur_code_ptr,
            &cur_fail_descrs,
            cur_num_ref_roots,
            cur_max_output_slots,
            &current_inputs,
            &attachments,
        );
        let fail_index = exec.fail_index;
        let direct_descr = exec.direct_descr.clone();
        let outputs = exec.extract_outputs(cur_max_output_slots.max(1));

        if let Some(frame) = maybe_take_call_assembler_deadframe(fail_index, &outputs) {
            // `RegisteredLoopTarget` is always a root-loop entry; root
            // registers never set `source_guard`, so pass `None` directly.
            // Bridges carry their own `BridgeData.source_guard` and never
            // reach this `execute_registered_loop_target` path.
            return wrap_call_assembler_deadframe_with_caller_prefix(
                frame,
                target.trace_id,
                target.header_pc,
                None,
                &target.inputarg_types,
                &current_inputs,
                target.caller_prefix_layout.as_ref(),
            );
        }

        let fail_descr_arc =
            direct_descr.unwrap_or_else(|| cur_fail_descrs[fail_index as usize].clone());
        let fail_descr = &fail_descr_arc;
        fail_descr.increment_fail_count();
        let bridge_guard = fail_descr.bridge_ref();
        if let Some(ref bridge) = *bridge_guard {
            if bridge.loop_reentry {
                // loop_reentry: use raw fail_args (same as run_compiled_code
                // bridge dispatch). rebuild_state_after_failure transforms
                // outputs for blackhole resume which may change Vec length.
                let raw_outputs = outputs[..bridge.num_inputs].to_vec();
                let bridge_frame = CraneliftBackend::execute_bridge(
                    bridge,
                    &raw_outputs,
                    &fail_descr.fail_arg_types,
                    &attachments,
                );
                let _ = bridge_guard;
                let bridge_descr = get_latest_descr_from_deadframe(&bridge_frame)
                    .expect("bridge deadframe must have descriptor");
                // llgraph/runner.py:1130-1140 Jump exception on external JUMP:
                // switch to the target loop identified by its TargetToken.
                // assembler.py:2456-2462 closing_jump parity.
                if bridge_descr.is_external_jump() {
                    let target_entry = bridge_descr
                        .target_descr()
                        .and_then(lookup_loop_target)
                        .expect("external JUMP target must be a registered LoopTargetDescr");
                    current_inputs = raw_values_from_deadframe_typed(
                        &bridge_frame,
                        bridge_descr.fail_arg_types(),
                    )
                    .unwrap_or_else(|err| {
                        panic!("bridge loop-reentry deadframe decode failed: {err}")
                    });
                    cur_code_ptr = target_entry.code_ptr;
                    cur_fail_descrs = target_entry.fail_descrs;
                    cur_num_ref_roots = target_entry.num_ref_roots;
                    cur_max_output_slots = target_entry.max_output_slots;
                    continue;
                }
                return bridge_frame;
            }
            // Non-loop-reentry: materialize virtuals then dispatch.
            let mut mat_outputs = outputs.clone();
            rebuild_state_after_failure(
                &mut mat_outputs,
                &fail_descr.fail_arg_types,
                fail_descr.recovery_layout_ref().as_ref(),
                bridge.num_inputs,
            );
            return CraneliftBackend::execute_bridge(
                bridge,
                &mat_outputs,
                &fail_descr.fail_arg_types,
                &attachments,
            );
        }
        let _ = bridge_guard;

        // Bridge compilation is decided by MetaInterp.must_compile()
        // (compile.py:783-784 jitcounter.tick), not by backend fail_count.

        // jf_savedata already correct in jf_frame memory.
        // jf_guard_exc already written by emit_guard_exit
        // (_build_failure_recovery parity).

        return deadframe_from_jitframe(exec.jf_gcref, fail_descr.clone(), exec.heap_owner);
    } // end loop
}

/// RPython assembler_call_helper parity: handle guard failure from
/// direct call_assembler path. Checks for bridge first, falls back
/// to force_fn. Called from codegen's direct non-finish block.
///
/// This avoids the full shim → execute_registered_loop_target overhead
/// while still supporting bridge dispatch.
/// RPython assembler_call_helper parity: handle guard failure from
/// assembler.py:1122-1128 _call_header_shadowstack — inline Cranelift IR.
///
/// Emits 5 inline instructions matching RPython's x86 sequence:
///   MOV ebx, [root_stack_top_addr]     // load top pointer
///   MOV [ebx], 1                       // is_minor marker
///   MOV [ebx + WORD], ebp              // jf_ptr
///   ADD ebx, 2*WORD                    // advance
///   MOV [root_stack_top_addr], ebx     // store new top
fn emit_call_header_shadowstack(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    jf_ptr: CValue,
) {
    let word = std::mem::size_of::<usize>() as i64;
    let rst_addr_val = builder.ins().iconst(
        ptr_type,
        majit_gc::shadow_stack::get_root_stack_top_addr() as i64,
    );
    // MOV ebx, [root_stack_top_addr]
    let rst = builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), rst_addr_val, 0);
    // MOV [ebx], 1   — is_minor marker
    let one = builder.ins().iconst(ptr_type, 1);
    builder.ins().store(MemFlags::trusted(), one, rst, 0);
    // MOV [ebx + WORD], ebp
    let jf_as_ptr = if builder.func.dfg.value_type(jf_ptr) != ptr_type {
        builder.ins().ireduce(ptr_type, jf_ptr)
    } else {
        jf_ptr
    };
    builder
        .ins()
        .store(MemFlags::trusted(), jf_as_ptr, rst, word as i32);
    // ADD ebx, 2*WORD
    let new_rst = builder.ins().iadd_imm(rst, 2 * word);
    // MOV [root_stack_top_addr], ebx
    builder
        .ins()
        .store(MemFlags::trusted(), new_rst, rst_addr_val, 0);
}

/// assembler.py:1130-1136 _call_footer_shadowstack — inline Cranelift IR.
///
/// Emits 3 inline instructions matching RPython's x86 sequence:
///   MOV ebx, [root_stack_top_addr]     // load (or reuse known addr)
///   SUB ebx, 2*WORD                    // decrement
///   MOV [root_stack_top_addr], ebx     // store
fn emit_call_footer_shadowstack(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
) {
    let word = std::mem::size_of::<usize>() as i64;
    let rst_addr_val = builder.ins().iconst(
        ptr_type,
        majit_gc::shadow_stack::get_root_stack_top_addr() as i64,
    );
    let rst = builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), rst_addr_val, 0);
    let new_rst = builder.ins().iadd_imm(rst, -(2 * word));
    builder
        .ins()
        .store(MemFlags::trusted(), new_rst, rst_addr_val, 0);
}

/// direct call_assembler path. Ultra-lightweight: just increments
/// fail count, checks bridge (atomic + mutex only when bridge exists),
/// and defers bridge compilation. Falls back to force_fn.
///
/// `cpu_handle` is the heap-stable `Arc<RwLock<CpuDescrAttachments>>`
/// address baked as an immediate at emit time so the runtime
/// finish-descr classifier can resolve
/// `DoneWithThisFrame*` identity against the owning cpu's attachments
/// (`compile.py:665 setattr(cpu, name, descr)`).  Never null in
/// production — `CompiledLoop.cpu_attachments` keeps the allocation
/// alive for the lifetime of the emitted code.
#[inline(never)]
extern "C" fn call_assembler_guard_failure(
    cpu_handle: *const std::sync::RwLock<CpuDescrAttachments>,
    token_number: u64,
    fail_descr_ptr: i64,
    frame_ptr: i64,
    outputs_ptr: *const i64,
    inputs_ptr: *const i64,
) -> i64 {
    // warmspot.py:1021-1028 assembler_call_helper parity: Rust
    // equivalent of RPython's assembler_call_helper. Handles ALL
    // non-finish cases: bridge dispatch, bridge compilation, and
    // blackhole resume.
    //
    // No alternate-stack switch: upstream runs on the caller's stack
    // so the compiled prologue's inline SP probe
    // (`_call_header_with_stack_check`,
    // rpython/jit/backend/x86/assembler.py:1085) measures against the
    // same `PYRE_STACKTOOBIG.stack_end` the rest of the runtime uses.
    call_assembler_guard_failure_inner(
        cpu_handle,
        token_number,
        fail_descr_ptr,
        frame_ptr,
        outputs_ptr,
        inputs_ptr,
    )
}

fn call_assembler_guard_failure_inner(
    cpu_handle: *const std::sync::RwLock<CpuDescrAttachments>,
    token_number: u64,
    fail_descr_ptr: i64,
    frame_ptr: i64,
    outputs_ptr: *const i64,
    inputs_ptr: *const i64,
) -> i64 {
    // `compile.py:665 setattr(cpu, name, descr)` — borrow the read lock
    // for the duration of this call so finish-descr classification sees
    // the owning cpu's attachments without cloning Arc<DescrRef>s on
    // every guard failure.  Setters (`set_done_with_this_frame_descr_*`
    // etc.) only run at `MetaInterpStaticData.finish_setup` so write
    // contention is effectively nil.
    let attachments_guard;
    let default_attachments;
    let attachments: &CpuDescrAttachments = if cpu_handle.is_null() {
        default_attachments = CpuDescrAttachments::default();
        &default_attachments
    } else {
        attachments_guard = unsafe { (*cpu_handle).read().unwrap() };
        &*attachments_guard
    };
    // Handle deadframe sentinel (nested CALL_ASSEMBLER propagation).
    if fail_descr_ptr == CALL_ASSEMBLER_DEADFRAME_SENTINEL as i64 {
        let frame = take_call_assembler_deadframe_from_outputs(unsafe {
            std::slice::from_raw_parts(outputs_ptr, 16)
        });
        let handle = store_call_assembler_deadframe(frame);
        return handle as i64;
    }

    let target = unsafe { &*fast_lookup_ca_target(token_number) };

    // Fast path: read the attached bridge directly from the fail_descr
    // pointer (which IS jf_descr from the callee's guard exit). Skip
    // fail_count increment when bridge is available, as patched PyPy
    // code jumps to the bridge instead of re-entering the guard helper.
    let fail_descr_ref = unsafe { &*(fail_descr_ptr as *const CraneliftFailDescr) };
    if let Some(ref bridge) = *fail_descr_ref.bridge_ref() {
        let raw_num = fail_descr_ref.fail_arg_types.len();
        let parent_outputs = unsafe { std::slice::from_raw_parts(outputs_ptr, raw_num) };
        let frame = CraneliftBackend::execute_bridge(
            bridge,
            parent_outputs,
            &fail_descr_ref.fail_arg_types,
            attachments,
        );
        if let Some(result) = call_assembler_finish_or_blackhole_deadframe(frame) {
            return result;
        }
        return 0;
    }

    // Slow path: target lookup + fail_count increment.
    //
    // RPython handle_fail is a method on the failed descriptor object.
    // Do not re-index through the root CALL_ASSEMBLER target here: when a
    // bridge has just returned another guard failure, `fail_descr_ptr` is
    // the bridge descriptor, not necessarily `target.fail_descrs[fail_index]`.
    let fail_index = fail_descr_ref.fail_index();
    let fail_descr = fail_descr_ref;
    // pyjitpl.py:2897-2899 parity: recover the owning Arc<JitCellToken>
    // identity from the descr.  When the weakref is dead (memmgr-evicted
    // JCT — "should be rare" per upstream), `compile.giveup()` raises
    // `SwitchToBlackhole(ABORT_BRIDGE)` (compile.py:27-29), caught at
    // pyjitpl.py:2906-2907, falling through to blackhole resume.  Pyre
    // mirrors this by gating bridge tracing on `Some(jct)` and passing
    // `green_key=0` to the blackhole callback so it derives the key from
    // the deadframe's pyframe pointer (call_jit.rs:1694-1704); never
    // substitute the parent CALL_ASSEMBLER target's green_key here — that
    // would mis-route resume storage and trip
    // `compile_bridge`'s `debug_assert_eq!(source_jct.green_key, green_key)`
    // (pyjitpl/mod.rs:8297-8301).
    let owning_jct = majit_backend::descr_owning_jct(fail_descr);
    let trace_id = fail_descr.trace_id;
    fail_descr.increment_fail_count();

    // compile.py:701-717 handle_fail → must_compile → bridge tracing.
    // Check jitcounter threshold; if reached, trace alternate path and
    // compile bridge. The bridge is attached to fail_descr for fast
    // dispatch on subsequent guard failures.  Skipped on giveup (None).
    if let (Some(jct), Some(bridge_fn)) = (owning_jct.as_ref(), CALL_ASSEMBLER_BRIDGE_FN.get()) {
        let raw_num = fail_descr.fail_arg_types.len();
        if bridge_fn(
            jct.green_key,
            trace_id,
            fail_index,
            outputs_ptr,
            raw_num,
            fail_descr_ptr as usize,
        ) {
            // compile.py:704-716 / dynasm parity: the hook traces and
            // attaches the bridge; the current occurrence still resumes
            // through blackhole instead of re-entering the new bridge.
        }
    }
    let green_key = owning_jct.as_ref().map(|j| j.green_key).unwrap_or(0);

    // resume.py:1312 blackhole_from_resumedata parity: materialize
    // virtuals before blackhole resume.
    //
    // RPython-orthodox (compile.py:701-716 handle_fail): when
    // must_compile + !stack_almost_full, trace+attach a bridge;
    // OTHERWISE resume_in_blackhole. There is NO force_fn fallback —
    // handle_fail is declared `assert 0, "unreachable"` after both
    // paths (bridge always raises via compile_trace's "raises in case
    // it works", blackhole always raises via JitException). Dynasm's
    // `call_assembler_helper_trampoline` mirrors this: blackhole result
    // OR 0, no force_fn. The previous force_fn fallback created a new
    // PyFrame with empty locals via PyFrame::new_for_call(code, &[], ...)
    // in jit_force_callee_frame, which left locals_w[0] uninitialized;
    // when portal_runner re-entered the JIT via try_function_entry_jit,
    // the stale `n` value drove the self-recursive CA into an unbounded
    // loop, blowing the shadow stack (see fib_recursive_plan_2026_04_20).
    // Dynasm's call_assembler_helper_trampoline (majit-backend-dynasm/src/lib.rs
    // :189-202) passes the raw deadframe values as BOTH fail_values AND
    // raw_deadframe to bh_fn. blackhole_resume_via_rd_numb
    // (pyre-jit/src/call_jit.rs:1351) only consumes `raw_deadframe`, so the
    // rebuild_state_after_failure preprocessing step is functionally a
    // no-op for the blackhole path — but when it empties `bh_outputs`
    // (recovery.frames.slots order mismatched with raw_num), bh_fn sees
    // `num_fail_values == 0` and returns None immediately
    // (call_jit.rs:1293), which previously drove the force_fn fallback
    // into garbage-frame territory.
    if let Some(bh_fn) = CALL_ASSEMBLER_BLACKHOLE_FN.get() {
        let raw_num = fail_descr.fail_arg_types.len();
        if let Some(result) = bh_fn(
            green_key,
            trace_id,
            fail_index,
            outputs_ptr,
            raw_num,
            outputs_ptr,
            raw_num,
        ) {
            // warmspot.py:988-996: DoneWithThisFrame{Int,Ref,Float} returns
            // e.result as-is. warmspot.py:982: ContinueRunningNormally
            // applies unspecialize_value (identity for GCREF portal).
            // warmspot.py:998: ExitFrameWithExceptionRef sets JIT_EXC_VALUE
            // inside handle_blackhole_result; value here is garbage.
            // No backend-side conversion — handle_jitexception handles all types.
            return result;
        }
    }
    let _ = inputs_ptr;
    let _ = frame_ptr;
    0
}

/// Fast path for call_assembler when a force callback is available.
/// Runs compiled code with stack-allocated buffers, avoiding all heap
/// allocation per call (no Vec, no DeadFrame, no Box).
fn call_assembler_fast_path(
    target: &RegisteredLoopTarget,
    inputs: &[i64],
    outcome: *mut i64,
    force_fn: extern "C" fn(i64) -> i64,
) -> u64 {
    // RPython parity: compile_tmp_callback. When target is pending
    // (code_ptr not yet set), fall back to force_fn which runs the
    // interpreter. The force_fn receives the callee frame pointer
    // from the first input arg.
    if target.code_ptr.is_null() {
        let frame_ptr = inputs.get(0).copied().unwrap_or(0);
        // Set pending_force_local0 for lazy frame creation.
        // When create_frame is elided, frame_ptr is the caller frame.
        // force_fn uses pending_force_local0 to create a callee frame.
        let fli = target.num_scalar_inputargs;
        if inputs.len() > fli {
            PENDING_FORCE_LOCAL0.with(|c| c.set(Some(inputs[fli])));
        }
        let result = force_fn(frame_ptr);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        return result as u64;
    }

    call_assembler_fast_path_heap(target, inputs, outcome, force_fn)
}

/// Heap path for call_assembler — uses heap-allocated JitFrame
/// via run_compiled_code.
fn call_assembler_fast_path_heap(
    target: &RegisteredLoopTarget,
    inputs: &[i64],
    outcome: *mut i64,
    force_fn: extern "C" fn(i64) -> i64,
) -> u64 {
    let actual_outputs = target.max_output_slots.max(1);
    let attachments_guard = target.cpu_attachments.read().unwrap();
    let attachments: &CpuDescrAttachments = &*attachments_guard;
    let exec = run_compiled_code(
        target.code_ptr,
        &target.fail_descrs,
        target.num_ref_roots,
        target.max_output_slots,
        inputs,
        attachments,
    );
    let fail_index = exec.fail_index;
    let outputs = exec.extract_outputs(actual_outputs);

    if fail_index == CALL_ASSEMBLER_DEADFRAME_SENTINEL {
        let frame = take_call_assembler_deadframe_from_outputs(&outputs);

        let handle = store_call_assembler_deadframe(frame);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_DEADFRAME;
            *outcome.add(1) = handle as i64;
        }
        return 0;
    }

    let fail_descr = &target.fail_descrs[fail_index as usize];

    if fail_descr.is_finish() {
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        return match fail_descr.fail_arg_types() {
            [] | [Type::Void] => 0,
            // compile.py:649 DoneWithThisFrameDescr*.get_result:
            // cpu.get_{int,ref,float}_value(deadframe, 0)
            [Type::Int] | [Type::Float] | [Type::Ref] => outputs[0] as u64,
            other => unreachable!("call_assembler finish with unsupported layout: {other:?}"),
        };
    }

    // Guard failure — check for bridge, then fall back to force
    fail_descr.increment_fail_count();

    let bridge_guard = fail_descr.bridge_ref();
    if let Some(ref bridge) = *bridge_guard {
        // rebuild_state_after_failure decodes recovery_layout to match
        // what the bridge tracer saw via rebuild_from_resumedata.
        let mut bridge_outputs = outputs;
        rebuild_state_after_failure(
            &mut bridge_outputs,
            &fail_descr.fail_arg_types,
            fail_descr.recovery_layout_ref().as_ref(),
            bridge.num_inputs,
        );
        let mut frame = CraneliftBackend::execute_bridge(
            bridge,
            &bridge_outputs,
            &fail_descr.fail_arg_types,
            &attachments,
        );
        let bridge_descr = get_latest_descr_from_deadframe(&frame)
            .expect("bridge deadframe must have a descriptor");
        if bridge_descr.is_finish() {
            unsafe {
                *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
                *outcome.add(1) = 0;
            }
            return finish_result_from_deadframe(&mut frame)
                .expect("finish_result_from_deadframe failed") as u64;
        }
        let df_handle = store_call_assembler_deadframe(frame);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_DEADFRAME;
            *outcome.add(1) = df_handle as i64;
        }
        return 0;
    }
    let _ = bridge_guard;

    // resume.py:1312 blackhole_from_resumedata parity.
    if let Some(bh_fn) = CALL_ASSEMBLER_BLACKHOLE_FN.get() {
        let green_key = target.green_key;
        let trace_id = target.trace_id;
        let raw_num = fail_descr.fail_arg_types.len();
        let raw_outputs = outputs.to_vec();
        let mut bh_outputs = outputs.to_vec();
        rebuild_state_after_failure(
            &mut bh_outputs,
            &fail_descr.fail_arg_types,
            fail_descr.recovery_layout_ref().as_ref(),
            raw_num,
        );
        let num_outputs = bh_outputs.len();
        if let Some(result) = bh_fn(
            green_key,
            trace_id,
            fail_index,
            bh_outputs.as_ptr(),
            num_outputs,
            raw_outputs.as_ptr(),
            raw_num,
        ) {
            unsafe {
                *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
                *outcome.add(1) = 0;
            }
            // warmspot.py:988-996: DoneWithThisFrame* returns typed result as-is.
            // warmspot.py:998: exception already raised via jit_exc_raise.
            return result as u64;
        }
    }

    // RPython assembler_call_helper: force_fn receives the callee frame.
    // outputs[0] holds the virtualizable frame (fail_args[0] = caller),
    // but force_fn needs the callee frame which is inputs[0].
    let callee_frame_ptr = inputs[0];
    let result = force_fn(callee_frame_ptr);
    unsafe {
        *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
        *outcome.add(1) = 0;
    }
    result as u64
}

extern "C" fn call_assembler_shim(
    target_token: u64,
    args_ptr: u64,
    outcome_ptr: u64,
    _expected_result_kind: u64,
) -> u64 {
    // warmspot.py:1017-1024 assembler_call_helper parity: handle_fail
    // always raises JitException, never returns silently.
    //
    // No alternate-stack switch here. Upstream RPython runs the shim
    // on the caller's native stack so the compiled prologue's inline
    // probe (`_call_header_with_stack_check`,
    // rpython/jit/backend/x86/assembler.py:1085 parity) measures SP
    // against the same `PYRE_STACKTOOBIG.stack_end` the rest of the
    // runtime uses. Growing the stack via `stacker::maybe_grow` here
    // would read `stack_end` from a different guard page, breaking the
    // budget comparison.
    call_assembler_shim_inner(target_token, args_ptr, outcome_ptr, _expected_result_kind)
}

fn call_assembler_shim_inner(
    target_token: u64,
    args_ptr: u64,
    outcome_ptr: u64,
    _expected_result_kind: u64,
) -> u64 {
    // No wrapper-level stack probe here. assembler.py:2254-2265
    // `_genop_call_assembler` emits a plain CALL to the target trace;
    // the prologue of the target (`_call_header_with_stack_check`,
    // assembler.py:1080) runs the inline SP probe. Pyre's Cranelift
    // prologue emits the same probe at trace compile time via
    // `prologue_probe_addr()`, so any extra probe here would duplicate
    // the check the callee is about to perform on the same stack.
    let outcome = outcome_ptr as usize as *mut i64;

    let target = unsafe { &*fast_lookup_ca_target(target_token) };

    let input_slice =
        unsafe { std::slice::from_raw_parts(args_ptr as usize as *const i64, target.num_inputs) };
    assert!(
        !outcome.is_null(),
        "call_assembler shim outcome buffer must be non-null"
    );

    if std::env::var_os("MAJIT_LOG").is_some() {
        let i0 = input_slice.first().copied().unwrap_or(-1);
        eprintln!(
            "[ca-shim] entering trace_id={} num_inputs={} input0={:#x}",
            target.trace_id, target.num_inputs, i0
        );
    }
    let mut frame = execute_registered_loop_target(target, input_slice);
    let descr =
        get_latest_descr_from_deadframe(&frame).expect("get_latest_descr_from_deadframe failed");
    if descr.is_finish() {
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        let result = finish_result_from_deadframe(&mut frame)
            .expect("finish_result_from_deadframe failed") as u64;
        return result;
    }

    // RPython resume_in_blackhole parity: use blackhole to resume from
    // the guard failure point instead of re-executing from scratch.
    let fail_index = descr.fail_index();
    let fail_types = descr.fail_arg_types();
    let fail_values: Vec<i64> = fail_types
        .iter()
        .enumerate()
        .map(|(i, tp)| match tp {
            Type::Int => get_int_from_deadframe(&frame, i).unwrap_or(0),
            Type::Ref => get_ref_from_deadframe(&frame, i)
                .map(|value| value.0 as i64)
                .unwrap_or(0),
            Type::Float => get_float_from_deadframe(&frame, i)
                .map(|value| value.to_bits() as i64)
                .unwrap_or(0),
            Type::Void => 0,
        })
        .collect();
    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!(
            "[ca-shim] guard fail_idx={} nvals={}",
            fail_index,
            fail_values.len()
        );
    }
    if let Some(bh_fn) = CALL_ASSEMBLER_BLACKHOLE_FN.get() {
        // resume.py:1312 blackhole_from_resumedata parity.
        let raw_num = fail_types.len();
        let raw_outputs = fail_values.clone();
        let mut bh_outputs = fail_values;
        let recovery = target
            .fail_descrs
            .get(fail_index as usize)
            .and_then(|d| d.recovery_layout_ref().as_ref().cloned());
        rebuild_state_after_failure(&mut bh_outputs, fail_types, recovery.as_ref(), raw_num);
        let num_outputs = bh_outputs.len();
        if let Some(result) = bh_fn(
            target.green_key,
            target.trace_id,
            fail_index,
            bh_outputs.as_ptr(),
            num_outputs,
            raw_outputs.as_ptr(),
            raw_num,
        ) {
            unsafe {
                *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
                *outcome.add(1) = 0;
            }
            // warmspot.py:988-996: typed result as-is.
            return result as u64;
        }
    }
    // Fallback: force_fn re-executes from scratch.
    if let Some(force_fn) = CALL_ASSEMBLER_FORCE_FN.get() {
        let callee_frame_ptr = input_slice[0];
        let fli = target.num_scalar_inputargs;
        if input_slice.len() > fli {
            PENDING_FORCE_LOCAL0.with(|c| c.set(Some(input_slice[fli])));
        }
        let result = force_fn(callee_frame_ptr);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        return result as u64;
    }

    let handle = store_call_assembler_deadframe(frame);
    unsafe {
        *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_DEADFRAME;
        *outcome.add(1) = handle as i64;
    }
    0
}

/// RPython parity: alloc shims dispatch through the active
/// `cpu.gc_ll_descr` (`llmodel.py:58`). pyre publishes that slot in
/// the thread-local `CRANELIFT_ACTIVE_GC`, mirroring dynasm's
/// `DYNASM_ACTIVE_GC`. Roots are persistently registered by
/// `run_compiled_code` via `add_root` on jf_frame ref slot addresses
/// (`_call_header_shadowstack` parity).
extern "C" fn gc_alloc_nursery_shim(size: u64) -> u64 {
    with_cranelift_gc_required(|gc| gc.alloc_nursery(size as usize).0 as u64)
}

/// Plain malloc fallback for New() when no GC runtime is configured.
extern "C" fn plain_malloc_zeroed_shim(size: u64) -> u64 {
    let layout = std::alloc::Layout::from_size_align(size as usize, 8)
        .unwrap_or(std::alloc::Layout::new::<u8>());
    unsafe { std::alloc::alloc_zeroed(layout) as u64 }
}

/// `gc.py:51` malloc-helper OOM signaling — Cranelift counterpart of
/// `runner::oom_signal_if_zero`.  `do_malloc_fixedsize_clear` raises
/// `MemoryError` on failure; translation lowers that to "store the
/// singleton in `cpu.pos_exc_value`, return NULL".  pyre malloc
/// helpers return 0 directly, so this wrapper performs the equivalent
/// thread-local store before propagating the NULL up to the
/// JIT-emitted `CHECK_MEMORY_ERROR` (compiler.rs:7997).
#[inline]
fn oom_signal_if_zero(result: u64) -> u64 {
    if result == 0 {
        let v = majit_backend::memory_error_singleton_ref();
        if v != 0 {
            JIT_EXC_VALUE.store(v, std::sync::atomic::Ordering::Relaxed);
            // llmodel.py:194-199 _store_exception parity: typeptr is
            // value's first word.  Safe because the singleton is a
            // valid `W_ExceptionObject` allocated by pyre-object.
            let exc_type = unsafe { *(v as *const i64) };
            JIT_EXC_TYPE.store(exc_type, std::sync::atomic::Ordering::Relaxed);
        }
    }
    result
}

extern "C" fn gc_alloc_typed_nursery_shim(type_id: u64, size: u64) -> u64 {
    // RPython rewrite.py: CALL_MALLOC_NURSERY fallback path.
    // Use no-collect allocation to avoid triggering GC during compiled code
    // execution. When nursery is full, falls back to old-gen allocation.
    with_cranelift_gc_required(|gc| {
        gc.alloc_nursery_no_collect_typed(type_id as u32, size as usize)
            .0 as u64
    })
}

extern "C" fn gc_alloc_varsize_shim(base_size: u64, item_size: u64, length: u64) -> u64 {
    with_cranelift_gc_required(|gc| {
        gc.alloc_varsize(base_size as usize, item_size as usize, length as usize)
            .0 as u64
    })
}

fn raw_varsize_alloc_typed_and_set_len(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
) -> u64 {
    let Some(var_bytes) = item_size.checked_mul(length) else {
        return 0;
    };
    let Some(payload_size) = base_size.checked_add(var_bytes) else {
        return 0;
    };
    let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
        return 0;
    };
    // gc.py:51 contract: malloc helper returns NULL on OOM or invalid
    // layout; the caller's CHECK_MEMORY_ERROR converts NULL into a
    // MemoryError.  Falling back to a 1-byte layout would silently
    // succeed-with-a-bogus-allocation and then corrupt the caller's
    // header/length store.
    let Ok(layout) = std::alloc::Layout::from_size_align(total_size, 8) else {
        return 0;
    };
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        return 0;
    }
    unsafe {
        *(raw as *mut GcHeader) = GcHeader::new(type_id);
    }
    let obj = unsafe { raw.add(GcHeader::SIZE) };
    unsafe {
        *(obj.add(length_ofs) as *mut usize) = length;
    }
    obj as u64
}

fn active_runtime_alloc_varsize_typed_and_set_len(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
) -> u64 {
    let result = with_cranelift_gc(|gc| {
        let obj = gc.alloc_varsize_typed(type_id, base_size, item_size, length);
        if obj.is_null() {
            0
        } else {
            unsafe {
                *((obj.0 as *mut u8).add(length_ofs) as *mut usize) = length;
            }
            obj.0 as u64
        }
    });
    match result {
        Some(v) => v,
        None => {
            raw_varsize_alloc_typed_and_set_len(type_id, base_size, item_size, length_ofs, length)
        }
    }
}

extern "C" fn gc_malloc_array_helper(item_size: u64, type_id: u64, num_elem: u64) -> u64 {
    oom_signal_if_zero(active_runtime_alloc_varsize_typed_and_set_len(
        type_id as u32,
        std::mem::size_of::<usize>(),
        item_size as usize,
        0,
        num_elem as usize,
    ))
}

extern "C" fn gc_malloc_array_nonstandard_helper(
    base_size: u64,
    item_size: u64,
    length_ofs: u64,
    type_id: u64,
    num_elem: u64,
) -> u64 {
    oom_signal_if_zero(active_runtime_alloc_varsize_typed_and_set_len(
        type_id as u32,
        base_size as usize,
        item_size as usize,
        length_ofs as usize,
        num_elem as usize,
    ))
}

fn raw_fixedsize_alloc_typed(type_id: u32, size: usize) -> u64 {
    let Some(total_size) = GcHeader::SIZE.checked_add(size) else {
        return 0;
    };
    let Ok(layout) = std::alloc::Layout::from_size_align(total_size, 8) else {
        return 0;
    };
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        return 0;
    }
    unsafe {
        *(raw as *mut GcHeader) = GcHeader::new(type_id);
    }
    let obj = unsafe { raw.add(GcHeader::SIZE) };
    obj as u64
}

fn active_runtime_alloc_oldgen_typed(type_id: u32, payload_size: usize) -> u64 {
    // gc.py:51 contract: helper returns NULL on OOM so
    // CHECK_MEMORY_ERROR can convert it into a MemoryError.  Only
    // `None` (no active runtime — typical for unit tests) falls back
    // to a raw alloc; `Some(0)` (a real OOM from the registered
    // allocator) propagates as 0 unchanged, mirroring the existing
    // CALL_MALLOC_NURSERY slowpath at runner.rs:184 / compiler.rs:8959.
    match with_cranelift_gc(|gc| gc.alloc_oldgen_typed(type_id, payload_size).0 as u64) {
        None => raw_fixedsize_alloc_typed(type_id, payload_size),
        Some(v) => v,
    }
}

/// gc.py:481-490 `malloc_big_fixedsize(size, tid)` — fixed-size object
/// large enough to skip the nursery, allocated directly in the old gen
/// via `do_malloc_fixedsize_clear`.  Header is stamped with the type
/// id so callers MUST NOT emit a separate `gen_initialize_tid`.
extern "C" fn gc_malloc_big_fixedsize_helper(size: u64, type_id: u64) -> u64 {
    // The CALL_R arg is `total = payload + GcHeader::SIZE` (built by
    // `handle_new` at rewrite.rs:857 to include the GC header).  The
    // runtime allocators (`alloc_oldgen_typed`, `alloc_nursery_typed`)
    // and the raw fallback both prepend the GC header themselves and
    // expect a payload-only size, so subtract `HDR` here once —
    // mirroring the existing CALL_MALLOC_NURSERY slowpath at
    // runner.rs:184 (`alloc_nursery(total_size - gc_hdr)`).
    let payload = (size as usize).saturating_sub(GcHeader::SIZE);
    oom_signal_if_zero(active_runtime_alloc_oldgen_typed(type_id as u32, payload))
}

/// gc.py:460 `malloc_str(length)` — but the upstream closure captures
/// `str_type_id` from `self.str_descr.tid` at generate-time.  `extern
/// "C" fn` cannot capture, so the type id is threaded through the
/// CALL_R as an explicit Signed arg and the calldescr's first param is
/// it (see `make_malloc_str_calldescr`).
extern "C" fn gc_malloc_str_helper(type_id: u64, length: u64) -> u64 {
    oom_signal_if_zero(active_runtime_alloc_varsize_typed_and_set_len(
        type_id as u32,
        BUILTIN_STR_TOKEN_BASE_SIZE,
        1,
        BUILTIN_STRING_LEN_OFFSET,
        length as usize,
    ))
}

/// gc.py:469 `malloc_unicode(length)` — see `gc_malloc_str_helper` for
/// the closure-vs-extern type-id threading rationale.
extern "C" fn gc_malloc_unicode_helper(type_id: u64, length: u64) -> u64 {
    oom_signal_if_zero(active_runtime_alloc_varsize_typed_and_set_len(
        type_id as u32,
        BUILTIN_UNICODE_TOKEN_BASE_SIZE,
        4,
        BUILTIN_STRING_LEN_OFFSET,
        length as usize,
    ))
}

/// aarch64/assembler.py:342 get_write_barrier_fn()
/// → incminimark.py:1569 jit_remember_young_pointer
///
/// JIT-emitted write barriers fire on JITFrame slot writes. Backend-only
/// unit tests run compiled code without a registered GC runtime: those
/// frames live outside any managed heap and need no barrier, so behave
/// like `write_barrier_if_managed` and silently skip.
extern "C" fn gc_write_barrier_shim(obj: u64) {
    with_cranelift_gc(|gc| gc.write_barrier(GcRef(obj as usize)));
}

/// aarch64/assembler.py:352 get_write_barrier_from_array_fn()
/// → incminimark.py:1606 jit_remember_young_pointer_from_array
extern "C" fn gc_jit_remember_young_pointer_from_array_shim(obj: u64) {
    with_cranelift_gc(|gc| {
        gc.jit_remember_young_pointer_from_array(GcRef(obj as usize));
    });
}

thread_local! {
    static DECLARED_VARS_DEBUG: std::cell::RefCell<Option<std::collections::HashSet<u32>>> = const { std::cell::RefCell::new(None) };
    /// Op-result variable positions: ops that define a result via def_var.
    /// When an OpRef collides (in both constants map and op-result set),
    /// the variable takes precedence over the constant.
    static OP_RESULT_VARS: std::cell::RefCell<Option<std::collections::HashSet<u32>>> = const { std::cell::RefCell::new(None) };
}

fn opref_is_op_result_var(opref: OpRef) -> bool {
    OP_RESULT_VARS.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|rv| rv.contains(&opref.raw()))
    })
}

fn resolve_opref(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    opref: OpRef,
) -> CValue {
    // RPython parity (`llgraph/runner.py:1124, 1177`): both op args and
    // failargs flow through `env[box]`, so a missing/none binding is an
    // immediate invariant failure (KeyError) — never a silent zero
    // substitution. `OpRef::NONE` reaching here means the optimizer
    // emitted an op or failarg without a real binding; surface that
    // at the producing site rather than masking it.
    assert!(
        !opref.is_none(),
        "cranelift resolve_opref: OpRef::NONE reached arg-resolution — \
         RPython env[box] would KeyError. The optimizer site that \
         emitted this null OpRef must bind it (or rewrite the op) \
         instead of relying on a zero-load fallback"
    );
    // RPython boxes/inputargs are distinct from Consts even when pyre's
    // compact OpRef numbering collides with an entry in the constant pool.
    // Once an OpRef has a declared variable, the variable is authoritative.
    let is_op_result = opref_is_op_result_var(opref);
    if is_op_result {
        return builder.use_var(var(opref.raw()));
    }
    if let Some(&c) = constants.get(&opref.raw()) {
        return builder.ins().iconst(cl_types::I64, c);
    }
    // RPython parity guard: every Box reaching codegen MUST be bound —
    // an inputarg, a label arg, a constant, or a previous op result
    // (regalloc.py invariant). Pyre's `validate_oprefs_for_compile`
    // (compiler.rs:4013) already rejects undefined OpRefs that flow
    // into a dereferencing first arg with `BackendError::CompilationFailed`
    // (the equivalent of RPython's `InvalidLoop` abort, routing the
    // bridge to the blackhole resume path). Anything that reaches this
    // point without a `def_var` is a binding-invariant violation —
    // panic so the parity hole surfaces at the optimizer site that
    // emitted the unbound OpRef instead of being papered over with
    // `iconst(0)`.
    let is_declared = DECLARED_VARS_DEBUG.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|dv| dv.contains(&opref.raw()))
    });
    if !is_declared {
        panic!(
            "cranelift resolve_opref: OpRef({}) (raw={}) has no def_var \
             and is not in constants — RPython parity violation: every \
             Box reaching codegen must be bound. The optimizer site that \
             emitted this unbound OpRef must be fixed (or InvalidLoop'd in \
             validate_oprefs_for_compile) instead of falling back to a \
             zero load",
            opref.raw(),
            opref.raw()
        );
    }
    builder.use_var(var(opref.raw()))
}

fn resolve_binop(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    op: &Op,
) -> (CValue, CValue) {
    let a = resolve_opref(builder, constants, op.arg(0));
    let b = resolve_opref(builder, constants, op.arg(1));
    (a, b)
}

fn emit_icmp(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    cc: IntCC,
    op: &Op,
    vi: u32,
) {
    let (a, b) = resolve_binop(builder, constants, op);
    let cmp = builder.ins().icmp(cc, a, b);
    let r = builder.ins().uextend(cl_types::I64, cmp);
    builder.def_var(var(vi), r);
}

fn emit_fcmp(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    cc: FloatCC,
    op: &Op,
    vi: u32,
) {
    let (a, b) = resolve_binop(builder, constants, op);
    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
    let cmp = builder.ins().fcmp(cc, fa, fb);
    let r = builder.ins().uextend(cl_types::I64, cmp);
    builder.def_var(var(vi), r);
}

/// Map a field size (in bytes) to the corresponding Cranelift type.
fn cl_type_for_size(size: usize) -> cranelift_codegen::ir::Type {
    match size {
        1 => cl_types::I8,
        2 => cl_types::I16,
        4 => cl_types::I32,
        8 => cl_types::I64,
        _ => cl_types::I64,
    }
}

/// `symbolic.get_array_token(rstr.STR/UNICODE, ...)` token triple wrapped
/// as an `ArrayDescr`.  See the parallel helper in
/// `majit-backend-dynasm/src/runner.rs` for rationale.
fn builtin_string_array_descr(opcode: OpCode) -> Option<majit_ir::DescrRef> {
    let (base_size, item_size) = match opcode {
        OpCode::Newstr
        | OpCode::Strlen
        | OpCode::Strgetitem
        | OpCode::Strsetitem
        | OpCode::Copystrcontent => (BUILTIN_STR_TOKEN_BASE_SIZE, 1),
        OpCode::Newunicode
        | OpCode::Unicodelen
        | OpCode::Unicodegetitem
        | OpCode::Unicodesetitem
        | OpCode::Copyunicodecontent => (BUILTIN_UNICODE_TOKEN_BASE_SIZE, 4),
        _ => return None,
    };

    let len_descr = Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_LEN_OFFSET,
        field_size: BUILTIN_STRING_HASH_SIZE,
        field_type: Type::Int,
        signed: false,
    });
    Some(Arc::new(BuiltinArrayDescr {
        base_size,
        item_size,
        type_id: 0,
        item_type: Type::Int,
        signed: false,
        len_descr,
    }))
}

/// `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...)` wrapped as a
/// FieldDescr.  Injected for STRHASH/UNICODEHASH so rewrite.py:283-294
/// can decode it with `unpack_fielddescr(...)`.
fn builtin_string_hash_field_descr(opcode: OpCode) -> Option<majit_ir::DescrRef> {
    if !matches!(opcode, OpCode::Strhash | OpCode::Unicodehash) {
        return None;
    }
    Some(Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_HASH_OFFSET,
        field_size: BUILTIN_STRING_HASH_SIZE,
        field_type: Type::Int,
        // rewrite.py:288,293 pass `sign=True`.
        signed: true,
    }))
}

fn checked_cl_type_for_size(
    size: usize,
    opcode: OpCode,
    detail: &str,
) -> Result<cranelift_codegen::ir::Type, BackendError> {
    match size {
        1 | 2 | 4 | 8 => Ok(cl_type_for_size(size)),
        _ => Err(unsupported_semantics(opcode, detail)),
    }
}

fn op_var_index(op: &Op, op_idx: usize, num_inputs: usize) -> usize {
    if op.pos.is_none() {
        num_inputs + op_idx
    } else {
        op.pos.raw() as usize
    }
}

/// rewrite.py:397-407 `optimize_GUARD_CLASS` parity:
/// Pre-flight check that the OpRefs feeding pointer-dereferencing
/// guard ops are bound to a defined value. RPython's box-identity
/// scheme makes the equivalent of an "undefined OpRef" impossible —
/// every Box used as a guard argument is tracked through regalloc.
/// pyre uses OpRef indices instead of identity, and pyre's optimizer
/// occasionally emits ops that reference OpRefs no preceding op
/// produced. Cranelift's `use_var` on an undeclared variable then
/// silently returns the lazy default value (zero) and the
/// constant-folder turns it into `mov x0, #0; ldr x0, [x0]` (a null
/// vtable read) which crashes at runtime.
///
/// This validator catches that pattern at compile time for the
/// pointer-dereferencing opcodes — `Guard{Class,NonnullClass,Nonnull}`,
/// `GetfieldGc{,Pure}{I,R,F}`, `GetarrayitemGc*` etc. — and returns
/// `BackendError::CompilationFailed` so the caller falls back to the
/// blackhole resume path. Non-dereferencing opcodes (`IntAddOvf` etc.)
/// are deliberately left alone: they tolerate the legacy
/// silent-iconst-zero fallback (the value is wrong but the call does
/// not crash), and validating them would reject many bridges that
/// pyre's existing optimizer happens to emit with undefined OpRef
/// references. Pyre's optimizer should ultimately stop producing
/// undefined OpRefs (RPython parity), but until that lands, the
/// validator only protects the crash-causing paths.
///
/// `is_rewriter_immediate_arg` mirrors the GC-rewriter convention
/// used by `resolve_rewriter_immediate_i64`: a handful of opcodes
/// encode numeric immediates (allocation sizes, scale factors) by
/// stuffing the raw value directly into the OpRef field instead of
/// going via the constant pool. Those args are NOT real OpRef
/// references and must be skipped by the validator.
fn is_rewriter_immediate_arg(opcode: majit_ir::OpCode, arg_idx: usize) -> bool {
    use majit_ir::OpCode;
    match opcode {
        // ZeroArray(base, start_imm, size_imm, scale_start_imm, scale_size_imm)
        OpCode::ZeroArray => arg_idx >= 1 && arg_idx <= 4,
        _ => false,
    }
}

/// True for opcodes that dereference their first arg as a pointer.
/// An undefined OpRef in `arg(0)` of one of these ops becomes a null
/// pointer dereference at runtime — the validator must reject these.
fn op_dereferences_first_arg(opcode: majit_ir::OpCode) -> bool {
    use majit_ir::OpCode;
    matches!(
        opcode,
        // assembler.py:1880-1891 _cmp_guard_class: CMP(mem(loc_ptr, offset), classptr)
        OpCode::GuardClass
            | OpCode::GuardNonnullClass
            | OpCode::GuardSubclass
            | OpCode::GuardGcType
            | OpCode::GuardIsObject
            // GetfieldGc*: load from [loc_ptr + offset]
            | OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldGcPureI
            | OpCode::GetfieldGcPureR
            | OpCode::GetfieldGcPureF
            // SetfieldGc: store to [loc_ptr + offset]
            | OpCode::SetfieldGc
            // GetArrayItemGc / SetArrayItemGc: load/store from arrays
            | OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::SetarrayitemGc
    )
}

fn validate_oprefs_for_compile(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &HashMap<u32, i64>,
) -> Result<(), BackendError> {
    let num_inputs = inputargs.len();
    // RPython rewrite.py:397 + regalloc invariant: at the current op,
    // every dereferenced argument MUST be currently live (bound by an
    // earlier op result, an inputarg, a label arg seen at this PC, or
    // a constant). Forward references — boxes that will only be
    // defined later in the trace — are a structural bug, not a
    // recoverable case. Walk the ops in trace order with a single
    // `seen` set; do NOT pre-collect a `defined` set that lets
    // forward-references slip through.
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for input in inputargs {
        seen.insert(input.index);
    }
    for (op_idx, op) in ops.iter().enumerate() {
        if op.opcode == majit_ir::OpCode::Label {
            // LABEL params are introduced at the label block.
            for &arg in &op.args {
                if !arg.is_none() {
                    seen.insert(arg.raw());
                }
            }
            // The label op result itself (if any) is bound after the
            // label arg insertion below; fall through.
        }
        if op_dereferences_first_arg(op.opcode) {
            let arg = op.arg(0);
            let bound = arg.is_none()
                || constants.contains_key(&arg.raw())
                || is_rewriter_immediate_arg(op.opcode, 0)
                || seen.contains(&arg.raw());
            if !bound {
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[validate-oprefs] op[{}] {:?} dereferences undefined OpRef::from_raw({}) — InvalidLoop",
                        op_idx,
                        op.opcode,
                        arg.raw()
                    );
                }
                return Err(BackendError::CompilationFailed(format!(
                    "InvalidLoop: op[{}] {:?} dereferences undefined OpRef::from_raw({})",
                    op_idx,
                    op.opcode,
                    arg.raw()
                )));
            }
        }
        if op.result_type() != majit_ir::Type::Void {
            seen.insert(op_var_index(op, op_idx, num_inputs) as u32);
        }
    }
    Ok(())
}

fn unsupported_semantics(opcode: OpCode, detail: &str) -> BackendError {
    BackendError::Unsupported(format!(
        "opcode {:?} is not supported in the Cranelift backend yet: {detail}",
        opcode
    ))
}

/// Strict `+1` paired-guard check matching RPython
/// (`rpython/jit/backend/x86/assembler.py:2225-2244`):
/// `_genop_call_may_force` / `_genop_call_release_gil` both invoke
/// `_store_force_index(self._find_nearby_operation(+1))`, and
/// `_store_force_index` asserts `isinstance(guard_op, GuardOp)` with
/// `getopnum() in {GUARD_NOT_FORCED, GUARD_NOT_FORCED_2}`. Upstream
/// preserves this invariant by ensuring `optimizeopt` never schedules
/// any op between a force-able call and its paired guard — pyre's
/// optimizer must do the same.
///
/// Forward-looking note (review feedback): the wrong shape to watch
/// for is `CMF -> NEW_WITH_VTABLE -> SETFIELD_GC -> GUARD_NOT_FORCED`,
/// where a virtual escapes *after* the call. RPython's
/// `optimizeopt/virtualize.py` always materializes escaping virtuals
/// *before* the force-able call, producing
/// `NEW_WITH_VTABLE -> SETFIELD_GC -> CMF -> GUARD_NOT_FORCED` so that
/// every box reachable from `GuardNotForced.fail_args` already lives
/// in memory at call time. If this strict check ever rejects a real
/// trace, the fix is in the optimizer (force virtuals before the
/// call), not by relaxing the backend assertion — the backend's
/// `_store_force_index` runs once, on the operation at +1, so a
/// post-call materialization would leave fail_args pointing at boxes
/// the force path cannot read coherently.
fn check_paired_guard_not_forced(
    ops: &[Op],
    op_idx: usize,
    opcode: OpCode,
    label: &str,
) -> Result<(), BackendError> {
    let Some(next) = ops.get(op_idx + 1) else {
        return Err(unsupported_semantics(
            opcode,
            &format!("{label}: no paired guard_not_forced(_2) at +1 — trace ends"),
        ));
    };
    match next.opcode {
        OpCode::GuardNotForced | OpCode::GuardNotForced2 => Ok(()),
        other => Err(unsupported_semantics(
            opcode,
            &format!(
                "{label}: expected guard_not_forced(_2) at +1, found {other:?} — \
                 optimizer must not schedule ops between force-able call and its \
                 paired guard (RPython x86/assembler.py:2225-2244 invariant)"
            ),
        )),
    }
}

fn missing_gc_runtime(opcode: OpCode) -> BackendError {
    BackendError::Unsupported(format!(
        "opcode {:?} requires a configured GC runtime in the Cranelift backend",
        opcode
    ))
}

/// rewrite.py:613-666 `gen_malloc_frame` / `handle_call_assembler` —
/// the GC rewriter mediates the callee jitframe allocation and writes
/// the rewritten descrs into the call op. With no active GC runtime
/// or registered jitframe layout, the rewrite never runs, so the
/// emitted code would dispatch on a null callee jitframe. Refuse to
/// compile with the same error shape RPython surfaces from
/// `gc_ll_descr.gen_malloc_frame` when the descrs are missing.
fn validate_call_assembler_rewrite_prereqs(ops: &[Op]) -> Result<(), BackendError> {
    for op in ops {
        if !matches!(
            op.opcode,
            OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN
        ) {
            continue;
        }
        if !cranelift_gc_active() {
            return Err(missing_gc_runtime(op.opcode));
        }
        if JITFRAME_LAYOUT
            .get()
            .and_then(|info| info.jitframe_descrs.as_ref())
            .is_none()
        {
            return Err(unsupported_semantics(
                op.opcode,
                "call-assembler requires a registered jitframe layout for rewrite parity",
            ));
        }
    }
    Ok(())
}

fn resolve_call_assembler_target(
    opcode: OpCode,
    call_descr: &dyn CallDescr,
) -> Result<Option<RegisteredLoopTarget>, BackendError> {
    let target_token = call_descr.call_target_token().ok_or_else(|| {
        unsupported_semantics(
            opcode,
            "call-assembler descriptor must provide a compiled target token",
        )
    })?;
    let Some(target) = lookup_call_assembler_target(target_token) else {
        return Ok(None);
    };

    // Pending targets (null code_ptr) are placeholders — no compiled code
    // or finish descriptors yet. Return None so codegen uses shim fallback.
    // At runtime, call_assembler_fast_path detects null code_ptr and calls
    // force_fn (RPython compile_tmp_callback parity).
    if target.code_ptr.is_null() {
        return Ok(None);
    }

    // RPython rewrite.py:665-695 handle_call_assembler: when VableExpansion
    // is present, the CALL_ASSEMBLER op carries fewer args than the callee's
    // inputarg_types. The EXPANDED arity (1 frame + scalar_fields + array_items)
    // must match the target. Without VableExpansion, op.args matches directly.
    let expected_arity = if let Some(exp) = call_descr.vable_expansion() {
        1 + exp.scalar_fields.len() + exp.num_array_items
    } else {
        call_descr.arg_types().len()
    };
    if target.inputarg_types.len() != expected_arity {
        // PRE-EXISTING-ADAPTATION: pyre's `direct_assembler_call`
        // emits CALL_ASSEMBLER with `arg_types = [frame]` (or
        // `[frame, ec]` once `extra_reds` flips) while the callee's
        // compiled trace still exposes the full vable-expanded
        // inputarg list (`frame + scalars + items`, e.g. 8 for fib).
        // The shrink-to-reds happens in
        // `patch_new_loop_to_load_virtualizable_fields`
        // (compile.py:425-461), which only fires when
        // `state.rs:4058 driver_descriptor()` returns `Some(...)`.
        // Pyre holds it disabled pending Task #21 vable
        // heap-writeback (see state.rs comment for the full
        // prerequisite chain). Until then, when the target exposes
        // scalar header slots (`num_scalar_inputargs >` caller's red
        // count), the mismatch IS the vable-expansion case; treat it
        // as unresolved so the runtime falls through to the helper
        // (force_fn) path, matching how the pending-token path
        // already handles `code_ptr = null`. The `Err` branch is
        // reserved for cases that look like a real layout bug —
        // typically when the target's scalar header is SMALLER than
        // the caller's red count, which would not be explained by
        // pending vable expansion.
        if target.num_scalar_inputargs > call_descr.arg_types().len() {
            return Ok(None);
        }
        return Err(unsupported_semantics(
            opcode,
            &format!(
                "call-assembler target arity {} does not match expected {}{}",
                target.inputarg_types.len(),
                expected_arity,
                if call_descr.vable_expansion().is_some() {
                    " (with VableExpansion)"
                } else {
                    ""
                },
            ),
        ));
    }
    let finish_descr = target
        .fail_descrs
        .iter()
        .find(|descr| descr.is_finish())
        .ok_or_else(|| {
            unsupported_semantics(
                opcode,
                "call-assembler target must expose at least one finish exit",
            )
        })?;
    let finish_types = finish_descr.fail_arg_types();

    // Validate that the finish result type matches the call descriptor.
    // When force_token_slots are present the finish output type is Ref
    // (the raw force-token handle), so accept that as matching a Ref
    // result descriptor even though the value isn't a real GC ref.
    match call_descr.result_type() {
        Type::Void => {
            if !finish_types.is_empty() {
                return Err(unsupported_semantics(
                    opcode,
                    "void call-assembler targets must finish without result values",
                ));
            }
        }
        result_type => {
            if finish_types != [result_type] {
                // Type mismatch between caller's expected result and target's
                // actual finish type. Treat as unresolved — runtime will use
                // the helper fallback path.
                return Ok(None);
            }
        }
    }

    Ok(Some(target))
}

fn expected_call_assembler_result_kind(
    call_descr: &dyn CallDescr,
    target: Option<&RegisteredLoopTarget>,
) -> Result<u64, BackendError> {
    match call_descr.result_type() {
        Type::Void => Ok(CALL_ASSEMBLER_RESULT_VOID),
        Type::Int => Ok(CALL_ASSEMBLER_RESULT_INT),
        Type::Float => Ok(CALL_ASSEMBLER_RESULT_FLOAT),
        Type::Ref => {
            let Some(target) = target else {
                return Ok(CALL_ASSEMBLER_RESULT_REF);
            };
            let finish_descr = target
                .fail_descrs
                .iter()
                .find(|descr| descr.is_finish())
                .ok_or_else(|| {
                    unsupported_semantics(
                        OpCode::CallAssemblerR,
                        "call-assembler target must expose at least one finish exit",
                    )
                })?;
            Ok(
                if finish_descr.fail_arg_types() == [Type::Ref]
                    && finish_descr.force_token_slots == [0]
                {
                    CALL_ASSEMBLER_RESULT_FORCE_TOKEN_REF
                } else {
                    CALL_ASSEMBLER_RESULT_REF
                },
            )
        }
    }
}

fn build_known_values_set(inputargs: &[InputArg], ops: &[Op]) -> HashSet<u32> {
    let mut known = HashSet::new();
    for input in inputargs {
        known.insert(input.index);
    }
    for op in ops {
        if op.result_type() != Type::Void && !op.pos.is_none() {
            known.insert(op.pos.raw());
        }
    }
    known
}

fn build_force_token_set(inputargs: &[InputArg], ops: &[Op]) -> Result<HashSet<u32>, BackendError> {
    let mut force_tokens = HashSet::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if op.pos.is_none() {
            continue;
        }
        let result_var = op_var_index(op, op_idx, inputargs.len()) as u32;
        if op.opcode == OpCode::ForceToken {
            force_tokens.insert(result_var);
            continue;
        }
        if op.opcode == OpCode::CallAssemblerR {
            let descr = op.descr.as_ref().ok_or_else(|| {
                unsupported_semantics(op.opcode, "call-assembler op must have a descriptor")
            })?;
            let call_descr = descr.as_call_descr().ok_or_else(|| {
                unsupported_semantics(op.opcode, "call-assembler descriptor must be a CallDescr")
            })?;
            if let Some(target) = resolve_call_assembler_target(op.opcode, call_descr)? {
                if let Some(finish_descr) =
                    target.fail_descrs.iter().find(|descr| descr.is_finish())
                {
                    if finish_descr.fail_arg_types() == [Type::Ref]
                        && finish_descr.force_token_slots == [0]
                    {
                        force_tokens.insert(result_var);
                    }
                }
            }
        }
    }
    Ok(force_tokens)
}

/// Auxiliary LABEL type overrides on top of `OpTypeIndex`.
///
/// `OpTypeIndex` covers the RPython `box.type` parity surface — every
/// OpRef yields its declared type from the corresponding inputarg/op/const
/// object.  The only pyre-only flat-OpRef effect handled here is the
/// JUMP→LABEL edge rewrite: when a JUMP delivers a value of a different
/// runtime type to a LABEL arg, the label arg's value at runtime carries
/// the jump's type, not the inputarg's declared type. RPython's Box.type
/// does not need this because each iteration allocates fresh Boxes with
/// fresh `.type` attributes.
///
/// `op_def_positions` records the op_index where an OpRef was redefined
/// from its inputarg type (or by a JUMP→LABEL rewrite). It drives the
/// "guard fired before redefinition → use original inputarg type"
/// fallback in `resolve_fail_arg_types`.
fn build_type_overrides(
    inputargs: &[InputArg],
    ops: &[Op],
    type_index: &OpTypeIndex<'_>,
) -> (HashMap<u32, Type>, HashMap<u32, usize>) {
    let mut overrides: HashMap<u32, Type> = HashMap::new();
    let mut op_def_positions: HashMap<u32, usize> = HashMap::new();

    for (op_idx, op) in ops.iter().enumerate() {
        let result_type = op.result_type();
        if result_type != Type::Void {
            // history.py:220 Box.type parity: an op's `.pos` is the typed
            // OpRef the recorder assigned to its result. Use it directly
            // — `inputarg_type` looks up by raw position so the variant
            // tag is variant-blind, and an op without `.pos` (synthetic)
            // can never collide with an inputarg slot anyway.
            if op.pos.is_none() {
                continue;
            }
            let var_idx = op.pos.raw();
            if let Some(ia_type) = type_index.inputarg_type(op.pos) {
                if ia_type != result_type {
                    op_def_positions.insert(var_idx, op_idx);
                }
            }
        }
    }

    // Box identity parity: when a JUMP sends a value to a LABEL arg,
    // the LABEL arg's type changes to the JUMP arg's type. RPython's
    // Box objects carry their own types; in our flat OpRef namespace we
    // must propagate types through the JUMP→LABEL edge explicitly.
    // Collect LABEL→descr and JUMP→descr, then match by descr index.
    let mut label_by_descr: HashMap<u32, usize> = HashMap::new();
    let mut jumps_by_descr: Vec<(u32, usize)> = Vec::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if op.opcode == OpCode::Label {
            if let Some(ref d) = op.descr {
                label_by_descr.insert(d.index(), op_idx);
            }
        } else if op.opcode == OpCode::Jump {
            if let Some(ref d) = op.descr {
                jumps_by_descr.push((d.index(), op_idx));
            }
        }
    }
    let lookup =
        |opref: OpRef, at_op_index: usize, fallback: Type, ovrs: &HashMap<u32, Type>| -> Type {
            if let Some(&tp) = ovrs.get(&opref.raw()) {
                return tp;
            }
            type_index
                .opref_type_at(opref, at_op_index)
                .unwrap_or(fallback)
        };
    for (descr_idx, jump_idx) in &jumps_by_descr {
        let Some(&label_idx) = label_by_descr.get(descr_idx) else {
            continue;
        };
        let label_op = &ops[label_idx];
        let jump_op = &ops[*jump_idx];
        for (i, &label_arg) in label_op.args.iter().enumerate() {
            if label_arg.is_none() {
                continue;
            }
            let Some(&jump_arg) = jump_op.args.get(i) else {
                continue;
            };
            if jump_arg.is_none() {
                continue;
            }
            let jump_type = lookup(jump_arg, *jump_idx, Type::Int, &overrides);
            let label_type = lookup(label_arg, label_idx, Type::Ref, &overrides);
            if jump_type != label_type {
                overrides.insert(label_arg.raw(), jump_type);
                op_def_positions.insert(label_arg.raw(), label_idx);
            }
        }
    }

    (overrides, op_def_positions)
}

#[inline]
fn lookup_type_at(
    type_index: &OpTypeIndex<'_>,
    overrides: &HashMap<u32, Type>,
    opref: OpRef,
    op_index: usize,
) -> Option<Type> {
    if let Some(&tp) = overrides.get(&opref.raw()) {
        return Some(tp);
    }
    type_index.opref_type_at(opref, op_index)
}

fn build_ref_root_slots(
    inputargs: &[InputArg],
    ops: &[Op],
    type_index: &OpTypeIndex<'_>,
    overrides: &HashMap<u32, Type>,
    force_tokens: &HashSet<u32>,
) -> Result<Vec<(u32, usize)>, BackendError> {
    let mut seen = HashSet::new();
    let mut slots = Vec::new();

    // RPython parity: when jump_to_preamble is used without
    // force_box_for_end_of_preamble, the body JUMP may pass Float/Int
    // values at Ref-typed inputarg positions. Build a set of inputarg
    // OpRef raw values that receive non-Ref values from the backedge JUMP.
    // These positions must NOT be treated as GC ref roots, because
    // (1) the GC would try to trace Float/Int bits as pointers, and
    // (2) the preamble guard check would dereference non-pointer values.
    //
    // Phase E.2b: bridge `InputArg.index` is shifted into the disjoint
    // `[bridge_inputarg_base..)` range, so `arg.0` and `inputargs[i].index`
    // are not interchangeable. RPython relies on Box identity here
    // (regalloc.py treats `box is inputarg` as the test); pyre's
    // line-by-line analog is `arg.0 == inputargs[i].index`. Key both
    // `non_ref_at_backedge` and `used_inputargs` by the InputArg's raw
    // OpRef value so the downstream loop's `contains(&input.index)` checks
    // line up with the keys we inserted.
    let inputarg_oprefs: HashSet<u32> = inputargs.iter().map(|ia| ia.index).collect();
    let mut non_ref_at_backedge: HashSet<u32> = HashSet::new();
    let mut has_float_at_ref_position = false;
    // Find the closing JUMP and check arg types against inputarg types
    if let Some((jump_idx, jump)) = ops
        .iter()
        .enumerate()
        .rfind(|(_, op)| op.opcode == OpCode::Jump)
    {
        let num_inputs = inputargs.len();
        for (i, &arg) in jump.args.iter().enumerate() {
            if i >= num_inputs {
                break;
            }
            if inputargs[i].tp != Type::Ref {
                continue;
            }
            // RPython regalloc.py: backedge JUMP arg `is` an inputarg Box →
            // GC roots already cover it. Pyre analog: arg.raw() matches some
            // InputArg.index whose .tp is Ref. Phase E.2b shifted bridge
            // inputargs into `[bridge_inputarg_base..)`, so a dense lookup
            // by position no longer maps to the right InputArg.
            if inputargs
                .iter()
                .any(|ia| ia.index == arg.raw() && ia.tp == Type::Ref)
            {
                continue; // inputarg reference — always safe
            }
            if let Some(actual_tp) = lookup_type_at(type_index, overrides, arg, jump_idx) {
                if actual_tp != Type::Ref {
                    non_ref_at_backedge.insert(inputargs[i].index);
                    if actual_tp == Type::Float {
                        has_float_at_ref_position = true;
                    }
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[ref-root] SKIP inputarg idx={}: backedge passes {:?} (arg={:?})",
                            inputargs[i].index, actual_tp, arg
                        );
                    }
                }
            }
        }
    }

    // RPython regalloc parity: only track ref roots that are actually
    // LIVE at some GC-triggering call site. Ref inputargs that the
    // optimizer replaced with constants (e.g. guard_value'd code/namespace)
    // are never referenced by ops — they don't need GC root slots.
    // Build the set of inputarg OpRef raw values actually used in ops.
    let mut used_inputargs: HashSet<u32> = HashSet::new();
    for op in ops.iter() {
        for &arg in op
            .args
            .iter()
            .chain(op.fail_args.iter().flat_map(|fa| fa.iter()))
        {
            if inputarg_oprefs.contains(&arg.raw()) {
                used_inputargs.insert(arg.raw());
            }
        }
    }
    for input in inputargs {
        if input.tp == Type::Ref
            && !force_tokens.contains(&input.index)
            && !non_ref_at_backedge.contains(&input.index)
            && used_inputargs.contains(&input.index)
            && seen.insert(input.index)
        {
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!("[ref-root] inputarg idx={} tp={:?}", input.index, input.tp);
            }
            slots.push((input.index, slots.len()));
        } else if input.tp == Type::Ref
            && !used_inputargs.contains(&input.index)
            && std::env::var_os("MAJIT_LOG").is_some()
        {
            eprintln!(
                "[ref-root] SKIP inputarg idx={}: not referenced by ops (constant-folded)",
                input.index
            );
        }
    }

    for (op_idx, op) in ops.iter().enumerate() {
        if op.result_type() == Type::Ref {
            let vi = op_var_index(op, op_idx, inputargs.len()) as u32;
            if !force_tokens.contains(&vi) && seen.insert(vi) {
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[ref-root] op idx={} opcode={:?} vi={}",
                        op_idx, op.opcode, vi
                    );
                }
                slots.push((vi, slots.len()));
            }
        }
    }

    // RPython parity note: this situation (non-Ref at Ref inputarg position)
    // does not arise in RPython because Box identity preserves types through
    // optimization. In RPython, force_box_for_end_of_preamble materializes
    // virtuals; the type of the Box never changes. majit's flat OpRef model
    // allows type substitution (SameAsF/SameAsI replacing Ref) which RPython
    // cannot express.
    //
    // Float at Ref is unsafe: IEEE754 bits are never valid pointers. Int at Ref
    // can be safe when the optimizer forwards a GC pointer through SameAsI
    // without changing the actual runtime value.
    if has_float_at_ref_position {
        return Err(BackendError::Unsupported(
            "jump_to_preamble: backedge passes Float at Ref-typed inputarg position".to_string(),
        ));
    }

    Ok(slots)
}

/// Simple normalization: assign sequential pos to ops without pos.
fn normalize_ops_for_codegen_simple(inputargs: &[InputArg], ops: &[Op]) -> Vec<Op> {
    let num_inputs = inputargs.len() as u32;
    ops.iter()
        .enumerate()
        .map(|(op_idx, op)| {
            let mut normalized = op.clone();
            let rt = normalized.result_type();
            if rt != Type::Void && normalized.pos.is_none() {
                // op_typed mints the typed Int/Float/Ref variant
                // (resoperation.py:564-638 AbstractResOp + IntOp/FloatOp/
                // RefOp mixins) — the Void branch is filtered above.
                normalized.pos = OpRef::op_typed(num_inputs + op_idx as u32, rt);
            }
            normalized
        })
        .collect()
}

fn inject_builtin_string_descrs(ops: &mut [Op]) {
    for op in ops {
        if op.descr.is_some() {
            continue;
        }
        if let Some(descr) = builtin_string_array_descr(op.opcode) {
            op.descr = Some(descr);
        } else if let Some(descr) = builtin_string_hash_field_descr(op.opcode) {
            op.descr = Some(descr);
        }
    }
}

fn resolve_opref_or_imm(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    known_values: &HashSet<u32>,
    opref: OpRef,
) -> CValue {
    if opref.is_none() {
        return builder.ins().iconst(cl_types::I64, 0);
    }
    if known_values.contains(&opref.raw()) || opref_is_op_result_var(opref) {
        return builder.use_var(var(opref.raw()));
    }
    if let Some(&c) = constants.get(&opref.raw()) {
        return builder.ins().iconst(cl_types::I64, c);
    }
    // Constant-namespace OpRef: the const_index IS the value
    // (RPython ConstInt(n).getint() == n).
    if opref.is_constant() {
        return builder
            .ins()
            .iconst(cl_types::I64, opref.const_index() as i64);
    }
    builder.ins().iconst(cl_types::I64, opref.raw() as i64)
}

fn resolve_failarg_opref(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    stale_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
    opref: OpRef,
) -> CValue {
    if let Some((_, root_slot)) = ref_root_slots
        .iter()
        .find(|(var_idx, _)| *var_idx == opref.raw() && stale_ref_vars.contains(var_idx))
    {
        let root_offset = ref_root_base_ofs + (*root_slot as i32) * 8;
        return builder
            .ins()
            .load(cl_types::I64, MemFlags::trusted(), jf_ptr, root_offset);
    }
    resolve_opref(builder, constants, opref)
}

fn resolve_constant_i64(
    constants: &HashMap<u32, i64>,
    known_values: &HashSet<u32>,
    opcode: OpCode,
    opref: OpRef,
    what: &str,
) -> Result<i64, BackendError> {
    if known_values.contains(&opref.raw()) || opref_is_op_result_var(opref) {
        return Err(unsupported_semantics(
            opcode,
            &format!("{what} must be a compile-time constant"),
        ));
    }
    if let Some(&c) = constants.get(&opref.raw()) {
        return Ok(c);
    }
    if opref.is_constant() {
        return Ok(opref.const_index() as i64);
    }
    Ok(opref.raw() as i64)
}

fn resolve_rewriter_immediate_i64(constants: &HashMap<u32, i64>, opref: OpRef) -> i64 {
    constants.get(&opref.raw()).copied().unwrap_or_else(|| {
        if opref.is_constant() {
            opref.const_index() as i64
        } else {
            opref.raw() as i64
        }
    })
}

fn type_for_opref(
    type_index: &OpTypeIndex<'_>,
    overrides: &HashMap<u32, Type>,
    known_values: &HashSet<u32>,
    opcode: OpCode,
    opref: OpRef,
    op_index: usize,
    what: &str,
) -> Result<Type, BackendError> {
    if let Some(tp) = lookup_type_at(type_index, overrides, opref, op_index) {
        return Ok(tp);
    }
    if known_values.contains(&opref.raw()) {
        return Err(unsupported_semantics(
            opcode,
            &format!("type information for {what} is missing"),
        ));
    }
    Ok(Type::Int)
}

fn emit_load_from_addr(
    builder: &mut FunctionBuilder,
    addr: CValue,
    value_type: Type,
    size: usize,
    signed: bool,
    opcode: OpCode,
) -> Result<CValue, BackendError> {
    // Use non-trusted MemFlags so Cranelift does NOT speculate loads
    // past guards. With trusted(), Cranelift can hoist a load before
    // a GuardNonnull branch, causing SEGFAULT on null pointers.
    // RPython's x86 backend emits in IR order (no scheduling), so
    // loads after guards are safe. Cranelift needs this annotation.
    let heap_flags = MemFlags::new();
    match value_type {
        Type::Float => {
            if size != 8 {
                return Err(unsupported_semantics(
                    opcode,
                    "float memory operations currently require 8-byte values",
                ));
            }
            let fval = builder.ins().load(cl_types::F64, heap_flags, addr, 0);
            Ok(builder.ins().bitcast(cl_types::I64, MemFlags::new(), fval))
        }
        Type::Int | Type::Ref => {
            if value_type == Type::Ref && size != 8 {
                return Err(unsupported_semantics(
                    opcode,
                    "reference memory operations currently require pointer-sized values",
                ));
            }
            let mem_ty = checked_cl_type_for_size(
                size,
                opcode,
                "memory operations only support 1-, 2-, 4-, and 8-byte values",
            )?;
            let raw = builder.ins().load(mem_ty, heap_flags, addr, 0);
            if mem_ty == cl_types::I64 {
                Ok(raw)
            } else if value_type == Type::Int && signed {
                Ok(builder.ins().sextend(cl_types::I64, raw))
            } else {
                Ok(builder.ins().uextend(cl_types::I64, raw))
            }
        }
        Type::Void => Err(unsupported_semantics(
            opcode,
            "void-typed memory loads are invalid",
        )),
    }
}

fn emit_store_to_addr(
    builder: &mut FunctionBuilder,
    addr: CValue,
    value: CValue,
    value_type: Type,
    size: usize,
    opcode: OpCode,
) -> Result<(), BackendError> {
    match value_type {
        Type::Float => {
            if size != 8 {
                return Err(unsupported_semantics(
                    opcode,
                    "float memory operations currently require 8-byte values",
                ));
            }
            let fval = builder.ins().bitcast(cl_types::F64, MemFlags::new(), value);
            builder.ins().store(MemFlags::trusted(), fval, addr, 0);
        }
        Type::Int | Type::Ref => {
            if value_type == Type::Ref && size != 8 {
                return Err(unsupported_semantics(
                    opcode,
                    "reference memory operations currently require pointer-sized values",
                ));
            }
            let mem_ty = checked_cl_type_for_size(
                size,
                opcode,
                "memory operations only support 1-, 2-, 4-, and 8-byte values",
            )?;
            let store_val = if mem_ty == cl_types::I64 {
                value
            } else {
                builder.ins().ireduce(mem_ty, value)
            };
            builder.ins().store(MemFlags::trusted(), store_val, addr, 0);
        }
        Type::Void => {
            return Err(unsupported_semantics(
                opcode,
                "void-typed memory stores are invalid",
            ));
        }
    }
    Ok(())
}

fn emit_dynamic_offset_addr(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    known_values: &HashSet<u32>,
    base_arg: OpRef,
    offset_arg: OpRef,
) -> CValue {
    let base = resolve_opref(builder, constants, base_arg);
    let offset = resolve_opref_or_imm(builder, constants, known_values, offset_arg);
    builder.ins().iadd(base, offset)
}

fn emit_scaled_index_addr(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    base_arg: OpRef,
    index_arg: OpRef,
    scale: i64,
    base_offset: i64,
) -> CValue {
    let base = resolve_opref(builder, constants, base_arg);
    let index = resolve_opref(builder, constants, index_arg);
    let scaled_index = match scale {
        0 => builder.ins().iconst(cl_types::I64, 0),
        1 => index,
        _ => {
            let scale_val = builder.ins().iconst(cl_types::I64, scale);
            builder.ins().imul(index, scale_val)
        }
    };
    let with_base_offset = if base_offset == 0 {
        scaled_index
    } else {
        builder.ins().iadd_imm(scaled_index, base_offset)
    };
    builder.ins().iadd(base, with_base_offset)
}

fn emit_host_call(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
    func_ptr: usize,
    args: &[CValue],
    return_type: Option<cranelift_codegen::ir::Type>,
) -> Option<CValue> {
    let mut sig = Signature::new(call_conv);
    for _ in args {
        sig.params.push(AbiParam::new(cl_types::I64));
    }
    if let Some(ret) = return_type {
        sig.returns.push(AbiParam::new(ret));
    }
    let sig_ref = builder.import_signature(sig);

    let raw_fptr = builder.ins().iconst(cl_types::I64, func_ptr as i64);
    let fptr = if ptr_type != cl_types::I64 {
        builder.ins().ireduce(ptr_type, raw_fptr)
    } else {
        raw_fptr
    };
    let call = builder.ins().call_indirect(sig_ref, fptr, args);
    return_type.map(|_| builder.inst_results(call)[0])
}

/// assembler.py:1348-1367 _push_all_regs_to_frame:
/// Save all defined GC ref variables to jf_frame slots before a call
/// that may trigger GC. The per-call gcmap (from get_gcmap) tells the
/// GC which of these slots are alive.
fn spill_ref_roots(
    builder: &mut FunctionBuilder,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    stale_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
) {
    for &(var_idx, slot) in ref_root_slots {
        if !defined_ref_vars.contains(&var_idx) || stale_ref_vars.contains(&var_idx) {
            continue;
        }
        let offset = ref_root_base_ofs + (slot as i32) * 8;
        let val = builder.use_var(var(var_idx));
        builder.ins().store(MemFlags::new(), val, jf_ptr, offset);
    }
}

/// RPython regalloc keeps many REF boxes frame-resident after
/// `_sync_var_to_stack` / `_bc_spill`.  When Cranelift has already
/// written a ref box to its jitframe root slot, re-storing it before a
/// CALL_ASSEMBLER is redundant: the GC updates that slot in-place.
fn spill_unsynced_ref_roots(
    builder: &mut FunctionBuilder,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    synced_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
) {
    for &(var_idx, slot) in ref_root_slots {
        if !defined_ref_vars.contains(&var_idx) || synced_ref_vars.contains(&var_idx) {
            continue;
        }
        let offset = ref_root_base_ofs + (slot as i32) * 8;
        let val = builder.use_var(var(var_idx));
        builder.ins().store(MemFlags::new(), val, jf_ptr, offset);
    }
}

/// assembler.py:1369-1377 _pop_all_regs_from_frame:
/// Reload all defined GC ref variables from jf_frame after a call.
/// The GC may have moved objects and updated the slots in-place.
fn reload_ref_roots(
    builder: &mut FunctionBuilder,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
) {
    for &(var_idx, slot) in ref_root_slots {
        if !defined_ref_vars.contains(&var_idx) {
            continue;
        }
        let offset = ref_root_base_ofs + (slot as i32) * 8;
        let val = builder
            .ins()
            .load(cl_types::I64, MemFlags::trusted(), jf_ptr, offset);
        builder.def_var(var(var_idx), val);
    }
}

fn sync_ref_root_var(
    builder: &mut FunctionBuilder,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    var_idx: u32,
    value: CValue,
    ref_root_base_ofs: i32,
    synced_ref_vars: &mut HashSet<u32>,
) {
    if let Some((_, slot)) = ref_root_slots.iter().find(|(idx, _)| *idx == var_idx) {
        let offset = ref_root_base_ofs + (*slot as i32) * 8;
        builder.ins().store(MemFlags::new(), value, jf_ptr, offset);
        synced_ref_vars.insert(var_idx);
    }
}

fn mark_ref_roots_synced(synced_ref_vars: &mut HashSet<u32>, ref_root_slots: &[(u32, usize)]) {
    for &(var_idx, _) in ref_root_slots {
        synced_ref_vars.insert(var_idx);
    }
}

fn mark_ref_roots_fresh(stale_ref_vars: &mut HashSet<u32>, ref_root_slots: &[(u32, usize)]) {
    for &(var_idx, _) in ref_root_slots {
        stale_ref_vars.remove(&var_idx);
    }
}

fn mark_ref_roots_after_selective_reload(
    stale_ref_vars: &mut HashSet<u32>,
    live_ref_root_slots: &[(u32, usize)],
    reloaded_ref_root_slots: &[(u32, usize)],
) {
    for &(var_idx, _) in live_ref_root_slots {
        if reloaded_ref_root_slots
            .iter()
            .any(|(idx, _)| *idx == var_idx)
        {
            stale_ref_vars.remove(&var_idx);
        } else {
            stale_ref_vars.insert(var_idx);
        }
    }
}

fn ptr_arg_as_i64(
    builder: &mut FunctionBuilder,
    ptr: CValue,
    ptr_type: cranelift_codegen::ir::Type,
) -> CValue {
    if ptr_type == cl_types::I64 {
        ptr
    } else {
        builder.ins().uextend(cl_types::I64, ptr)
    }
}

/// Compute per-call gcmap for ref root slots.
///
/// RPython regalloc.py get_gcmap (1092-1108): builds a bitmap of frame
/// slots that hold GC references at a given call site. The bitmap bit
/// position is the frame slot index (relative to jf_frame[0]).
///
/// regalloc.py:1089-1106 get_gcmap
///
/// Build a per-call-site gcmap bitmap marking only the ref root slots
/// that are still alive at `position`. A ref is alive when its
/// `last_usage >= position` (regalloc.py:380 is_still_alive).
///
/// Returns a leaked `[length, data]` gcmap array pointer (same layout
/// as RPython's allocate_gcmap), or 0 (NULLGCMAP) if no live refs.
fn get_gcmap(
    position: usize,
    max_output_slots: usize,
    ref_root_slots: &[(u32, usize)],
    longevity: &HashMap<u32, usize>,
    defined_ref_vars: &HashSet<u32>,
) -> i64 {
    // regalloc.py:1093-1105: iterate bindings, include only alive refs.
    let mut live_bit_positions: Vec<usize> = Vec::new();
    for &(var_idx, slot) in ref_root_slots {
        if !defined_ref_vars.contains(&var_idx) {
            continue;
        }
        // regalloc.py:1096/1102: box.type == REF and self.rm.is_still_alive(box)
        // is_still_alive: longevity[v].last_usage >= position
        if let Some(&last_usage) = longevity.get(&var_idx) {
            if last_usage >= position {
                live_bit_positions.push(max_output_slots + slot);
            }
        }
    }
    if live_bit_positions.is_empty() {
        return 0; // NULLGCMAP
    }
    let max_bit = live_bit_positions.iter().copied().max().unwrap() + 1;
    if max_bit <= 64 {
        let mut bitmap: usize = 0;
        for bit_pos in &live_bit_positions {
            bitmap |= 1usize << bit_pos;
        }
        let gcmap_arr = Box::leak(Box::new([1isize, bitmap as isize]));
        return gcmap_arr.as_ptr() as i64;
    }
    let num_words = (max_bit + 63) / 64;
    let mut gcmap: Vec<isize> = vec![0; 1 + num_words];
    gcmap[0] = num_words as isize;
    for bit_pos in &live_bit_positions {
        let word_idx = bit_pos / 64;
        let bit_idx = bit_pos % 64;
        gcmap[1 + word_idx] |= (1usize << bit_idx) as isize;
    }
    let leaked = gcmap.into_boxed_slice();
    let ptr = Box::leak(leaked).as_ptr();
    ptr as i64
}

fn live_ref_root_slots_at(
    position: usize,
    ref_root_slots: &[(u32, usize)],
    longevity: &HashMap<u32, usize>,
    defined_ref_vars: &HashSet<u32>,
) -> Vec<(u32, usize)> {
    ref_root_slots
        .iter()
        .filter(|(var_idx, _)| {
            defined_ref_vars.contains(var_idx)
                && longevity
                    .get(var_idx)
                    .is_some_and(|last_usage| *last_usage >= position)
        })
        .copied()
        .collect()
}

fn ref_root_slots_with_future_regular_uses(
    position: usize,
    ops: &[Op],
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
) -> Vec<(u32, usize)> {
    // RPython regalloc only reloads values it keeps in registers after a
    // collecting call.  References that are live only as future failargs can
    // remain frame-resident: the gcmap protects their slots, and failure
    // recovery reads them from the frame if needed.
    ref_root_slots
        .iter()
        .filter(|(var_idx, _)| {
            defined_ref_vars.contains(var_idx)
                && ops
                    .iter()
                    .skip(position + 1)
                    .flat_map(|op| op.args.iter())
                    .any(|arg| arg.raw() == *var_idx)
        })
        .copied()
        .collect()
}

/// assembler.py:1369-1377 _reload_frame_if_necessary:
///   MOV ecx, [rootstacktop]      // load shadow stack top pointer
///   MOV ebp, [ecx - WORD]        // load jf_ptr from topmost entry
///
/// After a collecting call, the GC may have copied the jitframe from
/// nursery to old gen. The shadow stack entry was updated by the GC.
/// Reload jf_ptr from the shadow stack top.
fn emit_reload_frame_if_necessary(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    _call_conv: cranelift_codegen::isa::CallConv,
) -> CValue {
    let word = std::mem::size_of::<usize>() as i32;
    // MOV ecx, [rootstacktop]
    let rst_addr = builder.ins().iconst(
        ptr_type,
        majit_gc::shadow_stack::get_root_stack_top_addr() as i64,
    );
    let rst = builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), rst_addr, 0);
    // MOV ebp, [ecx - WORD]  — jf_ptr is at top - WORD
    builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), rst, -word)
}

/// llsupport/llmodel.py:229-234 `insert_stack_check` plus
/// assembler.py:1080-1091 `_call_header_with_stack_check` parity.
///
/// Cranelift does not expose the native SP register directly, but
/// `stack_addr` gives the address of a stack slot in this frame.  Pyre's
/// interpreter-side `current_sp()` uses the address of a stack local too,
/// so this preserves the monotonic depth probe while keeping the common
/// path as load/sub/cmp instead of an extern call on every JIT entry.
fn emit_stack_check_result(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
) -> CValue {
    if let Some(addrs) = stack_check_addresses() {
        let probe_slot =
            builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
        let current_ptr = builder.ins().stack_addr(ptr_type, probe_slot, 0);
        let current = ptr_arg_as_i64(builder, current_ptr, ptr_type);

        let end_addr = builder.ins().iconst(ptr_type, addrs.end_adr as i64);
        let end = builder
            .ins()
            .load(cl_types::I64, MemFlags::trusted(), end_addr, 0);
        let length_addr = builder.ins().iconst(ptr_type, addrs.length_adr as i64);
        let length = builder
            .ins()
            .load(cl_types::I64, MemFlags::trusted(), length_addr, 0);
        let ofs = builder.ins().isub(end, current);
        let overflow_candidate = builder.ins().icmp(IntCC::UnsignedGreaterThan, ofs, length);

        let slow_block = builder.create_block();
        let done_block = builder.create_block();
        builder.append_block_param(done_block, cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.ins().brif(
            overflow_candidate,
            slow_block,
            &[],
            done_block,
            &[BlockArg::from(zero)],
        );

        builder.switch_to_block(slow_block);
        builder.seal_block(slow_block);
        let slow_result = emit_host_call(
            builder,
            ptr_type,
            call_conv,
            addrs.slowpath_addr,
            &[current],
            Some(cl_types::I8),
        )
        .expect("stack_check_slowpath_for_backend must return a u8");
        let slow_result = builder.ins().uextend(cl_types::I64, slow_result);
        builder
            .ins()
            .jump(done_block, &[BlockArg::from(slow_result)]);

        builder.switch_to_block(done_block);
        builder.seal_block(done_block);
        return builder.block_params(done_block)[0];
    }

    let probe_addr = prologue_probe_addr().unwrap_or(0);
    if probe_addr != 0 {
        return emit_host_call(
            builder,
            ptr_type,
            call_conv,
            probe_addr,
            &[],
            Some(cl_types::I64),
        )
        .expect("pyre_stack_check_for_jit_prologue must return an i64");
    }

    builder.ins().iconst(cl_types::I64, 0)
}

/// RPython push_gcmap (assembler.py:2017-2022, callbuilder.py:93-107):
///   MOV [ebp + jf_gcmap_ofs], gcmap_ptr
///
/// Writes the per-call gcmap to jf_gcmap before a GC-triggering call.
/// This tells the GC which jf_frame slots contain live refs.
fn emit_push_gcmap(builder: &mut FunctionBuilder, jf_ptr: CValue, per_call_gcmap: i64) {
    let gcmap_val = builder.ins().iconst(cl_types::I64, per_call_gcmap);
    builder
        .ins()
        .store(MemFlags::new(), gcmap_val, jf_ptr, JF_GCMAP_OFS);
}

/// RPython pop_gcmap (assembler.py:2024-2027):
///   MOV [ebp + jf_gcmap_ofs], 0
///
/// Clears jf_gcmap after the call returns.
fn emit_pop_gcmap(builder: &mut FunctionBuilder, jf_ptr: CValue, _per_call_gcmap: i64) {
    let zero = builder.ins().iconst(cl_types::I64, 0);
    builder
        .ins()
        .store(MemFlags::new(), zero, jf_ptr, JF_GCMAP_OFS);
}

fn emit_jitframe_write_barrier(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
    jf_ptr: CValue,
) {
    let jf_arg = ptr_arg_as_i64(builder, jf_ptr, ptr_type);
    let _ = emit_host_call(
        builder,
        ptr_type,
        call_conv,
        gc_write_barrier_shim as *const () as usize,
        &[jf_arg],
        None,
    );
}

extern "C" fn zero_memory_shim(base: u64, offset: u64, size: u64) {
    if size == 0 {
        return;
    }
    let addr = (base as usize).wrapping_add(offset as usize) as *mut u8;
    unsafe {
        std::ptr::write_bytes(addr, 0, size as usize);
    }
}

extern "C" fn copy_nonoverlapping_memory_shim(src: u64, dst: u64, size: u64) {
    if size == 0 {
        return;
    }

    let src = src as usize;
    let dst = dst as usize;
    let size = size as usize;
    let overlap = src < dst.saturating_add(size) && dst < src.saturating_add(size);
    assert!(
        !overlap,
        "copystrcontent/copyunicodecontent overlapping ranges are unsupported"
    );

    unsafe {
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
    }
}

/// RPython parity: emit a GC-collecting call with ref root spill/reload
/// and push_gcmap/pop_gcmap (callbuilder.py:93-122).
///
/// Sequence:
///   1. spill_ref_roots — save live refs to jf_frame
///   2. push_gcmap — MOV [ebp+jf_gcmap], gcmap_ptr
///   3. CALL (may trigger GC → GC reads jf_gcmap to find refs)
///   4. pop_gcmap — MOV [ebp+jf_gcmap], 0
///   5. reload_ref_roots — restore refs (GC may have updated them)
fn emit_collecting_gc_call(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    stale_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
    per_call_gcmap: i64,
    func_ptr: usize,
    extra_args: &[CValue],
    return_type: Option<cranelift_codegen::ir::Type>,
) -> Option<CValue> {
    spill_ref_roots(
        builder,
        jf_ptr,
        ref_root_slots,
        defined_ref_vars,
        stale_ref_vars,
        ref_root_base_ofs,
    );
    emit_push_gcmap(builder, jf_ptr, per_call_gcmap);

    let result = emit_host_call(
        builder,
        ptr_type,
        call_conv,
        func_ptr,
        extra_args,
        return_type,
    );

    // _reload_frame_if_necessary (assembler.py:405-412)
    let new_jf_ptr = emit_reload_frame_if_necessary(builder, ptr_type, call_conv);
    emit_pop_gcmap(builder, new_jf_ptr, per_call_gcmap);
    reload_ref_roots(
        builder,
        new_jf_ptr,
        ref_root_slots,
        defined_ref_vars,
        ref_root_base_ofs,
    );
    result
}

fn emit_indirect_call_from_parts(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    func_ref: OpRef,
    arg_refs: &[OpRef],
    call_descr: &dyn CallDescr,
    call_conv: cranelift_codegen::isa::CallConv,
    ptr_type: cranelift_codegen::ir::Type,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    stale_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
    per_call_gcmap: i64,
) -> Option<CValue> {
    let mut sig = Signature::new(call_conv);
    let arg_types = call_descr.arg_types();
    for at in arg_types {
        sig.params.push(AbiParam::new(cranelift_type_for(at)));
    }
    let result_type = call_descr.result_type();
    if result_type != Type::Void {
        sig.returns
            .push(AbiParam::new(cranelift_type_for(&result_type)));
    }
    let sig_ref = builder.import_signature(sig);

    let func_ptr_raw = resolve_opref(builder, constants, func_ref);
    let func_ptr = if ptr_type != cl_types::I64 {
        builder.ins().ireduce(ptr_type, func_ptr_raw)
    } else {
        func_ptr_raw
    };

    let mut args: Vec<CValue> = Vec::with_capacity(arg_refs.len());
    for (i, &arg_ref) in arg_refs.iter().enumerate() {
        let raw = resolve_opref(builder, constants, arg_ref);
        if i < arg_types.len() && arg_types[i] == Type::Float {
            args.push(builder.ins().bitcast(cl_types::F64, MemFlags::new(), raw));
        } else {
            args.push(raw);
        }
    }

    // RPython callbuilder.py parity:
    //   emit() [can_collect]: spill + push_gcmap + CALL + pop_gcmap + reload
    //   emit_no_collect(): bare CALL only
    let can_collect = call_descr.get_extra_info().check_can_collect();
    if can_collect {
        spill_ref_roots(
            builder,
            jf_ptr,
            ref_root_slots,
            defined_ref_vars,
            stale_ref_vars,
            ref_root_base_ofs,
        );
        emit_push_gcmap(builder, jf_ptr, per_call_gcmap);
    }
    let call = builder.ins().call_indirect(sig_ref, func_ptr, &args);
    if can_collect {
        // _reload_frame_if_necessary (assembler.py:405-412)
        let new_jf_ptr = emit_reload_frame_if_necessary(builder, ptr_type, call_conv);
        emit_pop_gcmap(builder, new_jf_ptr, per_call_gcmap);
        reload_ref_roots(
            builder,
            new_jf_ptr,
            ref_root_slots,
            defined_ref_vars,
            ref_root_base_ofs,
        );
    }

    if result_type != Type::Void {
        let result = builder.inst_results(call)[0];
        let stored = if result_type == Type::Float {
            builder
                .ins()
                .bitcast(cl_types::I64, MemFlags::new(), result)
        } else {
            result
        };
        Some(stored)
    } else {
        None
    }
}

/// Emit a guard/finish side-exit.
///
/// _build_failure_recovery (assembler.py:2080-2109) parity:
///   1. _push_all_regs_to_frame  — save live values to jf_frame slots
///   2. POP [ebp + jf_descr]     — store fail_descr index to jf_descr
///   3. _call_footer              — mov eax, ebp; ret (return jitframe)
///
/// x86/assembler.py:1880-1891 _cmp_guard_class:
///   loc_ptr = locs[0]
///   loc_classptr = locs[1]
///   offset = self.cpu.vtable_offset
///   if offset is not None:
///       self.mc.CMP(mem(loc_ptr, offset), loc_classptr)
///   else:
///       assert isinstance(loc_classptr, ImmedLoc)
///       classptr = loc_classptr.value
///       expected_typeid = (self.cpu.gc_ll_descr
///               .get_typeid_from_classptr_if_gcremovetypeptr(classptr))
///       self._cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
///
/// Returns a 1-bit boolean: 1 = guard fails (not-equal), 0 = guard passes.
///
/// `expected_classptr_imm` is the immediate value of the classptr argument
/// (RPython requires this to be an `ImmedLoc` in the gcremovetypeptr branch).
/// `gc_remove_typeptr_lookup` is the gc_ll_descr-equivalent typeid resolver;
/// when `vtable_offset is None` it must be set or the guard cannot be
/// emitted.
fn emit_cmp_guard_class(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    obj: CValue,
    expected_class: CValue,
    expected_classptr_imm: Option<i64>,
    vtable_offset: Option<usize>,
    gc_remove_typeptr_lookup: Option<&dyn Fn(usize) -> Option<u32>>,
) -> CValue {
    if let Some(off) = vtable_offset {
        // x86/assembler.py:1884-1885 vtable_offset path: full classptr CMP.
        let actual_class = builder
            .ins()
            .load(ptr_type, MemFlags::trusted(), obj, off as i32);
        builder
            .ins()
            .icmp(IntCC::NotEqual, actual_class, expected_class)
    } else {
        // x86/assembler.py:1886-1891 gcremovetypeptr fallback:
        //   assert isinstance(loc_classptr, ImmedLoc)
        //   classptr = loc_classptr.value
        //   expected_typeid = gc_ll_descr.
        //       get_typeid_from_classptr_if_gcremovetypeptr(classptr)
        //   _cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
        let classptr = expected_classptr_imm.unwrap_or_else(|| {
            panic!(
                "_cmp_guard_class: gcremovetypeptr requires loc_classptr to \
                 be an immediate (assert isinstance(loc_classptr, ImmedLoc) \
                 in x86/assembler.py:1887)"
            )
        });
        let lookup = gc_remove_typeptr_lookup.unwrap_or_else(|| {
            panic!(
                "_cmp_guard_class: vtable_offset is None but the backend \
                 has no gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr"
            )
        });
        let expected_typeid = lookup(classptr as usize).unwrap_or_else(|| {
            panic!(
                "get_typeid_from_classptr_if_gcremovetypeptr({:#x}) returned \
                 None — classptr is not a registered vtable",
                classptr
            )
        });
        // _cmp_guard_gc_type (x86/assembler.py:1893-1901): on x86_64 the
        // typeid is a 32-bit half-word at offset 0 of the object.
        let actual_typeid = builder
            .ins()
            .load(cl_types::I32, MemFlags::trusted(), obj, 0);
        let expected_typeid_val = builder.ins().iconst(cl_types::I32, expected_typeid as i64);
        builder
            .ins()
            .icmp(IntCC::NotEqual, actual_typeid, expected_typeid_val)
    }
}

fn emit_attached_bridge_dispatch(
    builder: &mut FunctionBuilder,
    jf_ptr: CValue,
    fail_descr_ptr: i64,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
) {
    // dynasm/x86 patches the failing guard's jump to the attached bridge.
    // Cranelift cannot patch finalized code, so guard exits do the
    // equivalent descriptor-cache dispatch before returning a deadframe to
    // the caller/interpreter.
    let fail_descr = unsafe { &*(fail_descr_ptr as *const CraneliftFailDescr) };
    let bridge_code_cache_addr =
        &fail_descr.bridge_code_ptr_cache as *const std::sync::atomic::AtomicUsize as usize;
    let bridge_frame_depth_cache_addr =
        &fail_descr.bridge_frame_depth_cache as *const std::sync::atomic::AtomicUsize as usize;
    let bridge_code_cache_ptr = builder
        .ins()
        .iconst(ptr_type, bridge_code_cache_addr as i64);
    let bridge_frame_depth_cache_ptr = builder
        .ins()
        .iconst(ptr_type, bridge_frame_depth_cache_addr as i64);
    let bridge_code = builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), bridge_code_cache_ptr, 0);
    let required_frame_len_raw = builder.ins().load(
        ptr_type,
        MemFlags::trusted(),
        bridge_frame_depth_cache_ptr,
        0,
    );
    let required_frame_len = if ptr_type == cl_types::I64 {
        required_frame_len_raw
    } else {
        builder.ins().uextend(cl_types::I64, required_frame_len_raw)
    };
    let frame_len = builder.ins().load(
        cl_types::I64,
        MemFlags::trusted(),
        jf_ptr,
        JF_FRAME_LENGTH_OFS,
    );
    let null_ptr = builder.ins().iconst(ptr_type, 0);
    let has_bridge = builder.ins().icmp(IntCC::NotEqual, bridge_code, null_ptr);
    let frame_fits = builder.ins().icmp(
        IntCC::UnsignedGreaterThanOrEqual,
        frame_len,
        required_frame_len,
    );
    let can_dispatch = builder.ins().band(has_bridge, frame_fits);
    let bridge_block = builder.create_block();
    builder.append_block_param(bridge_block, ptr_type);
    let miss_block = builder.create_block();
    builder.ins().brif(
        can_dispatch,
        bridge_block,
        &[BlockArg::from(bridge_code)],
        miss_block,
        &[],
    );

    builder.switch_to_block(bridge_block);
    builder.seal_block(bridge_block);
    let bridge_code_ptr = builder.block_params(bridge_block)[0];
    // Dynasm patches the guard branch into a jump to the bridge, so the
    // parent trace no longer contributes a separate shadowstack entry while
    // the bridge runs.  Pop before the call; the bridge prologue pushes the
    // same jitframe for its own execution and pops it on return.
    emit_call_footer_shadowstack(builder, ptr_type);
    let mut bridge_sig = Signature::new(call_conv);
    bridge_sig.params.push(AbiParam::new(ptr_type));
    bridge_sig.returns.push(AbiParam::new(ptr_type));
    let bridge_sig_ref = builder.import_signature(bridge_sig);
    let bridge_call = builder
        .ins()
        .call_indirect(bridge_sig_ref, bridge_code_ptr, &[jf_ptr]);
    let bridge_result_jf = builder.inst_results(bridge_call)[0];
    builder.ins().return_(&[bridge_result_jf]);

    builder.switch_to_block(miss_block);
    builder.seal_block(miss_block);
}

/// genop_finish (assembler.py:2114-2155) parity:
///   1. save result to jf_frame[0]
///   2. MOV [ebp + jf_descr], faildescrindex
///   3. _call_footer
fn emit_guard_exit(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    jf_ptr: CValue,
    info: &GuardInfo,
    ref_root_slots: &[(u32, usize)],
    stale_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
) {
    // _push_all_regs_to_frame / save_into_mem parity:
    // store fail_args to jf_frame[slot]
    //
    // vector_ext.py:119-156 _update_at_exit parity:
    // If accumulation is done in this loop, at the guard exit some vector
    // values must be reduced to scalars before storing to jf_frame.
    let accum_positions: HashMap<usize, &AccumInfo> = info
        .accum_info
        .iter()
        .map(|ai| (ai.failargs_pos, ai))
        .collect();

    for (slot, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
        let offset = JF_FRAME_ITEM0_OFS + (slot as i32) * 8;

        if let Some(accum) = accum_positions.get(&slot) {
            // _update_at_exit: reduce vector accumulator to scalar.
            // resume.py:28 + vector_ext.py:130: accum_info.location = vector SSA
            // resume.py:47 + vector_ext.py:132: accum_info.getoriginal() → scalar
            // type info only
            let vec_val = resolve_opref(builder, constants, accum.location);
            let val_type = builder.func.dfg.value_type(vec_val);

            let reduced = if val_type == cl_types::F64X2 {
                // _accum_reduce_sum (vector_ext.py:164-173): HADDPD
                // _accum_reduce_mul (vector_ext.py:158-162): SHUFPD + MULSD
                let lane0 = builder.ins().extractlane(vec_val, 0);
                let lane1 = builder.ins().extractlane(vec_val, 1);
                let scalar_f = match accum.accum_operation {
                    '+' => builder.ins().fadd(lane0, lane1),
                    '*' => builder.ins().fmul(lane0, lane1),
                    op => panic!("unsupported accum_operation '{op}'"),
                };
                builder
                    .ins()
                    .bitcast(cl_types::I64, MemFlags::new(), scalar_f)
            } else {
                // _accum_reduce_sum INT (vector_ext.py:174-179):
                // PEXTRQ lane0, PEXTRQ lane1, ADD
                let lane0 = builder.ins().extractlane(vec_val, 0);
                let lane1 = builder.ins().extractlane(vec_val, 1);
                match accum.accum_operation {
                    '+' => builder.ins().iadd(lane0, lane1),
                    '*' => builder.ins().imul(lane0, lane1),
                    op => panic!("unsupported accum_operation '{op}'"),
                }
            };
            builder
                .ins()
                .store(MemFlags::trusted(), reduced, jf_ptr, offset);
        } else {
            let val = resolve_failarg_opref(
                builder,
                constants,
                jf_ptr,
                ref_root_slots,
                stale_ref_vars,
                ref_root_base_ofs,
                arg_ref,
            );
            builder
                .ins()
                .store(MemFlags::trusted(), val, jf_ptr, offset);
        }
    }
    // assembler.py:2126 get_gcref_from_faildescr → MOV [ebp+jf_descr], gcref
    // Store FailDescr POINTER (not index) to jf_descr on the deadframe path.
    let descr_val = builder.ins().iconst(cl_types::I64, info.fail_descr_ptr);

    if info.can_have_bridge && !info.must_save_exception {
        // RPython/dynasm patched guards enter the bridge before failure
        // recovery. Cranelift still has to publish failargs into jf_frame
        // because bridge input locations are frame based, but a bridge hit
        // should skip the deadframe-only gcmap/write-barrier/descr stores.
        emit_attached_bridge_dispatch(builder, jf_ptr, info.fail_descr_ptr, ptr_type, call_conv);
    }

    // _build_failure_recovery (assembler.py:2102-2105) parity:
    //   POP [ebp + jf_gcmap]   — #2104
    //   POP [ebp + jf_descr]   — #2105
    // push_gcmap (assembler.py:2013): PUSH gcmap_ptr
    // Skip gcmap store when null (no ref fail_args) — saves 1 iconst + 1 store.
    if info.gcmap != 0 {
        // allocate_gcmap (gcmap.py:7-18): Array(Unsigned) [length, data...].
        let gcmap_arr = Box::leak(Box::new([1isize, info.gcmap as isize]));
        let gcmap_val = builder
            .ins()
            .iconst(cl_types::I64, gcmap_arr.as_ptr() as i64);
        builder
            .ins()
            .store(MemFlags::new(), gcmap_val, jf_ptr, JF_GCMAP_OFS); // #2104
    }
    // _build_failure_recovery (assembler.py:2089-2096) parity:
    // if exc: MOV ebx, [pos_exc_value]; MOV [pos_exception], 0;
    //         MOV [pos_exc_value], 0; MOV [jf_guard_exc], ebx
    if info.must_save_exception {
        // _store_and_reset_exception (assembler.py:1826-1842) parity:
        // Store pos_exc_value → jf_guard_exc, clear both globals to 0.
        // exc_class is derived from exc_value.typeptr (pyjitpl.py:3119-3123).
        let exc_addr = builder
            .ins()
            .iconst(cl_types::I64, jit_exc_value_addr() as i64);
        let exc_val = builder
            .ins()
            .load(cl_types::I64, MemFlags::trusted(), exc_addr, 0);
        builder
            .ins()
            .store(MemFlags::trusted(), exc_val, jf_ptr, JF_GUARD_EXC_OFS);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.ins().store(MemFlags::trusted(), zero, exc_addr, 0);
        let exc_type_addr = builder
            .ins()
            .iconst(cl_types::I64, jit_exc_type_addr() as i64);
        builder
            .ins()
            .store(MemFlags::trusted(), zero, exc_type_addr, 0);
    }
    if info.gcmap != 0 || info.must_save_exception {
        // aarch64/assembler.py:967-980 `_reload_frame_if_necessary`:
        // RPython emits a JITFrame write-barrier after any potentially-
        // allocating CALL when `gcrootmap and wbdescr` so promoted-frame
        // young-pointer slots remain trackable. Cranelift cannot patch
        // mid-trace, so the barrier is hoisted to every guard exit that
        // wrote Ref fail-args (gcmap != 0) or stashed an exception value;
        // both cases install young pointers in slots whose owner JITFrame
        // may have been promoted by an earlier nursery collection.
        emit_jitframe_write_barrier(builder, ptr_type, call_conv, jf_ptr);
    }
    if info.can_have_bridge && info.must_save_exception {
        // Exception guards need the exception payload saved before a bridge
        // sees the frame, so keep their bridge check after recovery stores.
        emit_attached_bridge_dispatch(builder, jf_ptr, info.fail_descr_ptr, ptr_type, call_conv);
    }
    builder
        .ins()
        .store(MemFlags::trusted(), descr_val, jf_ptr, JF_DESCR_OFS); // #2105
    // assembler.py:1101 _call_footer → _call_footer_shadowstack:
    // SUB [rootstacktop], 2*WORD — inline, no function call.
    emit_call_footer_shadowstack(builder, ptr_type);
    // _call_footer (assembler.py:1097): mov eax, ebp; ret
    builder.ins().return_(&[jf_ptr]);
}

// ---------------------------------------------------------------------------
// Compiled loop data
// ---------------------------------------------------------------------------

struct CompiledLoop {
    trace_id: u64,
    input_types: Vec<Type>,
    header_pc: u64,
    /// Green key hash from JitCellToken. Used by bridge threshold callback.
    green_key: u64,
    caller_prefix_layout: Option<ExitRecoveryLayout>,
    _func_id: FuncId,
    code_ptr: *const u8,
    code_size: usize,
    fail_descrs: Box<[Arc<CraneliftFailDescr>]>,
    terminal_exit_layouts: UnsafeCell<Vec<TerminalExitLayout>>,
    num_inputs: usize,
    num_ref_roots: usize,
    max_output_slots: usize,
    /// `compile.py:665 setattr(cpu, name, descr)` — heap-stable clone of
    /// the owning `CraneliftBackend`'s per-cpu descr attachments.  The
    /// JIT-emitted CALL_ASSEMBLER slow path bakes this Arc's address as
    /// an immediate; holding the clone here keeps the `RwLock`
    /// allocation alive for the lifetime of the compiled code.
    cpu_attachments: CpuDescrHandle,
}

unsafe impl Send for CompiledLoop {}
unsafe impl Sync for CompiledLoop {}

impl CompiledLoop {
    #[inline]
    fn terminal_exit_layouts_ref(&self) -> &Vec<TerminalExitLayout> {
        unsafe { &*self.terminal_exit_layouts.get() }
    }

    #[inline]
    fn terminal_exit_layouts_mut(&self) -> &mut Vec<TerminalExitLayout> {
        unsafe { &mut *self.terminal_exit_layouts.get() }
    }
}

fn find_trace_fail_descr_layouts_in_fail_descrs(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
) -> Option<Vec<majit_backend::FailDescrLayout>> {
    for descr in fail_descrs {
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if bridge.trace_id == trace_id {
                return Some(
                    bridge
                        .fail_descrs
                        .iter()
                        .map(|descr| descr.layout())
                        .collect(),
                );
            }
            if let Some(layouts) =
                find_trace_fail_descr_layouts_in_fail_descrs(&bridge.fail_descrs, trace_id)
            {
                return Some(layouts);
            }
        }
    }
    None
}

fn find_trace_terminal_exit_layouts_in_fail_descrs(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
) -> Option<Vec<majit_backend::TerminalExitLayout>> {
    for descr in fail_descrs {
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if bridge.trace_id == trace_id {
                return Some(bridge.terminal_exit_layouts_ref().clone());
            }
            if let Some(layouts) =
                find_trace_terminal_exit_layouts_in_fail_descrs(&bridge.fail_descrs, trace_id)
            {
                return Some(layouts);
            }
        }
    }
    None
}

fn find_trace_info_in_fail_descrs(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
) -> Option<CompiledTraceInfo> {
    for descr in fail_descrs {
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if bridge.trace_id == trace_id {
                return Some(CompiledTraceInfo {
                    trace_id: bridge.trace_id,
                    input_types: bridge.input_types.clone(),
                    header_pc: bridge.header_pc,
                    source_guard: Some(bridge.source_guard),
                });
            }
            if let Some(info) = find_trace_info_in_fail_descrs(&bridge.fail_descrs, trace_id) {
                return Some(info);
            }
        }
    }
    None
}

fn find_fail_descr_in_fail_descrs(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
    fail_index: u32,
) -> Option<Arc<CraneliftFailDescr>> {
    for descr in fail_descrs {
        if descr.trace_id == trace_id && descr.fail_index == fail_index {
            return Some(descr.clone());
        }
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if let Some(found) =
                find_fail_descr_in_fail_descrs(&bridge.fail_descrs, trace_id, fail_index)
            {
                return Some(found);
            }
        }
    }
    None
}

fn find_fail_descr_by_ptr(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    descr_ptr: usize,
) -> Option<Arc<CraneliftFailDescr>> {
    for descr in fail_descrs {
        if Arc::as_ptr(descr) as usize == descr_ptr {
            return Some(descr.clone());
        }
        if let Some(bridge) = descr.bridge_ref().as_ref() {
            if let Some(found) = find_fail_descr_by_ptr(&bridge.fail_descrs, descr_ptr) {
                return Some(found);
            }
        }
    }
    None
}

/// Result of `run_compiled_code` — RPython llmodel.py:328 parity.
///
/// Instead of copying values out of jf_frame into Vec<i64>,
/// the JitFrame GcRef is returned directly. Values stay in place.
struct JitExecResult {
    jf_gcref: GcRef,
    heap_owner: Option<Vec<i64>>,
    fail_index: u32,
    direct_descr: Option<Arc<CraneliftFailDescr>>,
}

impl JitExecResult {
    const HEADER_WORDS: usize = (JF_FRAME_ITEM0_OFS as usize) / 8;

    /// Read jf_frame[index] as i64.
    #[inline]
    fn get_jf_int(&self, index: usize) -> i64 {
        unsafe { *((self.jf_gcref.0 + JF_FRAME_ITEM0_OFS as usize + index * 8) as *const i64) }
    }

    /// Extract raw i64 values from jf_frame for legacy callers.
    fn extract_outputs(&self, arity: usize) -> Vec<i64> {
        (0..arity).map(|i| self.get_jf_int(i)).collect()
    }
}

fn run_compiled_code(
    code_ptr: *const u8,
    fail_descrs: &[Arc<CraneliftFailDescr>],
    num_ref_roots: usize,
    max_output_slots: usize,
    inputs: &[i64],
    attachments: &CpuDescrAttachments,
) -> JitExecResult {
    // No alternate-stack switch here. The compiled prologue's inline
    // SP probe (`_call_header_with_stack_check`,
    // rpython/jit/backend/x86/assembler.py:1085 parity) reads
    // `PYRE_STACKTOOBIG.stack_end` against the caller's real SP;
    // running the compiled code on a grown stack would shift SP onto
    // a different guard region and make the budget compare meaningless.
    run_compiled_code_inner(
        code_ptr,
        fail_descrs,
        num_ref_roots,
        max_output_slots,
        inputs,
        attachments,
    )
}

fn run_compiled_code_inner(
    code_ptr: *const u8,
    fail_descrs: &[Arc<CraneliftFailDescr>],
    num_ref_roots: usize,
    max_output_slots: usize,
    inputs: &[i64],
    attachments: &CpuDescrAttachments,
) -> JitExecResult {
    // RPython llmodel.py:298: frame = gc_ll_descr.malloc_jitframe(frame_info)
    // jitframe.py:48-52: jitframe_allocate(frame_info)
    let depth = max_output_slots.max(inputs.len()).max(1);
    let header_words = (JF_FRAME_ITEM0_OFS as usize) / 8; // 8 words = 64 bytes
    let frame_depth = depth + num_ref_roots;
    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!(
            "[jf-alloc] frame_depth={} depth={} max_output={} inputs={} ref_roots={}",
            frame_depth,
            depth,
            max_output_slots,
            inputs.len(),
            num_ref_roots
        );
    }
    let jf_total = header_words + frame_depth;
    let payload_bytes = jf_total * 8;

    // RPython gc.py:132 malloc_jitframe → jitframe_allocate:
    //   rgc.register_custom_trace_hook(JITFRAME, jitframe_trace)
    //   frame = lltype.malloc(JITFRAME, frame_depth)
    //
    // When GC has a type registry (MiniMarkGC), allocate from nursery with
    // JITFRAME type_id so the GC can copy+trace it correctly. The shadow
    // stack holds a GcRef that the GC updates in place when copying.
    // When GC is a stub (TrackingGc) or absent, fall back to Vec<i64>.
    // jitframe_allocate (jitframe.py:48) allocates from the nursery
    // via rgc.malloc. Nursery::alloc syncs from NURSERY_FREE_ADDR
    // (incminimark.py:676 gc_adr_of_nursery_free parity) so JIT
    // inline bumps and GC allocations share the same free pointer.
    // jitframe.py:48 parity: allocate jitframes from nursery when GC
    // type registry is available. The gcmap marks ref_root_slots, which
    // are spilled/reloaded around collecting calls. Input slots are read
    // once at entry (before any GC point) and their values live in
    // SSA/ref_root_slots afterward.
    let runtime_jitframe_tid = cranelift_jitframe_type_id();
    let use_gc_alloc = runtime_jitframe_tid.is_some();
    let (jf_gcref, heap_owner): (GcRef, Option<Vec<i64>>) = if use_gc_alloc {
        let type_id = runtime_jitframe_tid.unwrap();
        assert_ne!(
            type_id,
            u32::MAX,
            "JITFRAME GC type id not registered — frontend must call \
             set_jitframe_gc_type_id() or ensure_jitframe_type_registered() \
             before running compiled code"
        );
        let gcref = with_cranelift_gc_required(|gc| {
            gc.alloc_nursery_no_collect_typed(type_id, payload_bytes)
        });
        unsafe {
            // jitframe.py:48 parity: zero the header so jf_descr starts
            // as NULL (GuardNotForced checks jf_descr != 0).
            // RPython's nursery allocation zeros memory; ours may not.
            std::ptr::write_bytes(gcref.0 as *mut u8, 0, JF_FRAME_ITEM0_OFS as usize);
            *((gcref.0 + JF_FRAME_LENGTH_OFS as usize) as *mut usize) = frame_depth;
        }
        (gcref, None)
    } else {
        let mut buf = vec![0i64; jf_total];
        let gcref = GcRef(buf.as_mut_ptr() as usize);
        (gcref, Some(buf))
    };

    let jf_ptr = jf_gcref.0 as *mut i64;

    // llmodel.py:306-315: set arguments in frame
    for (i, &val) in inputs.iter().enumerate() {
        unsafe { *jf_ptr.add(header_words + i) = val };
    }
    if std::env::var_os("MAJIT_LOG").is_some() {
        let preview: Vec<i64> = inputs.iter().copied().take(10).collect();
        eprintln!("[pre-call-inputs] {:?}", preview);
    }
    // Debug: verify first input (frame ptr) is valid
    if std::env::var_os("MAJIT_VERIFY").is_some() && !inputs.is_empty() {
        let frame_ptr = inputs[0];
        if frame_ptr != 0 && (frame_ptr < 0x1000 || (frame_ptr as usize & 0x7) != 0) {
            eprintln!("[VERIFY] BAD frame_ptr input[0]={:#x}", frame_ptr);
        }
    }

    // llmodel.py:322: llop.gc_writebarrier(ll_frame)
    if use_gc_alloc {
        with_cranelift_gc_required(|gc| gc.write_barrier(jf_gcref));
    }

    // llmodel.py:323 parity: ll_frame = func(ll_frame)
    let func: unsafe extern "C" fn(*mut i64) -> *mut i64 = unsafe { std::mem::transmute(code_ptr) };

    let _jitted_guard = majit_backend::JittedGuard::enter();

    // assembler.py:1074 _call_header_shadowstack and assembler.py:1101
    // _call_footer_shadowstack are emitted as inline MOVs in compiled
    // code's prologue/epilogue. The caller does NOT touch root_stack_top.

    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!(
            "[pre-call] code_ptr={:p} jf_ptr={:p} inputs.len={} ref_roots={} max_output={}",
            code_ptr,
            jf_ptr,
            inputs.len(),
            num_ref_roots,
            max_output_slots
        );
    }
    let result_jf = unsafe { func(jf_ptr) };
    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!("[post-call] result_jf={:p}", result_jf);
    }

    // jitframe_resolve (jitframe.py:54-57):
    // Follow jf_forward chain — the compiled code may return the old
    // (nursery) jf_ptr, but the jitframe has been forwarded to old gen.
    let result_jf = jitframe_resolve(result_jf);

    // llmodel.py:412-420 get_latest_descr: read jf_descr pointer from frame.
    // RPython stores actual descr object pointer in jf_descr field and
    // retrieves it via get_latest_descr() — no index lookup needed.
    // We extract both the integer fail_index (for sentinel checks) and
    // the Arc pointer (for direct descr propagation).
    let result_jf = result_jf;
    let jf_descr_raw = unsafe { *result_jf.add(JF_DESCR_OFS as usize / 8) };

    // compile.py:1085-1098 `PropagateExceptionDescr.handle_fail`:
    //     exception = cpu.grab_exc_value(deadframe)
    //     if not exception:
    //         exception = cast_instance_to_gcref(memory_error)
    //     raise jitexc.ExitFrameWithExceptionRef(exception)
    //
    // Detect-and-transfer before the regular match chain so the
    // upstream "ExitFrameWithExceptionRef" tail can absorb the
    // exception ref through the ordinary slot-0 reader path.  The
    // emitted propagate path (compiler.rs OpCode::CheckMemoryError /
    // x86 / aarch64 assembler.rs) stores the raw value into
    // `jf_guard_exc`; here we mirror `cpu.grab_exc_value` (read +
    // clear) and stage the result into `jf_frame[0]` so
    // `EXIT_FRAME_WITH_EXCEPTION_DESCR_REF_CL`'s consumer path
    // (compile.py:660) reads it through the existing accessor.
    let propagate_descr_ptr = attachments
        .propagate_exception_descr
        .as_ref()
        .map_or(0, |arc| Arc::as_ptr(arc) as *const () as usize);
    if propagate_descr_ptr != 0 && jf_descr_raw as usize == propagate_descr_ptr {
        let header_words = JF_FRAME_ITEM0_OFS as usize / 8;
        let exc_val = unsafe {
            let slot = result_jf.add(JF_GUARD_EXC_OFS as usize / 8);
            let v = *slot;
            *slot = 0;
            v
        };
        let exc_val = if exc_val != 0 {
            exc_val
        } else {
            // compile.py:1095 `cast_instance_to_gcref(memory_error)`
            // — fallback when Layer 1's singleton store didn't fire
            // (no provider registered, or test setup that bypasses
            // pyre's malloc helpers).
            majit_backend::memory_error_singleton_ref()
        };
        unsafe {
            *result_jf.add(header_words) = exc_val;
        }
        // Reroute jf_descr to the attached
        // `exit_frame_with_exception_descr_ref` so the existing
        // match arm picks up the ExitFrameWithExceptionRef tail.
        // When unattached (unit-test setup), fall through to the
        // cranelift singleton via `direct_descr` below.
        let attached_exit = attachments
            .exit_frame_with_exception_descr_ref
            .as_ref()
            .map_or(0, |arc| Arc::as_ptr(arc) as *const () as usize);
        if attached_exit != 0 {
            unsafe {
                *result_jf.add(JF_DESCR_OFS as usize / 8) = attached_exit as i64;
            }
        }
    }
    let jf_descr_raw = unsafe { *result_jf.add(JF_DESCR_OFS as usize / 8) };

    let (fail_index, direct_descr) = if jf_descr_raw == CALL_ASSEMBLER_DEADFRAME_SENTINEL as i64 {
        (CALL_ASSEMBLER_DEADFRAME_SENTINEL, None)
    } else if jf_descr_raw == 0 {
        // Propagate-into-exit fall-through: when the metainterp
        // didn't attach an `exit_frame_with_exception_descr_ref`
        // (unit-test setups), the rewrite above leaves jf_descr at 0;
        // surface the cranelift singleton directly so the consumer
        // still sees `is_exit_frame_with_exception=true`.
        if propagate_descr_ptr != 0 {
            (
                u32::MAX,
                Some(EXIT_FRAME_WITH_EXCEPTION_DESCR_REF_CL.clone()),
            )
        } else {
            (0u32, None)
        }
    } else if let Some((metainterp_finish, cl_singleton)) =
        match_metainterp_finish_descr(jf_descr_raw, attachments)
    {
        // `pyjitpl.py:2222` match — the metainterp-side
        // `Arc<DoneWithThisFrameDescr*>` was written into jf_descr at
        // FINISH emission.  Its `fail_index()` on `MetaFailDescr`
        // semantics is u32::MAX (finish marker), which `run_compiled_code`
        // consumers rely on.  Return the cranelift-side singleton as
        // `direct_descr` so downstream `bridge_ref()` / `fail_arg_types`
        // access keeps the same type as the non-metainterp path.
        let _ = metainterp_finish;
        (u32::MAX, Some(cl_singleton))
    } else {
        let descr_ptr = jf_descr_raw as *const CraneliftFailDescr;
        let fi = unsafe { &*descr_ptr }.fail_index();
        let arc = find_fail_descr_by_ptr(fail_descrs, jf_descr_raw as usize);
        // compile.py:665-674 parity: done_with_this_frame_descr is a
        // global singleton — it won't appear in per-trace fail_descrs.
        let arc = arc.or_else(|| {
            let ptr = jf_descr_raw as usize;
            for global in [
                &*DONE_WITH_THIS_FRAME_DESCR_INT,
                &*DONE_WITH_THIS_FRAME_DESCR_FLOAT,
                &*DONE_WITH_THIS_FRAME_DESCR_REF,
                &*DONE_WITH_THIS_FRAME_DESCR_VOID,
            ] {
                if Arc::as_ptr(global) as usize == ptr {
                    return Some(global.clone());
                }
            }
            None
        });
        (fi, arc)
    };
    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!(
            "[post-call-descr] raw={:#x} fail_index={} matched_local={}",
            jf_descr_raw,
            fail_index,
            direct_descr.is_some()
        );
    }

    // llgraph/runner.py:1184-1191 — bridge dispatch is handled by the
    // caller (execute_with_inputs) in a flat loop, matching RPython's
    // LLFrame.execute() pattern where Jump exceptions change the current
    // lltrace. run_compiled_code simply returns the guard failure to the
    // caller without dispatching bridges.

    // RPython llmodel.py:328 parity: return ll_frame.
    // Values stay in jf_frame — no copying.
    let jf_gcref = GcRef(result_jf as usize);

    drop(_jitted_guard);
    JitExecResult {
        jf_gcref,
        heap_owner,
        fail_index,
        direct_descr,
    }
}

struct GuardInfo {
    fail_index: u32,
    can_have_bridge: bool,
    fail_arg_refs: Vec<OpRef>,
    /// assembler.py:40-44 must_save_exception(): true for
    /// GUARD_EXCEPTION, GUARD_NO_EXCEPTION, GUARD_NOT_FORCED.
    must_save_exception: bool,
    /// gcmap bitmap: bit i set ⇔ fail_arg[i] is Ref type.
    /// allocate_gcmap (gcmap.py:7-18) parity.
    gcmap: u64,
    /// RPython assembler.py:2126 get_gcref_from_faildescr parity:
    /// stores Arc::as_ptr(CraneliftFailDescr) as i64.
    /// The FailDescr GCREF pointer is written to jf_descr on guard exit.
    fail_descr_ptr: i64,
    /// vector_ext.py:119 _update_at_exit: accumulation metadata for vector
    /// reduction at guard exit. Each entry maps a fail_arg slot to its
    /// vector accumulator variable and reduction operator.
    accum_info: Vec<AccumInfo>,
}

fn identity_recovery_layout(
    trace_id: u64,
    header_pc: u64,
    guard_resume_pc: Option<u64>,
    source_guard: Option<(u64, u32)>,
    slot_types: &[Type],
    caller_layout: Option<&ExitRecoveryLayout>,
) -> ExitRecoveryLayout {
    let mut frames = caller_layout
        .map(|layout| layout.frames.clone())
        .unwrap_or_default();
    frames.push(ExitFrameLayout {
        trace_id: Some(trace_id),
        header_pc: Some(header_pc),
        source_guard,
        pc: guard_resume_pc.unwrap_or(header_pc),
        // Placeholder: this backend-side identity layout is produced at
        // compile time, then replaced by the resume_layout-derived layout
        // via `compile::patch_backend_guard_recovery_layouts_for_trace`
        // (compile.rs:1596). The guard-exit rd_numb path at compiler.rs
        // ~11035 additionally overwrites `frame.jitcode_index` with the
        // topmost rd_numb frame, covering guards whose resume_layout is
        // absent. `Snapshot::single_frame(jitcode_index, pc, ...)` is the
        // upstream source of truth.
        jitcode_index: 0,
        slots: (0..slot_types.len())
            .map(ExitValueSourceLayout::ExitValue)
            .collect(),
        slot_types: Some(slot_types.to_vec()),
    });
    ExitRecoveryLayout {
        vable_array: Vec::new(),
        vref_array: Vec::new(),
        frames,
        virtual_layouts: caller_layout
            .map(|layout| layout.virtual_layouts.clone())
            .unwrap_or_default(),
        pending_field_layouts: caller_layout
            .map(|layout| layout.pending_field_layouts.clone())
            .unwrap_or_default(),
    }
}

fn patch_fail_descr_recovery_layout(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
    fail_index: u32,
    recovery_layout: &ExitRecoveryLayout,
) -> bool {
    for descr in fail_descrs {
        if descr.trace_id == trace_id && descr.fail_index == fail_index {
            descr.set_recovery_layout(recovery_layout.clone());
            return true;
        }
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if patch_fail_descr_recovery_layout(
                &bridge.fail_descrs,
                trace_id,
                fail_index,
                recovery_layout,
            ) {
                return true;
            }
        }
    }
    false
}

fn patch_terminal_exit_recovery_layout_in_vec(
    terminal_exit_layouts: &UnsafeCell<Vec<TerminalExitLayout>>,
    trace_id: u64,
    op_index: usize,
    recovery_layout: &ExitRecoveryLayout,
) -> bool {
    let terminal_exit_layouts = unsafe { &mut *terminal_exit_layouts.get() };
    if let Some(layout) = terminal_exit_layouts
        .iter_mut()
        .find(|layout| layout.trace_id == trace_id && layout.op_index == op_index)
    {
        layout.recovery_layout = Some(recovery_layout.clone());
        return true;
    }
    false
}

fn patch_terminal_exit_recovery_layout(
    root_terminal_exit_layouts: &UnsafeCell<Vec<TerminalExitLayout>>,
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
    op_index: usize,
    recovery_layout: &ExitRecoveryLayout,
) -> bool {
    if patch_terminal_exit_recovery_layout_in_vec(
        root_terminal_exit_layouts,
        trace_id,
        op_index,
        recovery_layout,
    ) {
        return true;
    }

    for descr in fail_descrs {
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if patch_terminal_exit_recovery_layout(
                &bridge.terminal_exit_layouts,
                &bridge.fail_descrs,
                trace_id,
                op_index,
                recovery_layout,
            ) {
                return true;
            }
        }
    }
    false
}

fn infer_fail_arg_types(
    fail_arg_refs: &[OpRef],
    type_index: &OpTypeIndex<'_>,
    overrides: &HashMap<u32, Type>,
    op_index: usize,
) -> Result<Vec<Type>, BackendError> {
    let mut fail_arg_types = Vec::with_capacity(fail_arg_refs.len());
    for &opref in fail_arg_refs {
        if opref.is_none() {
            // resoperation.py Box.type parity: NONE marks a virtual object
            // slot. Virtual objects are GCREF (Ref). The backend stores null;
            // materialization uses rd_virtuals on guard failure.
            fail_arg_types.push(Type::Ref);
            continue;
        }
        // resoperation.py Box.type parity: default to Ref (GCREF).
        // RPython's Box carries an immutable .type attribute; unknown
        // boxes are RefOp-based (the most common case in pyre).
        fail_arg_types
            .push(lookup_type_at(type_index, overrides, opref, op_index).unwrap_or(Type::Ref));
    }
    Ok(fail_arg_types)
}

/// resoperation.py Box.type parity: determine fail_arg types.
///
/// RPython's Box.type is immutable — the backend reads box.type directly
/// (assembler.py:46 compute_gcmap). In pyre, fail_arg_types come from
/// the optimizer, which may assign Int to OpRefs that are defined AFTER
/// this guard. For such cases, use the inputarg type (pre-redefinition).
fn resolve_fail_arg_types(
    fail_arg_refs: &[OpRef],
    fd: Option<&dyn majit_ir::descr::FailDescr>,
    type_index: &OpTypeIndex<'_>,
    overrides: &HashMap<u32, Type>,
    op_def_positions: &HashMap<u32, usize>,
    guard_op_index: usize,
) -> Result<Vec<Type>, BackendError> {
    // Use descriptor types as base, then fix positional conflicts.
    let base = if let Some(fd) = fd {
        let dt = fd.fail_arg_types();
        if dt.len() == fail_arg_refs.len() {
            dt.to_vec()
        } else {
            infer_fail_arg_types(fail_arg_refs, type_index, overrides, guard_op_index)?
        }
    } else {
        infer_fail_arg_types(fail_arg_refs, type_index, overrides, guard_op_index)?
    };

    // Fix positional conflicts: when a guard fires BEFORE the operation
    // that redefines an OpRef, the fail_arg holds the PRE-redefinition
    // value. Use the inputarg type (not the post-redefinition type).
    Ok(base
        .into_iter()
        .enumerate()
        .map(|(i, tp)| {
            let opref = fail_arg_refs.get(i).copied().unwrap_or(OpRef::NONE);
            if opref.is_none() {
                return Type::Ref;
            }
            if let Some(&def_pos) = op_def_positions.get(&opref.raw()) {
                if guard_op_index < def_pos {
                    // Guard before redefinition: use inputarg type.
                    return type_index.inputarg_type(opref).unwrap_or(tp);
                }
            }
            tp
        })
        .collect())
}

// ---------------------------------------------------------------------------
// CraneliftBackend
// ---------------------------------------------------------------------------

pub struct CraneliftBackend {
    module: JITModule,
    func_ctx: FunctionBuilderContext,
    constants: HashMap<u32, i64>,
    /// rewrite.py:930 parity — type annotations for constant OpRefs.
    /// Set by `set_constant_types` before each compile call; used by
    /// the GC rewriter to check `v.type == 'r'` on constant values.
    constant_types: HashMap<u32, majit_ir::Type>,
    /// compile.py: self.metainterp_sd.callinfocollection — used by
    /// recovery_layout building for VStr/VUni Concat/Slice func ptr
    /// lookups (resume.py:1143-1188).
    callinfocollection: Option<Arc<majit_ir::CallInfoCollection>>,
    func_counter: u32,
    trace_counter: u64,
    next_trace_id: Option<u64>,
    next_header_pc: Option<u64>,
    registered_call_assembler_tokens: HashSet<u64>,
    registered_call_assembler_bridge_traces: HashSet<u64>,
    /// llmodel.py: self.vtable_offset — byte offset for vtable in objects.
    /// pyre PyObject layout: ob_type at offset 0.
    vtable_offset: Option<usize>,
    /// `compile.py:665` `setattr(cpu, name, descr)` per-cpu attachments.
    /// Each `Backend` instance owns its own copy of the metainterp-side
    /// `DoneWithThisFrameDescr*` / `ExitFrameWithExceptionDescrRef` /
    /// `PropagateExceptionDescr` singletons, held in a heap-pinned
    /// `Arc<RwLock<CpuDescrAttachments>>` so the address is stable even
    /// when `CraneliftBackend` is moved.  `CompiledLoop` carries an Arc
    /// clone so the attachments outlive this backend for the lifetime of
    /// emitted code that baked the handle as an immediate.
    descr_attachments: CpuDescrHandle,
}

impl CraneliftBackend {
    /// llmodel.py:467-478 read_int_at_mem(gcref, ofs, size, sign).
    ///
    /// PRE-EXISTING-ADAPTATION: RPython's `llop.raw_load(STYPE, gcref, ofs)`
    /// segfaults on a null `gcref`; the JIT relies on emitted guards
    /// (`guard_nonnull`, `guard_class`, etc.) to filter null inputs
    /// before reaching the helper. Cranelift currently calls this helper
    /// from sites that occasionally pass `addr == 0` (fannkuch fails
    /// when the guard is removed; dynasm's identical helper is null-
    /// agnostic and works because dynasm's call sites always guard).
    /// The convergence path is to audit cranelift's call sites and
    /// either add the missing `addr != 0` guard at emit time or migrate
    /// to a guarded helper variant; until then, the silent-zero return
    /// shadows the missing guard. dynasm parity (`runner.rs:434`) is
    /// already RPython-orthodox.
    fn read_int_at_mem(&self, addr: i64, offset: i64, size: usize, sign: bool) -> i64 {
        if addr == 0 {
            return 0;
        }
        let ptr = (addr as usize).wrapping_add(offset as usize);
        unsafe {
            match (size, sign) {
                (1, true) => (ptr as *const i8).read_unaligned() as i64,
                (1, false) => (ptr as *const u8).read_unaligned() as i64,
                (2, true) => (ptr as *const i16).read_unaligned() as i64,
                (2, false) => (ptr as *const u16).read_unaligned() as i64,
                (4, true) => (ptr as *const i32).read_unaligned() as i64,
                (4, false) => (ptr as *const u32).read_unaligned() as i64,
                _ => (ptr as *const i64).read_unaligned(),
            }
        }
    }

    /// llmodel.py:481-488 write_int_at_mem(gcref, ofs, size, newvalue).
    /// PRE-EXISTING-ADAPTATION: see `read_int_at_mem` above for the
    /// null-guard rationale (cranelift call-site auditing required).
    fn write_int_at_mem(&self, addr: i64, offset: i64, size: usize, newvalue: i64) {
        if addr == 0 {
            return;
        }
        let ptr = (addr as usize).wrapping_add(offset as usize);
        unsafe {
            match size {
                1 => (ptr as *mut u8).write_unaligned(newvalue as u8),
                2 => (ptr as *mut u16).write_unaligned(newvalue as u16),
                4 => (ptr as *mut u32).write_unaligned(newvalue as u32),
                _ => (ptr as *mut i64).write_unaligned(newvalue),
            }
        }
    }

    /// llmodel.py:490-491 read_float_at_mem(gcref, ofs).
    /// PRE-EXISTING-ADAPTATION: see `read_int_at_mem` above for the
    /// null-guard rationale.
    fn read_float_at_mem(&self, addr: i64, offset: i64) -> f64 {
        if addr == 0 {
            return 0.0;
        }
        let ptr = (addr as usize).wrapping_add(offset as usize);
        unsafe { (ptr as *const f64).read_unaligned() }
    }

    /// llmodel.py:493-494 write_float_at_mem(gcref, ofs, newvalue).
    /// PRE-EXISTING-ADAPTATION: see `read_int_at_mem` above for the
    /// null-guard rationale.
    fn write_float_at_mem(&self, addr: i64, offset: i64, newvalue: f64) {
        if addr == 0 {
            return;
        }
        let ptr = (addr as usize).wrapping_add(offset as usize);
        unsafe { (ptr as *mut f64).write_unaligned(newvalue) }
    }

    /// Propagate bridges from previous tokens' fail_descrs to the current
    /// token's fail_descrs. Called after finish_and_compile replaces the
    /// token — bridges compiled during main loop tracing were attached to
    /// the old token and need to be copied to the new one.
    pub fn propagate_bridges_from_previous_tokens(
        &self,
        current_token: &JitCellToken,
        previous_tokens: &[std::sync::Arc<JitCellToken>],
    ) {
        let compiled = match current_token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>())
        {
            Some(c) => c,
            None => return,
        };
        let new_descrs = &compiled.fail_descrs;
        for prev in previous_tokens {
            if let Some(prev_compiled) = prev
                .compiled
                .as_ref()
                .and_then(|c| c.downcast_ref::<CompiledLoop>())
            {
                for prev_d in &prev_compiled.fail_descrs {
                    if prev_d.has_bridge() {
                        let fi = prev_d.fail_index();
                        let tid = prev_d.trace_id;
                        if let Some(new_d) = find_fail_descr_in_fail_descrs(new_descrs, tid, fi) {
                            if !new_d.has_bridge() {
                                let bridge = prev_d.bridge_ref();
                                if let Some(ref b) = *bridge {
                                    if std::env::var_os("MAJIT_LOG").is_some() {
                                        eprintln!(
                                            "[jit] propagate bridge tid={} fi={} to new token",
                                            tid, fi,
                                        );
                                    }
                                    new_d.attach_bridge(BridgeData {
                                        trace_id: b.trace_id,
                                        input_types: b.input_types.clone(),
                                        header_pc: b.header_pc,
                                        source_guard: b.source_guard,
                                        caller_prefix_layout: b.caller_prefix_layout.clone(),
                                        code_ptr: b.code_ptr,
                                        fail_descrs: b.fail_descrs.clone(),
                                        num_inputs: b.num_inputs,
                                        num_ref_roots: b.num_ref_roots,
                                        max_output_slots: b.max_output_slots,
                                        terminal_exit_layouts: UnsafeCell::new(
                                            unsafe { &*b.terminal_exit_layouts.get() }.clone(),
                                        ),
                                        loop_reentry: b.loop_reentry,
                                        invalidated_arc: b.invalidated_arc.clone(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();

        let isa_builder = cranelift_native::builder().expect("host ISA not supported");
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(jit_builder);
        let func_ctx = FunctionBuilderContext::new();

        CraneliftBackend {
            module,
            func_ctx,
            constants: HashMap::new(),
            constant_types: HashMap::new(),
            callinfocollection: None,
            func_counter: 0,
            trace_counter: 1,
            next_trace_id: None,
            next_header_pc: None,
            registered_call_assembler_tokens: HashSet::new(),
            registered_call_assembler_bridge_traces: HashSet::new(),
            // llmodel.py:64-69: vtable_offset is None when gcremovetypeptr is
            // enabled; otherwise it comes from
            //   symbolic.get_field_token(rclass.OBJECT, 'typeptr', ...).
            // Callers configure pyre's PyObject layout via set_vtable_offset.
            vtable_offset: None,
            descr_attachments: Arc::new(std::sync::RwLock::new(CpuDescrAttachments::default())),
        }
    }

    /// compile.py plumbing: register the shared `CallInfoCollection` so
    /// that recovery_layout building can resolve OS_STR_CONCAT etc.
    /// function pointers for VStr/VUni Concat/Slice guard-exit
    /// materialization (resume.py:1143-1188).
    pub fn set_callinfocollection(&mut self, cic: Option<Arc<majit_ir::CallInfoCollection>>) {
        self.callinfocollection = cic;
    }

    /// llmodel.py:64-69 self.vtable_offset configuration.
    /// Frontend (e.g. pyre) sets the byte offset of the type pointer field
    /// inside instance objects, mirroring how RPython resolves the typeptr
    /// offset via `symbolic.get_field_token(rclass.OBJECT, 'typeptr', ...)`.
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.vtable_offset = offset;
    }

    /// `compile.py:665-674` `make_and_attach_done_descrs` parity: expose
    /// the six per-cpu-instance descrs as raw pointers for emission
    /// consumers (`collect_guards` FINISH sites, `register_call_assembler_target`).
    /// The metainterp attaches the real descrs through
    /// `Backend::set_done_with_this_frame_descr_*` during
    /// `MetaInterpStaticData.finish_setup` (pyjitpl.py:2222); the
    /// fallback cranelift singletons below answer for the
    /// backend-only integration tests that skip `MetaInterp::new` so
    /// every result-type bucket still has a non-zero pointer.
    pub(crate) fn attached_descr_ptrs(&self) -> majit_backend::AttachedDescrPtrs {
        let attached = self.descr_attachments.read().unwrap().descr_ptrs();
        attached_descr_ptrs_with_fallbacks(attached)
    }

    /// Heap-stable `Arc<RwLock<CpuDescrAttachments>>` address for embedding
    /// in cranelift-emitted CALL_ASSEMBLER helpers; the extern-C trampoline
    /// dereferences it to snapshot the attached descr pointers at dispatch.
    /// Matches dynasm's `cpu_handle_ptr()` semantics so both backends share
    /// the same runtime classifier contract.
    pub(crate) fn cpu_handle(&self) -> CpuDescrHandle {
        Arc::clone(&self.descr_attachments)
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Delegates to the installed `GcAllocator`. RPython resolves the
    /// typeid through `cpu.gc_ll_descr`; in majit the gc_ll_descr role
    /// is filled by the active GC runtime registered via set_gc_allocator.
    fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        with_cranelift_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr)).flatten()
    }

    pub fn with_gc_allocator(gc: Box<dyn GcAllocator>) -> Self {
        let mut backend = Self::new();
        backend.set_gc_allocator(gc);
        backend
    }

    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn GcAllocator>) {
        let jitframe_type_id = ensure_jitframe_type_registered(gc.as_mut());
        // gctypelayout.encode_type_shapes_now parity: close the
        // type-registration phase before any compile embeds the
        // type_info_group base address. After this, register_type
        // panics and every is_object type's subclassrange is filled
        // in by the preorder walk in assign_inheritance_ids.
        gc.freeze_types();
        let supports_guard_gc_type = gc.supports_guard_gc_type();
        // `llmodel.py:58` `self.gc_ll_descr = get_ll_description(...)` —
        // single owning slot per CPU. Replaces any prior allocator on
        // this thread; the implicit drop of the old `Box` is the
        // RPython `cpu.gc_ll_descr = …` assignment's parity.
        set_cranelift_active_gc(Some(gc));
        set_cranelift_jitframe_type_id(jitframe_type_id);
        // Publish the backend's GC-guard seam through the
        // backend-agnostic shims so the optimizer (majit-metainterp)
        // and blackhole executor can reach the live allocator without
        // taking a cranelift dependency. Mirrors RPython's
        // `self.optimizer.cpu.check_is_object(...)` access path.
        majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
            check_is_object: Some(check_is_object_via_active_runtime),
            get_actual_typeid: Some(get_actual_typeid_via_active_runtime),
            subclass_range: Some(subclass_range_via_active_runtime),
            typeid_subclass_range: Some(typeid_subclass_range_via_active_runtime),
            typeid_is_object: Some(typeid_is_object_via_active_runtime),
            supports_guard_gc_type,
        });
        majit_gc::set_active_alloc_nursery_typed(Some(alloc_nursery_typed_via_active_runtime));
        majit_gc::set_active_alloc_oldgen_typed(Some(alloc_oldgen_typed_via_active_runtime));
        majit_gc::set_active_root_hooks(
            Some(gc_add_root_via_active_runtime),
            Some(gc_remove_root_via_active_runtime),
        );
        majit_gc::set_active_gc_owns_object(Some(gc_owns_object_via_active_runtime));
    }

    // `set_constants`, `set_constant_types`, `set_next_trace_id`,
    // `set_next_header_pc` are provided via the `Backend` trait impl
    // below so `compile_tmp_callback` and other backend-agnostic
    // consumers can reach them through `&mut dyn Backend`.

    fn gc_rewriter(&self, constant_types: &HashMap<u32, majit_ir::Type>) -> Option<GcRewriterImpl> {
        let ct = constant_types.clone();
        with_cranelift_gc(|gc| GcRewriterImpl {
            nursery_free_addr: gc.nursery_free_addr(),
            nursery_top_addr: gc.nursery_top_addr(),
            max_nursery_size: gc.max_nursery_object_size(),
            // gc.py:259-283 WriteBarrierDescr parity.
            wb_descr: {
                let mut descr = WriteBarrierDescr::for_current_gc();
                let card_page_shift = gc.card_page_shift();
                if card_page_shift > 0 {
                    descr.jit_wb_card_page_shift = card_page_shift;
                } else {
                    // gc.py:283: no card marking → jit_wb_cards_set = 0
                    descr.jit_wb_cards_set = 0;
                    descr.jit_wb_card_page_shift = 0;
                    descr.jit_wb_cards_set_byteofs = 0;
                    descr.jit_wb_cards_set_singlebyte = 0;
                }
                descr
            },
            jitframe_info: JITFRAME_LAYOUT
                .get()
                .and_then(|info| info.jitframe_descrs.clone()),
            constant_types: ct,
            // llmodel.py:39 default. Cranelift lowers GcStoreIndexed via
            // ir::MemFlags and explicit offset arithmetic rather than an
            // ISA scaled addressing mode, so we keep the rewriter in the
            // "pre-scale everything" contract that the lowering expects.
            load_supported_factors: &[1],
            // x86/runner.py:22 + arm/runner.py:26 — both pyre cranelift
            // lowering paths emit LOAD_EFFECTIVE_ADDRESS as a single
            // Cranelift `iadd` chain (no LEA fallback expansion needed
            // at the IR level), so the rewriter takes the single-LEA
            // path (rewrite.py:1083-1088).
            supports_load_effective_address: true,
            // nursery.rs:68 `alloc_zeroed` + nursery.rs:105-110 `reset`
            // memset-to-zero on recycle mean the nursery payload is
            // always zero-filled at allocation time; `clear_gc_fields`
            // thus short-circuits per rewrite.py:499-500.
            malloc_zero_filled: true,
            // gc.py:39 `self.memcpy_fn = memcpy_fn` cast through
            // `cast_ptr_to_adr` + `cast_adr_to_int` (rewrite.py:1046-1047).
            memcpy_fn: majit_ir::memcpy_fn_addr(),
            // gc.py:40-43 `self.memcpy_descr = get_call_descr(...)`.
            memcpy_descr: majit_ir::make_memcpy_calldescr(),
            // gc.py:46 `self.str_descr = get_array_descr(self, rstr.STR)`.
            str_descr: builtin_string_array_descr(OpCode::Newstr)
                .expect("Newstr must produce a str ArrayDescr"),
            // gc.py:47 `self.unicode_descr = get_array_descr(self, rstr.UNICODE)`.
            unicode_descr: builtin_string_array_descr(OpCode::Newunicode)
                .expect("Newunicode must produce a unicode ArrayDescr"),
            // gc.py:48 `self.str_hash_descr = get_field_descr(self, rstr.STR, 'hash')`.
            str_hash_descr: builtin_string_hash_field_descr(OpCode::Strhash)
                .expect("Strhash must produce a str hash FieldDescr"),
            // gc.py:49 `self.unicode_hash_descr = get_field_descr(self, rstr.UNICODE, 'hash')`.
            unicode_hash_descr: builtin_string_hash_field_descr(OpCode::Unicodehash)
                .expect("Unicodehash must produce a unicode hash FieldDescr"),
            // gc.py:33-37 `self.fielddescr_vtable = get_field_descr(
            // self, rclass.OBJECT, 'typeptr')`.  pyre always emits a
            // typeptr slot (no `gcremovetypeptr` build), so we install
            // Some unconditionally.
            fielddescr_vtable: Some(majit_ir::make_vtable_field_descr()),
            // gc.py:394 `self.fielddescr_tid = get_field_descr(self,
            // self.GCClass.HDR, 'tid')` — framework GC.  pyre's GC is
            // always framework-style (incminimark-shaped), so install
            // Some unconditionally; the helper translates the descr's
            // offset by `-HDR_SIZE` because pyre's HDR sits before the
            // object pointer.
            fielddescr_tid: Some(majit_ir::make_tid_field_descr()),
            malloc_array_fn: gc_malloc_array_helper as *const () as i64,
            malloc_array_nonstandard_fn: gc_malloc_array_nonstandard_helper as *const () as i64,
            malloc_str_fn: gc_malloc_str_helper as *const () as i64,
            malloc_unicode_fn: gc_malloc_unicode_helper as *const () as i64,
            malloc_big_fixedsize_fn: gc_malloc_big_fixedsize_helper as *const () as i64,
            malloc_array_descr: majit_ir::make_malloc_array_calldescr(),
            malloc_array_nonstandard_descr: majit_ir::make_malloc_array_nonstandard_calldescr(),
            malloc_str_descr: majit_ir::make_malloc_str_calldescr(),
            malloc_unicode_descr: majit_ir::make_malloc_unicode_calldescr(),
            malloc_big_fixedsize_descr: majit_ir::make_malloc_big_fixedsize_calldescr(),
            standard_array_basesize: std::mem::size_of::<usize>(),
            standard_array_length_ofs: 0,
            // rewrite.py:673 — read compiled_loop_token._ll_initial_locs and
            // rewrite.py:669 — ptr2int(compiled_loop_token.frame_info),
            // both sourced directly from the CLT Arc on the target
            // (model.py:292-338).
            call_assembler_callee_locs: Some(Box::new(|token_number| {
                lookup_call_assembler_target(token_number).map(|t| {
                    let clt = &t.compiled_loop_token;
                    // JitFrameInfo is #[repr(C)] so `&JitFrameInfo as *const _`
                    // points at [jfi_frame_depth: i64, jfi_frame_size: i64];
                    // the generated CALL_ASSEMBLER code loads those words
                    // without taking the Mutex. Keeping the Arc alive via
                    // `RegisteredLoopTarget.compiled_loop_token` pins the
                    // allocation, matching RPython where the lltype-malloced
                    // JITFRAMEINFO outlives the loop.
                    let frame_info_ptr = {
                        let guard = clt.frame_info.lock();
                        &*guard as *const majit_backend::JitFrameInfo as usize
                    };
                    let ll_initial_locs = clt._ll_initial_locs.lock().clone();
                    majit_gc::rewrite::CallAssemblerCalleeLocs {
                        _ll_initial_locs: ll_initial_locs,
                        frame_depth: t.max_output_slots + t.num_ref_roots,
                        frame_info_ptr,
                        // pyjitpl.py:3605 — outermost_jitdriver_sd.index_of_virtualizable,
                        // propagated from JitCellToken at registration time.
                        index_of_virtualizable: t.index_of_virtualizable,
                    }
                })
            })),
        })
    }

    fn prepare_ops_for_compile(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        constants: &HashMap<u32, i64>,
        constant_types: &HashMap<u32, majit_ir::Type>,
    ) -> Vec<Op> {
        let mut normalized = normalize_ops_for_codegen_simple(inputargs, ops);
        inject_builtin_string_descrs(&mut normalized);
        // rewrite.py:930 `v.type` — RPython Box carries its type on the
        // object itself, so InputArg / ResOperation / Const all return
        // the same `.type` attribute.  Pyre's flat `OpRef` stores types
        // in side maps keyed by the raw u32; `constant_types` already
        // carries op-position and constant types (disjoint key spaces
        // via `OpRef::CONST_BIT = 1 << 31`).  Each InputArg's raw OpRef
        // value lives at `InputArg.index`: top-level loops occupy
        // `[0, num_inputs)`, while Phase E.2b bridges occupy
        // `[bridge_inputarg_base..)`. Key the merged map by the actual
        // `ia.index` so the rewriter's `v.type` lookup matches the OpRef
        // values that ops reference, regardless of namespace.
        let mut constant_types_with_inputargs = constant_types.clone();
        for ia in inputargs.iter() {
            constant_types_with_inputargs
                .entry(ia.index)
                .or_insert(ia.tp);
        }
        if let Some(rewriter) = self.gc_rewriter(&constant_types_with_inputargs) {
            let (result, new_constants) =
                rewriter.rewrite_for_gc_with_constants(&normalized, constants);
            // Merge GC rewriter's new constants into self.constants
            for (k, v) in new_constants {
                self.constants.entry(k).or_insert(v);
            }
            result
        } else {
            normalized
        }
    }

    /// Execute a compiled bridge, returning the DeadFrame from the bridge's
    /// llgraph/runner.py:1117-1145 LLFrame.execute() parity.
    ///
    /// Flat dispatch loop: bridge dispatch is handled here (like Jump
    /// exceptions in llgraph), not in run_compiled_code. When a guard
    /// fails with a bridge, switch to the bridge trace and continue.
    /// When a bridge FINISH with loop_reentry fires, switch back to the
    /// main loop.
    fn execute_with_inputs(compiled: &CompiledLoop, inputs: &[i64]) -> DeadFrame {
        // Current trace state (equivalent to LLFrame.lltrace)
        let mut cur_code_ptr = compiled.code_ptr;
        let mut cur_fail_descrs: Box<[Arc<CraneliftFailDescr>]> = compiled.fail_descrs.clone();
        let mut cur_num_ref_roots = compiled.num_ref_roots;
        let mut cur_max_output_slots = compiled.max_output_slots;
        let mut cur_inputs = inputs.to_vec();
        // `compile.py:665 setattr(cpu, name, descr)` — borrow the owning
        // cpu's descr set for the whole dispatch loop.  Holding the read
        // lock is safe: `Backend::set_done_with_this_frame_descr_*` only
        // fires during `MetaInterpStaticData.finish_setup`, long before
        // compiled code runs.
        let attachments_guard = compiled.cpu_attachments.read().unwrap();
        let attachments: &CpuDescrAttachments = &*attachments_guard;

        loop {
            let exec = run_compiled_code(
                cur_code_ptr,
                &cur_fail_descrs,
                cur_num_ref_roots,
                cur_max_output_slots,
                &cur_inputs,
                attachments,
            );
            let fail_index = exec.fail_index;
            let direct_descr = exec.direct_descr.clone();
            let outputs = exec.extract_outputs(cur_max_output_slots.max(1));

            // CALL_ASSEMBLER deadframe interception.
            if let Some(frame) = maybe_take_call_assembler_deadframe(fail_index, &outputs) {
                return wrap_call_assembler_deadframe_with_caller_prefix(
                    frame,
                    compiled.trace_id,
                    compiled.header_pc,
                    None,
                    &compiled.input_types,
                    &cur_inputs,
                    compiled.caller_prefix_layout.as_ref(),
                );
            }

            // llmodel.py:412-420 get_latest_descr: resolve fail_descr from
            // jf_descr pointer (direct_descr) or fail_index lookup.
            let fail_descr = if let Some(descr) = direct_descr {
                descr
            } else if (fail_index as usize) < cur_fail_descrs.len() {
                cur_fail_descrs[fail_index as usize].clone()
            } else {
                // Search bridge fail_descrs for nested guard failures.
                let found = compiled.fail_descrs.iter().find_map(|d| {
                    let guard = d.bridge_ref();
                    guard.as_ref().and_then(|b| {
                        find_fail_descr_in_fail_descrs(&b.fail_descrs, b.trace_id, fail_index)
                    })
                });
                found.unwrap_or_else(|| {
                    cur_fail_descrs
                        .last()
                        .cloned()
                        .unwrap_or_else(|| compiled.fail_descrs[0].clone())
                })
            };
            let fail_descr = &fail_descr;

            // llgraph/runner.py:1130-1140 Jump exception caught by execute():
            // cross-loop JUMP — switch to the target loop trace identified
            // by the TargetToken stored on the fail descriptor.
            // assembler.py:2456-2462 closing_jump: raw JMP to
            // `target_token._ll_loop_code`. Cranelift can't emit inter-
            // function JMPs, so we return and re-enter the target loop here.
            if fail_descr.is_external_jump {
                let target_entry = fail_descr
                    .target_descr
                    .as_ref()
                    .and_then(lookup_loop_target)
                    .expect("external JUMP target must be a registered LoopTargetDescr");
                cur_code_ptr = target_entry.code_ptr;
                cur_fail_descrs = target_entry.fail_descrs;
                cur_num_ref_roots = target_entry.num_ref_roots;
                cur_max_output_slots = target_entry.max_output_slots;
                cur_inputs = outputs;
                continue;
            }
            // llgraph/runner.py:1200-1201 execute_finish → ExecutionFinished.
            if fail_descr.is_finish {
                // Real FINISH — function completed.
                // jf_savedata already correct in jf_frame memory.
                // jf_guard_exc already written by emit_guard_exit.
                return deadframe_from_jitframe(exec.jf_gcref, fail_descr.clone(), exec.heap_owner);
            }

            fail_descr.increment_fail_count();

            // llgraph/runner.py:1184-1191 fail_guard: if bridge attached,
            // raise Jump(target, values).
            // llgraph's values are concrete (self.env[box]). In Cranelift,
            // the jitframe uses recovery_layout encoding for virtuals.
            // rebuild_state_after_failure decodes this to match what the
            // bridge tracer saw via rebuild_from_resumedata (resume.py:1042).
            let bridge_guard = fail_descr.bridge_ref();
            if let Some(ref bridge) = *bridge_guard {
                let n = bridge.num_inputs.min(outputs.len());
                let bridge_inputs = outputs[..n].to_vec();
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[bridge-dispatch] fail_idx={} n={} bridge_inputs={:?} types={:?}",
                        fail_index, n, &bridge_inputs, &bridge.input_types
                    );
                }
                cur_code_ptr = bridge.code_ptr;
                cur_fail_descrs = bridge.fail_descrs.clone();
                cur_num_ref_roots = bridge.num_ref_roots;
                cur_max_output_slots = bridge.max_output_slots;

                cur_inputs = bridge_inputs;

                continue;
            }
            let _ = bridge_guard;

            // llgraph/runner.py:1192-1194 fail_guard without bridge →
            // ExecutionFinished(LLDeadFrame).
            return deadframe_from_jitframe(exec.jf_gcref, fail_descr.clone(), exec.heap_owner);
        } // end loop
    }

    ///
    /// If the bridge itself hits a guard that has another bridge attached,
    /// this chains through until a final exit is reached.
    fn execute_bridge(
        bridge: &BridgeData,
        parent_outputs: &[i64],
        parent_types: &[Type],
        attachments: &CpuDescrAttachments,
    ) -> DeadFrame {
        // The bridge's inputs are the parent guard's fail args.
        let num_bridge_inputs = bridge.num_inputs.min(parent_types.len());
        let bridge_inputs = &parent_outputs[..num_bridge_inputs];
        let exec = run_compiled_code(
            bridge.code_ptr,
            &bridge.fail_descrs,
            bridge.num_ref_roots,
            bridge.max_output_slots,
            bridge_inputs,
            attachments,
        );
        let fail_index = exec.fail_index;
        let direct_descr = exec.direct_descr.clone();
        let outputs = exec.extract_outputs(bridge.max_output_slots.max(1));

        if let Some(frame) = maybe_take_call_assembler_deadframe(fail_index, &outputs) {
            return wrap_call_assembler_deadframe_with_caller_prefix(
                frame,
                bridge.trace_id,
                bridge.header_pc,
                Some(bridge.source_guard),
                &bridge.input_types,
                bridge_inputs,
                bridge.caller_prefix_layout.as_ref(),
            );
        }

        let fail_descr_arc =
            direct_descr.unwrap_or_else(|| bridge.fail_descrs[fail_index as usize].clone());
        let fail_descr = &fail_descr_arc;

        // RPython parity: FINISH exits in bridges return directly,
        // just like in execute_with_inputs. Without this, the FINISH
        // bridge's exit is misinterpreted as a guard failure.
        if fail_descr.is_finish {
            // jf_guard_exc already written by emit_guard_exit.
            return deadframe_from_jitframe(exec.jf_gcref, fail_descr.clone(), exec.heap_owner);
        }

        fail_descr.increment_fail_count();

        let bridge_guard = fail_descr.bridge_ref();
        if let Some(ref next_bridge) = *bridge_guard {
            return Self::execute_bridge(
                next_bridge,
                &outputs,
                &fail_descr.fail_arg_types,
                attachments,
            );
        }
        let _ = bridge_guard;

        // jf_guard_exc already written by emit_guard_exit.
        deadframe_from_jitframe(exec.jf_gcref, fail_descr.clone(), exec.heap_owner)
    }

    fn do_compile(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        invalidation_flag_ptr: Option<usize>,
        source_guard: Option<(u64, u32)>,
        caller_layout: Option<&ExitRecoveryLayout>,
    ) -> Result<CompiledLoop, BackendError> {
        validate_call_assembler_rewrite_prereqs(ops)?;
        let prepared_ops = self.prepare_ops_for_compile(
            inputargs,
            ops,
            &self.constants.clone(),
            &self.constant_types.clone(),
        );
        let ops = prepared_ops.as_slice();
        // RPython parity: regalloc asserts that every Box used as an
        // argument or in fail_args is bound to a register or stack
        // location before code emission begins. The pyre/Cranelift
        // pipeline uses OpRef indices instead of Box identity, so the
        // equivalent invariant is "every referenced OpRef is either an
        // op result, an inputarg, a label arg, or a constant".
        // Violations indicate a structurally inconsistent trace
        // (e.g. unmapped short-preamble arg, see `unroll.py:404
        // _map_args`'s KeyError → InvalidLoop path) and must abort
        // compilation with `CompilationFailed` so the caller falls
        // back to the blackhole resume path. This pre-pass replaces
        // the silent `iconst(0)` fallback that previously turned
        // undefined OpRefs into runtime SIGSEGVs.
        validate_oprefs_for_compile(inputargs, ops, &self.constants)?;
        let trace_id = self.next_trace_id.take().unwrap_or_else(|| {
            let trace_id = self.trace_counter;
            self.trace_counter += 1;
            trace_id
        });
        let header_pc = self.next_header_pc.take().unwrap_or(0);
        let ptr_type = self.module.target_config().pointer_type();
        let call_conv = self.module.target_config().default_call_conv;

        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(ptr_type)); // jf_ptr (read inputs, write outputs)
        // RPython _call_footer (assembler.py:1097): mov eax, ebp; ret
        sig.returns.push(AbiParam::new(ptr_type)); // returned jf_ptr

        let func_name = format!("trace_{}", self.func_counter);
        self.func_counter += 1;

        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| BackendError::CompilationFailed(e.to_string()))?;

        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
            sig,
        );

        // Pre-scan
        let force_tokens = build_force_token_set(inputargs, ops)?;
        let mut fail_descrs: Vec<Arc<CraneliftFailDescr>> = Vec::new();
        let mut guard_infos: Vec<GuardInfo> = Vec::new();
        let mut max_output_slots: usize = 0;
        let attached_descrs = self.attached_descr_ptrs();
        collect_guards(
            ops,
            inputargs,
            &force_tokens,
            &mut fail_descrs,
            &mut guard_infos,
            &mut max_output_slots,
            trace_id,
            header_pc,
            source_guard,
            caller_layout,
            &self.constants,
            &self.constant_types,
            self.callinfocollection.as_deref(),
            attached_descrs,
        )?;
        // RPython jitframe layout parity: ref_root slots start AFTER all
        // output slots. max_output_slots must be >= inputs.len() so that
        // input loading (jf_frame[0..n]) and ref_root storage don't overlap.
        // store_final_boxes_in_guard may reduce fail_args (snapshot liveboxes)
        // below input count; ensure the minimum here.
        max_output_slots = max_output_slots.max(inputargs.len());
        let terminal_exit_layouts = collect_terminal_exit_layouts(
            ops,
            inputargs,
            &self.constant_types,
            &force_tokens,
            trace_id,
            header_pc,
            source_guard,
            caller_layout,
        )?;

        let num_inputs = inputargs.len();
        let known_values = build_known_values_set(inputargs, ops);
        let type_index = OpTypeIndex::new(inputargs, ops, &self.constant_types);
        let (type_overrides, _op_def_positions) = build_type_overrides(inputargs, ops, &type_index);
        let ref_root_slots =
            build_ref_root_slots(inputargs, ops, &type_index, &type_overrides, &force_tokens)?;
        let gc_nursery_addrs =
            with_cranelift_gc(|gc| (gc.nursery_free_addr(), gc.nursery_top_addr()));
        // llmodel.py:64-69 self.vtable_offset — backend property used by
        // bh_new_with_vtable. Capture for use in NEW_WITH_VTABLE codegen.
        let vtable_offset = self.vtable_offset;
        // llsupport/gc.py:563 GcLLDescr_framework
        //   .get_typeid_from_classptr_if_gcremovetypeptr — used by
        // _cmp_guard_class when vtable_offset is None. Reads through the
        // thread-local `CRANELIFT_ACTIVE_GC` (`llmodel.py:58`
        // `cpu.gc_ll_descr` parity), so the closure does not need a
        // reference to `self` and can be moved freely past borrow
        // conflicts with `func_ctx`.
        let gc_typeid_lookup = move |classptr: usize| -> Option<u32> {
            with_cranelift_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr))
                .flatten()
        };
        let mut defined_ref_vars: HashSet<u32> = inputargs
            .iter()
            .filter(|input| input.tp == Type::Ref && !force_tokens.contains(&input.index))
            .map(|input| input.index)
            .collect();
        let mut synced_ref_vars: HashSet<u32> = HashSet::new();
        let mut stale_ref_vars: HashSet<u32> = HashSet::new();

        // regalloc.py:1173-1213 compute_vars_longevity
        // Compute last_usage for each ref root variable. Used by get_gcmap
        // to build per-call-site gcmaps (only alive refs are marked).
        let longevity: HashMap<u32, usize> = {
            let mut m: HashMap<u32, usize> = HashMap::new();
            for (i, op) in ops.iter().enumerate() {
                for &arg in op
                    .args
                    .iter()
                    .chain(op.fail_args.iter().flat_map(|fa| fa.iter()))
                {
                    let idx = arg.raw();
                    if ref_root_slots.iter().any(|(vi, _)| *vi == idx) {
                        m.entry(idx)
                            .and_modify(|last| *last = (*last).max(i))
                            .or_insert(i);
                    }
                }
            }
            m
        };

        // Pre-scan for vector-producing ops (SIMD variable types)
        let vec_oprefs = build_vec_oprefs(ops, num_inputs);
        let vec_float_oprefs = build_vec_float_oprefs(ops, num_inputs);

        // pyjitpl.py:2283 `cpu.propagate_exception_descr` — snapshot the
        // attached descr pointer at compile time so the OpCode::CheckMemoryError
        // emitter can stamp it into `jf_descr` instead of unconditionally
        // trapping (`_build_propagate_exception_path` parity, x86/aarch64
        // assembler.py:328-345 / :559-577).  Snapshot taken before
        // `std::mem::take(&mut self.constants)` so `&self` is available.
        let propagate_exception_descr_ptr = self.attached_descr_ptrs().propagate_exception_descr;

        // Take constants out of self to avoid borrow conflicts with func_ctx
        let constants = std::mem::take(&mut self.constants);

        // Recursive tracing/bridge compilation can re-enter backend compilation
        // before an outer compile has restored its frontend context. Mirror the
        // effective RPython behavior by treating the frontend builder context as
        // compile-local state for the duration of this invocation.
        let mut func_ctx = std::mem::replace(&mut self.func_ctx, FunctionBuilderContext::new());
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // assembler.py:1080 _call_header_with_stack_check — inline
        // fast path (load stack_end/length, compare against current
        // stack slot address), calling the registered slowpath only on
        // a miss. If no stack-check addresses are registered
        // (backend-only tests), the probe is elided or falls back to
        // the combined helper.
        let initial_jf_ptr = builder.block_params(entry_block)[0];
        let stack_check_result = emit_stack_check_result(&mut builder, ptr_type, call_conv);
        let stack_overflow_block = builder.create_block();
        let stack_check_continue = builder.create_block();
        builder.ins().brif(
            stack_check_result,
            stack_overflow_block,
            &[],
            stack_check_continue,
            &[],
        );

        builder.switch_to_block(stack_overflow_block);
        builder.seal_block(stack_overflow_block);
        builder.ins().return_(&[initial_jf_ptr]);

        builder.switch_to_block(stack_check_continue);
        builder.seal_block(stack_check_continue);

        // jf_ptr serves as both inputs_ptr (entry) and the exit-frame base.
        // RPython: EBP = jitframe pointer throughout compiled code.
        // After a collecting call, _reload_frame_if_necessary reloads
        // jf_ptr from the shadow stack (GC may have moved the jitframe).
        let mut jf_ptr = initial_jf_ptr;
        let inputs_ptr = jf_ptr; // alias for entry loading

        // assembler.py:1074 _call_header_shadowstack — inline MOVs:
        //   MOV ebx, [root_stack_top_addr]
        //   MOV [ebx], 1            // is_minor marker
        //   MOV [ebx + WORD], ebp   // jf_ptr
        //   ADD ebx, 2*WORD
        //   MOV [root_stack_top_addr], ebx
        emit_call_header_shadowstack(&mut builder, ptr_type, jf_ptr);
        // Ref root slots live in jf_frame after output/fail_args area.
        // RPython: refs are always in jf_frame; gcmap marks which slots
        // are live at each GC point (regalloc.py get_gcmap).
        let ref_root_base_ofs = JF_FRAME_ITEM0_OFS + (max_output_slots as i32) * 8;
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[codegen] max_output={} ref_roots={} longevity={:?}",
                max_output_slots,
                ref_root_slots.len(),
                longevity,
            );
        }
        let debug_declares = std::env::var_os("MAJIT_DEBUG_DECLARES").is_some();

        let label_indices: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter_map(|(idx, op)| (op.opcode == OpCode::Label).then_some(idx))
            .collect();
        let has_entry_label = label_indices.first().copied() == Some(0);

        // descr.index() → label op's source-arity (args.len()).  Captured up
        // front because Cranelift's FunctionBuilder may auto-promote `def_var`
        // chains into block params later, growing the block's runtime
        // num_block_params past the original label arity. Used by the
        // OpCode::Jump handler to detect arity-mismatched local jumps and
        // lower them as external jumps (rewriter.py LABEL/JUMP redirect parity).
        let label_arity_by_descr: HashMap<u32, usize> = label_indices
            .iter()
            .filter_map(|&li| {
                ops[li]
                    .descr
                    .as_ref()
                    .map(|d| (d.index(), ops[li].args.len()))
            })
            .collect();

        if std::env::var_os("MAJIT_DUMP_CLIF").is_some() {
            for &li in &label_indices {
                eprintln!(
                    "[label-dump] idx={} args.len={} args={:?}",
                    li,
                    ops[li].args.len(),
                    ops[li].args
                );
            }
        }

        // Body-direct entry disabled — requires bridge to provide full body args.
        let body_direct_num_inputs = 0;

        // Collect all variable declarations into a map (index -> type)
        // before declaring them sequentially. Cranelift 0.130 declare_var
        // returns auto-assigned indices, so we must declare in order 0..max.
        let mut var_types: std::collections::HashMap<u32, cranelift_codegen::ir::Type> =
            std::collections::HashMap::new();
        let mut declared_vars = std::collections::HashSet::new();

        // Always declare inputarg variables, even when the trace starts
        // with a LABEL (preamble peeling). Preamble guards reference
        // inputarg OpRefs in fail_args and need declared variables.
        // The LABEL block later overrides these via block params + def_var.
        //
        // Box Identity Phase E.2b: bridge inputargs carry OpRefs in the
        // disjoint `[bridge_inputarg_base..)` namespace
        // (`prepare_bridge_trace_for_optimizer` in pyjitpl/mod.rs); use
        // each `InputArg.index` as the variable slot rather than a dense
        // `0..num_inputs` so the entry `def_var` lands on the slot the
        // ops actually reference.
        for ia in inputargs {
            if debug_declares {
                eprintln!("[jit][declare] input var{}", ia.index);
            }
            var_types.insert(ia.index, cl_types::I64);
            declared_vars.insert(ia.index);
        }
        // Declare variables for op results
        for (op_idx, op) in ops.iter().enumerate() {
            if op.result_type() != Type::Void {
                let vi = op_var_index(op, op_idx, num_inputs);
                if declared_vars.contains(&(vi as u32)) {
                    continue; // already declared
                }
                declared_vars.insert(vi as u32);
                let cl_type = if vec_oprefs.contains(&(vi as u32)) {
                    if is_vec_float_producing(op.opcode) {
                        cl_types::F64X2
                    } else {
                        cl_types::I64X2
                    }
                } else {
                    cl_types::I64
                };
                if debug_declares {
                    eprintln!("[jit][declare] op-result var{} opcode={:?}", vi, op.opcode);
                }
                var_types.insert(vi as u32, cl_type);
            }
            // Declare ALL referenced OpRefs: fail_args, op args, etc.
            for &arg in op
                .args
                .iter()
                .chain(op.fail_args.iter().flat_map(|fa| fa.iter()))
            {
                if !arg.is_none()
                    && !declared_vars.contains(&arg.raw())
                    && !constants.contains_key(&arg.raw())
                {
                    declared_vars.insert(arg.raw());
                    if debug_declares {
                        eprintln!(
                            "[jit][declare] ref-arg var{} owner={:?}",
                            arg.raw(),
                            op.opcode
                        );
                    }
                    var_types.insert(arg.raw(), cl_types::I64);
                }
            }
        }
        // RPython parity: LABEL args are independent box names bound by the
        // incoming JUMP, not necessarily inputargs or local op results.
        // The backend must declare them before the corresponding block param
        // is assigned in the LABEL handling path.
        for op in ops {
            if op.opcode != OpCode::Label {
                continue;
            }
            for &arg in &op.args {
                if arg.is_none() || declared_vars.contains(&arg.raw()) {
                    continue;
                }
                declared_vars.insert(arg.raw());
                if debug_declares {
                    eprintln!("[jit][declare] label-arg var{}", arg.raw());
                }
                var_types.insert(arg.raw(), cl_types::I64);
            }
        }

        // Compute loop_param_count early so legacy vars are included.
        let loop_param_count = if let Some(&li) = label_indices.last() {
            ops[li].args.len()
        } else {
            ops.iter()
                .rev()
                .find(|op| op.opcode == OpCode::Jump)
                .map(|op| op.args.len())
                .unwrap_or(num_inputs)
        };

        // Legacy no-LABEL traces can still need synthetic loop params.
        if label_indices.is_empty() && loop_param_count > num_inputs {
            for i in num_inputs..loop_param_count {
                if !declared_vars.contains(&(i as u32)) {
                    var_types.insert(i as u32, cl_types::I64);
                    declared_vars.insert(i as u32);
                }
            }
        }

        // regalloc.py:140-181 RegisterManager parity: build a sparse
        // OpRef → Variable map by declaring exactly the OpRefs that
        // var_types contains. Cranelift 0.130 declare_var(ty) issues
        // sequential Variable indices (0, 1, 2, ...) regardless of the
        // OpRef value, so we capture the returned Variable per OpRef into
        // OPREF_VAR_MAP. Iterate keys in sorted order so the resulting
        // index assignment is deterministic across runs.
        let mut var_keys: Vec<u32> = var_types.keys().copied().collect();
        var_keys.sort_unstable();
        let mut opref_var_map: std::collections::HashMap<u32, Variable> =
            std::collections::HashMap::with_capacity(var_keys.len());
        for opref_idx in var_keys {
            let ty = var_types.get(&opref_idx).copied().unwrap_or(cl_types::I64);
            let returned_var = builder.declare_var(ty);
            opref_var_map.insert(opref_idx, returned_var);
        }
        // Install the map BEFORE any var() lookup so subsequent calls
        // (jf_ptr_var, def_var loops, helper functions) see the dense
        // assignment. The Drop guard restores the previous mapping on
        // every return path so nested compiles (bridge compilation
        // re-entry) keep their own scoped maps.
        let saved_map = OPREF_VAR_MAP.with(|cell| cell.borrow_mut().replace(opref_var_map));
        let _opref_var_guard = OprefVarMapGuard { saved: saved_map };

        // RPython parity: EBP holds the jitframe pointer throughout the
        // assembled trace. After every collecting call, _reload_frame_if_necessary
        // (assembler.py:405-412) reloads ebp from the shadow stack — this is
        // a single mutable register, NOT new SSA values per reload site.
        // Cranelift's SSA model would otherwise require us to thread the
        // reloaded jf_ptr through every merge block manually; using a
        // Variable lets the FunctionBuilder insert the necessary block
        // parameters automatically so v66 (post-reload) does not get
        // referenced from a path that did not perform the reload.
        let jf_ptr_var = builder.declare_var(ptr_type);
        builder.def_var(jf_ptr_var, jf_ptr);

        // Debug: save declared_vars snapshot for resolve_opref checking.
        DECLARED_VARS_DEBUG.with(|cell| {
            *cell.borrow_mut() = Some(declared_vars.clone());
        });

        // Save op-result positions for resolve_opref collision handling.
        let mut op_result_positions = std::collections::HashSet::new();
        for ia in inputargs {
            op_result_positions.insert(ia.index);
        }
        for (op_idx, op) in ops.iter().enumerate() {
            if op.result_type() != Type::Void {
                let vi = op_var_index(op, op_idx, num_inputs) as u32;
                op_result_positions.insert(vi);
            }
        }
        OP_RESULT_VARS.with(|cell| {
            *cell.borrow_mut() = Some(op_result_positions);
        });

        let max_entry_inputs = num_inputs.max(body_direct_num_inputs);
        let load_count = if body_direct_num_inputs > 0 {
            max_entry_inputs + 1 // extra slot for entry_mode flag
        } else {
            num_inputs
        };
        let entry_input_vals: Vec<CValue> = (0..load_count)
            .map(|i| {
                let offset = JF_FRAME_ITEM0_OFS + (i as i32) * 8;
                builder
                    .ins()
                    .load(cl_types::I64, MemFlags::trusted(), inputs_ptr, offset)
            })
            .collect();

        // Always def_var inputargs in the entry block. When the trace
        // starts with LABEL, the LABEL block will override these via
        // block params. Preamble guards need the entry block values.
        //
        // Box Identity Phase E.2b: route the loaded jitframe slot value
        // into the variable indexed by `inputargs[i].index` so bridge
        // entries land in the `[bridge_inputarg_base..)` slot the ops
        // reference. Slots beyond `num_inputs` (synthesized loop padding
        // / `entry_mode` flag) keep the dense `var(i)` mapping.
        for (i, val) in entry_input_vals.iter().copied().enumerate() {
            let slot = if i < num_inputs {
                inputargs[i].index
            } else {
                i as u32
            };
            builder.def_var(var(slot), val);
            if i < num_inputs {
                sync_ref_root_var(
                    &mut builder,
                    jf_ptr,
                    &ref_root_slots,
                    slot,
                    val,
                    ref_root_base_ofs,
                    &mut synced_ref_vars,
                );
            }
        }

        // RPython parity: GC roots are registered by run_compiled_code()
        // before calling into compiled code, and unregistered after return.
        // No per-call registration/unregistration inside the compiled code.
        // Per-call spill/reload keeps the roots array up-to-date.

        // The jitframe allocation path is zero-filled (gc.py:30-34
        // malloc_zero_filled, pyre Nursery::new/reset).  RPython relies on
        // that allocator property instead of clearing every frame slot in
        // the trace prologue.  Undefined ref-root slots are also absent from
        // per-call gcmaps, so the GC will not trace them.

        // RPython compile.py sends loops to the backend as:
        // [start_label] + preamble_ops + [loop_label] + body_ops
        // and JUMPs target LABEL descrs. Model that directly by creating
        // a Cranelift block per LABEL descr.

        let mut label_blocks = Vec::with_capacity(label_indices.len());
        let mut label_blocks_by_descr = HashMap::new();
        for &label_idx in &label_indices {
            let block = builder.create_block();
            for _ in 0..ops[label_idx].args.len() {
                builder.append_block_param(block, cl_types::I64);
            }
            if let Some(descr_index) = ops[label_idx].descr.as_ref().map(|descr| descr.index()) {
                label_blocks_by_descr.insert(descr_index, block);
            }
            label_blocks.push((label_idx, block));
        }

        // RPython backend: Label ops define loop blocks.
        // Linear traces (no Label, no Jump) stay in the entry block.
        // Bridge traces (source_guard.is_some()) never self-loop — their
        // JUMP targets a different trace (the main loop). RPython compiles
        // bridges as linear code with a tail-call to the loop entry; no
        // separate loop block is needed. Creating an unnecessary loop_block
        // with padded zero parameters (for positions num_inputs..JUMP-args)
        // causes block-parameter transfer issues in the register allocator.
        let has_jump = ops.iter().any(|op| op.opcode == OpCode::Jump);
        let is_bridge = source_guard.is_some();
        let loop_block = if !label_blocks.is_empty() {
            label_blocks.last().map(|(_, block)| *block).unwrap()
        } else if has_jump && !is_bridge {
            // Legacy no-Label trace with Jump: need a loop block
            let block = builder.create_block();
            for _ in 0..loop_param_count {
                builder.append_block_param(block, cl_types::I64);
            }
            block
        } else {
            // Linear trace or bridge: no loop block needed, stay in entry_block
            entry_block
        };

        let mut guard_idx: usize = 0;
        let mut last_ovf_flag: Option<CValue> = None;
        let _nursery_inline_count: usize = 0; // reserved for future bridge inline support

        // RPython x86 backend parity: preamble runs ONCE per trace entry;
        // body runs many times. Mark preamble blocks cold so Cranelift
        // places body at a stable offset from function entry, matching
        // RPython's linear emission where body label is immediately after
        // the function prologue and preamble code is deferred to the end.
        // Tracks whether the current op-emission phase is in preamble.
        let mut preamble_phase = label_blocks.len() >= 2;
        if preamble_phase {
            if let Some(&(_, first_label_block)) = label_blocks.first() {
                builder.set_cold_block(first_label_block);
            }
        }

        let mut first_label_entered_at_entry = false;
        if let Some(&(entry_label_idx, entry_label_block)) = label_blocks.first() {
            if body_direct_num_inputs > 0 && label_blocks.len() >= 2 {
                // Dual entry: branch on entry_mode (last input slot).
                // entry_mode == 0 → preamble, entry_mode != 0 → body-direct.
                // brif directly targets label blocks with their params.
                let entry_mode = entry_input_vals[max_entry_inputs];
                let body_block = label_blocks.last().unwrap().1;

                let body_args = block_args(&entry_input_vals[..body_direct_num_inputs]);
                let preamble_args = block_args(&entry_input_vals[..num_inputs]);
                builder.ins().brif(
                    entry_mode,
                    body_block,
                    &body_args,
                    entry_label_block,
                    &preamble_args,
                );

                // Continue with preamble label block for var binding
                builder.switch_to_block(entry_label_block);
                for (i, &arg_ref) in ops[entry_label_idx].args.iter().enumerate() {
                    let param = builder.block_params(entry_label_block)[i];
                    if !arg_ref.is_none() {
                        builder.def_var(var(arg_ref.raw()), param);
                        let cur_jf = builder.use_var(jf_ptr_var);
                        sync_ref_root_var(
                            &mut builder,
                            cur_jf,
                            &ref_root_slots,
                            arg_ref.raw(),
                            param,
                            ref_root_base_ofs,
                            &mut synced_ref_vars,
                        );
                    }
                }
                first_label_entered_at_entry = true;
            } else if has_entry_label {
                let vals: Vec<CValue> = entry_input_vals.clone();
                builder.ins().jump(entry_label_block, &block_args(&vals));
                builder.switch_to_block(entry_label_block);
                for (i, &arg_ref) in ops[entry_label_idx].args.iter().enumerate() {
                    let param = builder.block_params(entry_label_block)[i];
                    if !arg_ref.is_none() {
                        builder.def_var(var(arg_ref.raw()), param);
                        let cur_jf = builder.use_var(jf_ptr_var);
                        sync_ref_root_var(
                            &mut builder,
                            cur_jf,
                            &ref_root_slots,
                            arg_ref.raw(),
                            param,
                            ref_root_base_ofs,
                            &mut synced_ref_vars,
                        );
                    }
                }
                first_label_entered_at_entry = true;
            }
        } else if has_jump && loop_block != entry_block {
            let zero = builder.ins().iconst(cl_types::I64, 0);
            let vals: Vec<CValue> = (0..loop_param_count)
                .map(|i| {
                    if i < num_inputs {
                        builder.use_var(var(i as u32))
                    } else {
                        zero
                    }
                })
                .collect();
            builder.ins().jump(loop_block, &block_args(&vals));
            builder.switch_to_block(loop_block);
            for i in 0..loop_param_count {
                let param = builder.block_params(loop_block)[i];
                builder.def_var(var(i as u32), param);
                let cur_jf = builder.use_var(jf_ptr_var);
                sync_ref_root_var(
                    &mut builder,
                    cur_jf,
                    &ref_root_slots,
                    i as u32,
                    param,
                    ref_root_base_ofs,
                    &mut synced_ref_vars,
                );
            }
        }
        // else: linear trace — already in entry_block with vars defined

        for op_idx in 0..ops.len() {
            if let Some((_, label_block)) = label_blocks
                .iter()
                .find(|(label_idx, _)| *label_idx == op_idx)
            {
                // The first LABEL is already the current block only when
                // trace entry jumped there above. Bridge/retrace fragments can
                // have real ops before their first LABEL; those ops must run
                // before the LABEL args are bound, matching RPython's linear
                // trace order.
                let first_label_idx = label_blocks.first().map(|(label_idx, _)| *label_idx);
                let label_already_current =
                    Some(op_idx) == first_label_idx && first_label_entered_at_entry;
                if !label_already_current {
                    let prev_terminated = op_idx
                        .checked_sub(1)
                        .and_then(|prev_idx| ops.get(prev_idx))
                        .map(|prev| prev.opcode == OpCode::Jump || prev.opcode == OpCode::Finish)
                        .unwrap_or(false);
                    if !prev_terminated {
                        let vals: Vec<CValue> = ops[op_idx]
                            .args
                            .iter()
                            .map(|&r| resolve_opref(&mut builder, &constants, r))
                            .collect();
                        builder.ins().jump(*label_block, &block_args(&vals));
                    }
                    builder.switch_to_block(*label_block);
                    for (i, &arg_ref) in ops[op_idx].args.iter().enumerate() {
                        let param = builder.block_params(*label_block)[i];
                        if !arg_ref.is_none() && !constants.contains_key(&arg_ref.raw()) {
                            builder.def_var(var(arg_ref.raw()), param);
                            let cur_jf = builder.use_var(jf_ptr_var);
                            sync_ref_root_var(
                                &mut builder,
                                cur_jf,
                                &ref_root_slots,
                                arg_ref.raw(),
                                param,
                                ref_root_base_ofs,
                                &mut synced_ref_vars,
                            );
                        }
                    }
                    // Entering the body LABEL (not the first label): body is
                    // the hot phase, any subsequent cont_blocks are hot.
                    preamble_phase = false;
                }
                continue;
            }
            let op = &ops[op_idx];
            let vi = op_var_index(op, op_idx, num_inputs) as u32;

            // RPython parity: ebp is the live register at every instruction
            // boundary. Refresh the cached jf_ptr CValue from
            // the Cranelift Variable so the FunctionBuilder threads the
            // correct value through any merge blocks introduced by the
            // previous opcode (LABEL, brif, etc.). Without this, opcode
            // handlers that emit IR directly using the cached locals can
            // reference an SSA value defined in a non-dominating block.
            jf_ptr = builder.use_var(jf_ptr_var);

            // regalloc.py:1089-1106 get_gcmap: per-call-site gcmap
            // marking only alive ref root slots at this position.
            let per_call_gcmap = get_gcmap(
                op_idx,
                max_output_slots,
                &ref_root_slots,
                &longevity,
                &defined_ref_vars,
            );
            let live_ref_root_slots =
                live_ref_root_slots_at(op_idx, &ref_root_slots, &longevity, &defined_ref_vars);

            match op.opcode {
                // ── Integer arithmetic ──
                OpCode::IntAdd => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().iadd(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntSub => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().isub(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntMul => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().imul(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntFloorDiv => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().sdiv(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntMod => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().srem(a, b);
                    builder.def_var(var(vi), r);
                }

                // ── Overflow arithmetic ──
                // Compute the result normally, then detect signed overflow
                // using the bit-manipulation formula:
                //   ovf = ((a ^ result) & (b ^ result)) >> 63   [for add]
                //   ovf = ((a ^ result) & ((a ^ b))) >> 63      [for sub]
                // For mul we use a widening approach: sign-extend to i128
                // then check if the result fits in i64.
                OpCode::IntAddOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().iadd(a, b);
                    builder.def_var(var(vi), r);
                    // ovf = ((a ^ r) & (b ^ r)) >> 63
                    let axr = builder.ins().bxor(a, r);
                    let bxr = builder.ins().bxor(b, r);
                    let both = builder.ins().band(axr, bxr);
                    let ovf = builder.ins().sshr_imm(both, 63);
                    last_ovf_flag = Some(ovf);
                }
                OpCode::IntSubOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().isub(a, b);
                    builder.def_var(var(vi), r);
                    // ovf = ((a ^ r) & (a ^ b)) >> 63
                    let axr = builder.ins().bxor(a, r);
                    let axb = builder.ins().bxor(a, b);
                    let both = builder.ins().band(axr, axb);
                    let ovf = builder.ins().sshr_imm(both, 63);
                    last_ovf_flag = Some(ovf);
                }
                OpCode::IntMulOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().imul(a, b);
                    builder.def_var(var(vi), r);
                    // x86 IMUL sets OF when the 128-bit signed product doesn't
                    // fit in 64 bits (assembler.py:1864-1866). Mirror that by
                    // comparing the high 64 bits against the sign-extension of
                    // the low 64 bits: ovf <=> smulhi(a,b) != (r >>s 63).
                    let hi = builder.ins().smulhi(a, b);
                    let sign = builder.ins().sshr_imm(r, 63);
                    let differ = builder.ins().icmp(IntCC::NotEqual, hi, sign);
                    let ovf = builder.ins().uextend(cl_types::I64, differ);
                    last_ovf_flag = Some(ovf);
                }
                OpCode::IntAnd => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().band(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntOr => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().bor(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntXor => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().bxor(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntLshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().ishl(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntRshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().sshr(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::UintRshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().ushr(a, b);
                    builder.def_var(var(vi), r);
                }

                // ── Unary integer ──
                OpCode::IntNeg => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let r = builder.ins().ineg(a);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntInvert => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let r = builder.ins().bnot(a);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntIsZero => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::Equal, a, zero);
                    let r = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntIsTrue => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::NotEqual, a, zero);
                    let r = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntForceGeZero => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, zero);
                    let r = builder.ins().select(cmp, zero, a);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntBetween => {
                    // int_between(a, b, c) => a <= b < c
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let b = resolve_opref(&mut builder, &constants, op.arg(1));
                    let c = resolve_opref(&mut builder, &constants, op.arg(2));
                    let cmp1 = builder.ins().icmp(IntCC::SignedLessThanOrEqual, a, b);
                    let cmp2 = builder.ins().icmp(IntCC::SignedLessThan, b, c);
                    let both = builder.ins().band(cmp1, cmp2);
                    let r = builder.ins().uextend(cl_types::I64, both);
                    builder.def_var(var(vi), r);
                }

                // ── Integer comparisons ──
                OpCode::IntLt => emit_icmp(&mut builder, &constants, IntCC::SignedLessThan, op, vi),
                OpCode::IntLe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::SignedLessThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::IntEq => emit_icmp(&mut builder, &constants, IntCC::Equal, op, vi),
                OpCode::IntNe => emit_icmp(&mut builder, &constants, IntCC::NotEqual, op, vi),
                OpCode::IntGt => {
                    emit_icmp(&mut builder, &constants, IntCC::SignedGreaterThan, op, vi)
                }
                OpCode::IntGe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::SignedGreaterThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::UintLt => {
                    emit_icmp(&mut builder, &constants, IntCC::UnsignedLessThan, op, vi)
                }
                OpCode::UintLe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::UnsignedLessThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::UintGt => {
                    emit_icmp(&mut builder, &constants, IntCC::UnsignedGreaterThan, op, vi)
                }
                OpCode::UintGe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::UnsignedGreaterThanOrEqual,
                    op,
                    vi,
                ),

                // ── Pointer comparisons ──
                OpCode::PtrEq | OpCode::InstancePtrEq => {
                    emit_icmp(&mut builder, &constants, IntCC::Equal, op, vi)
                }
                OpCode::PtrNe | OpCode::InstancePtrNe => {
                    emit_icmp(&mut builder, &constants, IntCC::NotEqual, op, vi)
                }

                // ── Float comparisons ──
                OpCode::FloatLt => emit_fcmp(&mut builder, &constants, FloatCC::LessThan, op, vi),
                OpCode::FloatLe => {
                    emit_fcmp(&mut builder, &constants, FloatCC::LessThanOrEqual, op, vi)
                }
                OpCode::FloatEq => emit_fcmp(&mut builder, &constants, FloatCC::Equal, op, vi),
                OpCode::FloatNe => emit_fcmp(&mut builder, &constants, FloatCC::NotEqual, op, vi),
                OpCode::FloatGt => {
                    emit_fcmp(&mut builder, &constants, FloatCC::GreaterThan, op, vi)
                }
                OpCode::FloatGe => emit_fcmp(
                    &mut builder,
                    &constants,
                    FloatCC::GreaterThanOrEqual,
                    op,
                    vi,
                ),

                // ── Identity / cast ──
                OpCode::SameAsI
                | OpCode::SameAsR
                | OpCode::SameAsF
                | OpCode::CastPtrToInt
                | OpCode::CastIntToPtr
                | OpCode::CastOpaquePtr => {
                    let a = if op.num_args() > 0 {
                        resolve_opref(&mut builder, &constants, op.arg(0))
                    } else if let Some(&c) = constants.get(&vi) {
                        builder.ins().iconst(cl_types::I64, c)
                    } else {
                        builder.ins().iconst(cl_types::I64, 0)
                    };
                    builder.def_var(var(vi), a);
                    if op.opcode == OpCode::SameAsR {
                        let cur_jf = builder.use_var(jf_ptr_var);
                        sync_ref_root_var(
                            &mut builder,
                            cur_jf,
                            &ref_root_slots,
                            vi,
                            a,
                            ref_root_base_ofs,
                            &mut synced_ref_vars,
                        );
                    }
                }

                // ── Guards ──
                OpCode::GuardTrue
                | OpCode::GuardFalse
                | OpCode::GuardNonnull
                | OpCode::GuardIsnull => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);

                    let exit_on_zero =
                        matches!(op.opcode, OpCode::GuardTrue | OpCode::GuardNonnull);
                    if exit_on_zero {
                        builder
                            .ins()
                            .brif(is_zero, exit_block, &[], cont_block, &[]);
                    } else {
                        builder
                            .ins()
                            .brif(is_zero, cont_block, &[], exit_block, &[]);
                    }

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardValue => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }

                    let neq = builder.ins().icmp(IntCC::NotEqual, a, b);
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardClass => {
                    // x86/assembler.py:1880-1891 _cmp_guard_class:
                    //   offset = self.cpu.vtable_offset
                    //   if offset is not None:
                    //       CMP(mem(loc_ptr, offset), loc_classptr)
                    //   else:
                    //       _cmp_guard_gc_type(loc_ptr, expected_typeid)
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (obj, expected_class) = resolve_binop(&mut builder, &constants, op);
                    // x86/assembler.py:1887 assert isinstance(loc_classptr, ImmedLoc)
                    // — pre-fetch the classptr immediate from the constant pool.
                    let expected_classptr_imm = constants.get(&op.arg(1).raw()).copied();
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }

                    let neq = emit_cmp_guard_class(
                        &mut builder,
                        ptr_type,
                        obj,
                        expected_class,
                        expected_classptr_imm,
                        vtable_offset,
                        Some(&gc_typeid_lookup),
                    );
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardNonnullClass => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (obj, expected_class) = resolve_binop(&mut builder, &constants, op);
                    let expected_classptr_imm = constants.get(&op.arg(1).raw()).copied();
                    let zero = builder.ins().iconst(ptr_type, 0);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let class_check_block = builder.create_block();
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }

                    let is_null = builder.ins().icmp(IntCC::Equal, obj, zero);
                    builder
                        .ins()
                        .brif(is_null, exit_block, &[], class_check_block, &[]);

                    builder.switch_to_block(class_check_block);
                    builder.seal_block(class_check_block);
                    // x86/assembler.py:1880-1891 _cmp_guard_class via vtable_offset.
                    let neq = emit_cmp_guard_class(
                        &mut builder,
                        ptr_type,
                        obj,
                        expected_class,
                        expected_classptr_imm,
                        vtable_offset,
                        Some(&gc_typeid_lookup),
                    );
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardNoException => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    // Direct memory load from global JIT_EXC_VALUE (no TLS/host call).
                    // RPython backend fuses this into the preceding CALL as an
                    // inline flag check. We use a plain load from a known address.
                    let exc_addr = builder.ins().iconst(ptr_type, jit_exc_value_addr() as i64);
                    let exc_val = builder.ins().load(
                        cl_types::I64,
                        cranelift_codegen::ir::MemFlags::trusted(),
                        exc_addr,
                        0,
                    );
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let has_exc = builder.ins().icmp(IntCC::NotEqual, exc_val, zero);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder
                        .ins()
                        .brif(has_exc, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardException => {
                    // x86/assembler.py:1808-1815 genop_guard_guard_exception:
                    //   MOV loc1, [pos_exception]
                    //   CMP loc1, expected
                    //   guard on E (equal)
                    //   _store_and_reset_exception → resloc = [pos_exc_value];
                    //     [pos_exception] = 0; [pos_exc_value] = 0
                    // All inline loads/stores, no host calls.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let expected_type = resolve_opref(&mut builder, &constants, op.arg(0));
                    // Inline: load pos_exception (exc type)
                    let exc_type_addr = builder.ins().iconst(ptr_type, jit_exc_type_addr() as i64);
                    let exc_type =
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), exc_type_addr, 0);
                    // CMP exc_type, expected
                    let mismatch = builder.ins().icmp(IntCC::NotEqual, exc_type, expected_type);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder
                        .ins()
                        .brif(mismatch, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);

                    // _store_and_reset_exception parity (inline):
                    //   resloc = [pos_exc_value]
                    //   [pos_exception] = 0
                    //   [pos_exc_value] = 0
                    let exc_val_addr = builder.ins().iconst(ptr_type, jit_exc_value_addr() as i64);
                    let exc_val =
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), exc_val_addr, 0);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero, exc_type_addr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero, exc_val_addr, 0);
                    let vi = op_var_index(op, op_idx, inputargs.len());
                    builder.def_var(var(vi as u32), exc_val);
                }

                OpCode::GuardNoOverflow => {
                    // RPython intbounds.py:217-220: if the preceding op
                    // is not an overflow op, the guard is redundant (the
                    // overflow was already proven impossible or optimized away).
                    if last_ovf_flag.is_none() {
                        guard_idx += 1; // consume guard_info slot
                        continue;
                    }
                    // Side-exit if overflow DID occur (ovf != 0).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let ovf = last_ovf_flag.take().unwrap();
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, ovf, zero);
                    // If ovf == 0 (no overflow), continue; otherwise side-exit.
                    builder
                        .ins()
                        .brif(is_zero, cont_block, &[], exit_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }
                OpCode::GuardOverflow => {
                    // Side-exit if overflow did NOT occur (ovf == 0).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let ovf = last_ovf_flag
                        .take()
                        .expect("GuardOverflow without preceding overflow op");
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, ovf, zero);
                    // If ovf == 0 (no overflow), side-exit; otherwise continue.
                    builder
                        .ins()
                        .brif(is_zero, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardNotForced => {
                    // x86/assembler.py:2228-2232 genop_guard_guard_not_forced:
                    //   ofs = self.cpu.get_ofs_of_frame_field('jf_descr')
                    //   self.mc.CMP_bi(ofs, 0)
                    //   self.guard_success_cc = rx86.Conditions['E']
                    //   self.implement_guard(guard_token)
                    // Pairing validation is done by CALL_MAY_FORCE (forward
                    // lookup, _find_nearby_operation(+1) parity).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let jf_descr = builder.ins().load(
                        cl_types::I64,
                        MemFlags::trusted(),
                        jf_ptr,
                        JF_DESCR_OFS,
                    );
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_forced = builder.ins().icmp(IntCC::NotEqual, jf_descr, zero);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder
                        .ins()
                        .brif(is_forced, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }
                OpCode::GuardNotForced2 => {
                    // x86/assembler.py:2662-2669 store_force_descr:
                    //   guard_token = implement_guard_recovery(...)
                    //   _store_force_index(op)
                    //   store_info_on_descr(0, guard_token)
                    // x86/regalloc.py:1411-1417 consider_guard_not_forced_2:
                    //   assembler.store_force_descr(op, fail_locs, frame_depth)
                    //
                    // RPython's recovery code (_update_at_exit) writes fail_arg
                    // values from registers to jf_frame[rd_locs[i]]. In majit,
                    // emit_guard_exit does this for inline guards. For
                    // GUARD_NOT_FORCED_2 (no inline branch), we write fail_args
                    // to jf_frame[0..n] here so force() sees correct values.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    // _store_force_index(op): store descr to jf_force_descr
                    let descr_val = builder.ins().iconst(cl_types::I64, info.fail_descr_ptr);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), descr_val, cur_jf, JF_FORCE_DESCR_OFS);

                    // implement_guard_recovery / _update_at_exit parity:
                    // Write fail_arg values to jf_frame[0..n] in fail_args order.
                    for (index, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
                        let raw = resolve_failarg_opref(
                            &mut builder,
                            &constants,
                            cur_jf,
                            &ref_root_slots,
                            &stale_ref_vars,
                            ref_root_base_ofs,
                            arg_ref,
                        );
                        builder.ins().store(
                            MemFlags::trusted(),
                            raw,
                            cur_jf,
                            JF_FRAME_ITEM0_OFS + (index as i32) * 8,
                        );
                    }
                    if info.gcmap != 0 {
                        emit_jitframe_write_barrier(&mut builder, ptr_type, call_conv, cur_jf);
                    }
                }

                OpCode::GuardNotInvalidated => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    if let Some(flag_addr) = invalidation_flag_ptr {
                        // Load the invalidation flag (AtomicBool, 1 byte) from
                        // the known address baked into the generated code.
                        let addr_val = builder.ins().iconst(cl_types::I64, flag_addr as i64);
                        let flag_val =
                            builder
                                .ins()
                                .load(cl_types::I8, MemFlags::trusted(), addr_val, 0);
                        let zero = builder.ins().iconst(cl_types::I8, 0);
                        let is_invalidated = builder.ins().icmp(IntCC::NotEqual, flag_val, zero);

                        let exit_block = builder.create_block();
                        builder.set_cold_block(exit_block);
                        let cont_block = builder.create_block();
                        if preamble_phase {
                            builder.set_cold_block(cont_block);
                        }

                        builder
                            .ins()
                            .brif(is_invalidated, exit_block, &[], cont_block, &[]);

                        builder.switch_to_block(exit_block);
                        builder.seal_block(exit_block);
                        let cur_jf = builder.use_var(jf_ptr_var);
                        emit_guard_exit(
                            &mut builder,
                            &constants,
                            cur_jf,
                            info,
                            &ref_root_slots,
                            &stale_ref_vars,
                            ref_root_base_ofs,
                            ptr_type,
                            call_conv,
                        );

                        builder.switch_to_block(cont_block);
                        builder.seal_block(cont_block);
                    }
                }

                OpCode::GuardFutureCondition => {
                    // Future condition: the guard condition is computed lazily.
                    // For now, behave like GuardTrue on args[0].
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let cond = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_false = builder.ins().icmp(IntCC::Equal, cond, zero);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder
                        .ins()
                        .brif(is_false, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardAlwaysFails => {
                    // Always-failing guard: unconditionally side-exit.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    // Create a continuation block for subsequent ops (dead code).
                    let dead_block = builder.create_block();
                    builder.switch_to_block(dead_block);
                    builder.seal_block(dead_block);
                }

                OpCode::GuardGcType => {
                    // args[0] = object ref, args[1] = expected type_id (as Int)
                    // Load the GC header (8 bytes before the obj pointer),
                    // extract the lower 32 bits (type_id), and compare.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let obj_ptr = resolve_opref(&mut builder, &constants, op.args[0]);
                    let expected_tid = resolve_opref(&mut builder, &constants, op.args[1]);

                    // Load header word from obj_ptr - GcHeader::SIZE
                    let hdr_addr = builder.ins().iadd_imm(obj_ptr, -(GcHeader::SIZE as i64));
                    let hdr_word =
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), hdr_addr, 0);
                    // Extract type_id (lower 32 bits)
                    let tid_mask = builder.ins().iconst(cl_types::I64, TYPE_ID_MASK as i64);
                    let actual_tid = builder.ins().band(hdr_word, tid_mask);

                    let neq = builder
                        .ins()
                        .icmp(IntCC::NotEqual, actual_tid, expected_tid);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardIsObject => {
                    // x86/assembler.py:1924-1943 genop_guard_guard_is_object.
                    //     assert self.cpu.supports_guard_gc_type
                    //     [loc_object, loc_typeid] = locs
                    //     if IS_X86_32:
                    //         self.mc.MOVZX16(loc_typeid, mem(loc_object, 0))
                    //     else:
                    //         self.mc.MOV32(loc_typeid, mem(loc_object, 0))
                    //     base_type_info, shift_by, sizeof_ti = (
                    //         self.cpu.gc_ll_descr
                    //             .get_translated_info_for_typeinfo())
                    //     infobits_offset, IS_OBJECT_FLAG = (
                    //         self.cpu.gc_ll_descr
                    //             .get_translated_info_for_guard_is_object())
                    //     loc_infobits = addr_add(imm(base_type_info),
                    //                             loc_typeid,
                    //                             scale=shift_by,
                    //                             offset=infobits_offset)
                    //     self.mc.TEST8(loc_infobits, imm(IS_OBJECT_FLAG))
                    //     self.guard_success_cc = rx86.Conditions['NZ']
                    //     self.implement_guard(guard_token)
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    if !cranelift_gc_active() {
                        return Err(missing_gc_runtime(op.opcode));
                    }
                    // assembler.py:1925 assert self.cpu.supports_guard_gc_type
                    assert!(
                        with_cranelift_gc_required(|gc| gc.supports_guard_gc_type()),
                        "x86/assembler.py:1925: assert self.cpu.\
                         supports_guard_gc_type (GcAllocator has not \
                         installed a TYPE_INFO layout)"
                    );

                    let loc_object = resolve_opref(&mut builder, &constants, op.args[0]);
                    // assembler.py:1931-1932 MOV32 loc_typeid, mem(loc_object, 0).
                    // majit's GC header sits at `obj - GcHeader::SIZE`
                    // (see the GuardGcType arm above); the typeid occupies
                    // the lower `TYPE_ID_BITS` of that header word.
                    let hdr_addr = builder.ins().iadd_imm(loc_object, -(GcHeader::SIZE as i64));
                    let hdr_word =
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), hdr_addr, 0);
                    let tid_mask = builder.ins().iconst(cl_types::I64, TYPE_ID_MASK as i64);
                    let loc_typeid = builder.ins().band(hdr_word, tid_mask);

                    // assembler.py:1934-1937 gc_ll_descr lookups.
                    let (base_type_info, shift_by, _sizeof_ti) =
                        with_cranelift_gc_required(|gc| gc.get_translated_info_for_typeinfo());
                    let (infobits_offset, is_object_flag) = with_cranelift_gc_required(|gc| {
                        gc.get_translated_info_for_guard_is_object()
                    });

                    // assembler.py:1938-1939 addr_add(imm(base_type_info),
                    //     loc_typeid, scale=shift_by, offset=infobits_offset)
                    let shifted_typeid = if shift_by > 0 {
                        builder.ins().ishl_imm(loc_typeid, shift_by as i64)
                    } else {
                        loc_typeid
                    };
                    let base_val = builder.ins().iconst(cl_types::I64, base_type_info as i64);
                    let addr_without_off = builder.ins().iadd(base_val, shifted_typeid);
                    let loc_infobits = builder
                        .ins()
                        .iadd_imm(addr_without_off, infobits_offset as i64);

                    // assembler.py:1940 TEST8 [loc_infobits], IS_OBJECT_FLAG.
                    let byte =
                        builder
                            .ins()
                            .load(cl_types::I8, MemFlags::trusted(), loc_infobits, 0);
                    let mask = builder.ins().iconst(cl_types::I8, is_object_flag as i64);
                    let masked = builder.ins().band(byte, mask);
                    let zero_i8 = builder.ins().iconst(cl_types::I8, 0);
                    // assembler.py:1942 guard_success_cc = Conditions['NZ']:
                    // the guard passes when the AND result is non-zero;
                    // the fail branch triggers when it is zero.
                    let fail = builder.ins().icmp(IntCC::Equal, masked, zero_i8);

                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder.ins().brif(fail, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardSubclass => {
                    // x86/assembler.py:1945-1980 genop_guard_guard_subclass.
                    //     assert self.cpu.supports_guard_gc_type
                    //     [loc_object, loc_check_against_class, loc_tmp] = locs
                    //     offset = self.cpu.vtable_offset
                    //     offset2 = self.cpu.subclassrange_min_offset
                    //     if offset is not None:
                    //         self.mc.MOV_rm(loc_tmp, (loc_object, offset))
                    //         self.mc.MOV_rm(loc_tmp, (loc_tmp, offset2))
                    //     else:
                    //         # read the typeid
                    //         self.mc.MOV32(loc_tmp, mem(loc_object, 0))
                    //         base_type_info, shift_by, sizeof_ti = (
                    //             gc_ll_descr.get_translated_info_for_typeinfo())
                    //         self.mc.MOV(loc_tmp, addr_add(
                    //             imm(base_type_info), loc_tmp,
                    //             scale=shift_by,
                    //             offset=sizeof_ti + offset2))
                    //     vtable_ptr = loc_check_against_class.getint()
                    //     vtable_ptr = rffi.cast(rclass.CLASSTYPE, vtable_ptr)
                    //     check_min = vtable_ptr.subclassrange_min
                    //     check_max = vtable_ptr.subclassrange_max
                    //     self.mc.SUB_ri(loc_tmp, check_min)
                    //     self.mc.CMP_ri(loc_tmp, check_max - check_min)
                    //     self.guard_success_cc = Conditions['B']
                    //     self.implement_guard(guard_token)
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    if !cranelift_gc_active() {
                        return Err(missing_gc_runtime(op.opcode));
                    }
                    // assembler.py:1946 assert self.cpu.supports_guard_gc_type
                    assert!(
                        with_cranelift_gc_required(|gc| gc.supports_guard_gc_type()),
                        "x86/assembler.py:1946: assert self.cpu.\
                         supports_guard_gc_type (GcAllocator has not \
                         installed a TYPE_INFO / rclass.CLASSTYPE layout)"
                    );

                    let loc_object = resolve_opref(&mut builder, &constants, op.args[0]);
                    // assembler.py:1971 vtable_ptr = loc_check_against_class
                    //   .getint(): the bounds are resolved at codegen time,
                    //   so arg1 must be an immediate class pointer.
                    let loc_check_against_class = constants
                        .get(&op.args[1].raw())
                        .copied()
                        .unwrap_or_else(|| {
                            panic!(
                                "x86/assembler.py:1971 vtable_ptr = \
                                 loc_check_against_class.getint(): \
                                 GUARD_SUBCLASS requires arg1 to be an \
                                 immediate class pointer"
                            )
                        });

                    // assembler.py:1950-1951: cpu.vtable_offset /
                    // cpu.subclassrange_min_offset.
                    let offset_vtable = vtable_offset;
                    let offset2 = with_cranelift_gc_required(|gc| gc.subclassrange_min_offset());

                    // loc_tmp: majit uses a cranelift value as the temp;
                    // x86 allocates a register.
                    let loc_tmp = if let Some(vtable_off) = offset_vtable {
                        // assembler.py:1953-1956:
                        //     self.mc.MOV_rm(loc_tmp, (loc_object, offset))
                        //     self.mc.MOV_rm(loc_tmp, (loc_tmp, offset2))
                        let vtable_ptr_val = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            loc_object,
                            vtable_off as i32,
                        );
                        builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            vtable_ptr_val,
                            offset2 as i32,
                        )
                    } else {
                        // assembler.py:1957-1969 gcremovetypeptr path.
                        //     MOV32 loc_tmp, mem(loc_object, 0)
                        //     base_type_info, shift_by, sizeof_ti = ...
                        //     MOV loc_tmp, [base_type_info
                        //         + (loc_tmp << shift_by)
                        //         + sizeof_ti + offset2]
                        let hdr_addr = builder.ins().iadd_imm(loc_object, -(GcHeader::SIZE as i64));
                        let hdr_word =
                            builder
                                .ins()
                                .load(cl_types::I64, MemFlags::trusted(), hdr_addr, 0);
                        let tid_mask = builder.ins().iconst(cl_types::I64, TYPE_ID_MASK as i64);
                        let typeid = builder.ins().band(hdr_word, tid_mask);
                        let (base_type_info, shift_by, sizeof_ti) =
                            with_cranelift_gc_required(|gc| gc.get_translated_info_for_typeinfo());
                        let shifted = if shift_by > 0 {
                            builder.ins().ishl_imm(typeid, shift_by as i64)
                        } else {
                            typeid
                        };
                        let base_val = builder.ins().iconst(cl_types::I64, base_type_info as i64);
                        let addr_base = builder.ins().iadd(base_val, shifted);
                        let addr = builder
                            .ins()
                            .iadd_imm(addr_base, (sizeof_ti + offset2) as i64);
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), addr, 0)
                    };

                    // assembler.py:1971-1974 read the bounds from the
                    // expected class pointer at codegen time.
                    let (check_min, check_max) = with_cranelift_gc_required(|gc| {
                        gc.subclass_range(loc_check_against_class as usize)
                    })
                    .unwrap_or_else(|| {
                        panic!(
                            "x86/assembler.py:1973-1974 vtable_ptr.\
                             subclassrange_min/max: GcAllocator has no \
                             rclass.CLASSTYPE entry for classptr {:#x}",
                            loc_check_against_class
                        )
                    });

                    // assembler.py:1976-1978 unsigned comparison:
                    //     (loc_tmp - check_min) <u (check_max - check_min)
                    let sub = builder.ins().iadd_imm(loc_tmp, -check_min);
                    let limit = builder.ins().iconst(cl_types::I64, check_max - check_min);
                    // assembler.py:1979 guard_success_cc = Conditions['B']:
                    // the guard passes when sub <u limit; the fail branch
                    // triggers when sub >=u limit.
                    let fail = builder
                        .ins()
                        .icmp(IntCC::UnsignedGreaterThanOrEqual, sub, limit);

                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder.ins().brif(fail, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Exception operations ──
                OpCode::SaveException => {
                    // x86/assembler.py:1820-1821 genop_save_exception:
                    //   _store_and_reset_exception → resloc = [pos_exc_value];
                    //   [pos_exception] = 0; [pos_exc_value] = 0
                    let exc_val_addr = builder.ins().iconst(ptr_type, jit_exc_value_addr() as i64);
                    let exc_val =
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), exc_val_addr, 0);
                    let exc_type_addr = builder.ins().iconst(ptr_type, jit_exc_type_addr() as i64);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero, exc_type_addr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero, exc_val_addr, 0);
                    let vi = op_var_index(op, op_idx, inputargs.len());
                    builder.def_var(var(vi as u32), exc_val);
                }
                OpCode::SaveExcClass => {
                    // x86/assembler.py:1817-1818 genop_save_exc_class:
                    //   MOV resloc, [pos_exception]
                    let exc_type_addr = builder.ins().iconst(ptr_type, jit_exc_type_addr() as i64);
                    let exc_type =
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), exc_type_addr, 0);
                    let vi = op_var_index(op, op_idx, inputargs.len());
                    builder.def_var(var(vi as u32), exc_type);
                }
                OpCode::RestoreException => {
                    // x86/assembler.py:1845-1850 _restore_exception:
                    //   MOV [pos_exc_value], excvalloc
                    //   MOV [pos_exception], exctploc
                    let exc_type = resolve_opref(&mut builder, &constants, op.args[0]);
                    let value = resolve_opref(&mut builder, &constants, op.args[1]);
                    let exc_val_addr = builder.ins().iconst(ptr_type, jit_exc_value_addr() as i64);
                    let exc_type_addr = builder.ins().iconst(ptr_type, jit_exc_type_addr() as i64);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, exc_val_addr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), exc_type, exc_type_addr, 0);
                }
                OpCode::CheckMemoryError => {
                    // x86/assembler.py:1630-1641 `genop_discard_check_memory_error`
                    // — emit `is_null?` branch into the propagate path (a tail
                    // that mirrors `_build_propagate_exception_path`,
                    // x86/assembler.py:328-345 / aarch64/assembler.py:559-577).
                    let ptr_val = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, ptr_val, zero);
                    let propagate_block = builder.create_block();
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder
                        .ins()
                        .brif(is_null, propagate_block, &[], cont_block, &[]);

                    builder.switch_to_block(propagate_block);
                    builder.seal_block(propagate_block);
                    if propagate_exception_descr_ptr == 0 {
                        // Unattached `propagate_exception_descr` — typical for
                        // unit tests that bypass `MetaInterp::finish_setup`.
                        // In production (`pyjitpl.py:2283
                        // self.cpu.propagate_exception_descr = exc_descr`) the
                        // descr is always set before `compile_loop` runs.
                        builder
                            .ins()
                            .trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
                    } else {
                        // _store_and_reset_exception (assembler.py:1826-1843):
                        // read JIT_EXC_VALUE → tmp, clear both globals.
                        let exc_val_addr =
                            builder.ins().iconst(ptr_type, jit_exc_value_addr() as i64);
                        let exc_val =
                            builder
                                .ins()
                                .load(cl_types::I64, MemFlags::trusted(), exc_val_addr, 0);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), zero, exc_val_addr, 0);
                        let exc_type_addr =
                            builder.ins().iconst(ptr_type, jit_exc_type_addr() as i64);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), zero, exc_type_addr, 0);
                        // assembler.py:336-337 — MOV [jf_guard_exc], tmp
                        // so `cpu.grab_exc_value(deadframe)` in
                        // `PropagateExceptionDescr.handle_fail`
                        // (compile.py:1095) can read it back.
                        let cur_jf = builder.use_var(jf_ptr_var);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), exc_val, cur_jf, JF_GUARD_EXC_OFS);
                        // assembler.py:339-340 — MOV [jf_descr], descr.
                        let descr_val = builder
                            .ins()
                            .iconst(ptr_type, propagate_exception_descr_ptr as i64);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), descr_val, cur_jf, JF_DESCR_OFS);
                        // assembler.py:1101 _call_footer_shadowstack
                        // + assembler.py:1097 _call_footer.
                        emit_call_footer_shadowstack(&mut builder, ptr_type);
                        builder.ins().return_(&[cur_jf]);
                    }

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Call operations ──
                //
                // Regular calls, pure calls, and loop-invariant calls compile
                // as ordinary indirect calls. More specialized call families
                // with extra runtime semantics are rejected below until wired.
                OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN => {
                    let descr = op.descr.as_ref().expect("call op must have a descriptor");
                    let call_descr = descr
                        .as_call_descr()
                        .expect("call op descriptor must be a CallDescr");

                    let call_result = emit_indirect_call_from_parts(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        &op.args[1..],
                        call_descr,
                        call_conv,
                        ptr_type,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                    );
                    if let Some(result) = call_result {
                        builder.def_var(var(vi), result);
                    }
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    builder.def_var(jf_ptr_var, jf_ptr);
                    if let Some(result) = call_result {
                        if op.result_type() == Type::Ref && !force_tokens.contains(&vi) {
                            sync_ref_root_var(
                                &mut builder,
                                jf_ptr,
                                &ref_root_slots,
                                vi,
                                result,
                                ref_root_base_ofs,
                                &mut synced_ref_vars,
                            );
                        }
                    }
                }

                OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN => {
                    let descr = op.descr.as_ref().ok_or_else(|| {
                        unsupported_semantics(op.opcode, "call-assembler op must have a descriptor")
                    })?;
                    let call_descr = descr.as_call_descr().ok_or_else(|| {
                        unsupported_semantics(
                            op.opcode,
                            "call-assembler descriptor must be a CallDescr",
                        )
                    })?;
                    let resolved_target = resolve_call_assembler_target(op.opcode, call_descr)?;
                    // rewrite.py:685-695 handle_call_assembler replaces the
                    // original red-arg list with [callee_jitframe] plus the
                    // optional virtualizable.  The descriptor's arg_types are
                    // the pre-rewrite CALL_ASSEMBLER shape and must not be
                    // compared with the rewritten op arity here.
                    if op.args.is_empty() || op.args.len() > 2 {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "post-rewrite call-assembler must carry callee frame plus optional virtualizable",
                        ));
                    }

                    // x86/assembler.py:2260 _store_force_index(ops[pos+1]):
                    // If next op is GUARD_NOT_FORCED, store its fail descr
                    // to jf_force_descr BEFORE the call. GC rewriter elides
                    // CallN(drop_frame), so GUARD_NOT_FORCED is ops[idx+1].
                    if let Some(next) = ops.get(op_idx + 1) {
                        if next.opcode == OpCode::GuardNotForced
                            || next.opcode == OpCode::GuardNotForced2
                        {
                            let info = &guard_infos[guard_idx];
                            let descr_val =
                                builder.ins().iconst(cl_types::I64, info.fail_descr_ptr);
                            let cur_jf = builder.use_var(jf_ptr_var);
                            builder.ins().store(
                                MemFlags::trusted(),
                                descr_val,
                                cur_jf,
                                JF_FORCE_DESCR_OFS,
                            );
                            // Reset jf_descr to 0: a previous guard exit
                            // (e.g., base-case bridge) may have left jf_descr
                            // non-zero. RPython's _call_header resets it.
                            let zero = builder.ins().iconst(cl_types::I64, 0);
                            builder
                                .ins()
                                .store(MemFlags::trusted(), zero, cur_jf, JF_DESCR_OFS);
                        }
                    }

                    // rewrite.py:613-653 gen_malloc_frame parity:
                    // Allocate callee jitframe from nursery (heap), not stack.
                    // The callee's prologue pushes jf_ptr onto shadow stack,
                    // so GC tracks it during callee execution. After return,
                    let (args_ptr, args_data_ptr) = if let Some(expansion) =
                        call_descr.vable_expansion()
                    {
                        let callee_depth = resolved_target
                            .as_ref()
                            .map_or(16, |t| (t.max_output_slots + t.num_ref_roots).max(1));
                        let num_expanded_items =
                            1 + expansion.scalar_fields.len() + expansion.num_array_items;
                        let jf_depth = num_expanded_items.max(callee_depth).max(1);
                        let jf_bytes = (JF_FRAME_ITEM0_OFS as u32) + (jf_depth as u32) * 8;
                        let args_slot = builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot,
                            jf_bytes,
                            3,
                        ));
                        let args_ptr = builder.ins().stack_addr(ptr_type, args_slot, 0);
                        let frame_val = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.ins().store(
                            MemFlags::trusted(),
                            frame_val,
                            args_ptr,
                            JF_FRAME_ITEM0_OFS,
                        );
                        for (i, &(offset, _tp)) in expansion.scalar_fields.iter().enumerate() {
                            let slot = i + 1;
                            let ofs = JF_FRAME_ITEM0_OFS + (slot as i32) * 8;
                            if let Some(&(_, cval)) =
                                expansion.const_overrides.iter().find(|(s, _)| *s == slot)
                            {
                                let cv = builder.ins().iconst(cl_types::I64, cval);
                                builder.ins().stack_store(cv, args_slot, ofs);
                            } else {
                                let val = builder.ins().load(
                                    cl_types::I64,
                                    MemFlags::trusted(),
                                    frame_val,
                                    offset as i32,
                                );
                                builder.ins().stack_store(val, args_slot, ofs);
                            }
                        }
                        let num_scalars_with_frame = 1 + expansion.scalar_fields.len();
                        // Only load array data pointer if at least one item
                        // needs to be read from the frame (not covered by
                        // const_overrides or arg_overrides).
                        let needs_array_load = (0..expansion.num_array_items).any(|i| {
                            let slot = num_scalars_with_frame + i;
                            !expansion.const_overrides.iter().any(|(s, _)| *s == slot)
                                && !expansion.arg_overrides.iter().any(|(s, _)| *s == slot)
                        });
                        let arr_data_ptr_val = if needs_array_load {
                            let arr_struct_addr = builder
                                .ins()
                                .iadd_imm(frame_val, expansion.array_struct_offset as i64);
                            builder.ins().load(
                                ptr_type,
                                MemFlags::trusted(),
                                arr_struct_addr,
                                expansion.array_ptr_offset as i32,
                            )
                        } else {
                            builder.ins().iconst(ptr_type, 0)
                        };
                        for i in 0..expansion.num_array_items {
                            let slot = num_scalars_with_frame + i;
                            let ofs = JF_FRAME_ITEM0_OFS + (slot as i32) * 8;
                            // Match dynasm x86 assembler.rs:4544 / aarch64
                            // assembler.rs:4673: const_overrides take precedence
                            // for any slot, including array items.
                            if let Some(&(_, cval)) =
                                expansion.const_overrides.iter().find(|(s, _)| *s == slot)
                            {
                                let cv = builder.ins().iconst(cl_types::I64, cval);
                                builder.ins().stack_store(cv, args_slot, ofs);
                            } else if let Some(&(_, arg_idx)) =
                                expansion.arg_overrides.iter().find(|(s, _)| *s == slot)
                            {
                                let val = resolve_opref(&mut builder, &constants, op.args[arg_idx]);
                                builder.ins().stack_store(val, args_slot, ofs);
                            } else {
                                let val = builder.ins().load(
                                    cl_types::I64,
                                    MemFlags::trusted(),
                                    arr_data_ptr_val,
                                    (i * 8) as i32,
                                );
                                builder.ins().stack_store(val, args_slot, ofs);
                            }
                        }
                        let args_data_ptr =
                            builder
                                .ins()
                                .stack_addr(ptr_type, args_slot, JF_FRAME_ITEM0_OFS);
                        (args_ptr, args_data_ptr)
                    } else {
                        let args_ptr = resolve_opref(&mut builder, &constants, op.args[0]);
                        let args_data_ptr =
                            builder.ins().iadd_imm(args_ptr, JF_FRAME_ITEM0_OFS as i64);
                        (args_ptr, args_data_ptr)
                    };
                    let target_token = builder.ins().iconst(
                        cl_types::I64,
                        call_descr.call_target_token().unwrap() as i64,
                    );
                    let args_ptr_i64 = ptr_arg_as_i64(&mut builder, args_data_ptr, ptr_type);
                    let outcome_slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        16,
                        3,
                    ));
                    let outcome_ptr = builder.ins().stack_addr(ptr_type, outcome_slot, 0);
                    let outcome_ptr_i64 = ptr_arg_as_i64(&mut builder, outcome_ptr, ptr_type);
                    let expected_result_kind = builder.ins().iconst(
                        cl_types::I64,
                        expected_call_assembler_result_kind(call_descr, resolved_target.as_ref())?
                            as i64,
                    );

                    let ca_merge_block = builder.create_block();
                    builder.append_block_param(ca_merge_block, cl_types::I64);

                    // Direct call via dispatch table: load code_ptr from a
                    // stable slot (AtomicPtr). For self-recursion (target not
                    // yet registered), pre-create the slot with null — compile
                    // completion fills it. Runtime null check falls back to shim.
                    let token_val = call_descr.call_target_token().unwrap_or(0);
                    let dispatch_slot_addr = if token_val != 0 {
                        let code_ptr = resolved_target
                            .as_ref()
                            .map(|target| target.code_ptr)
                            .unwrap_or(std::ptr::null());
                        // Self-recursion and other pending targets are
                        // registered as Some(target) with a null code_ptr.
                        // Still emit the direct dispatch block: the stable
                        // slot is filled by register_call_assembler_target
                        // after compilation, while runtime null/unknown
                        // checks keep the current execution on the shim.
                        Some(ca_dispatch_slot(token_val, code_ptr) as usize)
                    } else {
                        None
                    };

                    let use_direct = dispatch_slot_addr.is_some();

                    if use_direct {
                        let slot_addr = dispatch_slot_addr.unwrap();

                        // No separate out_slot: callee writes outputs to args_slot (shared).

                        // Load code_ptr from the dispatch entry.  RPython's
                        // call_assembler path compares the returned jf_descr
                        // against the CPU's DoneWithThisFrame* descr for the
                        // CALL_ASSEMBLER result type; it is not target-local.
                        // Keep the finish descr as a compile-time immediate
                        // like dynasm instead of reloading it on every
                        // recursive call.
                        // RPython done_with_this_frame parity: compare jf_descr
                        // with the finish FailDescr pointer directly.
                        let entry_ptr = builder.ins().iconst(ptr_type, slot_addr as i64);
                        let code_addr =
                            builder
                                .ins()
                                .load(ptr_type, MemFlags::trusted(), entry_ptr, 0);
                        let expected_finish_descr_ptr = self
                            .attached_descr_ptrs()
                            .done_with_this_frame_descr_ptr_for_type(op.opcode.result_type())
                            as i64;
                        let expected_finish_descr = builder
                            .ins()
                            .iconst(cl_types::I64, expected_finish_descr_ptr);
                        // assembler.py:call_assembler() parity:
                        // CMP + conditional branch: null → shim, else → direct call.
                        let direct_call_block = builder.create_block();
                        let shim_fallback_block = builder.create_block();
                        let null_ptr = builder.ins().iconst(ptr_type, 0);
                        let is_null = builder.ins().icmp(IntCC::Equal, code_addr, null_ptr);
                        builder.ins().brif(
                            is_null,
                            shim_fallback_block,
                            &[],
                            direct_call_block,
                            &[],
                        );

                        // ── Direct call to compiled code ──
                        // assembler.py:_call_assembler_emit_call
                        builder.switch_to_block(direct_call_block);
                        builder.seal_block(direct_call_block);

                        // assembler.py:2267-2269 _call_assembler_emit_call:
                        // Spill caller's GC refs, then direct-call the callee.
                        // The callee's prologue pushes its own shadow stack
                        // entry (_call_header_shadowstack), and its epilogue
                        // pops it (_call_footer_shadowstack). The caller does
                        // NOT touch the shadow stack here.
                        jf_ptr = builder.use_var(jf_ptr_var);
                        spill_unsynced_ref_roots(
                            &mut builder,
                            jf_ptr,
                            &live_ref_root_slots,
                            &defined_ref_vars,
                            &synced_ref_vars,
                            ref_root_base_ofs,
                        );
                        emit_push_gcmap(&mut builder, jf_ptr, per_call_gcmap);

                        // fn(jf_ptr) → jf_ptr  (simple_call parity)
                        let mut sig = Signature::new(call_conv);
                        sig.params.push(AbiParam::new(ptr_type)); // jf_ptr
                        sig.returns.push(AbiParam::new(ptr_type)); // returned jf_ptr
                        let sig_ref = builder.import_signature(sig);
                        let call_inst =
                            builder.ins().call_indirect(sig_ref, code_addr, &[args_ptr]);
                        let result_jf = builder.inst_results(call_inst)[0];

                        // _reload_frame_if_necessary: GC may have moved
                        // the caller's jitframe during the call.
                        jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                        builder.def_var(jf_ptr_var, jf_ptr);
                        emit_pop_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                        let post_call_reload_ref_root_slots =
                            ref_root_slots_with_future_regular_uses(
                                op_idx,
                                ops,
                                &live_ref_root_slots,
                                &defined_ref_vars,
                            );
                        reload_ref_roots(
                            &mut builder,
                            jf_ptr,
                            &post_call_reload_ref_root_slots,
                            &defined_ref_vars,
                            ref_root_base_ofs,
                        );
                        mark_ref_roots_after_selective_reload(
                            &mut stale_ref_vars,
                            &live_ref_root_slots,
                            &post_call_reload_ref_root_slots,
                        );
                        // _call_assembler_check_descr (assembler.py:2274-2278):
                        //   CMP [eax + jf_descr_ofs], done_with_this_frame_descr
                        let fail_idx_raw = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            result_jf,
                            JF_DESCR_OFS,
                        );

                        // RPython: check jf_descr == done_with_this_frame_descr
                        let is_direct_finish =
                            builder
                                .ins()
                                .icmp(IntCC::Equal, fail_idx_raw, expected_finish_descr);
                        // assembler.py:295-360 call_assembler parity:
                        //   Path B: JE done_descr → load result from frame[0]
                        //   Path A: CALL assembler_helper_adr(deadframe, vloc)
                        // Attached bridge dispatch happens in the callee's
                        // guard-exit path, which is the Cranelift equivalent
                        // of dynasm's patched guard jump. If a deadframe still
                        // reaches here, fall back to assembler_call_helper.
                        let direct_finish_block = builder.create_block();
                        let helper_block = builder.create_block();
                        builder.ins().brif(
                            is_direct_finish,
                            direct_finish_block,
                            &[],
                            helper_block,
                            &[],
                        );

                        // ── Path B: finish — load result from frame[0] ──
                        // _call_assembler_load_result (assembler.py:2303):
                        //   MOV eax, [eax + ofs] — load from RETURNED frame.
                        builder.switch_to_block(direct_finish_block);
                        builder.seal_block(direct_finish_block);
                        let direct_result = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            result_jf,
                            JF_FRAME_ITEM0_OFS,
                        );
                        builder
                            .ins()
                            .jump(ca_merge_block, &[BlockArg::from(direct_result)]);

                        // ── Path A: assembler_helper (handles bridges + blackhole) ──
                        // warmspot.py:1021-1028 assembler_call_helper parity:
                        //   fail_descr.handle_fail(deadframe, metainterp_sd, jd)
                        builder.switch_to_block(helper_block);
                        builder.seal_block(helper_block);
                        builder.set_cold_block(helper_block);
                        // `call_assembler_guard_failure_inner` follows
                        // RPython handle_fail via faildescr/resumedata and
                        // does not consume the old vloc/frame_ptr argument.
                        // Keep the argument slot for ABI parity without
                        // loading the callee frame on the normal finish path.
                        let frame_ptr = builder.ins().iconst(cl_types::I64, 0);
                        // CALL_ASSEMBLER helper can compile/execute bridges
                        // and allocate. Treat it as a collecting call for
                        // the caller frame, matching the pending-gcmap
                        // sequence used by the dynasm backend.
                        jf_ptr = builder.use_var(jf_ptr_var);
                        spill_unsynced_ref_roots(
                            &mut builder,
                            jf_ptr,
                            &live_ref_root_slots,
                            &defined_ref_vars,
                            &synced_ref_vars,
                            ref_root_base_ofs,
                        );
                        emit_push_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                        // RPython assembler_call_helper receives deadframe as an
                        // RPython function parameter → RPython's stack scanning
                        // keeps it as a GC root. Pyre's helper is a Rust function
                        // whose stack is NOT scanned by GC. Push the callee's
                        // jitframe onto the shadow stack so GC can trace it during
                        // the helper call. Pop after the helper returns.
                        emit_call_header_shadowstack(&mut builder, ptr_type, result_jf);
                        let result_jf_data =
                            builder.ins().iadd_imm(result_jf, JF_FRAME_ITEM0_OFS as i64);
                        let result_jf_data_i64 =
                            ptr_arg_as_i64(&mut builder, result_jf_data, ptr_type);
                        // `compile.py:665 setattr(cpu, name, descr)` — bake the
                        // heap-stable `Arc<RwLock<CpuDescrAttachments>>` address
                        // so the trampoline's runtime classifier resolves
                        // `DoneWithThisFrame*` identity against the owning cpu.
                        // `CompiledLoop.cpu_attachments` keeps the allocation
                        // alive for the lifetime of this emitted code.
                        let cpu_handle_imm = builder
                            .ins()
                            .iconst(cl_types::I64, Arc::as_ptr(&self.descr_attachments) as i64);
                        // RPython assembler.py:349-350: assembler_helper(tmploc, vloc).
                        let force_result = emit_host_call(
                            &mut builder,
                            ptr_type,
                            call_conv,
                            call_assembler_guard_failure as *const () as usize,
                            &[
                                cpu_handle_imm,
                                target_token,
                                fail_idx_raw,
                                frame_ptr,
                                result_jf_data_i64,
                                result_jf_data_i64,
                            ],
                            Some(cl_types::I64),
                        );
                        // Pop callee jf from shadow stack. _reload_frame_if_necessary
                        // then reloads the CALLER's jitframe (which is underneath).
                        emit_call_footer_shadowstack(&mut builder, ptr_type);
                        // _reload_frame_if_necessary: GC may have moved
                        // the caller's jitframe during the helper call.
                        jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                        builder.def_var(jf_ptr_var, jf_ptr);
                        emit_pop_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                        reload_ref_roots(
                            &mut builder,
                            jf_ptr,
                            &live_ref_root_slots,
                            &defined_ref_vars,
                            ref_root_base_ofs,
                        );
                        mark_ref_roots_fresh(&mut stale_ref_vars, &live_ref_root_slots);
                        builder
                            .ins()
                            .jump(ca_merge_block, &[BlockArg::from(force_result.unwrap())]);

                        // ── Fallback: null code_ptr, unknown finish, or deadframe ──
                        builder.switch_to_block(shim_fallback_block);
                        builder.seal_block(shim_fallback_block);
                    }

                    // Shim call (always present as fallback, or sole path
                    // when target isn't resolved or has non-primitive result)

                    // After switch_to_block(shim_fallback_block), the cached
                    // jf_ptr local may correspond to a value defined in a
                    // different block on the upstream tree. Read the fresh
                    // jitframe pointer through the Variable so Cranelift
                    // threads it through the necessary block params.
                    jf_ptr = builder.use_var(jf_ptr_var);
                    spill_unsynced_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &live_ref_root_slots,
                        &defined_ref_vars,
                        &synced_ref_vars,
                        ref_root_base_ofs,
                    );
                    emit_push_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                    let result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        call_assembler_shim as *const () as usize,
                        &[
                            target_token,
                            args_ptr_i64,
                            outcome_ptr_i64,
                            expected_result_kind,
                        ],
                        Some(cl_types::I64),
                    );
                    // _reload_frame_if_necessary (assembler.py:405-412):
                    // GC may have moved the jitframe during the shim call.
                    // Reload jf_ptr from shadow stack before reading from it.
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    builder.def_var(jf_ptr_var, jf_ptr);
                    emit_pop_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                    reload_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &live_ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                    );
                    mark_ref_roots_fresh(&mut stale_ref_vars, &live_ref_root_slots);

                    let outcome_kind = builder.ins().stack_load(cl_types::I64, outcome_slot, 0);
                    let finish_kind = builder
                        .ins()
                        .iconst(cl_types::I64, CALL_ASSEMBLER_OUTCOME_FINISH);
                    let is_finish = builder.ins().icmp(IntCC::Equal, outcome_kind, finish_kind);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    builder.ins().brif(
                        is_finish,
                        ca_merge_block,
                        &[BlockArg::from(result.unwrap())],
                        exit_block,
                        &[],
                    );

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let deadframe_handle = builder.ins().stack_load(cl_types::I64, outcome_slot, 8);
                    builder.ins().store(
                        MemFlags::trusted(),
                        deadframe_handle,
                        jf_ptr,
                        JF_FRAME_ITEM0_OFS,
                    );
                    let sentinel = builder
                        .ins()
                        .iconst(cl_types::I64, CALL_ASSEMBLER_DEADFRAME_SENTINEL as i64);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), sentinel, jf_ptr, JF_DESCR_OFS);
                    // assembler.py:1130-1136 _call_footer_shadowstack — inline SUB
                    emit_call_footer_shadowstack(&mut builder, ptr_type);
                    builder.ins().return_(&[jf_ptr]);

                    // ── Merge: result from cache or shim ──
                    builder.switch_to_block(ca_merge_block);
                    builder.seal_block(ca_merge_block);
                    let merged_result = builder.block_params(ca_merge_block)[0];

                    if op.result_type() != Type::Void {
                        builder.def_var(var(vi), merged_result);
                        if op.result_type() == Type::Ref && !force_tokens.contains(&vi) {
                            let cur_jf = builder.use_var(jf_ptr_var);
                            sync_ref_root_var(
                                &mut builder,
                                cur_jf,
                                &ref_root_slots,
                                vi,
                                merged_result,
                                ref_root_base_ofs,
                                &mut synced_ref_vars,
                            );
                        }
                    }
                    mark_ref_roots_synced(&mut synced_ref_vars, &live_ref_root_slots);
                }

                OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN => {
                    // x86/assembler.py:2234-2235 _genop_call_may_force:
                    //   self._store_force_index(self._find_nearby_operation(+1))
                    //   self._genop_call(op, arglocs, result_loc)
                    check_paired_guard_not_forced(ops, op_idx, op.opcode, "call_may_force")?;
                    let info = &guard_infos[guard_idx];
                    // x86/assembler.py _store_force_index parity:
                    // Store the GUARD_NOT_FORCED fail descriptor pointer
                    // into jf_force_descr before the call. If the callee
                    // forces the frame, force() reads this to set jf_descr.
                    let descr_val = builder.ins().iconst(cl_types::I64, info.fail_descr_ptr);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), descr_val, cur_jf, JF_FORCE_DESCR_OFS);

                    // regalloc.py before_call() parity: spill fail_args to
                    // jf_frame so force_token_to_dead_frame() reads correct
                    // values if the callee forces the frame.
                    for (index, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
                        // The call result (vi) is not yet available — write 0.
                        let raw = if arg_ref.raw() == vi && !constants.contains_key(&arg_ref.raw())
                        {
                            builder.ins().iconst(cl_types::I64, 0)
                        } else {
                            resolve_failarg_opref(
                                &mut builder,
                                &constants,
                                cur_jf,
                                &ref_root_slots,
                                &stale_ref_vars,
                                ref_root_base_ofs,
                                arg_ref,
                            )
                        };
                        builder.ins().store(
                            MemFlags::trusted(),
                            raw,
                            cur_jf,
                            JF_FRAME_ITEM0_OFS + (index as i32) * 8,
                        );
                    }
                    if info.gcmap != 0 {
                        emit_jitframe_write_barrier(&mut builder, ptr_type, call_conv, cur_jf);
                    }

                    // x86/assembler.py:2236: self._genop_call(op, arglocs, result_loc)
                    let descr = op.descr.as_ref().expect("call op must have a descriptor");
                    let call_descr = descr
                        .as_call_descr()
                        .expect("call op descriptor must be a CallDescr");

                    if let Some(result) = emit_indirect_call_from_parts(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        &op.args[1..],
                        call_descr,
                        call_conv,
                        ptr_type,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                    ) {
                        builder.def_var(var(vi), result);
                    }
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    builder.def_var(jf_ptr_var, jf_ptr);
                }

                OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                    // x86/assembler.py:2242-2244 _genop_call_release_gil:
                    //   self._store_force_index(self._find_nearby_operation(+1))
                    //   self._genop_call(op, arglocs, result_loc, is_call_release_gil=True)
                    check_paired_guard_not_forced(ops, op_idx, op.opcode, "call_release_gil")?;
                    let info = &guard_infos[guard_idx];
                    let descr_val = builder.ins().iconst(cl_types::I64, info.fail_descr_ptr);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), descr_val, cur_jf, JF_FORCE_DESCR_OFS);

                    // regalloc.py before_call() parity: spill fail_args to
                    // jf_frame so force_token_to_dead_frame() reads correct
                    // values if the callee forces the frame.
                    for (index, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
                        let raw = if arg_ref.raw() == vi && !constants.contains_key(&arg_ref.raw())
                        {
                            builder.ins().iconst(cl_types::I64, 0)
                        } else {
                            resolve_failarg_opref(
                                &mut builder,
                                &constants,
                                cur_jf,
                                &ref_root_slots,
                                &stale_ref_vars,
                                ref_root_base_ofs,
                                arg_ref,
                            )
                        };
                        builder.ins().store(
                            MemFlags::trusted(),
                            raw,
                            cur_jf,
                            JF_FRAME_ITEM0_OFS + (index as i32) * 8,
                        );
                    }
                    if info.gcmap != 0 {
                        emit_jitframe_write_barrier(&mut builder, ptr_type, call_conv, cur_jf);
                    }

                    let descr = op
                        .descr
                        .as_ref()
                        .expect("call_release_gil op must have a descriptor");
                    let call_descr = descr
                        .as_call_descr()
                        .expect("call_release_gil descriptor must be a CallDescr");

                    // Spill GC roots before the call
                    spill_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                    );
                    emit_push_gcmap(&mut builder, jf_ptr, per_call_gcmap);

                    // Release GIL (call the pre-hook)
                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_release_gil_shim as *const () as usize,
                        &[],
                        None,
                    );

                    // Make the actual call
                    let mut sig = Signature::new(call_conv);
                    let arg_types = call_descr.arg_types();
                    for at in arg_types {
                        sig.params.push(AbiParam::new(cranelift_type_for(at)));
                    }
                    let result_type = call_descr.result_type();
                    if result_type != Type::Void {
                        sig.returns
                            .push(AbiParam::new(cranelift_type_for(&result_type)));
                    }
                    let sig_ref = builder.import_signature(sig);

                    // pyjitpl.py:3679 direct_call_release_gil records
                    // `[savebox, funcbox_real] + argboxes[1:]`, so for
                    // CallReleaseGil{I,F,N} the function pointer lives
                    // at `op.arg(1)` and the C arguments start at
                    // `op.args[2..]`. The leading `op.arg(0)` (savebox)
                    // is consumed by the GIL-release pre-hook in
                    // RPython (`backend/x86/assembler.py:2242-2244
                    // _genop_call_release_gil`'s `_store_force_index` /
                    // `is_call_release_gil=True` offset). Pyre's
                    // dynasm backend mirrors that with
                    // `func_index = 3 + is_call_release_gil` at
                    // `backend-dynasm/x86/assembler.rs:4804`.
                    // PRE-EXISTING-ADAPTATION: pyre does not yet
                    // forward `saveerr` to the GIL pre/post-hooks, so
                    // `op.arg(0)` is still consumed as a const_int
                    // operand at trace replay rather than threaded
                    // into the backend hook signature.
                    let func_ptr_raw = resolve_opref(&mut builder, &constants, op.arg(1));
                    let func_ptr_val = if ptr_type != cl_types::I64 {
                        builder.ins().ireduce(ptr_type, func_ptr_raw)
                    } else {
                        func_ptr_raw
                    };

                    let mut args: Vec<CValue> = Vec::with_capacity(op.args.len() - 2);
                    for (i, &arg_ref) in op.args[2..].iter().enumerate() {
                        let raw = resolve_opref(&mut builder, &constants, arg_ref);
                        if i < arg_types.len() && arg_types[i] == Type::Float {
                            args.push(builder.ins().bitcast(cl_types::F64, MemFlags::new(), raw));
                        } else {
                            args.push(raw);
                        }
                    }

                    let call = builder.ins().call_indirect(sig_ref, func_ptr_val, &args);
                    let result = if result_type != Type::Void {
                        Some(builder.inst_results(call)[0])
                    } else {
                        None
                    };

                    // Reacquire GIL (call the post-hook)
                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_reacquire_gil_shim as *const () as usize,
                        &[],
                        None,
                    );

                    emit_pop_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                    // Reload roots (may have been updated by GC during call)
                    reload_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                    );

                    if let Some(result) = result {
                        let result_val = if result_type == Type::Float {
                            builder
                                .ins()
                                .bitcast(cl_types::I64, MemFlags::new(), result)
                        } else {
                            result
                        };
                        builder.def_var(var(vi), result_val);
                    }
                }

                // ── Conditional call (void result) ──
                // args[0] = condition, args[1] = func_ptr, args[2..] = call args
                // If condition != 0, perform the call.
                OpCode::CondCallN => {
                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let call_block = builder.create_block();
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);
                    builder
                        .ins()
                        .brif(is_zero, cont_block, &[], call_block, &[]);

                    builder.switch_to_block(call_block);
                    builder.seal_block(call_block);

                    if let Some(descr) = op.descr.as_ref() {
                        if let Some(call_descr) = descr.as_call_descr() {
                            let _ = emit_indirect_call_from_parts(
                                &mut builder,
                                &constants,
                                op.arg(1),
                                &op.args[2..],
                                call_descr,
                                call_conv,
                                ptr_type,
                                jf_ptr,
                                &ref_root_slots,
                                &defined_ref_vars,
                                &stale_ref_vars,
                                ref_root_base_ofs,
                                per_call_gcmap,
                            );
                            jf_ptr =
                                emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                            builder.def_var(jf_ptr_var, jf_ptr);
                        }
                    }

                    builder.ins().jump(cont_block, &[]);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Conditional call with value result ──
                // args[0] = condition, args[1] = func_ptr, args[2..] = call args
                // If condition != 0: result = call(func_ptr, args...)
                // Else: result = condition (0)
                OpCode::CondCallValueI | OpCode::CondCallValueR => {
                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let call_block = builder.create_block();
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder.append_block_param(cont_block, cl_types::I64);

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);
                    builder.ins().brif(
                        is_zero,
                        cont_block,
                        &[BlockArg::from(cond)],
                        call_block,
                        &[],
                    );

                    builder.switch_to_block(call_block);
                    builder.seal_block(call_block);

                    let mut call_result = cond; // fallback
                    if let Some(descr) = op.descr.as_ref() {
                        if let Some(call_descr) = descr.as_call_descr() {
                            if let Some(result) = emit_indirect_call_from_parts(
                                &mut builder,
                                &constants,
                                op.arg(1),
                                &op.args[2..],
                                call_descr,
                                call_conv,
                                ptr_type,
                                jf_ptr,
                                &ref_root_slots,
                                &defined_ref_vars,
                                &stale_ref_vars,
                                ref_root_base_ofs,
                                per_call_gcmap,
                            ) {
                                call_result = result;
                            }
                            jf_ptr =
                                emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                            builder.def_var(jf_ptr_var, jf_ptr);
                        }
                    }

                    builder
                        .ins()
                        .jump(cont_block, &[BlockArg::from(call_result)]);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);

                    let phi = builder.block_params(cont_block)[0];
                    builder.def_var(var(vi), phi);
                }

                // ── GC allocation calls ──
                OpCode::CallMallocNursery => {
                    // x86/assembler.py:2556-2565 malloc_cond parity.
                    // RPython: inline nursery bump alloc for BOTH loops and
                    // bridges — there is no loop-vs-bridge distinction.
                    // The x86 slow path (_build_malloc_slowpath) does
                    // _push_all_regs_to_frame / _pop_all_regs_from_frame,
                    // so the caller never sees register changes.
                    //
                    // Cranelift equivalent: spill ref roots BEFORE brif,
                    // reload AFTER merge. No SSA variable redefinitions
                    // inside fast_block or slow_block — avoids phi issues.
                    let use_inline = cranelift_gc_active();

                    if use_inline {
                        // x86/assembler.py:2556-2565 malloc_cond parity:
                        // inline nursery bump alloc for both loops and bridges.
                        //
                        // RPython pattern:
                        //   inline: CMP+JA → fast: MOV(bump alloc)
                        //   slow stub: push_gcmap, CALL trampoline
                        //   trampoline: _push_all_regs, GC call,
                        //               _pop_all_regs, pop_gcmap, RET
                        //
                        // Fast path: ZERO overhead (no spill/gcmap).
                        // Slow path: all GC work inside CALL/RET boundary.
                        // After RET, registers restored from jitframe.
                        //
                        // Cranelift equivalent: explicit block params carry
                        // all live ref values + jf_ptr through the merge.
                        // Fast path passes originals (no memory ops).
                        // Slow path passes GC-updated values from jitframe.
                        let (nf_addr, nt_addr) =
                            gc_nursery_addrs.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                        let flags = MemFlags::trusted();
                        let size_val = constants
                            .get(&op.arg(0).raw())
                            .copied()
                            .unwrap_or(op.arg(0).raw() as i64);
                        let size_total = builder.ins().iconst(cl_types::I64, size_val);
                        let nf_ptr = builder.ins().iconst(ptr_type, nf_addr as i64);
                        let nt_ptr = builder.ins().iconst(ptr_type, nt_addr as i64);
                        let free = builder.ins().load(ptr_type, flags, nf_ptr, 0);
                        let new_free = builder.ins().iadd(free, size_total);
                        let top = builder.ins().load(ptr_type, flags, nt_ptr, 0);
                        let fits =
                            builder
                                .ins()
                                .icmp(IntCC::UnsignedLessThanOrEqual, new_free, top);

                        // Collect live ref vars for block param passing.
                        let live_refs: Vec<(u32, usize)> = ref_root_slots
                            .iter()
                            .filter(|(var_idx, _)| defined_ref_vars.contains(var_idx))
                            .copied()
                            .collect();

                        let fast_block = builder.create_block();
                        let slow_block = builder.create_block();
                        let merge_block = builder.create_block();
                        // Block params: [result, jf_ptr, ref0, ref1, ...]
                        builder.append_block_param(merge_block, ptr_type); // result
                        builder.append_block_param(merge_block, ptr_type); // jf_ptr
                        for _ in &live_refs {
                            builder.append_block_param(merge_block, cl_types::I64);
                        }
                        builder.ins().brif(fits, fast_block, &[], slow_block, &[]);

                        // fast: inline bump alloc, no GC — zero spill overhead
                        // (malloc_cond: MOV [nursery_free], edx; continue)
                        builder.switch_to_block(fast_block);
                        builder.seal_block(fast_block);
                        builder.ins().store(flags, new_free, nf_ptr, 0);
                        let zero_hdr = builder.ins().iconst(cl_types::I64, 0);
                        builder.ins().store(MemFlags::trusted(), zero_hdr, free, 0);
                        let hdr_sz = builder.ins().iconst(ptr_type, GcHeader::SIZE as i64);
                        let obj = builder.ins().iadd(free, hdr_sz);
                        // Pass original values through block params
                        let mut fast_args: Vec<BlockArg> =
                            vec![BlockArg::from(obj), BlockArg::from(jf_ptr)];
                        for &(var_idx, _) in &live_refs {
                            fast_args.push(BlockArg::from(builder.use_var(var(var_idx))));
                        }
                        builder.ins().jump(merge_block, &fast_args);

                        // slow: aarch64 _build_malloc_slowpath parity.
                        // Fast path is bare bump-allocation; only the
                        // overflow path spills refs and installs jf_gcmap
                        // before calling the helper.
                        builder.switch_to_block(slow_block);
                        builder.seal_block(slow_block);
                        builder.set_cold_block(slow_block);
                        spill_ref_roots(
                            &mut builder,
                            jf_ptr,
                            &ref_root_slots,
                            &defined_ref_vars,
                            &stale_ref_vars,
                            ref_root_base_ofs,
                        );
                        emit_push_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                        let ps = builder.ins().iadd_imm(size_total, -(GcHeader::SIZE as i64));
                        let slow_r = emit_host_call(
                            &mut builder,
                            ptr_type,
                            call_conv,
                            gc_alloc_nursery_shim as *const () as usize,
                            &[ps],
                            Some(cl_types::I64),
                        )
                        .expect("alloc");
                        // _build_malloc_slowpath: _reload_frame, _pop_all_regs, pop_gcmap
                        let jf_ptr_slow =
                            emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                        emit_pop_gcmap(&mut builder, jf_ptr_slow, per_call_gcmap);
                        reload_ref_roots(
                            &mut builder,
                            jf_ptr_slow,
                            &ref_root_slots,
                            &defined_ref_vars,
                            ref_root_base_ofs,
                        );
                        // Pass GC-updated values through block params
                        let mut slow_args: Vec<BlockArg> =
                            vec![BlockArg::from(slow_r), BlockArg::from(jf_ptr_slow)];
                        for &(var_idx, _) in &live_refs {
                            slow_args.push(BlockArg::from(builder.use_var(var(var_idx))));
                        }
                        builder.ins().jump(merge_block, &slow_args);

                        // merge: receive values from block params
                        builder.switch_to_block(merge_block);
                        builder.seal_block(merge_block);
                        let params = builder.block_params(merge_block).to_vec();
                        let result = params[0];
                        jf_ptr = params[1];
                        builder.def_var(jf_ptr_var, jf_ptr);
                        for (i, &(var_idx, _)) in live_refs.iter().enumerate() {
                            builder.def_var(var(var_idx), params[2 + i]);
                        }
                        builder.def_var(var(vi), result);
                    } else {
                        // No active GC — refuse compile, matching the
                        // pre-existing contract that CallMallocNursery
                        // requires a configured `cpu.gc_ll_descr`.
                        return Err(missing_gc_runtime(op.opcode));
                    }
                }
                OpCode::CallMallocNurseryVarsize => {
                    if !cranelift_gc_active() {
                        return Err(missing_gc_runtime(op.opcode));
                    }
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("CallMallocNurseryVarsize must have an ArrayDescr");
                    let ad = descr
                        .as_array_descr()
                        .expect("CallMallocNurseryVarsize descr must be an ArrayDescr");
                    let base_size = builder.ins().iconst(cl_types::I64, ad.base_size() as i64);
                    let item_size = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                    // rewrite.py:858: args = [ConstInt(kind), ConstInt(itemsize), v_length]
                    let length =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(2));
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        gc_alloc_varsize_shim as *const () as usize,
                        &[base_size, item_size, length],
                        Some(cl_types::I64),
                    )
                    .expect("GC varsize allocation helper must return a value");
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    builder.def_var(jf_ptr_var, jf_ptr);
                    builder.def_var(var(vi), result);
                }
                OpCode::CallMallocNurseryVarsizeFrame => {
                    // x86/assembler.py:2567-2582 malloc_cond_varsize_frame:
                    // inline nursery bump alloc with variable size.
                    // Same fast-path/slow-path split as CallMallocNursery:
                    // the fast path is a bare bump allocation; only the
                    // slow path installs the gcmap and spills/reloads roots.
                    let (nf_addr, nt_addr) =
                        gc_nursery_addrs.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let flags = MemFlags::trusted();
                    let size_total =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(0));

                    let nf_ptr = builder.ins().iconst(ptr_type, nf_addr as i64);
                    let nt_ptr = builder.ins().iconst(ptr_type, nt_addr as i64);
                    let free = builder.ins().load(ptr_type, flags, nf_ptr, 0);
                    let new_free = builder.ins().iadd(free, size_total);
                    let top = builder.ins().load(ptr_type, flags, nt_ptr, 0);
                    let fits = builder
                        .ins()
                        .icmp(IntCC::UnsignedLessThanOrEqual, new_free, top);

                    // Same block-param pattern as CallMallocNursery.
                    let live_refs: Vec<(u32, usize)> = ref_root_slots
                        .iter()
                        .filter(|(var_idx, _)| defined_ref_vars.contains(var_idx))
                        .copied()
                        .collect();

                    let fast_block = builder.create_block();
                    let slow_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, ptr_type); // result
                    builder.append_block_param(merge_block, ptr_type); // jf_ptr
                    for _ in &live_refs {
                        builder.append_block_param(merge_block, cl_types::I64);
                    }

                    builder.ins().brif(fits, fast_block, &[], slow_block, &[]);

                    // fast: bump free pointer, zero GcHeader, return payload
                    builder.switch_to_block(fast_block);
                    builder.seal_block(fast_block);
                    builder.ins().store(flags, new_free, nf_ptr, 0);
                    let zero_hdr = builder.ins().iconst(cl_types::I64, 0);
                    builder.ins().store(MemFlags::trusted(), zero_hdr, free, 0);
                    let header_size = builder.ins().iconst(ptr_type, GcHeader::SIZE as i64);
                    let obj_ptr = builder.ins().iadd(free, header_size);
                    let mut fast_args: Vec<BlockArg> =
                        vec![BlockArg::from(obj_ptr), BlockArg::from(jf_ptr)];
                    for &(var_idx, _) in &live_refs {
                        fast_args.push(BlockArg::from(builder.use_var(var(var_idx))));
                    }
                    builder.ins().jump(merge_block, &fast_args);

                    // slow: host call (malloc_slowpath), may trigger GC.
                    builder.switch_to_block(slow_block);
                    builder.seal_block(slow_block);
                    builder.set_cold_block(slow_block);
                    if !cranelift_gc_active() {
                        return Err(missing_gc_runtime(op.opcode));
                    }
                    spill_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                    );
                    emit_push_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                    let size = builder.ins().iadd_imm(size_total, -(GcHeader::SIZE as i64));
                    let slow_result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        gc_alloc_nursery_shim as *const () as usize,
                        &[size],
                        Some(cl_types::I64),
                    )
                    .expect("GC frame allocation helper must return a value");
                    let jf_ptr_slow =
                        emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    emit_pop_gcmap(&mut builder, jf_ptr_slow, per_call_gcmap);
                    reload_ref_roots(
                        &mut builder,
                        jf_ptr_slow,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                    );
                    let mut slow_args: Vec<BlockArg> =
                        vec![BlockArg::from(slow_result), BlockArg::from(jf_ptr_slow)];
                    for &(var_idx, _) in &live_refs {
                        slow_args.push(BlockArg::from(builder.use_var(var(var_idx))));
                    }
                    builder.ins().jump(merge_block, &slow_args);

                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let params = builder.block_params(merge_block).to_vec();
                    let result = params[0];
                    jf_ptr = params[1];
                    builder.def_var(jf_ptr_var, jf_ptr);
                    for (i, &(var_idx, _)) in live_refs.iter().enumerate() {
                        builder.def_var(var(var_idx), params[2 + i]);
                    }
                    builder.def_var(var(vi), result);
                }

                // ── GC write barriers ──
                // aarch64/opassembler.py:912-1021 _write_barrier_fastpath parity:
                // Inline flag check → skip if flag not set → slow path call.
                // CondCallGcWbArray: card marking + post-helper re-test.
                OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                    if !cranelift_gc_active() {
                        return Err(missing_gc_runtime(op.opcode));
                    }
                    let obj = resolve_opref(&mut builder, &constants, op.arg(0));
                    let is_array = op.opcode == OpCode::CondCallGcWbArray;

                    // Load flag byte from object header.
                    let rw = self.gc_rewriter(&HashMap::new());
                    let wb_byteofs = rw
                        .as_ref()
                        .map(|r| r.wb_descr.jit_wb_if_flag_byteofs as i32)
                        .unwrap_or(0);
                    let wb_mask_raw = rw
                        .as_ref()
                        .map(|r| r.wb_descr.jit_wb_if_flag_singlebyte)
                        .unwrap_or(0);
                    let wb_cards_set = rw
                        .as_ref()
                        .map(|r| r.wb_descr.jit_wb_cards_set)
                        .unwrap_or(0);
                    let wb_card_shift = rw
                        .as_ref()
                        .map(|r| r.wb_descr.jit_wb_card_page_shift)
                        .unwrap_or(0);
                    let wb_cards_singlebyte = rw
                        .as_ref()
                        .map(|r| r.wb_descr.jit_wb_cards_set_singlebyte)
                        .unwrap_or(0);
                    drop(rw);

                    // opassembler.py:921-929: mask includes CARDS_SET singlebyte for array ops.
                    let wb_mask = if is_array && wb_cards_set != 0 {
                        (wb_mask_raw as i64) | (wb_cards_singlebyte as i64)
                    } else {
                        wb_mask_raw as i64
                    };

                    let flag_byte =
                        builder
                            .ins()
                            .load(cl_types::I8, MemFlags::trusted(), obj, wb_byteofs);
                    let flag_ext = builder.ins().uextend(cl_types::I64, flag_byte);
                    let mask_val = builder.ins().iconst(cl_types::I64, wb_mask & 0xFF);
                    let test = builder.ins().band(flag_ext, mask_val);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let needs_wb = builder.ins().icmp(IntCC::NotEqual, test, zero);

                    let slow_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(needs_wb, slow_block, &[], cont_block, &[]);

                    builder.switch_to_block(slow_block);
                    builder.seal_block(slow_block);
                    builder.set_cold_block(slow_block);

                    if is_array && wb_cards_set != 0 {
                        // opassembler.py:941-1021 card marking path.
                        let helper_call_block = builder.create_block();
                        let card_mark_block = builder.create_block();

                        // opassembler.py:944-949: pre-call CARDS_SET test.
                        let cards_mask_val = (wb_cards_singlebyte as u8) as i64;
                        let cards_mask = builder.ins().iconst(cl_types::I64, cards_mask_val);
                        let cards_test = builder.ins().band(flag_ext, cards_mask);
                        let has_cards = builder.ins().icmp(IntCC::NotEqual, cards_test, zero);
                        builder
                            .ins()
                            .brif(has_cards, card_mark_block, &[], helper_call_block, &[]);

                        // opassembler.py:953-980: array-specific helper call.
                        builder.switch_to_block(helper_call_block);
                        builder.seal_block(helper_call_block);
                        let _ = emit_host_call(
                            &mut builder,
                            ptr_type,
                            call_conv,
                            gc_jit_remember_young_pointer_from_array_shim as *const () as usize,
                            &[obj],
                            None,
                        );

                        // opassembler.py:982-987: post-helper re-load + re-test CARDS_SET.
                        let flag_byte2 =
                            builder
                                .ins()
                                .load(cl_types::I8, MemFlags::trusted(), obj, wb_byteofs);
                        let flag_ext2 = builder.ins().uextend(cl_types::I64, flag_byte2);
                        let cards_test2 = builder.ins().band(flag_ext2, cards_mask);
                        let has_cards2 = builder.ins().icmp(IntCC::NotEqual, cards_test2, zero);
                        builder
                            .ins()
                            .brif(has_cards2, card_mark_block, &[], cont_block, &[]);

                        // opassembler.py:994-1015: inline card bit setting.
                        builder.switch_to_block(card_mark_block);
                        builder.seal_block(card_mark_block);
                        let index = resolve_opref(&mut builder, &constants, op.arg(1));
                        let shift_plus_3 = (wb_card_shift + 3) as i64;
                        let shifted = builder.ins().ushr_imm(index, shift_plus_3);
                        let byteofs_val = builder.ins().bnot(shifted);
                        let bit_idx = builder.ins().ushr_imm(index, wb_card_shift as i64);
                        let seven = builder.ins().iconst(cl_types::I64, 7);
                        let bit_idx = builder.ins().band(bit_idx, seven);
                        let one = builder.ins().iconst(cl_types::I64, 1);
                        let bit_mask = builder.ins().ishl(one, bit_idx);
                        let card_addr = builder.ins().iadd(obj, byteofs_val);
                        let card_byte =
                            builder
                                .ins()
                                .load(cl_types::I8, MemFlags::trusted(), card_addr, 0);
                        let card_ext = builder.ins().uextend(cl_types::I64, card_byte);
                        let card_new = builder.ins().bor(card_ext, bit_mask);
                        let card_trunc = builder.ins().ireduce(cl_types::I8, card_new);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), card_trunc, card_addr, 0);
                        builder.ins().jump(cont_block, &[]);
                    } else {
                        // Simple write barrier (no card marking).
                        let _ = emit_host_call(
                            &mut builder,
                            ptr_type,
                            call_conv,
                            gc_write_barrier_shim as *const () as usize,
                            &[obj],
                            None,
                        );
                        builder.ins().jump(cont_block, &[]);
                    }

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Generic GC/raw memory loads ──
                OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF => {
                    let item_size = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(2),
                        "GC_LOAD itemsize",
                    )?;
                    let value_type = match op.opcode {
                        OpCode::GcLoadI => Type::Int,
                        OpCode::GcLoadR => Type::Ref,
                        OpCode::GcLoadF => Type::Float,
                        _ => unreachable!(),
                    };
                    if value_type != Type::Int && item_size < 0 {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "negative GC_LOAD itemsize is only valid for integer loads",
                        ));
                    }
                    let addr = emit_dynamic_offset_addr(
                        &mut builder,
                        &constants,
                        &known_values,
                        op.arg(0),
                        op.arg(1),
                    );
                    let result = emit_load_from_addr(
                        &mut builder,
                        addr,
                        value_type,
                        item_size.unsigned_abs() as usize,
                        item_size < 0,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), result);
                    if value_type == Type::Ref {
                        let cur_jf = builder.use_var(jf_ptr_var);
                        sync_ref_root_var(
                            &mut builder,
                            cur_jf,
                            &ref_root_slots,
                            vi,
                            result,
                            ref_root_base_ofs,
                            &mut synced_ref_vars,
                        );
                    }
                }
                OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                    let scale = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(2),
                        "GC_LOAD_INDEXED scale",
                    )?;
                    let base_offset = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(3),
                        "GC_LOAD_INDEXED base offset",
                    )?;
                    let item_size = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(4),
                        "GC_LOAD_INDEXED itemsize",
                    )?;
                    let value_type = match op.opcode {
                        OpCode::GcLoadIndexedI => Type::Int,
                        OpCode::GcLoadIndexedR => Type::Ref,
                        OpCode::GcLoadIndexedF => Type::Float,
                        _ => unreachable!(),
                    };
                    if value_type != Type::Int && item_size < 0 {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "negative GC_LOAD_INDEXED itemsize is only valid for integer loads",
                        ));
                    }
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        scale,
                        base_offset,
                    );
                    let result = emit_load_from_addr(
                        &mut builder,
                        addr,
                        value_type,
                        item_size.unsigned_abs() as usize,
                        item_size < 0,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), result);
                    if value_type == Type::Ref {
                        let cur_jf = builder.use_var(jf_ptr_var);
                        sync_ref_root_var(
                            &mut builder,
                            cur_jf,
                            &ref_root_slots,
                            vi,
                            result,
                            ref_root_base_ofs,
                            &mut synced_ref_vars,
                        );
                    }
                }
                OpCode::RawLoadI | OpCode::RawLoadF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("raw load op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("raw load descriptor must be an ArrayDescr");
                    let addr = emit_dynamic_offset_addr(
                        &mut builder,
                        &constants,
                        &known_values,
                        op.arg(0),
                        op.arg(1),
                    );
                    let value_type = match op.opcode {
                        OpCode::RawLoadI => Type::Int,
                        OpCode::RawLoadF => Type::Float,
                        _ => unreachable!(),
                    };
                    let signed = value_type == Type::Int && ad.is_item_signed();
                    let result = emit_load_from_addr(
                        &mut builder,
                        addr,
                        value_type,
                        ad.item_size(),
                        signed,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), result);
                }

                // ── GC stores ──
                OpCode::GcStore => {
                    // rewrite.py:140-158 emit_gc_store_or_indexed parity.
                    // 4-arg form: GC_STORE(base, ConstInt(offset), value, ConstInt(size))
                    if op.args.len() >= 4 {
                        let item_size = resolve_constant_i64(
                            &constants,
                            &known_values,
                            op.opcode,
                            op.arg(3),
                            "GC_STORE itemsize",
                        )?;
                        if std::env::var_os("MAJIT_CL_GCSTORE_LOG").is_some() {
                            let offset_dbg =
                                constants.get(&op.arg(1).raw()).copied().or_else(|| {
                                    op.arg(1)
                                        .is_constant()
                                        .then(|| op.arg(1).const_index() as i64)
                                });
                            eprintln!(
                                "[cl-gcstore] op_idx={} base={:?} offset_arg={:?} offset={:?} value_arg={:?} size={}",
                                op_idx,
                                op.arg(0),
                                op.arg(1),
                                offset_dbg,
                                op.arg(2),
                                item_size
                            );
                        }
                        let value_type = type_for_opref(
                            &type_index,
                            &type_overrides,
                            &known_values,
                            op.opcode,
                            op.arg(2),
                            op_idx,
                            "GC_STORE value",
                        )?;
                        let addr = emit_dynamic_offset_addr(
                            &mut builder,
                            &constants,
                            &known_values,
                            op.arg(0),
                            op.arg(1),
                        );
                        let value = resolve_opref_or_imm(
                            &mut builder,
                            &constants,
                            &known_values,
                            op.arg(2),
                        );
                        emit_store_to_addr(
                            &mut builder,
                            addr,
                            value,
                            value_type,
                            item_size.unsigned_abs() as usize,
                            op.opcode,
                        )?;
                    } else {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "GC_STORE expects 4-arg form: [base, offset, value, size]",
                        ));
                    }
                }
                OpCode::GcStoreIndexed => {
                    let scale = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(3),
                        "GC_STORE_INDEXED scale",
                    )?;
                    let base_offset = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(4),
                        "GC_STORE_INDEXED base offset",
                    )?;
                    let item_size = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(5),
                        "GC_STORE_INDEXED itemsize",
                    )?;
                    let value_type = type_for_opref(
                        &type_index,
                        &type_overrides,
                        &known_values,
                        op.opcode,
                        op.arg(2),
                        op_idx,
                        "GC_STORE_INDEXED value",
                    )?;
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        scale,
                        base_offset,
                    );
                    let value =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(2));
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        value,
                        value_type,
                        item_size.unsigned_abs() as usize,
                        op.opcode,
                    )?;
                }

                // ── Field access (getfield) ──
                // All getfield variants load from base + offset.
                // The loaded value is sign/zero-extended from its field_size to I64,
                // or bitcast from F64 for float fields.
                OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
                | OpCode::GetfieldGcPureI
                | OpCode::GetfieldGcPureR
                | OpCode::GetfieldGcPureF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("getfield op must have a descriptor");
                    let fd = descr
                        .as_field_descr()
                        .expect("getfield descriptor must be a FieldDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let addr = builder.ins().iadd_imm(base, fd.offset() as i64);
                    let r = emit_load_from_addr(
                        &mut builder,
                        addr,
                        fd.field_type(),
                        fd.field_size(),
                        fd.is_field_signed(),
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), r);
                }

                // ── Field access (setfield) ──
                // args[0] = base, args[1] = value
                OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("setfield op must have a descriptor");
                    let fd = descr
                        .as_field_descr()
                        .expect("setfield descriptor must be a FieldDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let val = resolve_opref(&mut builder, &constants, op.arg(1));
                    let addr = builder.ins().iadd_imm(base, fd.offset() as i64);
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        fd.field_type(),
                        fd.field_size(),
                        op.opcode,
                    )?;
                }

                // ── Array access (getarrayitem) ──
                // args[0] = base, args[1] = index
                // address = base + base_size + index * item_size
                OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemRawF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("getarrayitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("getarrayitem descriptor must be an ArrayDescr");

                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        ad.base_size() as i64,
                    );
                    let signed = ad.item_type() == Type::Int && ad.is_item_signed();
                    let r = emit_load_from_addr(
                        &mut builder,
                        addr,
                        ad.item_type(),
                        ad.item_size(),
                        signed,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), r);
                }

                // ── Array access (setarrayitem) ──
                // args[0] = base, args[1] = index, args[2] = value
                OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("setarrayitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("setarrayitem descriptor must be an ArrayDescr");

                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        ad.base_size() as i64,
                    );
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        ad.item_type(),
                        ad.item_size(),
                        op.opcode,
                    )?;
                }

                // ── Interior field access ──
                OpCode::GetinteriorfieldGcI
                | OpCode::GetinteriorfieldGcR
                | OpCode::GetinteriorfieldGcF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("getinteriorfield op must have a descriptor");
                    let id = descr
                        .as_interior_field_descr()
                        .expect("getinteriorfield descriptor must be an InteriorFieldDescr");
                    let ad = id.array_descr();
                    let fd = id.field_descr();
                    let base_offset = (ad.base_size() + fd.offset()) as i64;
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        base_offset,
                    );
                    let r = emit_load_from_addr(
                        &mut builder,
                        addr,
                        fd.field_type(),
                        fd.field_size(),
                        fd.is_field_signed(),
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), r);
                }

                OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("setinteriorfield op must have a descriptor");
                    let id = descr
                        .as_interior_field_descr()
                        .expect("setinteriorfield descriptor must be an InteriorFieldDescr");
                    let ad = id.array_descr();
                    let fd = id.field_descr();
                    let base_offset = (ad.base_size() + fd.offset()) as i64;
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        base_offset,
                    );
                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        fd.field_type(),
                        fd.field_size(),
                        op.opcode,
                    )?;
                }

                OpCode::RawStore => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("raw store op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("raw store descriptor must be an ArrayDescr");
                    let addr = emit_dynamic_offset_addr(
                        &mut builder,
                        &constants,
                        &known_values,
                        op.arg(0),
                        op.arg(1),
                    );
                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        ad.item_type(),
                        ad.item_size(),
                        op.opcode,
                    )?;
                }

                // ── Array/string length ──
                // These load the length field from the object header using
                // the array descriptor's len_descr.
                OpCode::ArraylenGc => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("arraylen op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("arraylen descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    if let Some(ld) = ad.len_descr() {
                        let addr = builder.ins().iadd_imm(base, ld.offset() as i64);
                        let r = emit_load_from_addr(
                            &mut builder,
                            addr,
                            ld.field_type(),
                            ld.field_size(),
                            ld.is_field_signed(),
                            op.opcode,
                        )?;
                        builder.def_var(var(vi), r);
                    } else {
                        // No len_descr: return 0 as a fallback.
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var(vi), zero);
                    }
                }

                OpCode::Strlen | OpCode::Unicodelen => {
                    // These use the array descriptor attached to the op.
                    // The length is at len_descr().offset() from the base pointer.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("strlen/unicodelen op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("strlen/unicodelen descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    if let Some(ld) = ad.len_descr() {
                        let addr = builder.ins().iadd_imm(base, ld.offset() as i64);
                        let r = emit_load_from_addr(
                            &mut builder,
                            addr,
                            ld.field_type(),
                            ld.field_size(),
                            ld.is_field_signed(),
                            op.opcode,
                        )?;
                        builder.def_var(var(vi), r);
                    } else {
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var(vi), zero);
                    }
                }

                OpCode::Strhash | OpCode::Unicodehash => {
                    // rewrite.py:283-294 parity — `get_field_token(rstr.STR,
                    // 'hash', ...)`.  Injected upstream of GC rewriter; see
                    // `builtin_string_hash_field_descr` above.  The assert
                    // `size == WORD` at rewrite.py:286,292 mirrors here.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("strhash/unicodehash op must have a descriptor");
                    let fd = descr
                        .as_field_descr()
                        .expect("strhash/unicodehash descriptor must be a FieldDescr");
                    assert_eq!(fd.field_size(), std::mem::size_of::<usize>());
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let addr = builder.ins().iadd_imm(base, fd.offset() as i64);
                    let hash = emit_load_from_addr(
                        &mut builder,
                        addr,
                        fd.field_type(),
                        fd.field_size(),
                        fd.is_field_signed(),
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), hash);
                }

                // ── String/unicode item access ──
                // Strgetitem/Unicodegetitem: args[0] = base, args[1] = index
                // Treated as array item access using the descriptor's base_size/item_size.
                OpCode::Strgetitem | OpCode::Unicodegetitem => {
                    // rewrite.py:295-306 parity — `get_array_token(rstr.STR,
                    // ...)` returns `basesize` already inclusive of the STR
                    // `extra_item_after_alloc=1` null terminator, and
                    // rewrite.py:299/311 subtracts it before emitting the
                    // address arithmetic.  UNICODE has no extra item.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("str/unicodegetitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("str/unicodegetitem descriptor must be an ArrayDescr");
                    let itemsize = ad.item_size() as i64;
                    let basesize = if itemsize == 1 {
                        ad.base_size() as i64 - 1
                    } else {
                        ad.base_size() as i64
                    };

                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        itemsize,
                        basesize,
                    );
                    let r = emit_load_from_addr(
                        &mut builder,
                        addr,
                        Type::Int,
                        ad.item_size(),
                        false,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), r);
                }

                // Strsetitem/Unicodesetitem: args[0] = base, args[1] = index, args[2] = value
                OpCode::Strsetitem | OpCode::Unicodesetitem => {
                    // rewrite.py:307-318 parity.  See STRGETITEM note above
                    // for the `basesize -= 1` rationale.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("str/unicodesetitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("str/unicodesetitem descriptor must be an ArrayDescr");
                    let itemsize = ad.item_size() as i64;
                    let basesize = if itemsize == 1 {
                        ad.base_size() as i64 - 1
                    } else {
                        ad.base_size() as i64
                    };

                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        itemsize,
                        basesize,
                    );
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        Type::Int,
                        ad.item_size(),
                        op.opcode,
                    )?;
                }

                OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                    // rstr.py:1204-1215 parity — the source/destination
                    // addresses skip the extra null character for STR, just
                    // like STR{GET,SET}ITEM.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("copystr/unicodecontent op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("copystr/unicodecontent descriptor must be an ArrayDescr");
                    let itemsize = ad.item_size() as i64;
                    let basesize = if itemsize == 1 {
                        ad.base_size() as i64 - 1
                    } else {
                        ad.base_size() as i64
                    };

                    let src_addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(2),
                        itemsize,
                        basesize,
                    );
                    let dst_addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(1),
                        op.arg(3),
                        itemsize,
                        basesize,
                    );
                    let length =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(4));
                    let byte_length = if ad.item_size() == 1 {
                        length
                    } else {
                        let scale = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                        builder.ins().imul(length, scale)
                    };

                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        copy_nonoverlapping_memory_shim as *const () as usize,
                        &[src_addr, dst_addr, byte_length],
                        None,
                    );
                }

                // ── Zero array range ──
                // args = [base, start, size, scale_start, scale_size]
                // zeroes bytes in
                // [base + descr.base_size + start*scale_start,
                //  base + descr.base_size + start*scale_start + size*scale_size)
                OpCode::ZeroArray => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("zero_array op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("zero_array descriptor must be an ArrayDescr");
                    let scale_start = resolve_rewriter_immediate_i64(&constants, op.arg(3));
                    let scale_size = resolve_rewriter_immediate_i64(&constants, op.arg(4));
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let start = builder.ins().iconst(
                        cl_types::I64,
                        resolve_rewriter_immediate_i64(&constants, op.arg(1)),
                    );
                    let size = builder.ins().iconst(
                        cl_types::I64,
                        resolve_rewriter_immediate_i64(&constants, op.arg(2)),
                    );

                    let start_bytes = match scale_start {
                        0 => builder.ins().iconst(cl_types::I64, 0),
                        1 => start,
                        _ => {
                            let scale = builder.ins().iconst(cl_types::I64, scale_start);
                            builder.ins().imul(start, scale)
                        }
                    };
                    let byte_offset = builder.ins().iadd_imm(start_bytes, ad.base_size() as i64);
                    let byte_size = match scale_size {
                        0 => builder.ins().iconst(cl_types::I64, 0),
                        1 => size,
                        _ => {
                            let scale = builder.ins().iconst(cl_types::I64, scale_size);
                            builder.ins().imul(size, scale)
                        }
                    };

                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        zero_memory_shim as *const () as usize,
                        &[base, byte_offset, byte_size],
                        None,
                    );
                }

                // ── Nursery pointer increment ──
                // args[0] = base ptr, args[1] = byte offset
                OpCode::NurseryPtrIncrement => {
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let offset = resolve_opref(&mut builder, &constants, op.arg(1));
                    let r = builder.ins().iadd(base, offset);
                    builder.def_var(var(vi), r);
                }

                // ── Control flow ──
                OpCode::Jump => {
                    let vals: Vec<CValue> = op
                        .args
                        .iter()
                        .map(|&r| resolve_opref(&mut builder, &constants, r))
                        .collect();
                    // Resolve target: try descr-based lookup first, then
                    // fall back to loop_block (Jump without descr targets
                    // the loop header, matching RPython's self-loop).
                    // rewriter.py LABEL/JUMP redirect parity: a local target
                    // whose original label arity differs from the JUMP arg
                    // count lowers as external jump (Cranelift's
                    // FunctionBuilder may have auto-promoted `def_var` chains
                    // into block params, so the runtime num_block_params is
                    // unreliable — compare against the arity captured before
                    // codegen).
                    let local_target_arity_matches = op.descr.as_ref().is_some_and(|d| {
                        label_arity_by_descr
                            .get(&d.index())
                            .copied()
                            .is_some_and(|arity| arity == op.args.len())
                    });
                    let target_block = if local_target_arity_matches {
                        op.descr
                            .as_ref()
                            .and_then(|descr| label_blocks_by_descr.get(&descr.index()).copied())
                    } else if op.descr.is_some() {
                        // Descr present but either points outside this function
                        // (bridge → main loop) or to a local label with
                        // mismatched arity — compile as external jump.
                        None
                    } else if loop_block != entry_block {
                        // Implicit self-loop: Jump without descr.
                        Some(loop_block)
                    } else {
                        None
                    };
                    if let Some(target_block) = target_block {
                        builder.ins().jump(target_block, &block_args(&vals));
                    } else {
                        // External JUMP (bridge → loop body) — emit as
                        // Finish exit. The dispatcher will re-enter the
                        // target loop with these output values.
                        let info = &guard_infos[guard_idx];
                        guard_idx += 1;
                        let cur_jf = builder.use_var(jf_ptr_var);
                        emit_guard_exit(
                            &mut builder,
                            &constants,
                            cur_jf,
                            info,
                            &ref_root_slots,
                            &stale_ref_vars,
                            ref_root_base_ofs,
                            ptr_type,
                            call_conv,
                        );
                    }
                }

                OpCode::Finish => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );
                }

                OpCode::Label => {}

                // ── Float arithmetic ──
                OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let fr = match op.opcode {
                        OpCode::FloatAdd => builder.ins().fadd(fa, fb),
                        OpCode::FloatSub => builder.ins().fsub(fa, fb),
                        OpCode::FloatMul => builder.ins().fmul(fa, fb),
                        OpCode::FloatTrueDiv => builder.ins().fdiv(fa, fb),
                        _ => unreachable!(),
                    };
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatFloorDiv => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let fdiv = builder.ins().fdiv(fa, fb);
                    let fr = builder.ins().floor(fdiv);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatMod => {
                    // Python's float mod: a - floor(a/b) * b
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let fdiv = builder.ins().fdiv(fa, fb);
                    let ffloor = builder.ins().floor(fdiv);
                    let prod = builder.ins().fmul(ffloor, fb);
                    let fr = builder.ins().fsub(fa, prod);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatNeg => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fr = builder.ins().fneg(fa);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatAbs => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fr = builder.ins().fabs(fa);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }

                // ── Casts ──
                OpCode::CastFloatToInt => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let r = builder.ins().fcvt_to_sint(cl_types::I64, fa);
                    builder.def_var(var(vi), r);
                }
                OpCode::CastIntToFloat => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fr = builder.ins().fcvt_from_sint(cl_types::F64, a);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::ConvertFloatBytesToLonglong | OpCode::ConvertLonglongBytesToFloat => {
                    // Both are identity in our I64 storage scheme.
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), a);
                }

                // ── Debug / no-op operations ──
                OpCode::DebugMergePoint
                | OpCode::EnterPortalFrame
                | OpCode::LeavePortalFrame
                | OpCode::JitDebug
                | OpCode::Keepalive
                | OpCode::ForceSpill
                | OpCode::VirtualRefFinish
                | OpCode::RecordExactClass
                | OpCode::RecordExactValueR
                | OpCode::RecordExactValueI
                | OpCode::RecordKnownResult
                | OpCode::QuasiimmutField
                | OpCode::AssertNotNone
                | OpCode::IncrementDebugCounter => {
                    // No-op markers or optimizer hints.
                }

                // ── ForceToken ──
                OpCode::ForceToken => {
                    // x86/assembler.py genop_force_token: mov resloc, ebp
                    // FORCE_TOKEN returns the JitFrame pointer itself.
                    // resoperation.py:1090: "returns the jitframe"
                    let cur_jf = builder.use_var(jf_ptr_var);
                    builder.def_var(var(vi), cur_jf);
                }

                // ── VirtualRef operations ──
                // When VirtualRef survives the optimizer, the backend
                // materializes it by returning the underlying object reference.
                // args[0] = the real object, args[1] = vref_id
                OpCode::VirtualRefI | OpCode::VirtualRefR => {
                    let obj = resolve_opref(&mut builder, &constants, op.args[0]);
                    builder.def_var(var(vi), obj);
                    if op.opcode == OpCode::VirtualRefR {
                        let cur_jf = builder.use_var(jf_ptr_var);
                        sync_ref_root_var(
                            &mut builder,
                            cur_jf,
                            &ref_root_slots,
                            vi,
                            obj,
                            ref_root_base_ofs,
                            &mut synced_ref_vars,
                        );
                    }
                }

                // ── Vector guards ──
                OpCode::VecGuardTrue => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let cond = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_false = builder.ins().icmp(IntCC::Equal, cond, zero);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder
                        .ins()
                        .brif(is_false, exit_block, &[], cont_block, &[]);
                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }
                OpCode::VecGuardFalse => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let cond = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_true = builder.ins().icmp(IntCC::NotEqual, cond, zero);
                    let exit_block = builder.create_block();
                    builder.set_cold_block(exit_block);
                    let cont_block = builder.create_block();
                    if preamble_phase {
                        builder.set_cold_block(cont_block);
                    }
                    builder
                        .ins()
                        .brif(is_true, exit_block, &[], cont_block, &[]);
                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let cur_jf = builder.use_var(jf_ptr_var);
                    emit_guard_exit(
                        &mut builder,
                        &constants,
                        cur_jf,
                        info,
                        &ref_root_slots,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        ptr_type,
                        call_conv,
                    );
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Vector integer arithmetic ──
                // Native SIMD path: I64X2 (128-bit, 2x i64) / F64X2 (128-bit, 2x f64).
                // Scalar emulation fallback for portability.
                OpCode::VecIntAdd
                | OpCode::VecIntSub
                | OpCode::VecIntMul
                | OpCode::VecIntAnd
                | OpCode::VecIntOr
                | OpCode::VecIntXor => {
                    if USE_NATIVE_SIMD {
                        let a = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let b = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[1],
                        );
                        let result = match op.opcode {
                            OpCode::VecIntAdd => builder.ins().iadd(a, b),
                            OpCode::VecIntSub => builder.ins().isub(a, b),
                            OpCode::VecIntMul => builder.ins().imul(a, b),
                            OpCode::VecIntAnd => builder.ins().band(a, b),
                            OpCode::VecIntOr => builder.ins().bor(a, b),
                            OpCode::VecIntXor => builder.ins().bxor(a, b),
                            _ => unreachable!(),
                        };
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let b = resolve_opref(&mut builder, &constants, op.args[1]);
                        let result = match op.opcode {
                            OpCode::VecIntAdd => builder.ins().iadd(a, b),
                            OpCode::VecIntSub => builder.ins().isub(a, b),
                            OpCode::VecIntMul => builder.ins().imul(a, b),
                            OpCode::VecIntAnd => builder.ins().band(a, b),
                            OpCode::VecIntOr => builder.ins().bor(a, b),
                            OpCode::VecIntXor => builder.ins().bxor(a, b),
                            _ => unreachable!(),
                        };
                        builder.def_var(var(vi), result);
                    }
                }

                // ── Vector float arithmetic ──
                OpCode::VecFloatAdd
                | OpCode::VecFloatSub
                | OpCode::VecFloatMul
                | OpCode::VecFloatTrueDiv => {
                    if USE_NATIVE_SIMD {
                        let a = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let b = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[1],
                        );
                        let result = match op.opcode {
                            OpCode::VecFloatAdd => builder.ins().fadd(a, b),
                            OpCode::VecFloatSub => builder.ins().fsub(a, b),
                            OpCode::VecFloatMul => builder.ins().fmul(a, b),
                            OpCode::VecFloatTrueDiv => builder.ins().fdiv(a, b),
                            _ => unreachable!(),
                        };
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let b = resolve_opref(&mut builder, &constants, op.args[1]);
                        let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                        let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                        let fresult = match op.opcode {
                            OpCode::VecFloatAdd => builder.ins().fadd(fa, fb),
                            OpCode::VecFloatSub => builder.ins().fsub(fa, fb),
                            OpCode::VecFloatMul => builder.ins().fmul(fa, fb),
                            OpCode::VecFloatTrueDiv => builder.ins().fdiv(fa, fb),
                            _ => unreachable!(),
                        };
                        let result = builder
                            .ins()
                            .bitcast(cl_types::I64, MemFlags::new(), fresult);
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::VecFloatNeg => {
                    if USE_NATIVE_SIMD {
                        let a = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let result = builder.ins().fneg(a);
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                        let fresult = builder.ins().fneg(fa);
                        let result = builder
                            .ins()
                            .bitcast(cl_types::I64, MemFlags::new(), fresult);
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::VecFloatAbs => {
                    if USE_NATIVE_SIMD {
                        let a = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let result = builder.ins().fabs(a);
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                        let fresult = builder.ins().fabs(fa);
                        let result = builder
                            .ins()
                            .bitcast(cl_types::I64, MemFlags::new(), fresult);
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::VecFloatXor => {
                    if USE_NATIVE_SIMD {
                        // XOR on the raw bits: operate on I64X2 representation
                        let a = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let b = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[1],
                        );
                        let xored = builder.ins().bxor(a, b);
                        // Result is declared as F64X2, bitcast from I64X2
                        let result = builder
                            .ins()
                            .bitcast(cl_types::F64X2, MemFlags::new(), xored);
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let b = resolve_opref(&mut builder, &constants, op.args[1]);
                        let result = builder.ins().bxor(a, b);
                        builder.def_var(var(vi), result);
                    }
                }

                // ── Vector comparison/test operations ──
                // These always produce scalar results (I64), not vectors.
                OpCode::VecFloatEq => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let cmp = builder.ins().fcmp(FloatCC::Equal, fa, fb);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecFloatNe => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let cmp = builder.ins().fcmp(FloatCC::NotEqual, fa, fb);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecIntIsTrue => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::NotEqual, a, zero);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecIntEq => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let cmp = builder.ins().icmp(IntCC::Equal, a, b);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecIntNe => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let cmp = builder.ins().icmp(IntCC::NotEqual, a, b);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecIntSignext => {
                    // Sign-extend a narrower integer value to i64
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    builder.def_var(var(vi), a); // already i64
                }

                // ── Vector cast operations ──
                // These operate on scalar elements, not full vectors.
                OpCode::VecCastFloatToInt => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let result = builder.ins().fcvt_to_sint(cl_types::I64, fa);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecCastIntToFloat => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let fresult = builder.ins().fcvt_from_sint(cl_types::F64, a);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I64, MemFlags::new(), fresult);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecCastFloatToSinglefloat => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let f32val = builder.ins().fdemote(cl_types::F32, fa);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I32, MemFlags::new(), f32val);
                    let result_ext = builder.ins().uextend(cl_types::I64, result);
                    builder.def_var(var(vi), result_ext);
                }

                OpCode::VecCastSinglefloatToFloat => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let a_trunc = builder.ins().ireduce(cl_types::I32, a);
                    let f32val = builder
                        .ins()
                        .bitcast(cl_types::F32, MemFlags::new(), a_trunc);
                    let f64val = builder.ins().fpromote(cl_types::F64, f32val);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I64, MemFlags::new(), f64val);
                    builder.def_var(var(vi), result);
                }

                // ── Vector pack/unpack/expand ──
                OpCode::VecI | OpCode::VecF => {
                    if USE_NATIVE_SIMD {
                        // Zero-initialized vector
                        if op.opcode == OpCode::VecF {
                            let zero_i = builder.ins().iconst(cl_types::I64, 0);
                            let zero_f =
                                builder
                                    .ins()
                                    .bitcast(cl_types::F64, MemFlags::new(), zero_i);
                            let result = builder.ins().splat(cl_types::F64X2, zero_f);
                            builder.def_var(var(vi), result);
                        } else {
                            let zero = builder.ins().iconst(cl_types::I64, 0);
                            let result = builder.ins().splat(cl_types::I64X2, zero);
                            builder.def_var(var(vi), result);
                        }
                    } else {
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var(vi), zero);
                    }
                }

                OpCode::VecPackI => {
                    if USE_NATIVE_SIMD {
                        // vec_pack(vec, scalar, lane_const, count_const)
                        // insertlane(vec, scalar, lane_idx)
                        let vec_val = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let scalar = resolve_opref(&mut builder, &constants, op.args[1]);
                        let lane = constants.get(&op.args[2].raw()).copied().unwrap_or(0) as u8;
                        let result = builder.ins().insertlane(vec_val, scalar, lane);
                        builder.def_var(var(vi), result);
                    } else {
                        let scalar = resolve_opref(&mut builder, &constants, op.args[1]);
                        builder.def_var(var(vi), scalar);
                    }
                }

                OpCode::VecPackF => {
                    if USE_NATIVE_SIMD {
                        // vec_pack(vec, scalar, lane_const, count_const)
                        // insertlane(vec, scalar, lane_idx)
                        let vec_val = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let scalar_i = resolve_opref(&mut builder, &constants, op.args[1]);
                        let scalar_f =
                            builder
                                .ins()
                                .bitcast(cl_types::F64, MemFlags::new(), scalar_i);
                        let lane = constants.get(&op.args[2].raw()).copied().unwrap_or(0) as u8;
                        let result = builder.ins().insertlane(vec_val, scalar_f, lane);
                        builder.def_var(var(vi), result);
                    } else {
                        let scalar = resolve_opref(&mut builder, &constants, op.args[1]);
                        builder.def_var(var(vi), scalar);
                    }
                }

                OpCode::VecUnpackI => {
                    if USE_NATIVE_SIMD {
                        // vec_unpack(vec, lane_const, count_const) → scalar i64
                        let vec_val = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let lane = constants.get(&op.args[1].raw()).copied().unwrap_or(0) as u8;
                        let result = builder.ins().extractlane(vec_val, lane);
                        builder.def_var(var(vi), result);
                    } else {
                        let vec_val = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.def_var(var(vi), vec_val);
                    }
                }

                OpCode::VecUnpackF => {
                    if USE_NATIVE_SIMD {
                        // vec_unpack(vec, lane_const, count_const) → scalar f64 as i64
                        let vec_val = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let lane = constants.get(&op.args[1].raw()).copied().unwrap_or(0) as u8;
                        let scalar_f = builder.ins().extractlane(vec_val, lane);
                        let result =
                            builder
                                .ins()
                                .bitcast(cl_types::I64, MemFlags::new(), scalar_f);
                        builder.def_var(var(vi), result);
                    } else {
                        let vec_val = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.def_var(var(vi), vec_val);
                    }
                }

                OpCode::VecExpandI => {
                    if USE_NATIVE_SIMD {
                        // Broadcast scalar to all lanes
                        let scalar = resolve_opref(&mut builder, &constants, op.args[0]);
                        let result = builder.ins().splat(cl_types::I64X2, scalar);
                        builder.def_var(var(vi), result);
                    } else {
                        let scalar = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.def_var(var(vi), scalar);
                    }
                }

                OpCode::VecExpandF => {
                    if USE_NATIVE_SIMD {
                        // Broadcast scalar f64 to all lanes
                        let scalar_i = resolve_opref(&mut builder, &constants, op.args[0]);
                        let scalar_f =
                            builder
                                .ins()
                                .bitcast(cl_types::F64, MemFlags::new(), scalar_i);
                        let result = builder.ins().splat(cl_types::F64X2, scalar_f);
                        builder.def_var(var(vi), result);
                    } else {
                        let scalar = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.def_var(var(vi), scalar);
                    }
                }

                // ── Vector load/store ──
                OpCode::VecLoadI | OpCode::VecLoadF => {
                    if USE_NATIVE_SIMD {
                        // Load 128 bits (2x i64 or 2x f64) from memory
                        let base = resolve_opref(&mut builder, &constants, op.args[0]);
                        let offset_val = if op.args.len() > 1 {
                            resolve_opref(&mut builder, &constants, op.args[1])
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };
                        let addr = builder.ins().iadd(base, offset_val);
                        let load_type = if op.opcode == OpCode::VecLoadF {
                            cl_types::F64X2
                        } else {
                            cl_types::I64X2
                        };
                        let result = builder.ins().load(load_type, MemFlags::trusted(), addr, 0);
                        builder.def_var(var(vi), result);
                    } else {
                        let base = resolve_opref(&mut builder, &constants, op.args[0]);
                        let offset_val = if op.args.len() > 1 {
                            resolve_opref(&mut builder, &constants, op.args[1])
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };
                        let addr = builder.ins().iadd(base, offset_val);
                        let result =
                            builder
                                .ins()
                                .load(cl_types::I64, MemFlags::trusted(), addr, 0);
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::VecStore => {
                    if USE_NATIVE_SIMD {
                        // Store 128 bits to memory
                        let base = resolve_opref(&mut builder, &constants, op.args[0]);
                        let offset_val = if op.args.len() > 2 {
                            resolve_opref(&mut builder, &constants, op.args[1])
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };
                        let value_ref = op.args[op.args.len() - 1];
                        let value = if vec_oprefs.contains(&value_ref.raw()) {
                            builder.use_var(var(value_ref.raw()))
                        } else {
                            resolve_opref(&mut builder, &constants, value_ref)
                        };
                        let addr = builder.ins().iadd(base, offset_val);
                        builder.ins().store(MemFlags::trusted(), value, addr, 0);
                    } else {
                        let base = resolve_opref(&mut builder, &constants, op.args[0]);
                        let offset_val = if op.args.len() > 2 {
                            resolve_opref(&mut builder, &constants, op.args[1])
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };
                        let value =
                            resolve_opref(&mut builder, &constants, op.args[op.args.len() - 1]);
                        let addr = builder.ins().iadd(base, offset_val);
                        builder.ins().store(MemFlags::trusted(), value, addr, 0);
                    }
                }

                // ── Allocation opcodes ──
                // These are normally eliminated by the optimizer (virtualize pass)
                // or rewritten by the GC rewriter. If they reach the backend,
                // we call out to a runtime helper.
                OpCode::New | OpCode::NewWithVtable => {
                    let sd = op.descr.as_ref().and_then(|d| d.as_size_descr());
                    let (size, type_id, vtable) = sd.map_or((16, 0, 0usize), |sd| {
                        (sd.size() as i64, sd.type_id() as i64, sd.vtable())
                    });
                    let size_val = builder.ins().iconst(cl_types::I64, size);
                    let type_id_val = builder.ins().iconst(cl_types::I64, type_id);
                    // llmodel.py:778-782 bh_new_with_vtable:
                    //   res = self.gc_ll_descr.gc_malloc(sizedescr)
                    //   if self.vtable_offset is not None:
                    //       self.write_int_at_mem(res, self.vtable_offset, WORD,
                    //                             sizedescr.get_vtable())
                    //   return res
                    // The vtable is written at backend-configured `vtable_offset`,
                    // not at a fixed offset. When `vtable_offset is None` (e.g.
                    // gcremovetypeptr is enabled, llmodel.py:64-65), no write.
                    let write_vtable = op.opcode == OpCode::NewWithVtable
                        && vtable != 0
                        && vtable_offset.is_some();
                    let vtable_off_i32 = vtable_offset.unwrap_or(0) as i32;
                    if cranelift_gc_active() {
                        let cur_jf = builder.use_var(jf_ptr_var);
                        let result = emit_collecting_gc_call(
                            &mut builder,
                            ptr_type,
                            call_conv,
                            cur_jf,
                            &ref_root_slots,
                            &defined_ref_vars,
                            &stale_ref_vars,
                            ref_root_base_ofs,
                            per_call_gcmap,
                            gc_alloc_typed_nursery_shim as *const () as usize,
                            &[type_id_val, size_val],
                            Some(cl_types::I64),
                        )
                        .expect("GC allocation helper must return a value");
                        // assembler.py:405-412 _reload_frame_if_necessary:
                        // GC may have moved the jitframe during allocation.
                        // Reload jf_ptr so subsequent spill/reload use the correct address.
                        jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                        builder.def_var(jf_ptr_var, jf_ptr);
                        if write_vtable {
                            let vtable_val = builder.ins().iconst(cl_types::I64, vtable as i64);
                            builder.ins().store(
                                cranelift_codegen::ir::MemFlags::trusted(),
                                vtable_val,
                                result,
                                vtable_off_i32,
                            );
                        }
                        builder.def_var(var(vi), result);
                        sync_ref_root_var(
                            &mut builder,
                            jf_ptr,
                            &ref_root_slots,
                            vi,
                            result,
                            ref_root_base_ofs,
                            &mut synced_ref_vars,
                        );
                    } else {
                        // No GC runtime: plain malloc fallback for non-GC languages.
                        let alloc_fn = builder
                            .ins()
                            .iconst(ptr_type, plain_malloc_zeroed_shim as *const () as i64);
                        let sig = {
                            let mut sig = cranelift_codegen::ir::Signature::new(call_conv);
                            sig.params
                                .push(cranelift_codegen::ir::AbiParam::new(cl_types::I64));
                            sig.returns
                                .push(cranelift_codegen::ir::AbiParam::new(cl_types::I64));
                            builder.import_signature(sig)
                        };
                        let call = builder.ins().call_indirect(sig, alloc_fn, &[size_val]);
                        let result = builder.inst_results(call)[0];
                        if write_vtable {
                            let vtable_val = builder.ins().iconst(cl_types::I64, vtable as i64);
                            builder.ins().store(
                                cranelift_codegen::ir::MemFlags::trusted(),
                                vtable_val,
                                result,
                                vtable_off_i32,
                            );
                        }
                        builder.def_var(var(vi), result);
                        sync_ref_root_var(
                            &mut builder,
                            jf_ptr,
                            &ref_root_slots,
                            vi,
                            result,
                            ref_root_base_ofs,
                            &mut synced_ref_vars,
                        );
                    }
                }
                OpCode::NewArray | OpCode::NewArrayClear => {
                    if !cranelift_gc_active() {
                        return Err(missing_gc_runtime(op.opcode));
                    }
                    let length =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(0));
                    let (base_size, item_size) =
                        if let Some(ad) = op.descr.as_ref().and_then(|d| d.as_array_descr()) {
                            (
                                builder.ins().iconst(cl_types::I64, ad.base_size() as i64),
                                builder.ins().iconst(cl_types::I64, ad.item_size() as i64),
                            )
                        } else {
                            (
                                builder.ins().iconst(cl_types::I64, 16),
                                builder.ins().iconst(cl_types::I64, 8),
                            )
                        };
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        gc_alloc_varsize_shim as *const () as usize,
                        &[base_size, item_size, length],
                        Some(cl_types::I64),
                    )
                    .expect("GC varsize allocation helper must return a value");
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    builder.def_var(jf_ptr_var, jf_ptr);
                    builder.def_var(var(vi), result);
                    sync_ref_root_var(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        vi,
                        result,
                        ref_root_base_ofs,
                        &mut synced_ref_vars,
                    );
                }

                // ── Integer sign extension ──
                OpCode::IntSignext => {
                    let val = resolve_opref(&mut builder, &constants, op.arg(0));
                    let num_bits = resolve_opref(&mut builder, &constants, op.arg(1));

                    // Sign extend from `num_bits` to 64 bits
                    // shift left by (64 - num_bits), then arithmetic shift right by same
                    let sixty_four = builder.ins().iconst(cl_types::I64, 64);
                    let shift = builder.ins().isub(sixty_four, num_bits);
                    let shifted_left = builder.ins().ishl(val, shift);
                    let result = builder.ins().sshr(shifted_left, shift);
                    builder.def_var(var(vi), result);
                }

                // ── Unsigned multiply high ──
                OpCode::UintMulHigh => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let b = resolve_opref(&mut builder, &constants, op.arg(1));
                    let result = builder.ins().umulhi(a, b);
                    builder.def_var(var(vi), result);
                }

                // ── Float ↔ SingleFloat casts ──
                OpCode::CastFloatToSinglefloat => {
                    let val = resolve_opref(&mut builder, &constants, op.arg(0));
                    // i64-encoded f64 → f64 → f32 → zero-extend to i64
                    let f64_val = builder.ins().bitcast(cl_types::F64, MemFlags::new(), val);
                    let f32_val = builder.ins().fdemote(cl_types::F32, f64_val);
                    let i32_val = builder
                        .ins()
                        .bitcast(cl_types::I32, MemFlags::new(), f32_val);
                    let result = builder.ins().uextend(cl_types::I64, i32_val);
                    builder.def_var(var(vi), result);
                }

                OpCode::CastSinglefloatToFloat => {
                    let val = resolve_opref(&mut builder, &constants, op.arg(0));
                    // i64 (lower 32 bits = f32) → f32 → f64 → i64
                    let i32_val = builder.ins().ireduce(cl_types::I32, val);
                    let f32_val = builder
                        .ins()
                        .bitcast(cl_types::F32, MemFlags::new(), i32_val);
                    let f64_val = builder.ins().fpromote(cl_types::F64, f32_val);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I64, MemFlags::new(), f64_val);
                    builder.def_var(var(vi), result);
                }

                // ── Raw array item read (ref-typed) ──
                OpCode::GetarrayitemRawR => {
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let index = resolve_opref(&mut builder, &constants, op.arg(1));
                    let scale = builder.ins().iconst(cl_types::I64, 8);
                    let offset = builder.ins().imul(index, scale);
                    let addr = builder.ins().iadd(base, offset);
                    let result = builder
                        .ins()
                        .load(cl_types::I64, MemFlags::trusted(), addr, 0);
                    builder.def_var(var(vi), result);
                    sync_ref_root_var(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        vi,
                        result,
                        ref_root_base_ofs,
                        &mut synced_ref_vars,
                    );
                }

                // ── Thread-local reference get ──
                OpCode::ThreadlocalrefGet => {
                    // Load from a thread-local slot via runtime callback.
                    // arg(0) = offset (in bytes) into the TLS area.
                    let offset = resolve_opref(&mut builder, &constants, op.arg(0));
                    let result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_threadlocalref_get as *const () as usize,
                        &[offset],
                        Some(cl_types::I64),
                    )
                    .expect("jit_threadlocalref_get must return a value");
                    builder.def_var(var(vi), result);
                    sync_ref_root_var(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        vi,
                        result,
                        ref_root_base_ofs,
                        &mut synced_ref_vars,
                    );
                }

                // ── Load from GC table ──
                OpCode::LoadFromGcTable => {
                    // Load a constant pointer from the GC table.
                    // arg(0) = index into the gc table
                    let index = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), index);
                }

                // ── Load effective address ──
                // resoperation.py:1052-1054 — `[v_gcptr, v_index, c_baseofs,
                // c_shift]`, `res = arg0 + (arg1 << arg3) + arg2`.
                OpCode::LoadEffectiveAddress => {
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let index = resolve_opref(&mut builder, &constants, op.arg(1));
                    let baseofs = resolve_opref(&mut builder, &constants, op.arg(2));
                    let shift = resolve_opref(&mut builder, &constants, op.arg(3));
                    let shifted_index = builder.ins().ishl(index, shift);
                    let addr = builder.ins().iadd(base, shifted_index);
                    let result = builder.ins().iadd(addr, baseofs);
                    builder.def_var(var(vi), result);
                }

                // ── String allocation ──
                // Newstr: allocate a byte string of length args[0].
                // Newunicode: allocate a unicode string of length args[0].
                // Layout: 16-byte header + length * char_size bytes.
                OpCode::Newstr | OpCode::Newunicode => {
                    // get_array_token(rstr.STR/UNICODE, ...) basesize/itemsize.
                    // STR basesize = 17 (includes +1 extra_item_after_alloc);
                    // UNICODE basesize = 16. Allocated bytes per rstr.malloc
                    // are `basesize + n * itemsize`.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("newstr/newunicode op must have an ArrayDescr");
                    let ad = descr
                        .as_array_descr()
                        .expect("newstr/newunicode descriptor must be an ArrayDescr");
                    if !cranelift_gc_active() {
                        return Err(missing_gc_runtime(op.opcode));
                    }
                    let len = resolve_opref(&mut builder, &constants, op.arg(0));
                    let base_size = builder.ins().iconst(cl_types::I64, ad.base_size() as i64);
                    let item_size = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        &stale_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        gc_alloc_varsize_shim as *const () as usize,
                        &[base_size, item_size, len],
                        Some(cl_types::I64),
                    )
                    .expect("GC varsize allocation helper must return a value");
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    builder.def_var(jf_ptr_var, jf_ptr);
                    builder.def_var(var(vi), result);
                    sync_ref_root_var(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        vi,
                        result,
                        ref_root_base_ofs,
                        &mut synced_ref_vars,
                    );
                }
                // All OpCode variants are explicitly handled above.
                // This arm is unreachable but kept for forward-compatibility
                // when new opcodes are added to the IR.
                #[allow(unreachable_patterns)]
                _other => {
                    return Err(BackendError::Unsupported(format!(
                        "opcode {:?} has no backend lowering",
                        _other
                    )));
                }
            }

            if op.result_type() == Type::Ref && !force_tokens.contains(&vi) {
                defined_ref_vars.insert(vi);
            }
        }

        for (_, block) in &label_blocks {
            builder.seal_block(*block);
        }
        if label_blocks.is_empty() && loop_block != entry_block {
            builder.seal_block(loop_block);
        }
        builder.finalize();
        self.func_ctx = func_ctx;

        // Compile
        let mut ctx = Context::for_function(func);
        if std::env::var_os("MAJIT_DUMP_CLIF").is_some() {
            eprintln!(
                "[jit][clif-dump] trace_id={} header_pc={} num_inputs={} num_ops={}\n{}",
                trace_id,
                header_pc,
                inputargs.len(),
                ops.len(),
                ctx.func.display()
            );
        }
        if let Err(e) = self.module.define_function(func_id, &mut ctx) {
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!("[jit][clif-error] {e}\nCLIF IR:\n{}", ctx.func.display());
            }
            self.module.clear_context(&mut ctx);
            return Err(BackendError::CompilationFailed(format!("{e}\n{e:?}")));
        }
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().unwrap();

        let code_ptr = self.module.get_finalized_function(func_id);
        if std::env::var_os("MAJIT_LOG").is_some() {
            let fail_descr_preview: Vec<(u32, usize)> = fail_descrs
                .iter()
                .map(|descr| (descr.fail_index, Arc::as_ptr(descr) as usize))
                .collect();
            eprintln!(
                "[jit][compile-loop] trace_id={} header_pc={} code_ptr={:p} fail_descrs={:?}",
                trace_id, header_pc, code_ptr, fail_descr_preview
            );
        }

        let trace_info = CompiledTraceInfo {
            trace_id,
            input_types: inputargs.iter().map(|arg| arg.tp).collect(),
            header_pc,
            source_guard: None,
        };
        for descr in &fail_descrs {
            descr.set_trace_info(trace_info.clone());
        }
        // Freeze the fail_descrs vector. The collect_guards loop assigns
        // `fail_index = vec.len() as u32` immediately before pushing each
        // descr, so position equals fail_index by construction; no later
        // path mutates the vector. `Box<[T]>` reflects this contract in
        // the type system (mirrors RPython compile.py:183-203's frozen
        // descrs after record_loop_or_bridge).
        //
        // Load-bearing identity invariant for runtime dispatch: cranelift's
        // guard-fail handler indexes `compiled.fail_descrs[idx]` by the
        // descr-index slot stamped at compile time (compiler.rs guard
        // emission writes the per-loop `fail_index` into the deadframe).
        // Promoting from `debug_assert!` so a release build catches the
        // position/fail_index mismatch eagerly rather than dispatching to
        // the wrong descr at runtime.
        assert!(
            fail_descrs
                .iter()
                .enumerate()
                .all(|(i, d)| d.fail_index as usize == i),
            "fail_descrs position must equal fail_index"
        );
        let fail_descrs: Box<[Arc<CraneliftFailDescr>]> = fail_descrs.into_boxed_slice();
        // history.py:470-499 / x86/regalloc.py:1397 / x86/assembler.py:990-993
        // parity: set TargetToken._ll_loop_code on every Label in this
        // function, and register the entry in LOOP_TARGET_REGISTRY so that
        // an external JUMP whose descr is one of these Labels can re-enter
        // here (assembler.py:2456-2462 closing_jump).
        // Cranelift can't expose individual block addresses, so we use the
        // function's code_ptr for every Label in this function — re-entry
        // re-runs the preamble. PyPy's raw JMP would skip preamble; for
        // pyre's loops the preamble is just inputarg decoding (idempotent).
        let entry: LoopTargetEntry = LoopTargetEntry {
            code_ptr,
            fail_descrs: fail_descrs.clone(),
            num_inputs: inputargs.len(),
            num_ref_roots: ref_root_slots.len(),
            max_output_slots,
        };
        for op in ops.iter() {
            if op.opcode != OpCode::Label {
                continue;
            }
            if let Some(descr_ref) = op.descr.as_ref() {
                if let Some(target) = descr_ref.as_loop_target_descr() {
                    target.set_ll_loop_code(code_ptr as usize);
                }
                register_loop_target(descr_ref, entry.clone());
            }
        }
        Ok(CompiledLoop {
            trace_id,
            input_types: trace_info.input_types.clone(),
            header_pc,
            green_key: 0, // Set by compile_loop from token.green_key
            caller_prefix_layout: caller_layout.cloned(),
            _func_id: func_id,
            code_ptr,
            code_size: 0,
            fail_descrs,
            terminal_exit_layouts: UnsafeCell::new(terminal_exit_layouts),
            num_inputs: inputargs.len(),
            num_ref_roots: ref_root_slots.len(),
            max_output_slots,
            cpu_attachments: self.cpu_handle(),
        })
    }
}

impl Drop for CraneliftBackend {
    fn drop(&mut self) {
        for token_number in std::mem::take(&mut self.registered_call_assembler_tokens) {
            unregister_call_assembler_target(token_number);
        }
        for trace_id in std::mem::take(&mut self.registered_call_assembler_bridge_traces) {
            unregister_call_assembler_expectations(CallAssemblerCallerId::BridgeTrace(trace_id));
        }
        let _ = CALL_ASSEMBLER_DEADFRAMES.try_with(|map| map.borrow_mut().clear());
        let _ = NEXT_CALL_ASSEMBLER_DEADFRAME_HANDLE.try_with(|cell| cell.set(1));
        // `llmodel.py:58` `cpu.gc_ll_descr` ownership ends here. Drop the
        // active allocator so a subsequent backend is free to install
        // its own; matching dynasm's
        // `runner.rs DYNASM_ACTIVE_GC` reset on backend teardown.
        let _ = CRANELIFT_ACTIVE_GC.try_with(|cell| {
            *cell.borrow_mut() = None;
        });
        let _ = CRANELIFT_JITFRAME_TYPE_ID.try_with(|c| c.set(None));
    }
}

fn collect_guards(
    ops: &[Op],
    inputargs: &[InputArg],
    force_tokens: &HashSet<u32>,
    fail_descrs: &mut Vec<Arc<CraneliftFailDescr>>,
    guard_infos: &mut Vec<GuardInfo>,
    max_output_slots: &mut usize,
    trace_id: u64,
    header_pc: u64,
    source_guard: Option<(u64, u32)>,
    caller_layout: Option<&ExitRecoveryLayout>,
    constants: &HashMap<u32, i64>,
    constant_types: &HashMap<u32, Type>,
    callinfocollection: Option<&majit_ir::CallInfoCollection>,
    attached_descrs: majit_backend::AttachedDescrPtrs,
) -> Result<(), BackendError> {
    let num_inputs = inputargs.len();
    let type_index = OpTypeIndex::new(inputargs, ops, constant_types);
    let (type_overrides, op_def_positions) = build_type_overrides(inputargs, ops, &type_index);

    // Map Label descr index → block arity, used to distinguish internal vs
    // external JUMPs.  rewriter.py LABEL/JUMP redirect parity: a JUMP whose
    // descr targets a Label in this function but with a *different* arg arity
    // also lowers as an external jump (the target's stack frame layout
    // doesn't match, so we exit this trace and re-enter the target via the
    // dispatcher instead of jumping locally).
    let label_arity_by_descr: HashMap<u32, usize> = ops
        .iter()
        .filter(|op| op.opcode == OpCode::Label)
        .filter_map(|op| op.descr.as_ref().map(|d| (d.index(), op.args.len())))
        .collect();

    for (op_idx, op) in ops.iter().enumerate() {
        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;
        // External JUMP: target not in this function's Labels, or local
        // target with mismatched arity (treated as external for parity).
        let is_external_jump = op.opcode == OpCode::Jump
            && op
                .descr
                .as_ref()
                .map_or(false, |d| match label_arity_by_descr.get(&d.index()) {
                    None => true,
                    Some(&arity) => arity != op.args.len(),
                });

        if !is_guard && !is_finish && !is_external_jump {
            continue;
        }

        let fail_index = fail_descrs.len() as u32;

        let (fail_arg_refs, fail_arg_types) = if is_finish || is_external_jump {
            let refs: Vec<OpRef> = op.args.iter().copied().collect();
            // Use the descriptor's explicit types for FINISH args — these are
            // set by the tracer and represent the caller's view of the return
            // type, which may differ from the op's inferred type (e.g. New
            // produces Ref but the value is treated as Int by the caller).
            let types = if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_fail_descr()) {
                let dt = fd.fail_arg_types();
                if dt.len() == refs.len() {
                    dt.to_vec()
                } else {
                    infer_fail_arg_types(&refs, &type_index, &type_overrides, op_idx)?
                }
            } else {
                infer_fail_arg_types(&refs, &type_index, &type_overrides, op_idx)?
            };
            (refs, types)
        } else if let Some(ref fa) = op.fail_args {
            let refs: Vec<OpRef> = fa.iter().copied().collect();
            let descr_fd = op.descr.as_ref().and_then(|d| d.as_fail_descr());
            // store_final_boxes_in_guard (optimizeopt/mod.rs:3393-3404)
            // sets op.descr to a fresh ResumeGuardDescr whose
            // fail_arg_types match the reduced liveboxes — prefer the
            // descr as the post-numbering source of truth. Sharing-path
            // guards (optimizeopt/mod.rs:3068-3088) skip
            // store_final_boxes_in_guard and leave op.descr = None,
            // inheriting the donor's op.fail_arg_types instead; that is
            // the only path left where op-level types are load-bearing.
            let types = match (
                descr_fd.map(|fd| fd.fail_arg_types()),
                op.fail_arg_types.as_ref(),
            ) {
                (Some(dt), _) if dt.len() == refs.len() => dt.to_vec(),
                (_, Some(explicit)) if explicit.len() == refs.len() => explicit.clone(),
                _ => resolve_fail_arg_types(
                    &refs,
                    descr_fd,
                    &type_index,
                    &type_overrides,
                    &op_def_positions,
                    op_idx,
                )?,
            };
            (refs, types)
        } else {
            // RPython parity: a guard with no explicit fail_args carries the
            // current trace inputargs as live boxes. Phase E.2b shifted bridge
            // inputargs into `[bridge_inputarg_base..)`, so the fallback must
            // read each `InputArg.index` rather than assuming dense
            // `[0..num_inputs)`.  The outer match arm at 12399 already handles
            // `op.fail_args = Some(_)`; this branch is the `None` fallback.
            // resoperation.py:719/727/739 InputArg{Int,Float,Ref}.type — the
            // type for each synthesized fail_arg lives directly on the
            // InputArg, so mint a typed OpRef per entry.
            let refs: Vec<OpRef> = inputargs
                .iter()
                .map(|ia| OpRef::input_arg_typed(ia.index, ia.tp))
                .collect();
            let types = resolve_fail_arg_types(
                &refs,
                op.descr.as_ref().and_then(|d| d.as_fail_descr()),
                &type_index,
                &type_overrides,
                &op_def_positions,
                op_idx,
            )?;
            (refs, types)
        };

        let n = fail_arg_refs.len();
        if n > *max_output_slots {
            *max_output_slots = n;
        }

        let force_token_slots = fail_arg_refs
            .iter()
            .enumerate()
            .filter_map(|(slot, opref)| force_tokens.contains(&opref.raw()).then_some(slot))
            .collect();

        // RPython resume.py:396: on guard failure the resume PC is taken
        // from the rebuilt top frame (`RebuiltFrame.pc`). rd_numb/rd_consts
        // are populated by `store_final_boxes_in_guard` during optimizer
        // emit, so every non-finish, non-external-jump guard that reaches
        // the backend has them. We pre-compute the PC at compile time so
        // the ExitFrameLayout doesn't need to decode rd_numb at resume.
        let guard_resume_pc: Option<u64> = if !is_finish && !is_external_jump {
            if let (Some(rd_numb_bytes), Some(rd_consts_data)) =
                (op.resolved_rd_numb(), op.resolved_rd_consts())
            {
                use majit_ir::resumedata::{get_frame_value_count_fn, rebuild_from_numbering};
                let fvc = get_frame_value_count_fn();
                let fvc_ref: Option<&dyn Fn(i32, i32) -> usize> =
                    fvc.as_ref().map(|f| f as &dyn Fn(i32, i32) -> usize);
                let (_nfa, _vable, _vref, frames) =
                    rebuild_from_numbering(rd_numb_bytes, rd_consts_data, &fail_arg_types, fvc_ref);
                frames.last().map(|f| f.pc as u64)
            } else {
                None
            }
        } else {
            None
        };
        if std::env::var_os("MAJIT_LOG").is_some() && guard_resume_pc.is_some() {
            eprintln!(
                "[guard-resume] fail_index={} last_ref={:?} resume_pc={:?}",
                fail_index,
                fail_arg_refs.last(),
                guard_resume_pc
            );
        }
        let mut recovery_layout = identity_recovery_layout(
            trace_id,
            header_pc,
            guard_resume_pc,
            source_guard,
            &fail_arg_types,
            caller_layout,
        );
        // RPython parity: when rd_numb is present, rebuild frame slots from
        // rd_numb. rd_numb encodes ALL snapshot positions (including virtuals
        // as TAGVIRTUAL), while the identity recovery_layout only has
        // fail_args-count slots. The rd_numb-based layout is authoritative.
        if let (Some(rd_numb_bytes), Some(rd_consts_data)) =
            (op.resolved_rd_numb(), op.resolved_rd_consts())
        {
            let rd_vi = op.resolved_rd_virtuals();
            use majit_ir::resumedata::{self, RebuiltValue, rebuild_from_numbering};
            let rd_consts_ref: &[majit_ir::Const] = rd_consts_data;
            let fvc = majit_ir::resumedata::get_frame_value_count_fn();
            let fvc_ref: Option<&dyn Fn(i32, i32) -> usize> =
                fvc.as_ref().map(|f| f as &dyn Fn(i32, i32) -> usize);
            let (_num_failargs, _vable_values, _vref_values, frames) =
                rebuild_from_numbering(rd_numb_bytes, rd_consts_data, &fail_arg_types, fvc_ref);

            // Rebuild frame slots from rd_numb values.
            // Track Virtual(vidx) → slot_idx for target_slot in virtual_layouts.
            let mut vidx_to_slot: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            let mut new_slots: Vec<ExitValueSourceLayout> = Vec::new();
            for frame in &frames {
                for val in &frame.values {
                    new_slots.push(match val {
                        RebuiltValue::Box(idx, _) => ExitValueSourceLayout::ExitValue(*idx),
                        RebuiltValue::Virtual(vidx) => {
                            vidx_to_slot.insert(*vidx, new_slots.len());
                            ExitValueSourceLayout::Virtual(*vidx)
                        }
                        RebuiltValue::Const(c) => ExitValueSourceLayout::Constant(c.as_raw_i64()),
                        RebuiltValue::Unassigned => ExitValueSourceLayout::Uninitialized,
                    });
                }
            }
            if let Some(frame) = recovery_layout.frames.last_mut() {
                // Rebuild slot_types to match new slots length.
                // ExitValue → use original type (from fail_arg_types).
                // Virtual/Constant/Uninitialized → Ref (default for objects).
                let new_slot_types: Vec<Type> = new_slots
                    .iter()
                    .map(|slot| match slot {
                        ExitValueSourceLayout::ExitValue(idx) => {
                            fail_arg_types.get(*idx).copied().unwrap_or(Type::Ref)
                        }
                        _ => Type::Ref,
                    })
                    .collect();
                frame.slots = new_slots;
                frame.slot_types = Some(new_slot_types);
                // resume.py:250 jitcode_index parity: overwrite the
                // identity layout's default 0 with the innermost frame's
                // jitcode_index from rd_numb. After `opencoder.py:217`
                // `framestack.reverse()` parity rd_numb encodes frames
                // outermost-first (caller before callee), so the
                // innermost frame — the one the identity layout
                // represents — is `frames.last()`.
                if let Some(innermost) = frames.last() {
                    frame.jitcode_index = innermost.jitcode_index;
                }
            }

            // Build virtual_layouts from rd_virtuals.
            let total_fail_args = fail_arg_refs.len();
            let resolve_fieldnum = |fnum: i16| -> ExitValueSourceLayout {
                let (val, tagbits) = resumedata::untag(fnum);
                match tagbits {
                    // resume.py:1260 parity: TAGBOX indices can be negative
                    // (assigned by assign_number_to_box for virtual field values).
                    // RPython uses Python negative indexing (liveboxes[-1]);
                    // Rust needs explicit conversion to positive index.
                    resumedata::TAGBOX => {
                        let idx = if val >= 0 {
                            val as usize
                        } else {
                            (total_fail_args as i32 + val) as usize
                        };
                        ExitValueSourceLayout::ExitValue(idx)
                    }
                    resumedata::TAGVIRTUAL => ExitValueSourceLayout::Virtual(val as usize),
                    resumedata::TAGINT => ExitValueSourceLayout::Constant(val as i64),
                    resumedata::TAGCONST => {
                        let idx = (val - resumedata::TAG_CONST_OFFSET) as usize;
                        let c = rd_consts_ref
                            .get(idx)
                            .copied()
                            .unwrap_or(majit_ir::Const::Int(0));
                        ExitValueSourceLayout::Constant(c.as_raw_i64())
                    }
                    _ => ExitValueSourceLayout::Constant(0),
                }
            };
            let resolve_fieldnums = |fieldnums: &[i16],
                                     fielddescr_indices: &[u32]|
             -> Vec<(u32, ExitValueSourceLayout)> {
                fieldnums
                    .iter()
                    .enumerate()
                    .map(|(fi, &fnum)| {
                        let fdi = fielddescr_indices.get(fi).copied().unwrap_or(fi as u32);
                        (fdi, resolve_fieldnum(fnum))
                    })
                    .collect()
            };

            // Build virtual_layouts from rd_virtuals or rd_virtuals.
            recovery_layout.virtual_layouts.clear();
            if let Some(rd_vi_slice) = rd_vi {
                for (vidx, entry_rc) in rd_vi_slice.iter().enumerate() {
                    let entry: &majit_ir::RdVirtualInfo = entry_rc.as_ref();
                    let target_slot = vidx_to_slot.get(&vidx).copied();
                    let layout = match entry {
                        majit_ir::RdVirtualInfo::VirtualInfo {
                            descr,
                            type_id,
                            descr_index,
                            known_class,
                            fielddescrs,
                            fieldnums,
                            descr_size,
                        } => {
                            let indices: Vec<u32> = fielddescrs.iter().map(|fd| fd.index).collect();
                            ExitVirtualLayout::Object {
                                descr: descr.clone(),
                                type_id: *type_id,
                                descr_index: *descr_index,
                                known_class: *known_class,
                                fields: resolve_fieldnums(fieldnums, &indices),
                                target_slot,
                                fielddescrs: fielddescrs.clone(),
                                descr_size: *descr_size,
                            }
                        }
                        majit_ir::RdVirtualInfo::VStructInfo {
                            typedescr,
                            type_id,
                            descr_index,
                            fielddescrs,
                            fieldnums,
                            descr_size,
                        } => {
                            let indices: Vec<u32> = fielddescrs.iter().map(|fd| fd.index).collect();
                            ExitVirtualLayout::Struct {
                                typedescr: typedescr.clone(),
                                type_id: *type_id,
                                descr_index: *descr_index,
                                fields: resolve_fieldnums(fieldnums, &indices),
                                target_slot,
                                fielddescrs: fielddescrs.clone(),
                                descr_size: *descr_size,
                            }
                        }
                        majit_ir::RdVirtualInfo::VArrayInfoClear {
                            arraydescr: _,
                            descr_index,
                            kind,
                            fieldnums,
                        } => ExitVirtualLayout::Array {
                            descr_index: *descr_index,
                            clear: true,
                            kind: *kind,
                            items: fieldnums
                                .iter()
                                .map(|&fnum| resolve_fieldnum(fnum))
                                .collect(),
                        },
                        majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                            arraydescr: _,
                            descr_index,
                            kind,
                            fieldnums,
                        } => ExitVirtualLayout::Array {
                            descr_index: *descr_index,
                            clear: false,
                            kind: *kind,
                            items: fieldnums
                                .iter()
                                .map(|&fnum| resolve_fieldnum(fnum))
                                .collect(),
                        },
                        majit_ir::RdVirtualInfo::VArrayStructInfo {
                            arraydescr,
                            descr_index,
                            fielddescrs,
                            size,
                            fielddescr_indices,
                            fieldnums,
                            ..
                        } => {
                            let fpe = if *size > 0 {
                                fieldnums.len() / *size
                            } else {
                                0
                            };
                            ExitVirtualLayout::ArrayStruct {
                                descr_index: *descr_index,
                                arraydescr: arraydescr.clone(),
                                fielddescrs: fielddescrs.clone(),
                                element_fields: (0..*size)
                                    .map(|ei| {
                                        let s = ei * fpe;
                                        let e = (s + fpe).min(fieldnums.len());
                                        resolve_fieldnums(&fieldnums[s..e], fielddescr_indices)
                                    })
                                    .collect(),
                            }
                        }
                        majit_ir::RdVirtualInfo::VRawBufferInfo {
                            func,
                            size,
                            offsets,
                            descrs,
                            fieldnums,
                        } => ExitVirtualLayout::RawBuffer {
                            func: *func,
                            size: *size,
                            offsets: offsets.clone(),
                            descrs: descrs.clone(),
                            values: fieldnums
                                .iter()
                                .map(|&fnum| resolve_fieldnum(fnum))
                                .collect(),
                        },
                        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
                            ExitVirtualLayout::RawSlice {
                                offset: *offset,
                                base: fieldnums
                                    .first()
                                    .map(|&fnum| resolve_fieldnum(fnum))
                                    .unwrap_or(ExitValueSourceLayout::Constant(0)),
                            }
                        }
                        // resume.py:763 VStrPlainInfo / resume.py:817
                        // VUniPlainInfo — length = len(fieldnums).
                        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums } => {
                            let chars = fieldnums
                                .iter()
                                .map(|&fnum| resolve_fieldnum(fnum))
                                .collect();
                            ExitVirtualLayout::StrPlain {
                                is_unicode: false,
                                chars,
                            }
                        }
                        majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
                            let chars = fieldnums
                                .iter()
                                .map(|&fnum| resolve_fieldnum(fnum))
                                .collect();
                            ExitVirtualLayout::StrPlain {
                                is_unicode: true,
                                chars,
                            }
                        }
                        // resume.py:781 VStrConcatInfo / resume.py:836
                        // VUniConcatInfo — decoder.concat_strings(left, right).
                        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums }
                        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
                            let is_unicode =
                                matches!(entry, majit_ir::RdVirtualInfo::VUniConcatInfo { .. });
                            let oopspec = if is_unicode {
                                majit_ir::effectinfo::OopSpecIndex::UniConcat
                            } else {
                                majit_ir::effectinfo::OopSpecIndex::StrConcat
                            };
                            let cic = callinfocollection.expect(
                                "collect_guards: callinfocollection required for \
                                 VStr/VUni Concat guard-exit recovery",
                            );
                            let (calldescr, func) = cic
                                .callinfo_for_oopspec(oopspec)
                                .expect("callinfo_for_oopspec missing OS_STR_CONCAT/OS_UNI_CONCAT");
                            let left = resolve_fieldnum(fieldnums[0]);
                            let right = resolve_fieldnum(fieldnums[1]);
                            ExitVirtualLayout::StrConcat {
                                is_unicode,
                                func: *func as i64,
                                calldescr: calldescr.clone(),
                                left,
                                right,
                            }
                        }
                        // resume.py:801 VStrSliceInfo / resume.py:856
                        // VUniSliceInfo — decoder.slice_string(str, start, length).
                        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums }
                        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
                            let is_unicode =
                                matches!(entry, majit_ir::RdVirtualInfo::VUniSliceInfo { .. });
                            let oopspec = if is_unicode {
                                majit_ir::effectinfo::OopSpecIndex::UniSlice
                            } else {
                                majit_ir::effectinfo::OopSpecIndex::StrSlice
                            };
                            let cic = callinfocollection.expect(
                                "collect_guards: callinfocollection required for \
                                 VStr/VUni Slice guard-exit recovery",
                            );
                            let (calldescr, func) = cic
                                .callinfo_for_oopspec(oopspec)
                                .expect("callinfo_for_oopspec missing OS_STR_SLICE/OS_UNI_SLICE");
                            let str_src = resolve_fieldnum(fieldnums[0]);
                            let start = resolve_fieldnum(fieldnums[1]);
                            let length = resolve_fieldnum(fieldnums[2]);
                            ExitVirtualLayout::StrSlice {
                                is_unicode,
                                func: *func as i64,
                                calldescr: calldescr.clone(),
                                str_src,
                                start,
                                length,
                            }
                        }
                        majit_ir::RdVirtualInfo::Empty => continue,
                    };
                    recovery_layout.virtual_layouts.push(layout);
                }
            }
        }
        let recovery_layout = Some(recovery_layout);
        // Guard gcmap: bit i ⟺ fail_args[i] is Ref, stored at jf_frame[i].
        // No JITFRAME_FIXED_SIZE offset needed because fail_args start at
        // jf_frame[0] (not jf_frame[JITFRAME_FIXED_SIZE]).
        // assembler.py:2114-2155 genop_finish + regalloc.py:1190-1206 longevity:
        //   - GUARD: getfailargs() must not contain Const (regalloc.py:1206 assert).
        //   - FINISH: getarglist() may contain Const (regalloc.py:1192-1193 skip).
        //     The const value is loaded as immediate and stored into frame
        //     slot 0; gcmap_for_finish (assembler.py:160 = r_uint(1)) marks
        //     that slot so the GC traces the constant Ref.
        //   - JUMP (external/non-label): a normal op whose getarglist() may
        //     also contain Const (handled by regalloc.py:1192-1193 in the same
        //     loop). majit groups external JUMP with FINISH for fail_args
        //     bookkeeping; treat their gcmap the same way.
        let gcmap = {
            let mut bits: u64 = 0;
            for (i, tp) in fail_arg_types.iter().enumerate() {
                if *tp == Type::Ref {
                    let opref_id = fail_arg_refs.get(i).map(|r| r.raw()).unwrap_or(u32::MAX);
                    // regalloc.py:1206 — guard fail_args must never be Const.
                    debug_assert!(
                        is_finish || is_external_jump || !constants.contains_key(&opref_id),
                        "regalloc.py:1206: guard fail_args must not contain Const (slot={i}, opref={opref_id})"
                    );
                    if (i as u32) < 64 {
                        bits |= 1u64 << i;
                    }
                }
            }
            bits
        };
        let mut descr = if is_external_jump {
            // assembler.py:2456-2462 closing_jump parity: JUMP whose target
            // TargetToken lives in a different compiled function. The op.descr
            // is the target TargetToken (history.py:470) — the dispatcher
            // reads it to re-enter the target via lookup_loop_target.
            let target_descr = op
                .descr
                .as_ref()
                .cloned()
                .expect("external JUMP must carry a TargetToken descr");
            CraneliftFailDescr::new_external_jump(
                fail_index,
                trace_id,
                fail_arg_types,
                force_token_slots,
                recovery_layout,
                target_descr,
            )
        } else {
            CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
                fail_index,
                trace_id,
                fail_arg_types,
                is_finish,
                force_token_slots,
                recovery_layout,
            )
        };
        let accum_info = if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_fail_descr()) {
            let vi = fd.vector_info();
            descr.vector_info = vi.clone();
            // `compile.py:658` ExitFrameWithExceptionDescrRef identity
            // propagation: mirror the metainterp FailDescr flag onto the
            // backend-local CraneliftFailDescr so downstream runtime
            // classifiers (`is_exit_frame_with_exception`) keep parity.
            descr.is_exit_frame_with_exception = fd.is_exit_frame_with_exception();
            vi
        } else {
            Vec::new()
        };
        descr.set_source_op_index(op_idx);
        descr.green_key = header_pc;
        // Unified-Descr Port Epic Session 5b: capture the metainterp
        // ResumeGuardDescr Arc.  rd_numb/rd_consts/rd_virtuals/
        // rd_pendingfields readers (FailDescr trait impl) forward
        // through this Arc to the resume.py:450-488 storage on the
        // metainterp side, so no separate local copy is kept here.
        descr.meta_descr = op.descr.clone();
        let descr = Arc::new(descr);
        if std::env::var_os("MAJIT_LOG").is_some() && !is_finish && !is_external_jump {
            eprintln!(
                "[cl-guard-token] fail_index={} op_index={} opcode={:?} fail_args={:?} fail_arg_types={:?}",
                fail_index,
                op_idx,
                op.opcode,
                op.fail_args.as_ref(),
                descr.fail_arg_types,
            );
        }
        // store_hash is called after compile_loop by pyjitpl.rs using
        // jitcounter.fetch_next_hash() (compile.py:826-830 parity).
        //
        // regalloc.py:496-501 consider_guard_value / compile.py:813-824
        // make_a_counter_per_value: for GUARD_VALUE, encode fail_arg
        // index + type tag in status (overrides store_hash).
        if op.opcode == majit_ir::OpCode::GuardValue {
            if let Some(fa) = op.fail_args.as_ref() {
                let arg0 = op.arg(0);
                if let Some(idx) = fa.iter().position(|&r| r == arg0) {
                    let type_tag = match descr.fail_arg_types.get(idx) {
                        Some(majit_ir::Type::Ref) => CraneliftFailDescr::TY_REF,
                        Some(majit_ir::Type::Float) => CraneliftFailDescr::TY_FLOAT,
                        _ => CraneliftFailDescr::TY_INT,
                    };
                    descr.make_a_counter_per_value(idx as u32, type_tag);
                }
            }
        }
        // assembler.py:2126 get_gcref_from_faildescr parity:
        // store the FailDescr pointer (not index) in jf_descr.
        // compile.py:665-674 parity: Finish ops use the global singleton
        // (done_with_this_frame_descr) so that ALL traces share the same
        // pointer. This makes _call_assembler_check_descr (assembler.py:2274)
        // work across bridges (bridge's Finish descr == main trace's).
        let fail_descr_ptr = if is_finish {
            // `compile.py:665-674` + `compile.py:658` parity: write the
            // metainterp-attached `Arc<DoneWithThisFrameDescr*>` or
            // `Arc<ExitFrameWithExceptionDescrRef>` pointer into jf_descr.
            // The classifier path (`match_metainterp_finish_descr`) routes
            // the pointer back to a `CraneliftFailDescr` singleton for
            // downstream consumers that need `bridge_ref`/fail_arg_types.
            if descr.is_exit_frame_with_exception {
                attached_descrs.exit_frame_with_exception_descr_ref as i64
            } else {
                let result_type = descr.fail_arg_types.first().copied().unwrap_or(Type::Void);
                attached_descrs.done_with_this_frame_descr_ptr_for_type(result_type) as i64
            }
        } else {
            Arc::as_ptr(&descr) as i64
        };
        fail_descrs.push(descr);
        // assembler.py:40-44 must_save_exception parity:
        let must_save_exception = matches!(
            op.opcode,
            majit_ir::OpCode::GuardException
                | majit_ir::OpCode::GuardNoException
                | majit_ir::OpCode::GuardNotForced
        );
        guard_infos.push(GuardInfo {
            fail_index,
            can_have_bridge: is_guard,
            fail_arg_refs,
            must_save_exception,
            gcmap,
            fail_descr_ptr,
            accum_info,
        });
    }

    Ok(())
}

fn collect_terminal_exit_layouts(
    ops: &[Op],
    inputargs: &[InputArg],
    constant_types: &HashMap<u32, Type>,
    force_tokens: &HashSet<u32>,
    trace_id: u64,
    header_pc: u64,
    source_guard: Option<(u64, u32)>,
    caller_layout: Option<&ExitRecoveryLayout>,
) -> Result<Vec<TerminalExitLayout>, BackendError> {
    let type_index = OpTypeIndex::new(inputargs, ops, constant_types);
    let (type_overrides, _op_def_positions) = build_type_overrides(inputargs, ops, &type_index);
    let mut layouts = Vec::new();
    let mut fail_index = 0u32;

    for (op_index, op) in ops.iter().enumerate() {
        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;
        let is_jump = op.opcode == OpCode::Jump;

        if is_finish || is_jump {
            let exit_types =
                infer_fail_arg_types(op.args.as_slice(), &type_index, &type_overrides, op_index)?;
            let force_token_slots: Vec<usize> = op
                .args
                .iter()
                .enumerate()
                .filter_map(|(slot, opref)| force_tokens.contains(&opref.raw()).then_some(slot))
                .collect();
            let gc_ref_slots = exit_types
                .iter()
                .enumerate()
                .filter_map(|(slot, tp)| {
                    (*tp == Type::Ref && !force_token_slots.contains(&slot)).then_some(slot)
                })
                .collect();
            let recovery_layout = (is_jump || is_finish).then(|| {
                identity_recovery_layout(
                    trace_id,
                    header_pc,
                    None, // terminal exits don't need per-guard resume_pc
                    source_guard,
                    &exit_types,
                    caller_layout,
                )
            });
            layouts.push(TerminalExitLayout {
                op_index,
                trace_id,
                trace_info: Some(CompiledTraceInfo {
                    trace_id,
                    input_types: inputargs.iter().map(|arg| arg.tp).collect(),
                    header_pc,
                    source_guard: None,
                }),
                fail_index: if is_finish { fail_index } else { u32::MAX },
                exit_types,
                is_finish,
                gc_ref_slots,
                force_token_slots,
                recovery_layout,
                // Terminal exits (FINISH/JUMP) don't carry per-guard resume
                // data — RPython's jitdriver dispatches them via a separate
                // path (DoneWithThisFrame / ContinueRunningNormally).
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            });
        }

        if is_guard || is_finish {
            fail_index += 1;
        }
    }

    Ok(layouts)
}

// ---------------------------------------------------------------------------
// Backend trait implementation
// ---------------------------------------------------------------------------

impl majit_backend::Backend for CraneliftBackend {
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        token.inputarg_types = inputargs.iter().map(|ia| ia.tp).collect();
        // Pass the address of the invalidation flag so GUARD_NOT_INVALIDATED
        // can load from it at runtime.
        let flag_ptr = Arc::as_ptr(&token.invalidated) as *const AtomicBool as usize;
        let mut compiled = self.do_compile(inputargs, ops, Some(flag_ptr), None, None)?;
        compiled.green_key = token.green_key;
        let info = AsmInfo {
            code_addr: compiled.code_ptr as usize,
            code_size: compiled.code_size,
        };
        register_call_assembler_target(token, &compiled, self.attached_descr_ptrs())?;
        if let Err(err) =
            install_call_assembler_expectations(CallAssemblerCallerId::RootLoop(token.number), ops)
        {
            unregister_call_assembler_target(token.number);
            return Err(err);
        }
        self.registered_call_assembler_tokens.insert(token.number);

        // `compile.py:183-186 record_loop_or_bridge`: for each ResumeDescr
        // in the newly-compiled trace, stamp the owning CompiledLoopToken.
        // RPython gates on `isinstance(descr, ResumeDescr)` (`compile.py:185`);
        // pyre uses `is_resume_guard()` overridden on `CraneliftFailDescr`
        // to return `!is_finish` — backend descrs play the runtime role
        // of upstream `ResumeGuardDescr` for guard exits and of the
        // `DoneWithThisFrame*` family for finish exits.
        //
        // PRE-EXISTING-ADAPTATION: the actual stamp happens on the
        // backend-side `CraneliftFailDescr`, not the metainterp-side
        // `ResumeGuardDescr` that lives on the IR op.  Convergence
        // requires unifying the two descr objects per fail (Unified-Descr
        // Port Epic, Sessions 5+).  The metainterp-side
        // `op.descr.rd_loop_token_clt` is also stamped in
        // `pyjitpl/mod.rs::record_loop_or_bridge` (compile.py:185
        // line-by-line counterpart); the backend stamp here is the
        // pyre-side parallel write that survives the eventual unification.
        if let Some(clt) = token.compiled_loop_token.as_ref() {
            for descr in &compiled.fail_descrs {
                if !descr.is_resume_guard() {
                    continue;
                }
                descr.set_rd_loop_token_clt(std::sync::Arc::clone(clt));
            }
        }

        token.compiled = Some(Box::new(compiled));
        Ok(info)
    }

    fn set_constants(&mut self, constants: HashMap<u32, i64>) {
        self.constants = constants;
    }

    fn set_constant_types(&mut self, constant_types: HashMap<u32, majit_ir::Type>) {
        self.constant_types = constant_types;
    }

    fn set_next_trace_id(&mut self, trace_id: u64) {
        self.next_trace_id = Some(trace_id);
    }

    fn set_next_header_pc(&mut self, header_pc: u64) {
        self.next_header_pc = Some(header_pc);
    }

    fn set_done_with_this_frame_descr_void(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .done_with_this_frame_descr_void = Some(descr);
    }
    fn set_done_with_this_frame_descr_int(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .done_with_this_frame_descr_int = Some(descr);
    }
    fn set_done_with_this_frame_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .done_with_this_frame_descr_ref = Some(descr);
    }
    fn set_done_with_this_frame_descr_float(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .done_with_this_frame_descr_float = Some(descr);
    }
    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .exit_frame_with_exception_descr_ref = Some(descr);
    }
    fn set_propagate_exception_descr(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .propagate_exception_descr = Some(descr);
    }

    fn register_pending_target(
        &mut self,
        token_number: u64,
        input_types: Vec<majit_ir::Type>,
        num_inputs: usize,
        num_scalar_inputargs: usize,
        index_of_virtualizable: i32,
    ) {
        register_pending_call_assembler_target(
            token_number,
            input_types,
            num_inputs,
            num_scalar_inputargs,
            index_of_virtualizable,
            self.cpu_handle(),
        );
    }

    /// `compile.py:484 do_compile_bridge` — line-by-line.  Phase E.3+:
    /// the caller resolves `source_jct = descr_owning_jct(fail_descr)`
    /// (`majit-backend/src/lib.rs:969`) and passes it as `original_token`,
    /// so the source descr is always reachable from
    /// `original_token.compiled.fail_descrs` (no source-descr predecessor
    /// scan).  `previous_tokens` is still consumed below to attach the
    /// freshly compiled bridge to retired predecessor descrs whose machine
    /// code remains live (cranelift cannot patch in place); see the trait
    /// definition for the convergence path.
    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &JitCellToken,
        previous_tokens: &[std::sync::Arc<JitCellToken>],
    ) -> Result<AsmInfo, BackendError> {
        let invalidated_arc = original_token.invalidated.clone();
        let flag_ptr =
            Arc::as_ptr(&invalidated_arc) as *const std::sync::atomic::AtomicBool as usize;
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>())
            .ok_or_else(|| {
                BackendError::CompilationFailed("original token has no compiled loop".to_string())
            })?;
        // compile.py:569 / compile.py:593 line-by-line: the caller hands
        // `send_bridge_to_backend` the actual `faildescr` object;
        // `fail_descr.trace_id()` is the bridge origin's allocated id
        // (`alloc_trace_id` starts at 1, no sentinel) and identifies the
        // owning trace.  Phase E.3+ introduced
        // `descr_owning_jct(fail_descr)` (`majit-backend/src/lib.rs:969`),
        // which resolves the descr's owning JCT directly via
        // `rd_loop_token_clt`.  The metainterp caller now passes that
        // owning JCT as `original_token`, so the matching backend descr
        // is always reachable inside `original_compiled.fail_descrs` —
        // the prior `previous_tokens` scan was a NEW-DEVIATION compensating
        // for the absence of `descr_owning_jct` and has been retired.
        let source_trace_id = fail_descr.trace_id();
        let source_descr = find_fail_descr_in_fail_descrs(
            &original_compiled.fail_descrs,
            source_trace_id,
            fail_descr.fail_index_per_trace(),
        );
        if source_descr.is_none() {
            // RPython compile.py:569: send_bridge_to_backend always has a
            // valid faildescr. If source_descr is not found, the bridge
            // cannot be attached — return error for all fail_indices.
            return Err(BackendError::CompilationFailed(format!(
                "source fail descr not found for trace {} fail {}",
                source_trace_id,
                fail_descr.fail_index_per_trace()
            )));
        }
        let caller_layout = source_descr.as_ref().and_then(|d| {
            d.recovery_layout_ref().clone().map(|mut layout| {
                layout.frames.pop();
                layout
            })
        });
        let compiled = self.do_compile(
            inputargs,
            ops,
            Some(flag_ptr), // compile.py:186: bridges share parent's invalidation flag
            Some((source_trace_id, fail_descr.fail_index_per_trace())),
            caller_layout.as_ref(),
        );
        let compiled = compiled?;
        // x86/assembler.py:691-693 assemble_bridge parity:
        // bridge compilation can increase the frame depth required by
        // CALL_ASSEMBLER callee frames.  The GC rewriter reads the
        // CompiledLoopToken.frame_info pointer at runtime, so update the
        // original token in place just like RPython's update_frame_depth.
        let bridge_frame_depth = (compiled.max_output_slots + compiled.num_ref_roots) as i64;
        let baseofs = JF_FRAME_ITEM0_OFS as i64 + GcHeader::SIZE as i64;
        if let Some(clt) = original_token.compiled_loop_token.as_ref() {
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, bridge_frame_depth);
        }
        let info = AsmInfo {
            code_addr: compiled.code_ptr as usize,
            code_size: compiled.code_size,
        };
        install_call_assembler_expectations(
            CallAssemblerCallerId::BridgeTrace(compiled.trace_id),
            ops,
        )?;
        self.registered_call_assembler_bridge_traces
            .insert(compiled.trace_id);

        // Attach the bridge to the original guard's fail descriptor so that
        // execute_token can dispatch to it on subsequent guard failures.
        let bridge_trace_info = CompiledTraceInfo {
            trace_id: compiled.trace_id,
            input_types: compiled.input_types.clone(),
            header_pc: compiled.header_pc,
            source_guard: Some((source_trace_id, fail_descr.fail_index_per_trace())),
        };
        let clt_arc = original_token.compiled_loop_token.clone();
        for descr in &compiled.fail_descrs {
            descr.set_trace_info(bridge_trace_info.clone());
            // `compile.py:183-186 record_loop_or_bridge`: a bridge's
            // ResumeDescrs inherit the original loop's CompiledLoopToken.
            // `is_resume_guard()` restricts the stamp to guard descrs
            // per `compile.py:185 isinstance(descr, ResumeDescr)`.
            if !descr.is_resume_guard() {
                continue;
            }
            if let Some(clt) = clt_arc.as_ref() {
                descr.set_rd_loop_token_clt(std::sync::Arc::clone(clt));
            }
        }
        {
            let terminal_exit_layouts = compiled.terminal_exit_layouts_mut();
            for layout in terminal_exit_layouts.iter_mut() {
                layout.trace_info = Some(bridge_trace_info.clone());
            }
        }
        {
            if let Some(ref sd) = source_descr {
                let existing_bridge = sd.bridge_ref();
                if let Some(ref bridge) = *existing_bridge {
                    unregister_bridge_call_assembler_expectations(bridge);
                }
            }
        }
        if let Some(ref sd) = source_descr {
            let bridge_num_inputs = inputargs.len().max(compiled.input_types.len());
            // history.py:470-499 TargetToken parity: a bridge that ends with
            // an external JUMP re-enters its target via
            // `target_descr._ll_loop_code` (stored on the fail_descr at
            // collect_guards time). BridgeData.loop_reentry caches this for
            // the `execute_bridge` entry marshaling path; derive it from
            // the compiled fail_descrs' is_external_jump flag rather than
            // re-scanning ops + mutating descrs after the fact.
            let loop_reentry = compiled.fail_descrs.iter().any(|d| d.is_external_jump);
            sd.attach_bridge(BridgeData {
                trace_id: compiled.trace_id,
                input_types: compiled.input_types.clone(),
                header_pc: compiled.header_pc,
                source_guard: (source_trace_id, fail_descr.fail_index_per_trace()),
                caller_prefix_layout: compiled.caller_prefix_layout.clone(),
                code_ptr: compiled.code_ptr,
                fail_descrs: compiled.fail_descrs,
                terminal_exit_layouts: compiled.terminal_exit_layouts,
                loop_reentry,
                num_inputs: bridge_num_inputs,
                num_ref_roots: compiled.num_ref_roots,
                max_output_slots: compiled.max_output_slots,

                invalidated_arc: Some(invalidated_arc),
            });
            // Cranelift can't patch machine code like RPython's x86 backend.
            // After a retrace, the RUNNING machine code still references
            // fail_descrs from previous tokens. Attach the bridge to ALL
            // matching fail_descrs in previous_tokens so the running code
            // can dispatch to it.
            for prev_token in previous_tokens {
                if let Some(prev_compiled) = prev_token
                    .compiled
                    .as_ref()
                    .and_then(|c| c.downcast_ref::<CompiledLoop>())
                {
                    if let Some(prev_descr) = find_fail_descr_in_fail_descrs(
                        &prev_compiled.fail_descrs,
                        source_trace_id,
                        fail_descr.fail_index_per_trace(),
                    ) {
                        if !prev_descr.has_bridge() {
                            // Reconstruct a minimal BridgeData from the
                            // bridge already attached to `sd`.
                            let bridge = sd.bridge_ref();
                            if let Some(ref b) = *bridge {
                                prev_descr.attach_bridge(BridgeData {
                                    trace_id: b.trace_id,
                                    input_types: b.input_types.clone(),
                                    header_pc: b.header_pc,
                                    source_guard: b.source_guard,
                                    caller_prefix_layout: b.caller_prefix_layout.clone(),
                                    code_ptr: b.code_ptr,
                                    fail_descrs: b.fail_descrs.clone(),
                                    num_inputs: b.num_inputs,
                                    num_ref_roots: b.num_ref_roots,
                                    max_output_slots: b.max_output_slots,

                                    terminal_exit_layouts: UnsafeCell::new(
                                        unsafe { &*b.terminal_exit_layouts.get() }.clone(),
                                    ),
                                    loop_reentry: b.loop_reentry,
                                    invalidated_arc: b.invalidated_arc.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(info)
    }

    fn get_guard_status(
        &self,
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
    ) -> (u64, usize) {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>());
        if let Some(compiled) = compiled {
            if let Some(descr) =
                find_fail_descr_in_fail_descrs(&compiled.fail_descrs, trace_id, fail_index)
            {
                return (descr.get_status(), std::sync::Arc::as_ptr(&descr) as usize);
            }
        }
        (0, 0)
    }

    /// resume.py:1143-1188 parity — publish `metainterp.staticdata.
    /// callinfocollection` so the guard-exit recovery builder at
    /// `collect_guards` can resolve OS_STR_CONCAT / OS_UNI_CONCAT /
    /// OS_STR_SLICE / OS_UNI_SLICE func pointers for VStr/VUni
    /// Concat/Slice materialization. Forwards to the inherent
    /// [`CraneliftBackend::set_callinfocollection`].
    fn set_callinfocollection(&mut self, cic: Option<Arc<majit_ir::CallInfoCollection>>) {
        CraneliftBackend::set_callinfocollection(self, cic);
    }

    fn store_guard_hashes(&self, token: &JitCellToken, hashes: &[u64]) {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>());
        if let Some(compiled) = compiled {
            for (i, &hash) in hashes.iter().enumerate() {
                if let Some(descr) = compiled.fail_descrs.get(i) {
                    // Skip FINISH, external JUMP, and GUARD_VALUE
                    // (make_a_counter_per_value already set status).
                    if !descr.is_finish && descr.get_status() == 0 {
                        descr.store_hash(hash);
                    }
                }
            }
        }
    }

    fn store_bridge_guard_hashes(
        &self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
        hashes: &[u64],
    ) {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>());
        if let Some(compiled) = compiled {
            // Use recursive search matching compiled_bridge_fail_descr_layouts.
            let source_descr = find_fail_descr_in_fail_descrs(
                &compiled.fail_descrs,
                source_trace_id,
                source_fail_index,
            );
            if let Some(descr) = source_descr {
                let bridge_guard = descr.bridge_ref();
                if let Some(ref bridge) = *bridge_guard {
                    for (i, &hash) in hashes.iter().enumerate() {
                        if let Some(bd) = bridge.fail_descrs.get(i) {
                            if !bd.is_finish && bd.get_status() == 0 {
                                bd.store_hash(hash);
                            }
                        }
                    }
                }
            }
        }
    }

    fn read_descr_status(&self, descr_addr: usize) -> u64 {
        if descr_addr == 0 {
            return 0;
        }
        // Safety: descr_addr is Arc::as_ptr from a CraneliftFailDescr that is
        // alive in compiled_loops or previous_tokens. AtomicU64 read is safe.
        let descr = unsafe { &*(descr_addr as *const CraneliftFailDescr) };
        descr.get_status()
    }

    fn start_compiling_descr(&self, descr_addr: usize) {
        if descr_addr == 0 {
            return;
        }
        let descr = unsafe { &*(descr_addr as *const CraneliftFailDescr) };
        descr.start_compiling();
    }

    fn done_compiling_descr(&self, descr_addr: usize) {
        if descr_addr == 0 {
            return;
        }
        let descr = unsafe { &*(descr_addr as *const CraneliftFailDescr) };
        descr.done_compiling();
    }

    fn migrate_bridges(&self, old_token: &JitCellToken, new_token: &JitCellToken) {
        let old_compiled = old_token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>());
        let new_compiled = new_token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>());
        let (Some(old), Some(new)) = (old_compiled, new_compiled) else {
            return;
        };
        for old_fd in &old.fail_descrs {
            let bridge_guard = old_fd.bridge_ref();
            let Some(ref b) = *bridge_guard else {
                continue;
            };
            // Match by (trace_id, fail_index) — same key used by
            // find_fail_descr_in_fail_descrs and BridgeData.source_guard.
            let target = find_fail_descr_in_fail_descrs(
                &new.fail_descrs,
                old_fd.trace_id,
                old_fd.fail_index,
            );
            if let Some(new_fd) = target {
                if !new_fd.has_bridge() {
                    // Clone the bridge (don't take_bridge — old code may
                    // still reference old_fd and needs bridge dispatch).
                    new_fd.attach_bridge(BridgeData {
                        trace_id: b.trace_id,
                        input_types: b.input_types.clone(),
                        header_pc: b.header_pc,
                        source_guard: b.source_guard,
                        caller_prefix_layout: b.caller_prefix_layout.clone(),
                        code_ptr: b.code_ptr,
                        fail_descrs: b.fail_descrs.clone(),
                        num_inputs: b.num_inputs,
                        num_ref_roots: b.num_ref_roots,
                        max_output_slots: b.max_output_slots,

                        terminal_exit_layouts: UnsafeCell::new(
                            unsafe { &*b.terminal_exit_layouts.get() }.clone(),
                        ),
                        loop_reentry: b.loop_reentry,
                        invalidated_arc: b.invalidated_arc.clone(),
                    });
                }
            }
        }
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledLoop>()
            .expect("compiled data is not CompiledLoop");
        let mut inputs: Vec<i64> = Vec::with_capacity(compiled.num_inputs);
        for arg in args {
            inputs.push(match arg {
                Value::Int(v) => *v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.0 as i64,
                Value::Void => 0,
            });
        }

        Self::execute_with_inputs(compiled, &inputs)
    }

    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledLoop>()
            .expect("compiled data is not CompiledLoop");

        Self::execute_with_inputs(compiled, args)
    }

    fn execute_token_ints_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> majit_backend::RawExecResult {
        let compiled = token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledLoop>()
            .expect("compiled data is not CompiledLoop");

        // PRE-EXISTING-ADAPTATION (parity exception): RPython
        // `llgraph/runner.py:1120-1201` has only one `LLFrame.execute`
        // path that produces an `LLDeadFrame`; the deadframe is
        // decoded via the recovery layout (resume.py:1042
        // `rebuild_state_after_failure`) on every read, materialising
        // virtuals into concrete objects.  pyre's
        // `execute_with_inputs` mirrors that path and feeds
        // `MetaInterp.handle_*` callers that expect virtual rebuild.
        //
        // `RawExecResult` is the lightweight raw-output cousin used
        // by [`crate::pyre`] entry points that need the encoded
        // exit-slot bytes BEFORE virtual reconstruction (the
        // resume-trace builder rebuilds virtuals from the
        // `recovery_layout` itself).  Routing these consumers through
        // `execute_with_inputs` collapses the two semantics and
        // breaks fannkuch on cranelift (the recovery-layout decoder
        // allocates fresh per-iteration virtual lists, defeating the
        // raw resumetrace shortcut).  The dispatch loop below is kept
        // structurally aligned with `execute_with_inputs` (same
        // closing-jump / bridge-follow / FINISH branches) so a parity
        // edit to one stays mechanically applicable to the other.
        let mut cur_code_ptr = compiled.code_ptr;
        let mut cur_fail_descrs: Box<[Arc<CraneliftFailDescr>]> = compiled.fail_descrs.clone();
        let mut cur_num_ref_roots = compiled.num_ref_roots;
        let mut cur_max_output_slots = compiled.max_output_slots;
        let mut cur_inputs = args.to_vec();
        let attachments_guard = compiled.cpu_attachments.read().unwrap();
        let attachments: &CpuDescrAttachments = &*attachments_guard;

        loop {
            let exec = run_compiled_code(
                cur_code_ptr,
                &cur_fail_descrs,
                cur_num_ref_roots,
                cur_max_output_slots,
                &cur_inputs,
                attachments,
            );
            let fail_index = exec.fail_index;
            let direct_descr = exec.direct_descr.clone();
            let mut outputs = exec.extract_outputs(cur_max_output_slots.max(1));

            // CALL_ASSEMBLER deadframe interception — exits raw dispatch.
            if let Some(frame) = maybe_take_call_assembler_deadframe(fail_index, &outputs) {
                let frame = wrap_call_assembler_deadframe_with_caller_prefix(
                    frame,
                    compiled.trace_id,
                    compiled.header_pc,
                    None,
                    &compiled.input_types,
                    &cur_inputs,
                    compiled.caller_prefix_layout.as_ref(),
                );
                let descr = self.get_latest_descr(&frame);
                let exit_layout = self.describe_deadframe(&frame);
                let savedata = self.get_savedata_ref(&frame);
                let exception_value = self.grab_exc_value(&frame);
                let exit_arity = descr.fail_arg_types().len();
                let mut result = Vec::with_capacity(exit_arity);
                let mut typed_result = Vec::with_capacity(exit_arity);
                for (i, &tp) in descr.fail_arg_types().iter().enumerate() {
                    match tp {
                        Type::Int => {
                            let value = self.get_int_value(&frame, i);
                            result.push(value);
                            typed_result.push(Value::Int(value));
                        }
                        Type::Ref => {
                            let value = self.get_ref_value(&frame, i);
                            result.push(value.as_usize() as i64);
                            typed_result.push(Value::Ref(value));
                        }
                        Type::Float => {
                            let value = self.get_float_value(&frame, i);
                            result.push(value.to_bits() as i64);
                            typed_result.push(Value::Float(value));
                        }
                        Type::Void => {
                            result.push(0);
                            typed_result.push(Value::Void);
                        }
                    }
                }
                return majit_backend::RawExecResult {
                    outputs: result,
                    typed_outputs: typed_result,
                    exit_layout,
                    force_token_slots: descr.force_token_slots().to_vec(),
                    savedata,
                    exception_value,
                    fail_index: descr.fail_index(),
                    trace_id: descr.trace_id(),
                    is_finish: descr.is_finish(),
                    is_exit_frame_with_exception: descr.is_exit_frame_with_exception(),
                    status: descr.get_status(),
                    descr_addr: descr as *const dyn FailDescr as *const () as usize,
                };
            }

            // llmodel.py:412-420 get_latest_descr.
            let fail_descr_arc = if let Some(descr) = direct_descr {
                descr
            } else if (fail_index as usize) < cur_fail_descrs.len() {
                cur_fail_descrs[fail_index as usize].clone()
            } else {
                let found = compiled.fail_descrs.iter().find_map(|d| {
                    let guard = d.bridge_ref();
                    guard.as_ref().and_then(|b| {
                        find_fail_descr_in_fail_descrs(&b.fail_descrs, b.trace_id, fail_index)
                    })
                });
                found.unwrap_or_else(|| {
                    cur_fail_descrs
                        .last()
                        .cloned()
                        .unwrap_or_else(|| compiled.fail_descrs[0].clone())
                })
            };
            let fail_descr = &fail_descr_arc;

            // llgraph/runner.py:1130-1140 closing_jump: cross-loop JUMP
            // re-enters the target loop trace identified by the
            // TargetToken on the fail descriptor.
            if fail_descr.is_external_jump {
                let target_entry = fail_descr
                    .target_descr
                    .as_ref()
                    .and_then(lookup_loop_target)
                    .expect("external JUMP target must be a registered LoopTargetDescr");
                cur_code_ptr = target_entry.code_ptr;
                cur_fail_descrs = target_entry.fail_descrs;
                cur_num_ref_roots = target_entry.num_ref_roots;
                cur_max_output_slots = target_entry.max_output_slots;
                cur_inputs = outputs;
                continue;
            }

            // llgraph/runner.py:1200-1201 execute_finish.  Finish exits
            // the dispatch loop with the trace's payload — the raw path
            // reads outputs / savedata / exception directly from the
            // jitframe and emits a `RawExecResult`.
            if fail_descr.is_finish {
                let savedata = {
                    let raw =
                        unsafe { *((exec.jf_gcref.0 + JF_SAVEDATA_OFS as usize) as *const usize) };
                    if raw != 0 { Some(GcRef(raw)) } else { None }
                };
                let exc_raw =
                    unsafe { *((exec.jf_gcref.0 + JF_GUARD_EXC_OFS as usize) as *const usize) };
                let exception = GcRef(exc_raw);

                let exit_arity = fail_descr.fail_arg_types().len();
                outputs.truncate(exit_arity);
                let mut typed_outputs = Vec::with_capacity(exit_arity);
                for (&raw, &tp) in outputs.iter().zip(fail_descr.fail_arg_types().iter()) {
                    match tp {
                        Type::Int => typed_outputs.push(Value::Int(raw)),
                        Type::Ref => typed_outputs.push(Value::Ref(GcRef(raw as usize))),
                        Type::Float => typed_outputs.push(Value::Float(f64::from_bits(raw as u64))),
                        Type::Void => typed_outputs.push(Value::Void),
                    }
                }

                return majit_backend::RawExecResult {
                    outputs,
                    typed_outputs,
                    exit_layout: Some(fail_descr.layout()),
                    force_token_slots: fail_descr.force_token_slots().to_vec(),
                    savedata,
                    exception_value: exception,
                    fail_index,
                    trace_id: fail_descr.trace_id(),
                    is_finish: true,
                    is_exit_frame_with_exception: fail_descr.is_exit_frame_with_exception(),
                    status: fail_descr.get_status(),
                    descr_addr: Arc::as_ptr(&fail_descr_arc) as usize,
                };
            }

            fail_descr.increment_fail_count();

            // llgraph/runner.py:1184-1191 fail_guard with bridge.
            let bridge_guard = fail_descr.bridge_ref();
            if let Some(ref bridge) = *bridge_guard {
                let n = bridge.num_inputs.min(outputs.len());
                let bridge_inputs = outputs[..n].to_vec();
                cur_code_ptr = bridge.code_ptr;
                cur_fail_descrs = bridge.fail_descrs.clone();
                cur_num_ref_roots = bridge.num_ref_roots;
                cur_max_output_slots = bridge.max_output_slots;
                cur_inputs = bridge_inputs;
                continue;
            }
            let _ = bridge_guard;

            // llgraph/runner.py:1192-1194 fail_guard without bridge —
            // exit with the raw outputs.
            let savedata = {
                let raw =
                    unsafe { *((exec.jf_gcref.0 + JF_SAVEDATA_OFS as usize) as *const usize) };
                if raw != 0 { Some(GcRef(raw)) } else { None }
            };
            let exc_raw =
                unsafe { *((exec.jf_gcref.0 + JF_GUARD_EXC_OFS as usize) as *const usize) };
            let exception = GcRef(exc_raw);

            let exit_arity = fail_descr.fail_arg_types().len();
            outputs.truncate(exit_arity);
            let mut typed_outputs = Vec::with_capacity(exit_arity);
            for (&raw, &tp) in outputs.iter().zip(fail_descr.fail_arg_types().iter()) {
                match tp {
                    Type::Int => typed_outputs.push(Value::Int(raw)),
                    Type::Ref => typed_outputs.push(Value::Ref(GcRef(raw as usize))),
                    Type::Float => typed_outputs.push(Value::Float(f64::from_bits(raw as u64))),
                    Type::Void => typed_outputs.push(Value::Void),
                }
            }

            return majit_backend::RawExecResult {
                outputs,
                typed_outputs,
                exit_layout: Some(fail_descr.layout()),
                force_token_slots: fail_descr.force_token_slots().to_vec(),
                savedata,
                exception_value: exception,
                fail_index,
                trace_id: fail_descr.trace_id(),
                is_finish: false,
                is_exit_frame_with_exception: fail_descr.is_exit_frame_with_exception(),
                status: fail_descr.get_status(),
                descr_addr: Arc::as_ptr(&fail_descr_arc) as usize,
            };
        }
    }

    fn compiled_fail_descr_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        Some(
            compiled
                .fail_descrs
                .iter()
                .map(|descr| descr.layout())
                .collect(),
        )
    }

    fn compiled_bridge_fail_descr_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        // Use recursive search to find fail_descrs nested inside bridges.
        let source_descr = find_fail_descr_in_fail_descrs(
            &original_compiled.fail_descrs,
            source_trace_id,
            source_fail_index,
        )?;
        let bridge = source_descr.bridge_ref();
        let bridge = bridge.as_ref()?;
        Some(
            bridge
                .fail_descrs
                .iter()
                .map(|descr| descr.layout())
                .collect(),
        )
    }

    fn compiled_bridge_descr_arc(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Arc<dyn majit_ir::Descr>> {
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        let source_descr = find_fail_descr_in_fail_descrs(
            &original_compiled.fail_descrs,
            source_trace_id,
            source_fail_index,
        )?;
        Some(source_descr as Arc<dyn majit_ir::Descr>)
    }

    fn compiled_trace_fail_descr_layouts(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        if compiled.trace_id == trace_id {
            return Some(
                compiled
                    .fail_descrs
                    .iter()
                    .map(|descr| descr.layout())
                    .collect(),
            );
        }
        find_trace_fail_descr_layouts_in_fail_descrs(&compiled.fail_descrs, trace_id)
    }

    fn compiled_terminal_exit_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_backend::TerminalExitLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        Some(compiled.terminal_exit_layouts_ref().clone())
    }

    fn compiled_bridge_terminal_exit_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_backend::TerminalExitLayout>> {
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        let source_descr = original_compiled.fail_descrs.iter().find(|descr| {
            descr.fail_index == source_fail_index && descr.trace_id == source_trace_id
        })?;
        let bridge = source_descr.bridge_ref();
        let bridge = bridge.as_ref()?;
        Some(bridge.terminal_exit_layouts_ref().clone())
    }

    fn compiled_trace_terminal_exit_layouts(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<Vec<majit_backend::TerminalExitLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        if compiled.trace_id == trace_id {
            return Some(compiled.terminal_exit_layouts_ref().clone());
        }
        find_trace_terminal_exit_layouts_in_fail_descrs(&compiled.fail_descrs, trace_id)
    }

    fn compiled_trace_info(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<CompiledTraceInfo> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        if compiled.trace_id == trace_id {
            return Some(CompiledTraceInfo {
                trace_id: compiled.trace_id,
                input_types: compiled.input_types.clone(),
                header_pc: compiled.header_pc,
                source_guard: None,
            });
        }
        find_trace_info_in_fail_descrs(&compiled.fail_descrs, trace_id)
    }

    fn compiled_guard_frame_stacks(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<(u32, Vec<majit_backend::ExitFrameLayout>)>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        let mut result = Vec::new();
        for descr in &compiled.fail_descrs {
            let recovery = descr.recovery_layout_ref();
            if let Some(ref layout) = *recovery {
                result.push((descr.fail_index, layout.frames.clone()));
            }
        }
        Some(result)
    }

    fn describe_deadframe(&self, frame: &DeadFrame) -> Option<majit_backend::FailDescrLayout> {
        deadframe_layout(frame)
    }

    fn update_fail_descr_recovery_layout(
        &mut self,
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
        recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        let Some(compiled) = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())
        else {
            return false;
        };
        patch_fail_descr_recovery_layout(
            &compiled.fail_descrs,
            trace_id,
            fail_index,
            &recovery_layout,
        )
    }

    fn update_terminal_exit_recovery_layout(
        &mut self,
        token: &JitCellToken,
        trace_id: u64,
        op_index: usize,
        recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        let Some(compiled) = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())
        else {
            return false;
        };
        patch_terminal_exit_recovery_layout(
            &compiled.terminal_exit_layouts,
            &compiled.fail_descrs,
            trace_id,
            op_index,
            &recovery_layout,
        )
    }

    fn force(&self, force_token: GcRef) -> Option<DeadFrame> {
        // llmodel.py:270-274
        if force_token.0 == 0 {
            return None;
        }
        Some(force_token_to_dead_frame(force_token))
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        get_latest_descr_from_deadframe(frame).expect("get_latest_descr_from_deadframe failed")
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        get_int_from_deadframe(frame, index).expect("get_int_from_deadframe failed")
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        get_float_from_deadframe(frame, index).expect("get_float_from_deadframe failed")
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        get_ref_from_deadframe(frame, index).expect("get_ref_from_deadframe failed")
    }

    fn set_savedata_ref(&self, frame: &mut DeadFrame, data: GcRef) {
        set_savedata_ref_on_deadframe(frame, data).expect("set_savedata_ref_on_deadframe failed");
    }

    fn get_savedata_ref(&self, frame: &DeadFrame) -> Option<GcRef> {
        get_savedata_ref_from_deadframe(frame)
            .ok()
            .filter(|r| !r.is_null())
    }

    fn grab_exc_value(&self, frame: &DeadFrame) -> GcRef {
        grab_exc_value_from_deadframe(frame).expect("grab_exc_value_from_deadframe failed")
    }

    fn invalidate_loop(&self, token: &JitCellToken) {
        token.invalidate();
    }

    fn redirect_call_assembler(
        &self,
        old: &JitCellToken,
        new: &JitCellToken,
    ) -> Result<(), BackendError> {
        redirect_call_assembler_target(old.number, new.number)
    }

    fn free_loop(&mut self, token: &JitCellToken) {
        unregister_call_assembler_target(token.number);
        self.registered_call_assembler_tokens.remove(&token.number);
    }

    /// llmodel.py:775 bh_new(sizedescr) → gc_ll_descr.gc_malloc(sizedescr).
    fn bh_new(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        match with_cranelift_gc(|gc| gc.alloc_nursery_typed(sizedescr.get_type_id(), size).0 as i64)
        {
            Some(ptr) => ptr,
            None => {
                let layout = std::alloc::Layout::from_size_align(size, 8)
                    .unwrap_or(std::alloc::Layout::new::<u8>());
                unsafe { std::alloc::alloc_zeroed(layout) as i64 }
            }
        }
    }

    /// llmodel.py:778-782 bh_new_with_vtable(sizedescr).
    /// gc_malloc(sizedescr) + write vtable at vtable_offset.
    fn bh_new_with_vtable(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        let vtable = sizedescr.get_vtable();
        let ptr = match with_cranelift_gc(|gc| {
            gc.alloc_nursery_typed(sizedescr.get_type_id(), size).0 as i64
        }) {
            Some(p) => p,
            None => {
                let layout = std::alloc::Layout::from_size_align(size, 8)
                    .unwrap_or(std::alloc::Layout::new::<u8>());
                unsafe { std::alloc::alloc_zeroed(layout) as i64 }
            }
        };
        // llmodel.py:780-782: if self.vtable_offset is not None:
        //   self.write_int_at_mem(res, self.vtable_offset, WORD, sizedescr.get_vtable())
        if let Some(vt_off) = self.vtable_offset {
            if vtable != 0 && ptr != 0 {
                unsafe {
                    *((ptr as *mut u8).add(vt_off) as *mut usize) = vtable;
                }
            }
        }
        ptr
    }

    /// llmodel.py:816 bh_call_i: ABI-correct dispatch.
    ///
    /// ARM64/x86-64 C ABI assigns integer and float args to independent register
    /// files (x0-x7 + d0-d7 on ARM64; rdi,rsi,… + xmm0-xmm7 on x86-64).
    /// We construct `fn(ints…, floats…) -> i64` which places each group in the
    /// correct register file regardless of their original interleaving order.
    ///
    /// llmodel.py:816-820 bh_call_i(func, args_i, args_r, args_f, calldescr)
    /// calldescr.call_stub_i(func, args_i, args_r, args_f).
    ///
    /// On ARM64/x86-64, the C ABI assigns integer and floating-point args to
    /// independent register files (x0-x7 / d0-d7 on ARM64; rdi,rsi,... /
    /// xmm0-xmm7 on x86-64). So we can always construct the function pointer
    /// as `fn(ints..., floats...) -> i64` and get the correct register layout.
    fn bh_call_i(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        if func == 0 {
            return 0;
        }
        let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
            &calldescr.arg_classes,
            args_i,
            args_r,
            args_f,
        );
        unsafe {
            majit_backend::call_stub::bh_call_i_dispatch(func as usize, &int_args, &float_args)
        }
    }

    /// llmodel.py:818 bh_call_r: GcRef-returning parallel of `bh_call_i`.
    /// `lltype.Ptr(lltype.GcStruct, ...)` lowers to a host pointer that
    /// matches `i64` on 64-bit; transmute via the shared int dispatcher
    /// and wrap as `GcRef`. Without this override
    /// `bhimpl_residual_call_*_r` would silently no-op via the default
    /// trait impl at `majit-backend/lib.rs:1992`.
    fn bh_call_r(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        if func == 0 {
            return majit_ir::GcRef::NULL;
        }
        let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
            &calldescr.arg_classes,
            args_i,
            args_r,
            args_f,
        );
        let raw = unsafe {
            majit_backend::call_stub::bh_call_i_dispatch(func as usize, &int_args, &float_args)
        };
        majit_ir::GcRef(raw as usize)
    }

    /// llmodel.py:825 bh_call_f / descr.py:590-602 create_call_stub
    /// (`RESULT == lltype.Float`): route through the f64-typed
    /// dispatcher so the result lands in xmm0 / d0 rather than rax /
    /// x0. Without this override `bhimpl_residual_call_*_f` would
    /// silently no-op via the default trait impl at
    /// `majit-backend/lib.rs:2003`.
    fn bh_call_f(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> f64 {
        if func == 0 {
            return 0.0;
        }
        let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
            &calldescr.arg_classes,
            args_i,
            args_r,
            args_f,
        );
        unsafe {
            majit_backend::call_stub::bh_call_f_dispatch(func as usize, &int_args, &float_args)
        }
    }

    /// llmodel.py:834 bh_call_v / descr.py:590-602 create_call_stub
    /// (`RESULT == lltype.Void`): dispatch via the void-typed stub so
    /// the funcptr is transmuted to `extern "C" fn(...) -> ()`. Without
    /// this override the canonical `residual_call_*_v` walker would
    /// silently no-op via the default trait impl
    /// (`majit-backend/lib.rs:2013`).
    fn bh_call_v(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        // llmodel.py:834 bh_call_v / descr.py:590-602 create_call_stub
        // (`RESULT == lltype.Void`) parity: route through the void-typed
        // dispatcher so genuinely void C callees use the right C-ABI
        // signature instead of `extern "C" fn(...) -> i64` (which reads
        // garbage from rax/x0).
        if func == 0 {
            return;
        }
        let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
            &calldescr.arg_classes,
            args_i,
            args_r,
            args_f,
        );
        unsafe {
            majit_backend::call_stub::bh_call_v_dispatch(func as usize, &int_args, &float_args);
        }
    }

    /// llmodel.py:747-750 bh_raw_load_i(addr, offset, descr).
    fn bh_raw_load_i(
        &self,
        addr: i64,
        offset: i64,
        descr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        // llmodel.py:748-749: ofs, size, sign = self.unpack_arraydescr_size(descr)
        // ofs == 0 always for raw lengthless arrays (llmodel.py:749 assert)
        let size = descr.as_itemsize();
        let sign = descr.is_item_signed();
        // llmodel.py:750: return self.read_int_at_mem(addr, offset, size, sign)
        self.read_int_at_mem(addr, offset, size, sign)
    }

    /// llmodel.py:739-742 bh_raw_store_i(addr, offset, newvalue, descr).
    fn bh_raw_store_i(
        &self,
        addr: i64,
        offset: i64,
        newvalue: i64,
        descr: &majit_translate::jitcode::BhDescr,
    ) {
        // llmodel.py:740-741: ofs, size, _ = self.unpack_arraydescr_size(descr)
        // ofs == 0 always for raw lengthless arrays (llmodel.py:741 assert)
        let size = descr.as_itemsize();
        // llmodel.py:742: self.write_int_at_mem(addr, offset, size, newvalue)
        self.write_int_at_mem(addr, offset, size, newvalue);
    }

    /// llmodel.py:752-753 bh_raw_load_f(addr, offset, descr).
    fn bh_raw_load_f(
        &self,
        addr: i64,
        offset: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        // llmodel.py:753: return self.read_float_at_mem(addr, offset)
        self.read_float_at_mem(addr, offset)
    }

    /// llmodel.py:744-745 bh_raw_store_f(addr, offset, newvalue, descr).
    fn bh_raw_store_f(
        &self,
        addr: i64,
        offset: i64,
        newvalue: f64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
        // llmodel.py:745: self.write_float_at_mem(addr, offset, newvalue)
        self.write_float_at_mem(addr, offset, newvalue);
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves the typeid via the active GC runtime, mirroring how
    /// RPython looks the value up in `cpu.gc_ll_descr`.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        with_cranelift_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr)).flatten()
    }
}

// `bh_call_i_dispatch` arity table moved to
// `majit_backend::call_stub::bh_call_i_dispatch` (Slice 0d) so dynasm shares
// the same dispatch.

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use majit_backend::Backend;
    use majit_gc::collector::{GcConfig, MiniMarkGC};
    use majit_gc::flags;
    use majit_gc::header::{GcHeader, header_of};
    use majit_gc::trace::TypeInfo;
    use majit_ir::descr::{Descr, EffectInfo, ExtraEffect, SizeDescr};
    use std::collections::HashMap;

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut o = Op::new(opcode, args);
        o.pos = OpRef::from_raw(pos);
        o
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// Backend trait method must delegate to the active gc_ll_descr.
    /// This test verifies that vtables registered through the GC are
    /// resolvable via `Backend::get_typeid_from_classptr_if_gcremovetypeptr`.
    #[test]
    fn test_backend_typeid_from_classptr_via_gc_ll_descr() {
        let mut gc = MiniMarkGC::new();
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let int_vtable: usize = 0x1111_2200;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);

        let mut backend = CraneliftBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let resolved = backend.get_typeid_from_classptr_if_gcremovetypeptr(int_vtable);
        assert_eq!(resolved, Some(int_tid));
        let unknown = backend.get_typeid_from_classptr_if_gcremovetypeptr(0xCAFE_F00D);
        assert_eq!(unknown, None);
    }

    fn mk_op_with_descr(opcode: OpCode, args: &[OpRef], pos: u32, descr: majit_ir::DescrRef) -> Op {
        let mut o = Op::with_descr(opcode, args, descr);
        o.pos = OpRef::from_raw(pos);
        o
    }

    #[derive(Debug)]
    struct TestCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
        extra_info: EffectInfo,
    }

    #[derive(Debug)]
    struct TestLabelDescr {
        idx: u32,
    }

    impl Descr for TestLabelDescr {
        fn index(&self) -> u32 {
            self.idx
        }
    }

    impl Descr for TestCallDescr {
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            8
        }
        fn get_extra_info(&self) -> &EffectInfo {
            &self.extra_info
        }
    }

    fn make_call_descr(arg_types: Vec<Type>, result_type: Type) -> majit_ir::DescrRef {
        Arc::new(TestCallDescr {
            arg_types,
            result_type,
            extra_info: EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                oopspecindex: majit_ir::OopSpecIndex::None,
                ..Default::default()
            },
        })
    }

    fn make_call_assembler_descr(
        target: &JitCellToken,
        arg_types: Vec<Type>,
        result_type: Type,
    ) -> majit_ir::DescrRef {
        Arc::new(CallAssemblerDescr::new(
            target.number,
            arg_types,
            result_type,
        ))
    }

    fn make_label_descr(idx: u32) -> majit_ir::DescrRef {
        Arc::new(TestLabelDescr { idx })
    }

    extern "C" fn collect_nursery_via_runtime(_runtime_id: i64) -> i64 {
        with_cranelift_gc_required(|gc| gc.collect_nursery());
        123
    }

    extern "C" fn collect_nursery_via_runtime_void(_runtime_id: i64) {
        with_cranelift_gc_required(|gc| gc.collect_nursery());
    }

    fn may_force_void_values() -> &'static Mutex<Vec<i64>> {
        static VALUES: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_int_values() -> &'static Mutex<Vec<i64>> {
        static VALUES: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_float_values() -> &'static Mutex<Vec<u64>> {
        static VALUES: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_ref_values() -> &'static Mutex<Vec<usize>> {
        static VALUES: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn pending_call_assembler_force_values() -> &'static Mutex<Vec<(i64, Option<i64>)>> {
        static VALUES: OnceLock<Mutex<Vec<(i64, Option<i64>)>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    thread_local! {
        static TEST_EXCEPTION_VALUE: Cell<i64> = const { Cell::new(0) };
        static TEST_EXCEPTION_CALL_LOG: std::cell::RefCell<Vec<bool>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// RPython OBJECTPTR layout: typeptr at offset 0.
    #[repr(C)]
    struct FakeExcObject {
        typeptr: usize,
    }

    /// Allocate a fake exception object with typeptr at offset 0.
    /// Returns the address for use as exc_value.
    fn make_fake_exc(exc_type: i64) -> i64 {
        let obj = Box::leak(Box::new(FakeExcObject {
            typeptr: exc_type as usize,
        }));
        obj as *const FakeExcObject as usize as i64
    }

    /// pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
    fn exc_class_of(backend: &CraneliftBackend, frame: &DeadFrame) -> i64 {
        let exc = backend.grab_exc_value(frame);
        if exc.is_null() {
            0
        } else {
            unsafe { *(exc.0 as *const i64) }
        }
    }

    fn set_test_exception_state(value: i64) {
        TEST_EXCEPTION_VALUE.with(|cell| cell.set(value));
    }

    fn clear_test_exception_call_log() {
        TEST_EXCEPTION_CALL_LOG.with(|log| log.borrow_mut().clear());
    }

    fn test_exception_call_log_snapshot() -> Vec<bool> {
        TEST_EXCEPTION_CALL_LOG.with(|log| log.borrow().clone())
    }

    extern "C" fn maybe_raise_test_exception(flag: i64) {
        TEST_EXCEPTION_CALL_LOG.with(|log| log.borrow_mut().push(jit_exc_is_pending()));
        if flag != 0 {
            let value = TEST_EXCEPTION_VALUE.with(Cell::get);
            jit_exc_raise(value);
        }
    }

    extern "C" fn maybe_force_and_return_void(force_token: i64, flag: i64) {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let mut values = may_force_void_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            values.push(
                get_latest_descr_from_deadframe(&deadframe)
                    .unwrap()
                    .fail_index() as i64,
            );
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap());
            values.push(get_int_from_deadframe(&deadframe, 1).unwrap());
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xDADA)).unwrap();
        }
    }

    /// Like `maybe_force_and_return_int` but does not write to the global
    /// `may_force_int_values` vec, avoiding cross-test interference.
    extern "C" fn maybe_force_and_return_int_isolated(force_token: i64, flag: i64) -> i64 {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xBABA)).unwrap();
        }
        42
    }

    extern "C" fn maybe_force_and_return_int(force_token: i64, flag: i64) -> i64 {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let mut values = may_force_int_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap());
            values.push(get_int_from_deadframe(&deadframe, 2).unwrap());
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xBABA)).unwrap();
        }
        42
    }

    extern "C" fn maybe_force_and_return_float(force_token: i64, flag: i64) -> f64 {
        if flag != 0 {
            let deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let mut values = may_force_float_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap() as u64);
            values.push(get_float_from_deadframe(&deadframe, 1).unwrap().to_bits());
            values.push(get_int_from_deadframe(&deadframe, 2).unwrap() as u64);
        }
        12.5
    }

    extern "C" fn maybe_force_and_return_ref(
        force_token: i64,
        flag: i64,
        _runtime_id: i64,
        return_ref: i64,
    ) -> i64 {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let preview_live = get_ref_from_deadframe(&deadframe, 2).unwrap();
            set_savedata_ref_on_deadframe(&mut deadframe, preview_live).unwrap();
            with_cranelift_gc_required(|gc| gc.collect_nursery());
            let preview_result = get_ref_from_deadframe(&deadframe, 1).unwrap();
            let preview_live = get_ref_from_deadframe(&deadframe, 2).unwrap();
            let preview_return = get_ref_from_deadframe(&deadframe, 3).unwrap();
            let mut values = may_force_ref_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            values.push(preview_result.0);
            values.push(preview_live.0);
            values.push(preview_return.0);
            return get_ref_from_deadframe(&deadframe, 3).unwrap().0 as i64;
        }
        return_ref
    }

    extern "C" fn pending_call_assembler_force_probe(frame_ptr: i64) -> i64 {
        let pending = take_pending_force_local0();
        pending_call_assembler_force_values()
            .lock()
            .unwrap()
            .push((frame_ptr, pending));
        frame_ptr + pending.unwrap_or(0)
    }

    fn make_gc_backend() -> CraneliftBackend {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        // eval.rs registers regular GC types before installing the backend.
        // Keep this fixture on the same path so set_gc_allocator() can lazily
        // register JITFRAME instead of tripping the "type id not registered"
        // assertion that only exists for half-initialized test runtimes.
        gc.register_type(TypeInfo::simple(16));
        CraneliftBackend::with_gc_allocator(Box::new(gc))
    }

    #[derive(Default)]
    struct TrackingGcState {
        collecting_allocs: usize,
        no_collect_allocs: usize,
        write_barriers: usize,
        added_roots: usize,
        removed_roots: usize,
    }

    struct TrackingGc {
        state: Arc<Mutex<TrackingGcState>>,
        allocations: Vec<Box<[u8]>>,
        nursery_free: usize,
        nursery_top: usize,
    }

    impl TrackingGc {
        fn new(state: Arc<Mutex<TrackingGcState>>) -> Self {
            Self {
                state,
                allocations: Vec::new(),
                nursery_free: 0,
                nursery_top: 0,
            }
        }

        fn alloc_payload(&mut self, payload_size: usize, collecting: bool) -> GcRef {
            let total_size = GcHeader::SIZE + payload_size.max(std::mem::size_of::<usize>());
            let mut buf = vec![0u8; total_size].into_boxed_slice();
            let ptr = buf.as_mut_ptr();
            unsafe {
                *(ptr as *mut GcHeader) = GcHeader::new(0);
            }
            let obj = GcRef(ptr as usize + GcHeader::SIZE);
            self.allocations.push(buf);
            let mut state = self.state.lock().unwrap();
            if collecting {
                state.collecting_allocs += 1;
            } else {
                state.no_collect_allocs += 1;
            }
            obj
        }
    }

    impl GcAllocator for TrackingGc {
        fn alloc_nursery(&mut self, size: usize) -> GcRef {
            self.alloc_payload(size, true)
        }

        fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
            self.alloc_payload(size, false)
        }

        fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
            self.alloc_payload(base_size + item_size * length, true)
        }

        fn alloc_varsize_no_collect(
            &mut self,
            base_size: usize,
            item_size: usize,
            length: usize,
        ) -> GcRef {
            self.alloc_payload(base_size + item_size * length, false)
        }

        fn write_barrier(&mut self, _obj: GcRef) {
            self.state.lock().unwrap().write_barriers += 1;
        }

        fn jit_remember_young_pointer_from_array(&mut self, _obj: GcRef) {
            self.state.lock().unwrap().write_barriers += 1;
        }

        fn remember_young_pointer_from_array2(
            &mut self,
            _obj: GcRef,
            _index: usize,
            _card_page_shift: u32,
        ) {
            self.state.lock().unwrap().write_barriers += 1;
        }

        fn collect_nursery(&mut self) {}

        fn collect_full(&mut self) {}

        unsafe fn add_root(&mut self, _root: *mut GcRef) {
            self.state.lock().unwrap().added_roots += 1;
        }

        fn remove_root(&mut self, _root: *mut GcRef) {
            self.state.lock().unwrap().removed_roots += 1;
        }

        fn nursery_free(&self) -> *mut u8 {
            self.nursery_free as *mut u8
        }

        fn nursery_free_addr(&self) -> usize {
            std::ptr::addr_of!(self.nursery_free) as usize
        }

        fn nursery_top(&self) -> *const u8 {
            self.nursery_top as *const u8
        }

        fn nursery_top_addr(&self) -> usize {
            std::ptr::addr_of!(self.nursery_top) as usize
        }

        fn max_nursery_object_size(&self) -> usize {
            usize::MAX
        }
    }

    fn assert_compile_unsupported(
        inputargs: Vec<InputArg>,
        ops: Vec<Op>,
        token_number: u64,
        opcode: OpCode,
        expected_detail: &str,
    ) {
        let mut backend = CraneliftBackend::new();
        let mut token = JitCellToken::new(token_number);
        let err = backend
            .compile_loop(&inputargs, &ops, &mut token)
            .unwrap_err();
        match err {
            BackendError::Unsupported(msg) => {
                assert!(msg.contains(&format!("{opcode:?}")));
                assert!(
                    msg.contains(expected_detail),
                    "expected detail {expected_detail:?} in unsupported message {msg:?}"
                );
            }
            other => panic!("expected unsupported error, got {other:?}"),
        }
    }

    // ── Existing tests ──

    #[test]
    fn test_count_to_million() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(
                OpCode::IntLt,
                &[OpRef::from_raw(1), OpRef::from_raw(101)],
                2,
            ),
            mk_op(OpCode::GuardTrue, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
            mk_op(OpCode::Jump, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 1_000_000i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(0);
        let info = backend.compile_loop(&inputargs, &ops, &mut token).unwrap();
        assert!(info.code_addr != 0);

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 999_999);
    }

    #[test]
    fn test_simple_add_finish() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(40), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_finish_preserves_float_and_ref_types() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::Finish,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(1000);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(3.5), Value::Ref(GcRef(0x1234))]);
        assert_eq!(backend.get_float_value(&frame, 0), 3.5);
        assert_eq!(backend.get_ref_value(&frame, 1), GcRef(0x1234));
    }

    #[test]
    fn test_int_sub() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntSub, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(2);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(58)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(3);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_floor_div() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntFloorDiv,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(3);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42), Value::Int(6)]);
        assert_eq!(backend.get_int_value(&frame, 0), 7);
    }

    #[test]
    fn test_bitwise_ops() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntAnd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::IntOr, &[OpRef::from_raw(0), OpRef::from_raw(1)], 3),
            mk_op(OpCode::IntXor, &[OpRef::from_raw(0), OpRef::from_raw(1)], 4),
            mk_op(
                OpCode::Finish,
                &[OpRef::from_raw(2), OpRef::from_raw(3), OpRef::from_raw(4)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(4);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0xFF00), Value::Int(0x0FF0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0x0F00);
        assert_eq!(backend.get_int_value(&frame, 1), 0xFFF0);
        assert_eq!(backend.get_int_value(&frame, 2), 0xF0F0);
    }

    #[test]
    fn test_shift_ops() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntLshift,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(
                OpCode::IntRshift,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                3,
            ),
            mk_op(
                OpCode::UintRshift,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                4,
            ),
            mk_op(
                OpCode::Finish,
                &[OpRef::from_raw(2), OpRef::from_raw(3), OpRef::from_raw(4)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(5);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(-16), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), -64);
        assert_eq!(backend.get_int_value(&frame, 1), -4);
        let expected_ushr = ((-16i64 as u64) >> 2) as i64;
        assert_eq!(backend.get_int_value(&frame, 2), expected_ushr);
    }

    #[test]
    fn test_comparisons() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntLt, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::IntLe, &[OpRef::from_raw(0), OpRef::from_raw(1)], 3),
            mk_op(OpCode::IntEq, &[OpRef::from_raw(0), OpRef::from_raw(1)], 4),
            mk_op(OpCode::IntNe, &[OpRef::from_raw(0), OpRef::from_raw(1)], 5),
            mk_op(OpCode::IntGt, &[OpRef::from_raw(0), OpRef::from_raw(1)], 6),
            mk_op(OpCode::IntGe, &[OpRef::from_raw(0), OpRef::from_raw(1)], 7),
            mk_op(
                OpCode::Finish,
                &[
                    OpRef::from_raw(2),
                    OpRef::from_raw(3),
                    OpRef::from_raw(4),
                    OpRef::from_raw(5),
                    OpRef::from_raw(6),
                    OpRef::from_raw(7),
                ],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(6);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 1);
        assert_eq!(backend.get_int_value(&frame, 2), 0);
        assert_eq!(backend.get_int_value(&frame, 3), 1);
        assert_eq!(backend.get_int_value(&frame, 4), 0);
        assert_eq!(backend.get_int_value(&frame, 5), 0);
    }

    #[test]
    fn test_guard_false() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntEq,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(OpCode::GuardFalse, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntSub,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
                2,
            ),
            mk_op(OpCode::Jump, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0i64);
        constants.insert(101, 1i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(7);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(10)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    #[test]
    fn test_no_label_loop_uses_jump_arity_not_dead_extra_inputs() {
        let mut backend = CraneliftBackend::new();

        // Legacy no-LABEL traces can still carry stale extra input slots from
        // optimizer-owned virtual state, while the visible loop contract is the
        // terminal JUMP arity. This must still compile.
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
            InputArg::new_int(3),
            InputArg::new_int(4),
        ];
        let ops = vec![
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 5),
            mk_op(
                OpCode::Jump,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(70001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();
    }

    #[test]
    fn test_fail_descr() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(8);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_compiled_fail_descr_layouts_include_backend_recovery_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_ref(1)];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(1),
            OpRef::from_raw(0),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(76);
        backend.set_next_header_pc(1234);
        let mut token = JitCellToken::new(8008);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_fail_descr_layouts(&token)
            .expect("compiled layouts should exist");
        let layout = &layouts[0];
        assert_eq!(layout.source_op_index, Some(1));
        let recovery = layout
            .recovery_layout
            .as_ref()
            .expect("guard layout should include backend recovery layout");
        assert_eq!(recovery.frames.len(), 1);
        assert_eq!(recovery.frames[0].trace_id, Some(76));
        assert_eq!(recovery.frames[0].header_pc, Some(1234));
        assert_eq!(recovery.frames[0].pc, 1234);
        assert_eq!(
            recovery.frames[0].slots,
            vec![
                majit_backend::ExitValueSourceLayout::ExitValue(0),
                majit_backend::ExitValueSourceLayout::ExitValue(1),
            ]
        );
        assert_eq!(
            recovery.frames[0].slot_types,
            Some(vec![Type::Ref, Type::Int])
        );
    }

    #[test]
    fn test_compiled_terminal_exit_layouts_include_backend_jump_recovery_layout() {
        let mut backend = CraneliftBackend::new();

        // Int (not Float) at slot 0 so the JUMP backedge passes Int into a
        // Ref-typed inputarg position. build_ref_root_slots permits Int-at-Ref
        // (SameAsI may forward GC pointers without changing runtime value);
        // Float-at-Ref is rejected because IEEE754 bits are never valid
        // pointers.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::Jump,
                &[OpRef::from_raw(1), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
            ),
        ];

        backend.set_next_trace_id(77);
        backend.set_next_header_pc(1234);
        let mut token = JitCellToken::new(8009);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_terminal_exit_layouts(&token)
            .expect("compiled terminal layouts should exist");
        assert_eq!(layouts.len(), 1);
        let layout = &layouts[0];
        assert_eq!(layout.op_index, 1);
        assert_eq!(layout.trace_id, 77);
        assert_eq!(layout.fail_index, u32::MAX);
        assert_eq!(layout.exit_types, vec![Type::Ref, Type::Int]);
        assert!(!layout.is_finish);
        assert_eq!(layout.gc_ref_slots, vec![0]);
        assert!(layout.force_token_slots.is_empty());
        let recovery = layout
            .recovery_layout
            .as_ref()
            .expect("jump layout should include backend recovery layout");
        assert_eq!(recovery.frames.len(), 1);
        assert_eq!(recovery.frames[0].trace_id, Some(77));
        assert_eq!(recovery.frames[0].header_pc, Some(1234));
        assert_eq!(recovery.frames[0].pc, 1234);
        assert_eq!(
            recovery.frames[0].slots,
            vec![
                majit_backend::ExitValueSourceLayout::ExitValue(0),
                majit_backend::ExitValueSourceLayout::ExitValue(1),
            ]
        );
        assert_eq!(
            recovery.frames[0].slot_types,
            Some(vec![Type::Ref, Type::Int])
        );
    }

    #[test]
    fn test_compiled_terminal_exit_layouts_include_backend_finish_recovery_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::Finish,
                &[OpRef::from_raw(1), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
            ),
        ];

        backend.set_next_trace_id(79);
        backend.set_next_header_pc(5678);
        let mut token = JitCellToken::new(8011);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_terminal_exit_layouts(&token)
            .expect("compiled terminal layouts should exist");
        assert_eq!(layouts.len(), 1);
        let layout = &layouts[0];
        assert_eq!(layout.op_index, 1);
        assert_eq!(layout.trace_id, 79);
        assert_eq!(layout.fail_index, 0);
        assert_eq!(layout.exit_types, vec![Type::Ref, Type::Float]);
        assert!(layout.is_finish);
        assert_eq!(layout.gc_ref_slots, vec![0]);
        assert!(layout.force_token_slots.is_empty());
        let recovery = layout
            .recovery_layout
            .as_ref()
            .expect("finish layout should include backend recovery layout");
        assert_eq!(recovery.frames.len(), 1);
        assert_eq!(recovery.frames[0].trace_id, Some(79));
        assert_eq!(recovery.frames[0].header_pc, Some(5678));
        assert_eq!(recovery.frames[0].pc, 5678);
        assert_eq!(
            recovery.frames[0].slots,
            vec![
                majit_backend::ExitValueSourceLayout::ExitValue(0),
                majit_backend::ExitValueSourceLayout::ExitValue(1),
            ]
        );
        assert_eq!(
            recovery.frames[0].slot_types,
            Some(vec![Type::Ref, Type::Float])
        );
    }

    #[test]
    fn test_update_terminal_exit_recovery_layout_patches_compiled_terminal_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Jump, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(78);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8010);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let patched = majit_backend::ExitRecoveryLayout {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: vec![majit_backend::ExitFrameLayout {
                trace_id: None,
                header_pc: None,
                source_guard: None,
                pc: 4242,
                jitcode_index: 0,
                slots: vec![majit_backend::ExitValueSourceLayout::Constant(99)],
                slot_types: Some(vec![Type::Int]),
            }],
            virtual_layouts: Vec::new(),
            pending_field_layouts: Vec::new(),
        };
        assert!(backend.update_terminal_exit_recovery_layout(&token, 78, 1, patched.clone()));

        let layouts = backend
            .compiled_terminal_exit_layouts(&token)
            .expect("compiled terminal layouts should exist");
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].recovery_layout, Some(patched));
    }

    #[test]
    fn test_compiled_trace_layout_queries_find_attached_bridge_by_trace_id() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardFalse, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::smallvec![OpRef::from_raw(0)]);
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(90);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8012);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];
        let fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            90,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );

        backend.set_next_trace_id(91);
        backend.set_next_header_pc(2000);
        backend
            .compile_bridge(&fail_descr, &bridge_inputargs, &bridge_ops, &token, &[])
            .unwrap();

        let root_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 90)
            .expect("root trace layouts should exist");
        assert!(root_layouts.iter().all(|layout| layout.trace_id == 90));
        let root_info = backend
            .compiled_trace_info(&token, 90)
            .expect("root trace info should exist");
        assert_eq!(root_info.trace_id, 90);
        assert_eq!(root_info.input_types, vec![Type::Int]);
        assert_eq!(root_info.header_pc, 1000);
        assert_eq!(root_info.source_guard, None);

        let bridge_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 91)
            .expect("bridge trace layouts should exist");
        assert_eq!(bridge_layouts.len(), 1);
        assert_eq!(bridge_layouts[0].trace_id, 91);
        assert!(bridge_layouts[0].is_finish);
        let bridge_info = backend
            .compiled_trace_info(&token, 91)
            .expect("bridge trace info should exist");
        assert_eq!(bridge_info.trace_id, 91);
        assert_eq!(bridge_info.input_types, vec![Type::Int]);
        assert_eq!(bridge_info.header_pc, 2000);
        assert_eq!(bridge_info.source_guard, Some((90, 0)));

        let bridge_terminal_layouts = backend
            .compiled_trace_terminal_exit_layouts(&token, 91)
            .expect("bridge terminal layouts should exist");
        assert_eq!(bridge_terminal_layouts.len(), 1);
        assert_eq!(bridge_terminal_layouts[0].trace_id, 91);
        assert_eq!(
            bridge_terminal_layouts[0]
                .recovery_layout
                .as_ref()
                .expect("bridge terminal layout should have recovery")
                .frames[0]
                .pc,
            2000
        );
    }

    #[test]
    fn test_compiled_bridge_recovery_layouts_inherit_source_guard_caller_frames() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardFalse, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::smallvec![OpRef::from_raw(0)]);
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(190);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8013);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let source_layout = majit_backend::ExitRecoveryLayout {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: vec![
                majit_backend::ExitFrameLayout {
                    trace_id: Some(10),
                    header_pc: Some(900),
                    source_guard: Some((9, 0)),
                    pc: 900,
                    jitcode_index: 0,
                    slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
                majit_backend::ExitFrameLayout {
                    trace_id: Some(190),
                    header_pc: Some(1000),
                    source_guard: None,
                    pc: 1000,
                    jitcode_index: 0,
                    slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
            ],
            virtual_layouts: vec![majit_backend::ExitVirtualLayout::Array {
                descr_index: 17,
                clear: false,
                kind: 1,
                items: vec![
                    majit_backend::ExitValueSourceLayout::ExitValue(0),
                    majit_backend::ExitValueSourceLayout::Constant(44),
                ],
            }],
            pending_field_layouts: vec![majit_backend::ExitPendingFieldLayout {
                descr_index: 33,
                item_index: Some(1),
                is_array_item: true,
                target: majit_backend::ExitValueSourceLayout::Virtual(0),
                value: majit_backend::ExitValueSourceLayout::ExitValue(0),
                field_offset: 0,
                field_size: 8,
                field_type: Type::Int,
            }],
        };
        assert!(backend.update_fail_descr_recovery_layout(&token, 190, 0, source_layout.clone()));

        let bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            190,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );
        let mut bridge_guard = mk_op(OpCode::GuardFalse, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        bridge_guard.fail_args = Some(smallvec::smallvec![OpRef::from_raw(0)]);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            bridge_guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(191);
        backend.set_next_header_pc(2000);
        backend
            .compile_bridge(&bridge_fail_descr, &inputargs, &bridge_ops, &token, &[])
            .unwrap();

        let bridge_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 191)
            .expect("bridge trace layouts should exist");
        let guard_layout = bridge_layouts
            .iter()
            .find(|layout| !layout.is_finish)
            .expect("bridge guard layout should exist");
        let guard_recovery = guard_layout
            .recovery_layout
            .as_ref()
            .expect("bridge guard should carry backend recovery");
        assert_eq!(guard_recovery.frames.len(), 2);
        assert_eq!(guard_recovery.frames[0].trace_id, Some(10));
        assert_eq!(guard_recovery.frames[0].header_pc, Some(900));
        assert_eq!(guard_recovery.frames[0].source_guard, Some((9, 0)));
        assert_eq!(guard_recovery.frames[1].trace_id, Some(191));
        assert_eq!(guard_recovery.frames[1].header_pc, Some(2000));
        assert_eq!(guard_recovery.frames[1].source_guard, Some((190, 0)));
        assert_eq!(guard_recovery.frames[0].pc, 900);
        assert_eq!(guard_recovery.frames[1].pc, 2000);
        assert_eq!(guard_recovery.frames[0].slot_types, Some(vec![Type::Int]));
        assert_eq!(guard_recovery.frames[1].slot_types, Some(vec![Type::Int]));
        assert_eq!(
            guard_recovery.virtual_layouts,
            source_layout.virtual_layouts
        );
        assert_eq!(
            guard_recovery.pending_field_layouts,
            source_layout.pending_field_layouts
        );

        let bridge_terminal_layouts = backend
            .compiled_trace_terminal_exit_layouts(&token, 191)
            .expect("bridge terminal layouts should exist");
        let finish_recovery = bridge_terminal_layouts[0]
            .recovery_layout
            .as_ref()
            .expect("bridge finish should carry backend recovery");
        assert_eq!(finish_recovery.frames.len(), 2);
        assert_eq!(finish_recovery.frames[0].pc, 900);
        assert_eq!(finish_recovery.frames[1].pc, 2000);
        assert_eq!(
            finish_recovery.virtual_layouts,
            source_layout.virtual_layouts
        );
        assert_eq!(
            finish_recovery.pending_field_layouts,
            source_layout.pending_field_layouts
        );
    }

    #[test]
    fn test_nested_bridge_compilation_uses_source_trace_fail_descr_tree() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut root_guard = mk_op(OpCode::GuardFalse, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        root_guard.fail_args = Some(smallvec::smallvec![OpRef::from_raw(0)]);
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            root_guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(290);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8014);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let root_layout = majit_backend::ExitRecoveryLayout {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: vec![
                majit_backend::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 900,
                    jitcode_index: 0,
                    slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
                majit_backend::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 1000,
                    jitcode_index: 0,
                    slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
            ],
            virtual_layouts: vec![majit_backend::ExitVirtualLayout::Array {
                descr_index: 17,
                clear: false,
                kind: 1,
                items: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
            }],
            pending_field_layouts: vec![majit_backend::ExitPendingFieldLayout {
                descr_index: 33,
                item_index: Some(0),
                is_array_item: true,
                target: majit_backend::ExitValueSourceLayout::Virtual(0),
                value: majit_backend::ExitValueSourceLayout::ExitValue(0),
                field_offset: 0,
                field_size: 8,
                field_type: Type::Int,
            }],
        };
        assert!(backend.update_fail_descr_recovery_layout(&token, 290, 0, root_layout));

        let bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            290,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );
        let mut bridge_guard = mk_op(OpCode::GuardFalse, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        bridge_guard.fail_args = Some(smallvec::smallvec![OpRef::from_raw(0)]);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            bridge_guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(291);
        backend.set_next_header_pc(2000);
        backend
            .compile_bridge(&bridge_fail_descr, &inputargs, &bridge_ops, &token, &[])
            .unwrap();
        assert!(
            backend
                .compiled_trace_fail_descr_layouts(&token, 291)
                .is_some()
        );
        assert_eq!(
            backend
                .compiled_trace_info(&token, 291)
                .expect("first bridge trace info")
                .source_guard,
            Some((290, 0))
        );

        let bridge_source_layout = majit_backend::ExitRecoveryLayout {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: vec![
                majit_backend::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 444,
                    jitcode_index: 0,
                    slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
                majit_backend::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: Some((290, 0)),
                    pc: 2000,
                    jitcode_index: 0,
                    slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
            ],
            virtual_layouts: vec![majit_backend::ExitVirtualLayout::Array {
                descr_index: 99,
                clear: false,
                kind: 1,
                items: vec![
                    majit_backend::ExitValueSourceLayout::ExitValue(0),
                    majit_backend::ExitValueSourceLayout::Constant(55),
                ],
            }],
            pending_field_layouts: vec![majit_backend::ExitPendingFieldLayout {
                descr_index: 77,
                item_index: Some(1),
                is_array_item: true,
                target: majit_backend::ExitValueSourceLayout::Virtual(0),
                value: majit_backend::ExitValueSourceLayout::ExitValue(0),
                field_offset: 0,
                field_size: 8,
                field_type: Type::Int,
            }],
        };
        assert!(backend.update_fail_descr_recovery_layout(
            &token,
            291,
            0,
            bridge_source_layout.clone()
        ));

        let nested_bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            291,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );
        backend.set_next_trace_id(292);
        backend.set_next_header_pc(3000);
        backend
            .compile_bridge(
                &nested_bridge_fail_descr,
                &inputargs,
                &bridge_ops,
                &token,
                &[],
            )
            .unwrap();

        assert!(
            backend
                .compiled_trace_fail_descr_layouts(&token, 291)
                .is_some()
        );
        let nested_info = backend
            .compiled_trace_info(&token, 292)
            .expect("nested bridge trace info should exist");
        assert_eq!(nested_info.source_guard, Some((291, 0)));

        let nested_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 292)
            .expect("nested bridge layouts should exist");
        let nested_guard_layout = nested_layouts
            .iter()
            .find(|layout| !layout.is_finish)
            .expect("nested bridge guard layout should exist");
        let nested_guard_recovery = nested_guard_layout
            .recovery_layout
            .as_ref()
            .expect("nested bridge guard should carry backend recovery");
        assert_eq!(
            nested_guard_recovery
                .frames
                .iter()
                .map(|frame| frame.pc)
                .collect::<Vec<_>>(),
            vec![444, 3000]
        );
        assert_eq!(
            nested_guard_recovery
                .frames
                .iter()
                .map(|frame| frame.source_guard)
                .collect::<Vec<_>>(),
            vec![None, Some((291, 0))]
        );
        assert_eq!(
            nested_guard_recovery.virtual_layouts,
            bridge_source_layout.virtual_layouts
        );
        assert_eq!(
            nested_guard_recovery.pending_field_layouts,
            bridge_source_layout.pending_field_layouts
        );

        let nested_terminal_layouts = backend
            .compiled_trace_terminal_exit_layouts(&token, 292)
            .expect("nested bridge terminal layouts should exist");
        let nested_finish_recovery = nested_terminal_layouts[0]
            .recovery_layout
            .as_ref()
            .expect("nested bridge finish should carry backend recovery");
        assert_eq!(
            nested_finish_recovery
                .frames
                .iter()
                .map(|frame| frame.pc)
                .collect::<Vec<_>>(),
            vec![444, 3000]
        );
        assert_eq!(
            nested_finish_recovery.virtual_layouts,
            bridge_source_layout.virtual_layouts
        );
        assert_eq!(
            nested_finish_recovery.pending_field_layouts,
            bridge_source_layout.pending_field_layouts
        );
    }

    #[test]
    fn test_sum_loop() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(1), OpRef::from_raw(0)], 2),
            mk_op(
                OpCode::IntSub,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                3,
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(3), OpRef::from_raw(101)],
                4,
            ),
            mk_op(OpCode::GuardTrue, &[OpRef::from_raw(4)], OpRef::NONE.raw()),
            mk_op(
                OpCode::Jump,
                &[OpRef::from_raw(3), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 0i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(9);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 5049);
    }

    #[test]
    fn test_multi_output_finish() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(1)], 3),
            mk_op(
                OpCode::Finish,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(2),
                    OpRef::from_raw(3),
                ],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(10);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 3);
        assert_eq!(backend.get_int_value(&frame, 1), 7);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(backend.get_int_value(&frame, 3), 21);
    }

    // ── Call operation tests ──

    #[test]
    fn test_call_i_simple_add() {
        extern "C" fn add_two(a: i64, b: i64) -> i64 {
            a + b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::CallI,
                &[OpRef::from_raw(100), OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_two as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(20);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(40), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_call_pure_i() {
        extern "C" fn multiply(a: i64, b: i64) -> i64 {
            a * b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::CallPureI,
                &[OpRef::from_raw(100), OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, multiply as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(21);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_call_n_void_result() {
        static mut CALL_COUNTER: i64 = 0;

        extern "C" fn increment_counter(amount: i64) {
            unsafe {
                CALL_COUNTER += amount;
            }
        }

        unsafe {
            CALL_COUNTER = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, increment_counter as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(22);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(10)]);
        assert_eq!(backend.get_int_value(&frame, 0), 10);
        assert_eq!(unsafe { CALL_COUNTER }, 10);
    }

    #[test]
    fn test_call_f_double_result() {
        extern "C" fn add_doubles(a: f64, b: f64) -> f64 {
            a + b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Float, Type::Float], Type::Float);

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::CallF,
                &[OpRef::from_raw(100), OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_doubles as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(23);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(1.5), Value::Float(2.5)]);
        let result = backend.get_float_value(&frame, 0);
        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_call_in_loop() {
        extern "C" fn add_one(a: i64) -> i64 {
            a + 1
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallI,
                &[OpRef::from_raw(100), OpRef::from_raw(0)],
                1,
                descr,
            ),
            mk_op(
                OpCode::IntLt,
                &[OpRef::from_raw(1), OpRef::from_raw(101)],
                2,
            ),
            mk_op(OpCode::GuardTrue, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
            mk_op(OpCode::Jump, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_one as *const () as i64);
        constants.insert(101, 100i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(24);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
    }

    #[test]
    fn test_call_n_order_preserved_in_loop() {
        use std::sync::{Mutex, OnceLock};

        static EVENTS: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();

        extern "C" fn log_char(v: i64) {
            EVENTS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push(1000 + v);
        }

        extern "C" fn log_num(v: i64) {
            EVENTS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push(v);
        }

        EVENTS
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .clear();

        let mut backend = CraneliftBackend::new();
        let call_void = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
                OpRef::NONE.raw(),
                call_void.clone(),
            ),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(102), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                call_void,
            ),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(103)],
                1,
            ),
            mk_op(
                OpCode::IntLt,
                &[OpRef::from_raw(1), OpRef::from_raw(104)],
                2,
            ),
            mk_op(OpCode::GuardTrue, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
            mk_op(OpCode::Jump, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, log_char as *const () as i64);
        constants.insert(101, 32);
        constants.insert(102, log_num as *const () as i64);
        constants.insert(103, 1);
        constants.insert(104, 3);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(24_001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 2);
        assert_eq!(
            *EVENTS.get().unwrap().lock().unwrap(),
            vec![1032, 0, 1032, 1, 1032, 2]
        );
    }

    #[test]
    fn test_guard_exception_exact_match_returns_value_and_clears_deadframe_exception() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(make_fake_exc(0x1111));

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardException, &[OpRef::from_raw(101)], 1);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                descr,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(101, 0x1111);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(25);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        // GUARD_EXCEPTION matched — Finish returns the exception ref (fake object)
        assert!(!backend.get_ref_value(&frame, 0).is_null());
        assert_eq!(exc_class_of(&backend, &frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert!(!jit_exc_is_pending());

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
        assert_eq!(exc_class_of(&backend, &frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_guard_exception_exact_mismatch_preserves_deadframe_exception() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(make_fake_exc(0x2222));

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardException, &[OpRef::from_raw(101)], 1);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                descr,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(101, 0x1111);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(26);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(exc_class_of(&backend, &frame), 0x2222);
        assert!(!backend.grab_exc_value(&frame).is_null());
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_guard_no_exception_failure_preserves_deadframe_exception() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(make_fake_exc(0x1111));

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                descr,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(27);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(exc_class_of(&backend, &frame), 0x1111);
        assert!(!backend.grab_exc_value(&frame).is_null());
        assert!(!jit_exc_is_pending());

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 0);
        assert_eq!(exc_class_of(&backend, &frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_execute_token_ints_raw_preserves_exception_and_layout_metadata() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(make_fake_exc(0x1111));

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                descr,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(27_001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let raw = backend.execute_token_ints_raw(&token, &[1]);
        assert!(!raw.is_finish);
        assert_eq!(raw.fail_index, 0);
        assert_eq!(raw.outputs, vec![1]);
        assert_eq!(raw.typed_outputs, vec![Value::Int(1)]);
        assert_eq!(raw.savedata, None);
        // pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
        assert!(!raw.exception_value.is_null());
        assert_eq!(unsafe { *(raw.exception_value.0 as *const i64) }, 0x1111);
        let layout = raw
            .exit_layout
            .expect("raw exit should expose backend layout");
        assert_eq!(layout.fail_index, 0);
        assert_eq!(layout.source_op_index, Some(2));
        assert!(layout.recovery_layout.is_some());
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_save_restore_exception_roundtrip_matches_rpython_order() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(make_fake_exc(0x3333));

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardException, &[OpRef::from_raw(101)], 3);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                descr.clone(),
            ),
            mk_op(OpCode::SaveExcClass, &[], 1),
            mk_op(OpCode::SaveException, &[], 2),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(103)],
                OpRef::NONE.raw(),
                descr,
            ),
            mk_op(
                OpCode::RestoreException,
                &[OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(101, 0x3333);
        constants.insert(103, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(28);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        // GUARD_EXCEPTION matched — Finish returns the exception ref (fake object)
        assert!(!backend.get_ref_value(&frame, 0).is_null());
        assert_eq!(exc_class_of(&backend, &frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert_eq!(test_exception_call_log_snapshot(), vec![false, false]);
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_deadframe_exception_ref_survives_collection_after_execute_token() {
        jit_exc_clear();

        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));

        let exception_ref = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(exception_ref.0));
        unsafe {
            // RPython OBJECTPTR: typeptr at offset 0, payload at offset 8
            *(exception_ref.0 as *mut u64) = 0x4444;
            *((exception_ref.0 + 8) as *mut u64) = 0xCAFEBABE;
        }

        set_test_exception_state(exception_ref.0 as i64);
        clear_test_exception_call_log();

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef::from_raw(100), OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                descr,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(29);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        with_cranelift_gc_required(|gc| gc.collect_nursery());

        assert_eq!(exc_class_of(&backend, &frame), 0x4444);
        let moved = backend.grab_exc_value(&frame);
        assert!(!moved.is_null());
        assert_ne!(moved, exception_ref);
        // typeptr at offset 0, payload at offset 8
        assert_eq!(unsafe { *(moved.0 as *const u64) }, 0x4444);
        assert_eq!(unsafe { *((moved.0 + 8) as *const u64) }, 0xCAFEBABE);
    }

    // ── SameAs variants ──

    #[test]
    fn test_same_as_r() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::SameAsR, &[OpRef::from_raw(0)], 1),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(31);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(0x1234))]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0x1234));
    }

    #[test]
    fn test_same_as_f() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::SameAsF, &[OpRef::from_raw(0)], 1),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(32);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(3.14)]);
        let result = backend.get_float_value(&frame, 0);
        assert!((result - 3.14).abs() < 1e-10);
    }

    // ── Compile bridge test ──

    // ── Conditional call tests ──

    #[test]
    fn test_cond_call_n_calls_when_nonzero() {
        static mut COND_CALL_RESULT: i64 = 0;

        extern "C" fn set_value(v: i64) {
            unsafe {
                COND_CALL_RESULT = v;
            }
        }

        unsafe {
            COND_CALL_RESULT = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::CondCallN,
                &[OpRef::from_raw(0), OpRef::from_raw(100), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, set_value as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(40);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1), Value::Int(99)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
        assert_eq!(unsafe { COND_CALL_RESULT }, 99);
    }

    #[test]
    fn test_cond_call_n_skips_when_zero() {
        static mut COND_CALL_RESULT2: i64 = 0;

        extern "C" fn set_value2(v: i64) {
            unsafe {
                COND_CALL_RESULT2 = v;
            }
        }

        unsafe {
            COND_CALL_RESULT2 = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::CondCallN,
                &[OpRef::from_raw(0), OpRef::from_raw(100), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, set_value2 as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(41);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(99)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
        assert_eq!(unsafe { COND_CALL_RESULT2 }, 0);
    }

    #[test]
    fn test_cond_call_value_i_nonzero() {
        extern "C" fn compute(a: i64) -> i64 {
            a * 10
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::CondCallValueI,
                &[OpRef::from_raw(0), OpRef::from_raw(100), OpRef::from_raw(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, compute as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(42);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 50);
    }

    #[test]
    fn test_cond_call_value_i_zero() {
        extern "C" fn compute2(a: i64) -> i64 {
            a * 10
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::CondCallValueI,
                &[OpRef::from_raw(0), OpRef::from_raw(100), OpRef::from_raw(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, compute2 as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(43);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    // ── Guard variant tests ──

    #[test]
    fn test_guard_nonnull() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntSub,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(
                OpCode::GuardNonnull,
                &[OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Jump, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(44);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i0=5->i1=4, i0=4->i1=3, ..., i0=1->i1=0 (guard fails).
        // Guard saves the loop inputarg (i0), so saved value is 1.
        let frame = backend.execute_token(&token, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
    }

    #[test]
    fn test_guard_isnull_passes() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::GuardIsnull,
                &[OpRef::from_raw(0)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(45);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    #[test]
    fn test_guard_isnull_fails() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::GuardIsnull,
                &[OpRef::from_raw(0)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(46);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_guard_value() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::GuardValue,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        // Test: value matches -> guard passes, reaches Finish
        let mut constants = HashMap::new();
        constants.insert(100, 42i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(47);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish()); // Finish
        assert_eq!(backend.get_int_value(&frame, 0), 42);

        // Test: value doesn't match -> guard fails
        let mut constants2 = HashMap::new();
        constants2.insert(100, 42i64);
        backend.set_constants(constants2);

        let mut token2 = JitCellToken::new(48);
        backend.compile_loop(&inputargs, &ops, &mut token2).unwrap();
        let frame2 = backend.execute_token(&token2, &[Value::Int(99)]);
        let descr2 = backend.get_latest_descr(&frame2);
        assert_eq!(descr2.fail_index(), 0); // guard failure
        assert_eq!(backend.get_int_value(&frame2, 0), 99);
    }

    // ── Test descriptors for field/array ops ──

    use majit_ir::descr::{ArrayDescr, FieldDescr, InteriorFieldDescr};

    #[derive(Debug)]
    struct TestSizeDescr {
        size: usize,
        type_id: u32,
        vtable: usize,
    }

    impl Descr for TestSizeDescr {
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            self.size
        }

        fn type_id(&self) -> u32 {
            self.type_id
        }

        fn is_immutable(&self) -> bool {
            false
        }

        fn is_object(&self) -> bool {
            self.vtable != 0
        }

        fn vtable(&self) -> usize {
            self.vtable
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        offset: usize,
        field_size: usize,
        field_type: Type,
        signed: bool,
    }

    impl Descr for TestFieldDescr {
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }
        fn field_size(&self) -> usize {
            self.field_size
        }
        fn field_type(&self) -> Type {
            self.field_type
        }
        fn is_field_signed(&self) -> bool {
            self.signed
        }
    }

    #[derive(Debug)]
    struct TestArrayDescr {
        base_size: usize,
        item_size: usize,
        item_type: Type,
        signed: bool,
        len_descr: Option<Arc<TestFieldDescr>>,
    }

    #[derive(Debug)]
    struct TestInteriorFieldDescr {
        array_descr: Arc<TestArrayDescr>,
        field_descr: Arc<TestFieldDescr>,
    }

    impl Descr for TestArrayDescr {
        fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
            Some(self)
        }
    }

    impl ArrayDescr for TestArrayDescr {
        fn base_size(&self) -> usize {
            self.base_size
        }
        fn item_size(&self) -> usize {
            self.item_size
        }
        fn type_id(&self) -> u32 {
            0
        }
        fn item_type(&self) -> Type {
            self.item_type
        }
        fn is_item_signed(&self) -> bool {
            self.signed
        }
        fn len_descr(&self) -> Option<&dyn FieldDescr> {
            self.len_descr
                .as_ref()
                .map(|d| d.as_ref() as &dyn FieldDescr)
        }
    }

    impl Descr for TestInteriorFieldDescr {
        fn as_interior_field_descr(&self) -> Option<&dyn InteriorFieldDescr> {
            Some(self)
        }
    }

    impl InteriorFieldDescr for TestInteriorFieldDescr {
        fn array_descr(&self) -> &dyn ArrayDescr {
            self.array_descr.as_ref()
        }

        fn field_descr(&self) -> &dyn FieldDescr {
            self.field_descr.as_ref()
        }
    }

    fn make_field_descr(
        offset: usize,
        field_size: usize,
        field_type: Type,
        signed: bool,
    ) -> majit_ir::DescrRef {
        Arc::new(TestFieldDescr {
            offset,
            field_size,
            field_type,
            signed,
        })
    }

    fn make_size_descr(size: usize, type_id: u32) -> majit_ir::DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
        })
    }

    fn make_array_descr(
        base_size: usize,
        item_size: usize,
        item_type: Type,
        len_offset: Option<usize>,
    ) -> majit_ir::DescrRef {
        make_array_descr_with_signedness(base_size, item_size, item_type, true, len_offset)
    }

    fn make_array_descr_with_signedness(
        base_size: usize,
        item_size: usize,
        item_type: Type,
        signed: bool,
        len_offset: Option<usize>,
    ) -> majit_ir::DescrRef {
        let len_descr = len_offset.map(|off| {
            Arc::new(TestFieldDescr {
                offset: off,
                field_size: 8,
                field_type: Type::Int,
                signed: true,
            })
        });
        Arc::new(TestArrayDescr {
            base_size,
            item_size,
            item_type,
            signed,
            len_descr,
        })
    }

    fn make_interior_field_descr(
        array_base_size: usize,
        array_item_size: usize,
        array_item_type: Type,
        field_offset: usize,
        field_size: usize,
        field_type: Type,
        field_signed: bool,
    ) -> majit_ir::DescrRef {
        let array_descr = Arc::new(TestArrayDescr {
            base_size: array_base_size,
            item_size: array_item_size,
            item_type: array_item_type,
            signed: true,
            len_descr: None,
        });
        let field_descr = Arc::new(TestFieldDescr {
            offset: field_offset,
            field_size,
            field_type,
            signed: field_signed,
        });
        Arc::new(TestInteriorFieldDescr {
            array_descr,
            field_descr,
        })
    }

    // ── Float comparison tests ──

    #[test]
    fn test_float_comparisons() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::FloatLt,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(
                OpCode::FloatLe,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                3,
            ),
            mk_op(
                OpCode::FloatEq,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                4,
            ),
            mk_op(
                OpCode::FloatNe,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                5,
            ),
            mk_op(
                OpCode::FloatGt,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                6,
            ),
            mk_op(
                OpCode::FloatGe,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                7,
            ),
            mk_op(
                OpCode::Finish,
                &[
                    OpRef::from_raw(2),
                    OpRef::from_raw(3),
                    OpRef::from_raw(4),
                    OpRef::from_raw(5),
                    OpRef::from_raw(6),
                    OpRef::from_raw(7),
                ],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(60);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // 1.5 < 2.5
        let frame = backend.execute_token(&token, &[Value::Float(1.5), Value::Float(2.5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1); // lt
        assert_eq!(backend.get_int_value(&frame, 1), 1); // le
        assert_eq!(backend.get_int_value(&frame, 2), 0); // eq
        assert_eq!(backend.get_int_value(&frame, 3), 1); // ne
        assert_eq!(backend.get_int_value(&frame, 4), 0); // gt
        assert_eq!(backend.get_int_value(&frame, 5), 0); // ge
    }

    #[test]
    fn test_float_comparisons_equal() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::FloatEq,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(
                OpCode::FloatNe,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                3,
            ),
            mk_op(
                OpCode::FloatLe,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                4,
            ),
            mk_op(
                OpCode::FloatGe,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                5,
            ),
            mk_op(
                OpCode::Finish,
                &[
                    OpRef::from_raw(2),
                    OpRef::from_raw(3),
                    OpRef::from_raw(4),
                    OpRef::from_raw(5),
                ],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(61);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(3.14), Value::Float(3.14)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1); // eq
        assert_eq!(backend.get_int_value(&frame, 1), 0); // ne
        assert_eq!(backend.get_int_value(&frame, 2), 1); // le
        assert_eq!(backend.get_int_value(&frame, 3), 1); // ge
    }

    // ── Field access tests ──

    #[test]
    fn test_getfield_gc_i() {
        let mut backend = CraneliftBackend::new();

        // Simulate a struct: [padding(8), i64_field(8)]
        // The field is at offset 8, size 8, type Int.
        let fd = make_field_descr(8, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef::from_raw(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(70);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Allocate a fake struct on the heap
        let mut data: Vec<i64> = vec![0xDEAD, 42];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_setfield_gc() {
        let mut backend = CraneliftBackend::new();

        let fd = make_field_descr(8, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                fd,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(71);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0];
        let ptr = data.as_mut_ptr() as usize;

        backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(99)]);
        assert_eq!(data[1], 99);
    }

    #[test]
    fn test_getfield_small_signed() {
        let mut backend = CraneliftBackend::new();

        // i32 field at offset 0, signed
        let fd = make_field_descr(0, 4, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef::from_raw(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(72);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Write -1i32 into the buffer
        let mut data: Vec<u8> = vec![0; 8];
        let val: i32 = -1;
        data[..4].copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), -1i64);
    }

    // ── Array access tests ──

    #[test]
    fn test_getarrayitem_gc_i() {
        let mut backend = CraneliftBackend::new();

        // Array: base_size=16 (header), item_size=8, items are i64
        let ad = make_array_descr(16, 8, Type::Int, None);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(73);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Layout: 16 bytes header + items
        // Total: 16 + 3*8 = 40 bytes = 5 i64s
        let mut data: Vec<i64> = vec![0xAAAA, 0xBBBB, 10, 20, 30]; // header(2), items(3)
        let ptr = data.as_mut_ptr() as usize;

        // Get item at index 1 (should be 20)
        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(1)]);
        assert_eq!(backend.get_int_value(&frame, 0), 20);
    }

    #[test]
    fn test_setarrayitem_gc() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(16, 8, Type::Int, None);

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(74);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0, 0, 0, 0]; // header(2) + items(3)
        let ptr = data.as_mut_ptr() as usize;

        // Set item at index 2 to 42
        backend.execute_token(
            &token,
            &[Value::Ref(GcRef(ptr)), Value::Int(2), Value::Int(42)],
        );
        assert_eq!(data[4], 42); // header(2) + index 2 = slot 4
    }

    #[test]
    fn test_arraylen_gc() {
        let mut backend = CraneliftBackend::new();

        // Array with length at offset 8 (second i64 in header)
        let ad = make_array_descr(16, 8, Type::Int, Some(8));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(OpCode::ArraylenGc, &[OpRef::from_raw(0)], 1, ad),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(75);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // header: [type_id, length=5]
        let mut data: Vec<i64> = vec![0xAAAA, 5, 10, 20, 30, 40, 50];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 5);
    }

    // ── NurseryPtrIncrement test ──

    #[test]
    fn test_jump_target_label_args_are_declared_even_without_local_producer() {
        let mut backend = CraneliftBackend::new();
        let loop_descr = make_label_descr(9001);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                loop_descr.clone(),
            ),
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(50)],
                OpRef::NONE.raw(),
                loop_descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(50)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(76_001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_compile_loop_with_start_and_loop_labels_can_reuse_input_box_names() {
        let mut backend = CraneliftBackend::new();
        let start_descr = make_label_descr(76_100);
        let loop_descr = make_label_descr(76_101);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                start_descr,
            ),
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
                loop_descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(76_102);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(41), Value::Int(1)]);
        assert_eq!(backend.get_int_value(&frame, 0), 41);
    }

    #[test]
    fn test_guard_nonnull_class_checks_object_header_and_null() {
        let mut backend = CraneliftBackend::new();
        backend.set_vtable_offset(Some(0));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::GuardNonnullClass,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0xCAFE_BABEu64 as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(76_002);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut object_words = vec![0xCAFE_BABEu64 as i64, 123];
        let ptr = object_words.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(ptr));

        object_words[0] = 0xDEAD_BEEFu64 as i64;
        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(0))]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
    }

    // ── Overflow detection tests ──

    #[test]
    fn test_int_add_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntAddOvf,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(80);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // 10 + 20 = 30 (no overflow)
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(20)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish()); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 30);
    }

    #[test]
    fn test_int_add_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntAddOvf,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(81);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MAX + 1 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_int_sub_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntSubOvf,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(82);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MIN - 1 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MIN), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_int_sub_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntSubOvf,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(83);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(58)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish()); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntMulOvf,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(84);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish()); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntMulOvf,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(85);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MAX * 2 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(2)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_guard_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntAddOvf,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
            ),
            mk_op(OpCode::GuardOverflow, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(86);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // With overflow: guard_overflow passes (continues)
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish()); // Finish (overflow happened, guard passed)

        // Without overflow: guard_overflow fails (side-exits)
        let mut token2 = JitCellToken::new(87);
        backend.compile_loop(&inputargs, &ops, &mut token2).unwrap();
        let frame2 = backend.execute_token(&token2, &[Value::Int(1), Value::Int(2)]);
        let descr2 = backend.get_latest_descr(&frame2);
        assert_eq!(descr2.fail_index(), 0); // guard failure (no overflow)
    }

    // ── Getfield float test ──

    #[test]
    fn test_getfield_gc_f() {
        let mut backend = CraneliftBackend::new();

        // f64 field at offset 0
        let fd = make_field_descr(0, 8, Type::Float, false);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(OpCode::GetfieldGcF, &[OpRef::from_raw(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(90);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let val: f64 = 3.14;
        let mut data = vec![0u8; 8];
        data.copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        let result = backend.get_float_value(&frame, 0);
        assert!((result - 3.14).abs() < 1e-10);
    }

    // ── Getfield ref (pure) test ──

    #[test]
    fn test_getfield_gc_pure_r() {
        let mut backend = CraneliftBackend::new();

        // Ref field at offset 8
        let fd = make_field_descr(8, 8, Type::Ref, false);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(OpCode::GetfieldGcPureR, &[OpRef::from_raw(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(91);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0x42424242];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0x42424242usize));
    }

    // ── Setfield + getfield roundtrip ──

    #[test]
    fn test_setfield_getfield_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let fd = make_field_descr(0, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                fd.clone(),
            ),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef::from_raw(0)], 2, fd),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(92);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(12345)]);
        assert_eq!(backend.get_int_value(&frame, 0), 12345);
    }

    // ── Array getitem/setitem roundtrip ──

    #[test]
    fn test_setarrayitem_getarrayitem_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(0, 8, Type::Int, None);

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
                ad.clone(),
            ),
            mk_op_with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                3,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(93);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0, 0, 0];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(ptr)), Value::Int(2), Value::Int(777)],
        );
        assert_eq!(backend.get_int_value(&frame, 0), 777);
    }

    #[test]
    fn test_gc_load_i_signed_itemsize() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::GcLoadI,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(100),
                    OpRef::from_raw(101),
                ],
                1,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 4);
        constants.insert(101, -4);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(94);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 8];
        let val: i32 = -7;
        data[4..8].copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), -7);
    }

    #[test]
    fn test_gc_load_indexed_r_uses_scale_and_base_offset() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcLoadIndexedR,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(100),
                    OpRef::from_raw(101),
                    OpRef::from_raw(102),
                ],
                2,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 8);
        constants.insert(101, 16);
        constants.insert(102, 8);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(95);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<u64> = vec![0xAAAA, 0xBBBB, 0x1111_1111, 0xDEAD_BEEF];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(1)]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0xDEAD_BEEFusize));
    }

    #[test]
    fn test_gc_store_four_arg_form_stores_ref_value() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(100),
                    OpRef::from_raw(1),
                    OpRef::from_raw(101),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 8);
        constants.insert(101, 8);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(96);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<u64> = vec![0, 0];
        let ptr = data.as_mut_ptr() as usize;

        let stored = GcRef(0xFEED_FACEusize);
        backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Ref(stored)]);
        assert_eq!(data[1] as usize, stored.0);
    }

    #[test]
    fn test_gc_store_indexed_roundtrip_i32() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStoreIndexed,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(2),
                    OpRef::from_raw(100),
                    OpRef::from_raw(101),
                    OpRef::from_raw(102),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcLoadIndexedI,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(100),
                    OpRef::from_raw(101),
                    OpRef::from_raw(102),
                ],
                3,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 4);
        constants.insert(101, 8);
        constants.insert(102, 4);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(97);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 24];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[
                Value::Ref(GcRef(ptr)),
                Value::Int(2),
                Value::Int(0x1234_5678),
            ],
        );
        assert_eq!(backend.get_int_value(&frame, 0), 0x1234_5678);
    }

    #[test]
    fn test_raw_load_i_unsigned_zero_extends() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr_with_signedness(0, 1, Type::Int, false, None);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::RawLoadI,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(98);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0xFFu8];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 255);
    }

    #[test]
    fn test_raw_store_and_load_f_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(0, 8, Type::Float, None);
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::RawStore,
                &[OpRef::from_raw(0), OpRef::from_raw(100), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                ad.clone(),
            ),
            mk_op_with_descr(
                OpCode::RawLoadF,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                2,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 2);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(99);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 8];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Float(6.25)]);
        assert!((backend.get_float_value(&frame, 0) - 6.25).abs() < 1e-10);
    }

    #[test]
    fn test_gc_load_indexed_rejects_nonconstant_scale() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcLoadIndexedI,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(2),
                    OpRef::from_raw(100),
                    OpRef::from_raw(101),
                ],
                3,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 2);
        constants.insert(101, 8);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(100);
        let err = backend
            .compile_loop(&inputargs, &ops, &mut token)
            .unwrap_err();
        match err {
            BackendError::Unsupported(msg) => {
                assert!(msg.contains("scale must be a compile-time constant"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn test_getinteriorfield_gc_i_signed() {
        let mut backend = CraneliftBackend::new();

        let id = make_interior_field_descr(8, 16, Type::Int, 4, 4, Type::Int, true);
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::GetinteriorfieldGcI,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                2,
                id,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(101);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 40];
        let val: i32 = -11;
        let field_offset = 8 + 16 + 4;
        data[field_offset..field_offset + 4].copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(1)]);
        assert_eq!(backend.get_int_value(&frame, 0), -11);
    }

    #[test]
    fn test_setinteriorfield_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let id = make_interior_field_descr(8, 16, Type::Int, 8, 8, Type::Int, true);
        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::SetinteriorfieldGc,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
                id.clone(),
            ),
            mk_op_with_descr(
                OpCode::GetinteriorfieldGcI,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                3,
                id,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(102);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 40];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[
                Value::Ref(GcRef(ptr)),
                Value::Int(1),
                Value::Int(0x1234_5678_9ABC_DEF0),
            ],
        );
        assert_eq!(
            backend.get_int_value(&frame, 0),
            0x1234_5678_9ABC_DEF0u64 as i64
        );
    }

    // ── Bridge compilation and execution tests ──

    #[test]
    fn test_bridge_attaches_to_guard() {
        // Compile a loop: input(x) -> guard_true(x > 0) -> finish(x * 2)
        // When guard fails (x <= 0), compile a bridge that returns x + 100.
        // After attaching the bridge, executing with x <= 0 should run the
        // bridge and return x + 100 instead of falling back.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(OpCode::GuardTrue, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntMul,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
                2,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0i64); // guard: x > 0
        constants.insert(101, 2i64); // x * 2
        backend.set_constants(constants);

        let mut token = JitCellToken::new(200);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // First: verify guard fails when x = -5
        let frame = backend.execute_token(&token, &[Value::Int(-5)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard_true is guard index 0
        assert_eq!(backend.get_int_value(&frame, 0), -5);

        // Now compile a bridge for guard 0: bridge takes x, returns x + 100
        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(100, 100i64);
        backend.set_constants(bridge_constants);

        // compile.py:569 / compile.py:593: bridge attaches to the actual
        // FailDescr the running machine code holds, not a synthetic with
        // trace_id=0.  Pull the real descr off the compiled loop.
        let real_fail_descr = std::sync::Arc::clone(
            &token
                .compiled
                .as_ref()
                .unwrap()
                .downcast_ref::<CompiledLoop>()
                .unwrap()
                .fail_descrs[0],
        );
        let bridge_info = backend
            .compile_bridge(
                &*real_fail_descr,
                &bridge_inputargs,
                &bridge_ops,
                &token,
                &[],
            )
            .unwrap();
        assert!(bridge_info.code_addr != 0);

        // Now execute again with x = -5: the bridge should execute
        let frame = backend.execute_token(&token, &[Value::Int(-5)]);
        // Bridge returns -5 + 100 = 95
        assert_eq!(backend.get_int_value(&frame, 0), 95);

        // Execute with x = 0 (also fails guard): bridge returns 0 + 100 = 100
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 100);

        // Execute with x = 5 (guard passes): original path returns 5 * 2 = 10
        let frame = backend.execute_token(&token, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 10);
    }

    #[test]
    fn test_guard_failure_counting() {
        // Verify that guard failures are counted.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::GuardTrue, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(201);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Execute with x = 0 (guard fails) multiple times
        for i in 1..=5 {
            backend.execute_token(&token, &[Value::Int(0)]);
            let compiled = token
                .compiled
                .as_ref()
                .unwrap()
                .downcast_ref::<CompiledLoop>()
                .unwrap();
            let descr = &compiled.fail_descrs[0];
            assert_eq!(descr.get_fail_count(), i);
        }

        // Execute with x = 1 (guard passes, reaches finish)
        backend.execute_token(&token, &[Value::Int(1)]);
        let compiled = token
            .compiled
            .as_ref()
            .unwrap()
            .downcast_ref::<CompiledLoop>()
            .unwrap();
        // Guard descr count unchanged (guard didn't fail this time)
        let guard_descr = &compiled.fail_descrs[0];
        assert_eq!(guard_descr.get_fail_count(), 5);
    }

    #[test]
    fn test_bridge_with_explicit_fail_args() {
        // Test that when a guard has explicit fail_args, the bridge receives
        // those values as inputs.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        // Explicit fail_args: save both i0 and i1 plus the computed sum (i0+i1)
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(0),
            OpRef::from_raw(1),
        ]));

        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            guard_op,
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(203);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Without bridge: guard fails on x=0, y=42
        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 0); // x
        assert_eq!(backend.get_int_value(&frame, 1), 42); // y

        // Compile bridge that takes the fail_args (x, y) and returns x + y + 1000
        let bridge_inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let bridge_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(2), OpRef::from_raw(100)],
                3,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(100, 1000i64);
        backend.set_constants(bridge_constants);

        // compile.py:569: bridge attaches to the actual FailDescr.
        let real_fail_descr = std::sync::Arc::clone(
            &token
                .compiled
                .as_ref()
                .unwrap()
                .downcast_ref::<CompiledLoop>()
                .unwrap()
                .fail_descrs[0],
        );
        backend
            .compile_bridge(
                &*real_fail_descr,
                &bridge_inputargs,
                &bridge_ops,
                &token,
                &[],
            )
            .unwrap();

        // With bridge: guard fails on x=0, y=42 -> bridge returns 0 + 42 + 1000 = 1042
        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1042);

        // Guard passes on x=5, y=7 -> original path returns 5 + 7 = 12
        let frame = backend.execute_token(&token, &[Value::Int(5), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 12);
    }

    #[test]
    fn test_guard_fail_args_preserve_float_and_ref_types() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_float(1),
            InputArg::new_ref(2),
        ];
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(1),
            OpRef::from_raw(2),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(
            &token,
            &[Value::Int(0), Value::Float(9.25), Value::Ref(GcRef(0xBEEF))],
        );
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_float_value(&frame, 0), 9.25);
        assert_eq!(backend.get_ref_value(&frame, 1), GcRef(0xBEEF));
    }

    #[test]
    fn test_fail_descr_gc_map_tracks_ref_slots() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
            InputArg::new_ref(3),
        ];
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(0),
            OpRef::from_raw(1),
            OpRef::from_raw(2),
            OpRef::from_raw(3),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(2),
                    OpRef::from_raw(3),
                ],
                OpRef::NONE.raw(),
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1005);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let compiled = token
            .compiled
            .as_ref()
            .unwrap()
            .downcast_ref::<CompiledLoop>()
            .unwrap();
        let descr = &compiled.fail_descrs[0];
        assert!(!descr.gc_map().is_ref(0));
        assert!(descr.gc_map().is_ref(1));
        assert!(!descr.gc_map().is_ref(2));
        assert!(descr.gc_map().is_ref(3));
    }

    #[test]
    fn test_zero_array_clears_requested_range() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(16, 4, Type::Int, None);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::ZeroArray,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(100),
                    OpRef::from_raw(101),
                    OpRef::from_raw(102),
                    OpRef::from_raw(103),
                ],
                OpRef::NONE.raw(),
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 4);
        constants.insert(101, 8);
        constants.insert(102, 1);
        constants.insert(103, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1002);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<u8> = vec![
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
            0xAA, 0xAA, // header
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        ];
        let ptr = data.as_mut_ptr() as usize;

        backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(&data[16..20], &[1, 2, 3, 4]);
        assert_eq!(&data[20..28], &[0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_zero_array_uses_scaled_start_and_size() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(8, 4, Type::Int, None);
        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(
                OpCode::ZeroArray,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(2),
                    OpRef::from_raw(100),
                    OpRef::from_raw(101),
                ],
                OpRef::NONE.raw(),
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 4);
        constants.insert(101, 4);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1003);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![
            0xEEu8, 0xEE, 0xEE, 0xEE, 0xEE, 0xEE, 0xEE, 0xEE, // header
            1, 1, 1, 1, // item0
            2, 2, 2, 2, // item1
            3, 3, 3, 3, // item2
            4, 4, 4, 4, // item3
        ];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(ptr)), Value::Int(1), Value::Int(2)],
        );
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(&data[8..12], &[1, 1, 1, 1]);
        assert_eq!(&data[12..20], &[0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&data[20..24], &[4, 4, 4, 4]);
    }

    #[test]
    fn test_unsupported_void_opcode_errors() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::CallMayForceN, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1004);
        let err = backend
            .compile_loop(&inputargs, &ops, &mut token)
            .unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("CallMayForceN")),
            other => panic!("expected unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn test_call_may_force_n_guard_not_forced_and_savedata() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        may_force_void_values()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clear();

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Void);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(1),
            OpRef::from_raw(0),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceN,
                &[OpRef::from_raw(100), OpRef::from_raw(2), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_void as *const () as usize as i64,
        );
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_400);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(0)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 20);
        assert!(
            may_force_void_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .is_empty()
        );

        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 10);
        assert_eq!(
            *may_force_void_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner()),
            vec![0, 1, 10]
        );
        assert_eq!(backend.get_savedata_ref(&frame).unwrap(), GcRef(0xDADA));
    }

    #[test]
    fn test_execute_token_ints_raw_preserves_savedata_and_layout_for_call_may_force() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        may_force_void_values()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clear();

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Void);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(1),
            OpRef::from_raw(0),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceN,
                &[OpRef::from_raw(100), OpRef::from_raw(2), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_void as *const () as usize as i64,
        );
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_404);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let raw = backend.execute_token_ints_raw(&token, &[10, 1]);
        assert!(!raw.is_finish);
        assert_eq!(raw.fail_index, 0);
        assert_eq!(raw.outputs, vec![1, 10]);
        assert_eq!(raw.typed_outputs, vec![Value::Int(1), Value::Int(10)]);
        assert_eq!(raw.savedata, Some(GcRef(0xDADA)));
        assert_eq!(raw.exception_value, GcRef::NULL);
        let layout = raw
            .exit_layout
            .expect("raw exit should expose backend layout");
        assert_eq!(layout.fail_index, 0);
        assert_eq!(layout.source_op_index, Some(3));
        assert!(layout.recovery_layout.is_some());
        assert_eq!(
            *may_force_void_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner()),
            vec![0, 1, 10]
        );
    }

    #[test]
    fn test_call_may_force_with_intervening_ops() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        // RPython parity: `_genop_call_may_force` takes the paired guard
        // via `_find_nearby_operation(+1)` (x86/assembler.py:2225-2244)
        // — the optimizer is required to keep CALL_MAY_FORCE and
        // GUARD_NOT_FORCED adjacent. The cranelift backend mirrors the
        // assertion; any intervening op (here, a SameAsI between the call
        // and its guard) must be rejected at compile time.
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Int);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(1),
            OpRef::from_raw(3),
            OpRef::from_raw(0),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceI,
                &[OpRef::from_raw(100), OpRef::from_raw(2), OpRef::from_raw(1)],
                3,
                descr,
            ),
            mk_op(OpCode::SameAsI, &[OpRef::from_raw(3)], 4),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(4)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_int_isolated as *const () as usize as i64,
        );
        let mut backend = CraneliftBackend::new();
        backend.set_constants(constants);
        let mut token = JitCellToken::new(1500_410);
        let err = backend
            .compile_loop(&inputargs, &ops, &mut token)
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("expected guard_not_forced(_2) at +1") && msg.contains("SameAsI"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_call_may_force_i_guard_not_forced_uses_real_call_result() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        may_force_int_values()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clear();

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Int);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(1),
            OpRef::from_raw(3),
            OpRef::from_raw(0),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceI,
                &[OpRef::from_raw(100), OpRef::from_raw(2), OpRef::from_raw(1)],
                3,
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_force_and_return_int as *const () as usize as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_401);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(0)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 42);
        assert!(
            may_force_int_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .is_empty()
        );

        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 42);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(
            *may_force_int_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner()),
            vec![1, 10]
        );
        assert_eq!(backend.get_savedata_ref(&frame).unwrap(), GcRef(0xBABA));
    }

    #[test]
    fn test_call_may_force_f_guard_not_forced_uses_real_call_result() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        may_force_float_values()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clear();

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Float);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(1),
            OpRef::from_raw(3),
            OpRef::from_raw(0),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceF,
                &[OpRef::from_raw(100), OpRef::from_raw(2), OpRef::from_raw(1)],
                3,
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_float as *const () as usize as i64,
        );
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_402);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(0)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_float_value(&frame, 0), 12.5);
        assert!(
            may_force_float_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .is_empty()
        );

        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_float_value(&frame, 1), 12.5);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(
            *may_force_float_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner()),
            vec![1, 0.0f64.to_bits(), 10]
        );
    }

    #[test]
    #[ignore = "CALL_ASSEMBLER tests need GC runtime + jitframe layout setup (rewrite.py:613-695 handle_call_assembler / gen_malloc_frame parity); test harness still uses pre-rewrite shape from before validate_call_assembler_rewrite_prereqs landed"]
    fn test_call_assembler_i_executes_finish_only_target() {
        let mut backend = CraneliftBackend::new();

        let callee_inputargs = vec![InputArg::new_int(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut callee_constants = HashMap::new();
        callee_constants.insert(100, 2);
        backend.set_constants(callee_constants);

        let mut callee_token = JitCellToken::new(1500_200);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();

        let caller_inputargs = vec![InputArg::new_int(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef::from_raw(0)],
                1,
                make_call_assembler_descr(&callee_token, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        backend.set_constants(HashMap::new());

        let mut caller_token = JitCellToken::new(1500_201);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Int(40)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_call_assembler_fast_path_pending_target_uses_force_fallback() {
        pending_call_assembler_force_values()
            .lock()
            .unwrap()
            .clear();

        let token_number = 1500_199;
        register_pending_call_assembler_target(
            token_number,
            vec![Type::Int],
            2,
            1,
            -1,
            Arc::new(std::sync::RwLock::new(CpuDescrAttachments::default())),
        );

        let target_ptr = unsafe { fast_lookup_ca_target(token_number) };
        assert!(!target_ptr.is_null(), "pending target should be registered");
        let target = unsafe { &*target_ptr };
        let mut outcome = [0i64; 2];
        let result = call_assembler_fast_path(
            target,
            &[0x1234, 77],
            outcome.as_mut_ptr(),
            pending_call_assembler_force_probe,
        );

        assert_eq!(result as i64, 0x1234 + 77);
        assert_eq!(outcome, [CALL_ASSEMBLER_OUTCOME_FINISH, 0]);
        assert_eq!(
            *pending_call_assembler_force_values().lock().unwrap(),
            vec![(0x1234, Some(77))]
        );
        assert_eq!(take_pending_force_local0(), None);
    }

    #[test]
    #[ignore = "CALL_ASSEMBLER tests need GC runtime + jitframe layout setup (rewrite.py:613-695 handle_call_assembler / gen_malloc_frame parity); test harness still uses pre-rewrite shape from before validate_call_assembler_rewrite_prereqs landed"]
    fn test_call_assembler_compiles_before_target_is_registered() {
        let mut backend = CraneliftBackend::new();

        let mut deferred_target = JitCellToken::new(1500_240);
        backend.set_next_trace_id(1500_241);
        let caller_inputargs = vec![InputArg::new_int(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef::from_raw(0)],
                1,
                make_call_assembler_descr(&deferred_target, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut caller = JitCellToken::new(1500_241);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        backend.set_next_trace_id(1500_240);
        let callee_inputargs = vec![InputArg::new_int(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 7);
        backend.set_constants(constants);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut deferred_target)
            .unwrap();

        backend.set_constants(HashMap::new());
        let frame = backend.execute_token(&caller, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 12);
    }

    #[test]
    #[ignore = "CALL_ASSEMBLER tests need GC runtime + jitframe layout setup (rewrite.py:613-695 handle_call_assembler / gen_malloc_frame parity); test harness still uses pre-rewrite shape from before validate_call_assembler_rewrite_prereqs landed"]
    fn test_call_assembler_late_bound_ref_result_supports_plain_ref_finish() {
        let mut backend = CraneliftBackend::new();

        let mut deferred_target = JitCellToken::new(1500_245);
        let caller_inputargs = vec![InputArg::new_ref(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallAssemblerR,
                &[OpRef::from_raw(0)],
                1,
                make_call_assembler_descr(&deferred_target, vec![Type::Ref], Type::Ref),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut caller = JitCellToken::new(1500_246);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        backend.set_next_trace_id(1500_245);
        let callee_inputargs = vec![InputArg::new_ref(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];
        backend.set_constants(HashMap::new());
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut deferred_target)
            .unwrap();

        let root = GcRef(0xCAFE);
        let frame = backend.execute_token(&caller, &[Value::Ref(root)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert!(descr.force_token_slots().is_empty());
        assert_eq!(backend.get_ref_value(&frame, 0), root);
    }

    #[test]
    #[ignore = "CALL_ASSEMBLER tests need GC runtime + jitframe layout setup (rewrite.py:613-695 handle_call_assembler / gen_malloc_frame parity); test harness still uses pre-rewrite shape from before validate_call_assembler_rewrite_prereqs landed"]
    fn test_call_assembler_late_bound_ref_result_rejects_force_token_finish_shape() {
        let mut backend = CraneliftBackend::new();

        let mut deferred_target = JitCellToken::new(1500_247);
        let caller_inputargs = vec![InputArg::new_int(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallAssemblerR,
                &[OpRef::from_raw(0)],
                1,
                make_call_assembler_descr(&deferred_target, vec![Type::Int], Type::Ref),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut caller = JitCellToken::new(1500_248);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        backend.set_next_trace_id(1500_247);
        let callee_inputargs = vec![InputArg::new_int(0)];
        let mut guard_op = mk_op(OpCode::GuardNotForced2, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(1)]));
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(OpCode::ForceToken, &[], 2),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 10);
        backend.set_constants(constants);
        let err = backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut deferred_target)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("incompatible callee finish result kind"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[ignore = "CALL_ASSEMBLER tests need GC runtime + jitframe layout setup (rewrite.py:613-695 handle_call_assembler / gen_malloc_frame parity); test harness still uses pre-rewrite shape from before validate_call_assembler_rewrite_prereqs landed"]
    fn test_call_assembler_redirect_rejects_incompatible_force_token_result_shape() {
        let mut backend = CraneliftBackend::new();

        let ref_inputargs = vec![InputArg::new_ref(0)];
        let plain_ref_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];
        let mut plain_ref_target = JitCellToken::new(1500_349);
        backend
            .compile_loop(&ref_inputargs, &plain_ref_ops, &mut plain_ref_target)
            .unwrap();

        let mut guard_op = mk_op(OpCode::GuardNotForced2, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(1)]));
        let force_token_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::ForceToken, &[], 1),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut force_token_target = JitCellToken::new(1500_350);
        backend
            .compile_loop(&ref_inputargs, &force_token_ops, &mut force_token_target)
            .unwrap();

        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallAssemblerR,
                &[OpRef::from_raw(0)],
                1,
                make_call_assembler_descr(&plain_ref_target, vec![Type::Ref], Type::Ref),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut caller = JitCellToken::new(1500_351);
        backend
            .compile_loop(&ref_inputargs, &caller_ops, &mut caller)
            .unwrap();

        let err = backend
            .redirect_call_assembler(&plain_ref_target, &force_token_target)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("incompatible callee finish result kind"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[ignore = "CALL_ASSEMBLER tests need GC runtime + jitframe layout setup (rewrite.py:613-695 handle_call_assembler / gen_malloc_frame parity); test harness still uses pre-rewrite shape from before validate_call_assembler_rewrite_prereqs landed"]
    fn test_call_assembler_supports_direct_self_recursive_dispatch() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_250);
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
                1,
            ),
            mk_op(OpCode::GuardTrue, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntSub,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                2,
            ),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef::from_raw(2)],
                3,
                make_call_assembler_descr(&token, vec![Type::Int], Type::Int),
            ),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(3), OpRef::from_raw(100)],
                4,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(4)], OpRef::NONE.raw()),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let compiled = token
            .compiled
            .as_ref()
            .unwrap()
            .downcast_ref::<CompiledLoop>()
            .unwrap();
        let guard_descr = &compiled.fail_descrs[0];

        let failed = backend.execute_token(&token, &[Value::Int(0)]);
        let failed_descr = get_latest_descr_from_deadframe(&failed)
            .expect("base-case guard should produce a valid descr");
        assert_eq!(failed_descr.fail_index(), guard_descr.fail_index());
        assert_eq!(guard_descr.get_fail_count(), 1);

        backend.set_constants(HashMap::new());
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];
        backend
            .compile_bridge(failed_descr, &inputargs, &bridge_ops, &token, &[])
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(4)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 4);
        assert_eq!(
            guard_descr.get_fail_count(),
            1,
            "CALL_ASSEMBLER guard with attached bridge should dispatch directly without recounting the original faildescr",
        );

        let raw = backend.execute_token_ints_raw(&token, &[4]);
        assert!(raw.is_finish);
        assert_eq!(raw.outputs, vec![4]);
        assert_eq!(raw.typed_outputs, vec![Value::Int(4)]);
    }

    #[test]
    #[ignore = "CALL_ASSEMBLER tests need GC runtime + jitframe layout setup (rewrite.py:613-695 handle_call_assembler / gen_malloc_frame parity); test harness still uses pre-rewrite shape from before validate_call_assembler_rewrite_prereqs landed"]
    fn test_call_assembler_reused_token_resets_stale_pending_dispatch_slot() {
        let token_number = 1500_252;

        {
            let mut backend = CraneliftBackend::new();
            let inputargs = vec![InputArg::new_int(0)];
            let mut constants = HashMap::new();
            constants.insert(100, 1);
            constants.insert(101, 0);
            backend.set_constants(constants);

            let mut token = JitCellToken::new(token_number);
            let ops = vec![
                mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
                mk_op(
                    OpCode::IntGt,
                    &[OpRef::from_raw(0), OpRef::from_raw(101)],
                    1,
                ),
                mk_op(OpCode::GuardTrue, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
                mk_op(
                    OpCode::IntSub,
                    &[OpRef::from_raw(0), OpRef::from_raw(100)],
                    2,
                ),
                mk_op_with_descr(
                    OpCode::CallAssemblerI,
                    &[OpRef::from_raw(2)],
                    3,
                    make_call_assembler_descr(&token, vec![Type::Int], Type::Int),
                ),
                mk_op(
                    OpCode::IntAdd,
                    &[OpRef::from_raw(3), OpRef::from_raw(100)],
                    4,
                ),
                mk_op(OpCode::Finish, &[OpRef::from_raw(4)], OpRef::NONE.raw()),
            ];
            backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

            let failed = backend.execute_token(&token, &[Value::Int(0)]);
            let guard_descr = get_latest_descr_from_deadframe(&failed)
                .expect("base-case guard should produce a valid descr");
            backend.set_constants(HashMap::new());
            let bridge_ops = vec![
                mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
                mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            ];
            backend
                .compile_bridge(guard_descr, &inputargs, &bridge_ops, &token, &[])
                .unwrap();

            let frame = backend.execute_token(&token, &[Value::Int(4)]);
            assert_eq!(backend.get_int_value(&frame, 0), 4);
        }

        let mut backend = CraneliftBackend::new();
        let mut deferred_target = JitCellToken::new(token_number);
        let caller_inputargs = vec![InputArg::new_int(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef::from_raw(0)],
                1,
                make_call_assembler_descr(&deferred_target, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut caller = JitCellToken::new(1500_253);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        let callee_inputargs = vec![InputArg::new_int(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 7);
        backend.set_constants(constants);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut deferred_target)
            .unwrap();

        backend.set_constants(HashMap::new());
        let frame = backend.execute_token(&caller, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 12);
    }

    #[test]
    fn test_call_assembler_requires_gc_runtime_for_rewrite_parity() {
        let mut backend = CraneliftBackend::new();
        let mut callee_token = JitCellToken::new(1500_352);
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef::from_raw(0)],
                1,
                make_call_assembler_descr(&callee_token, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let err = backend
            .compile_loop(&inputargs, &ops, &mut callee_token)
            .unwrap_err();
        assert!(
            err.to_string().contains("requires a configured GC runtime"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_execute_bridge_follows_external_jump_to_compiled_target() {
        let mut backend = CraneliftBackend::new();
        let loop_descr = make_label_descr(1500_260);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(1)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let root_ops = vec![
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                loop_descr.clone(),
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut root_constants = HashMap::new();
        root_constants.insert(100, 0);
        backend.set_constants(root_constants);

        let mut token = JitCellToken::new(1500_261);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let failed = backend.execute_token(&token, &[Value::Int(0)]);
        let guard_descr =
            get_latest_descr_from_deadframe(&failed).expect("guard should produce a descr");

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(101, 5);
        backend.set_constants(bridge_constants);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
                1,
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                loop_descr,
            ),
        ];
        backend
            .compile_bridge(guard_descr, &inputargs, &bridge_ops, &token, &[])
            .unwrap();

        backend.set_constants(HashMap::new());
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 5);

        let raw = backend.execute_token_ints_raw(&token, &[0]);
        assert!(raw.is_finish);
        assert_eq!(raw.outputs, vec![5]);
        assert_eq!(raw.typed_outputs, vec![Value::Int(5)]);
    }

    #[test]
    fn test_bridge_executes_ops_before_first_label_before_external_jump() {
        let mut backend = CraneliftBackend::new();
        let loop_descr = make_label_descr(1500_259);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(1)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let root_ops = vec![
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                loop_descr.clone(),
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut root_constants = HashMap::new();
        root_constants.insert(100, 0);
        backend.set_constants(root_constants);

        let mut token = JitCellToken::new(1500_258);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let failed = backend.execute_token(&token, &[Value::Int(0)]);
        let guard_descr =
            get_latest_descr_from_deadframe(&failed).expect("guard should produce a descr");

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(101, 5);
        bridge_constants.insert(102, 1);
        backend.set_constants(bridge_constants);
        let bridge_ops = vec![
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
                2,
            ),
            mk_op(OpCode::Label, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(2), OpRef::from_raw(102)],
                3,
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(3)],
                OpRef::NONE.raw(),
                loop_descr,
            ),
        ];
        backend
            .compile_bridge(guard_descr, &inputargs, &bridge_ops, &token, &[])
            .unwrap();

        backend.set_constants(HashMap::new());
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 6);
    }

    #[test]
    fn test_execute_bridge_updates_later_ref_input_slots_for_external_jump() {
        let mut backend = CraneliftBackend::new();
        let loop_descr = make_label_descr(1500_262);
        let int_size = make_size_descr(16, 1500_263);
        let int_field = make_field_descr(0, 8, Type::Int, true);

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_ref(2),
        ];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(4)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(0),
            OpRef::from_raw(1),
            OpRef::from_raw(2),
        ]));
        let root_ops = vec![
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
                loop_descr.clone(),
            ),
            mk_op_with_descr(
                OpCode::GetfieldGcPureI,
                &[OpRef::from_raw(2)],
                3,
                int_field.clone(),
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(3), OpRef::from_raw(100)],
                4,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut root_constants = HashMap::new();
        root_constants.insert(100, 0);
        backend.set_constants(root_constants);

        let mut token = JitCellToken::new(1500_264);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let mut old_box = Vec::with_capacity(1);
        old_box.push(0i64);
        let old_ref = GcRef(old_box.as_mut_ptr() as usize);

        let failed = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(0)), Value::Int(77), Value::Ref(old_ref)],
        );
        let guard_descr =
            get_latest_descr_from_deadframe(&failed).expect("guard should produce a descr");

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(101, 5);
        backend.set_constants(bridge_constants);
        let bridge_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            mk_op_with_descr(OpCode::NewWithVtable, &[], 3, int_size.clone()),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef::from_raw(3), OpRef::from_raw(101)],
                OpRef::NONE.raw(),
                int_field.clone(),
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(3)],
                OpRef::NONE.raw(),
                loop_descr,
            ),
        ];
        backend
            .compile_bridge(guard_descr, &inputargs, &bridge_ops, &token, &[])
            .unwrap();

        backend.set_constants(HashMap::new());
        let frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(0)), Value::Int(77), Value::Ref(old_ref)],
        );
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 5);
    }

    #[test]
    #[ignore = "label-selector dispatch (assembler.py:990-993 TargetToken._ll_loop_code per-label entry) is not yet ported — re-entry runs from the function code_ptr, not the target label's address. Cranelift cannot expose internal block addresses, so closing this requires either a per-label dispatch prologue or a switch-on-selector entry; multi-session structural change."]
    fn test_execute_bridge_external_jump_to_second_label_skips_preamble() {
        let mut backend = CraneliftBackend::new();
        let start_descr = make_label_descr(1500_265);
        let body_descr = make_label_descr(1500_266);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(1)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let root_ops = vec![
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                start_descr,
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            guard,
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
                2,
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(2)],
                OpRef::NONE.raw(),
                body_descr.clone(),
            ),
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(3)],
                OpRef::NONE.raw(),
                body_descr.clone(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(3)], OpRef::NONE.raw()),
        ];

        let mut root_constants = HashMap::new();
        root_constants.insert(100, 0);
        root_constants.insert(101, 10);
        backend.set_constants(root_constants);

        let mut token = JitCellToken::new(1500_267);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let failed = backend.execute_token(&token, &[Value::Int(0)]);
        let guard_descr =
            get_latest_descr_from_deadframe(&failed).expect("guard should produce a descr");

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(102, 5);
        backend.set_constants(bridge_constants);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(102)],
                1,
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                body_descr,
            ),
        ];
        backend
            .compile_bridge(guard_descr, &inputargs, &bridge_ops, &token, &[])
            .unwrap();

        backend.set_constants(HashMap::new());
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 5);
    }

    #[test]
    #[ignore = "label-selector dispatch (assembler.py:990-993 TargetToken._ll_loop_code per-label entry) is not yet ported — re-entry runs from the function code_ptr, not the target label's address. Cranelift cannot expose internal block addresses, so closing this requires either a per-label dispatch prologue or a switch-on-selector entry; multi-session structural change."]
    fn test_execute_bridge_external_jump_to_middle_label_uses_label_selector() {
        let mut backend = CraneliftBackend::new();
        let start_descr = make_label_descr(1500_268);
        let middle_descr = make_label_descr(1500_269);
        let final_descr = make_label_descr(1500_270);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(1)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));
        let root_ops = vec![
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                start_descr,
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            guard,
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
                2,
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(2)],
                OpRef::NONE.raw(),
                middle_descr.clone(),
            ),
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(3)],
                OpRef::NONE.raw(),
                middle_descr.clone(),
            ),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(3), OpRef::from_raw(102)],
                4,
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(4)],
                OpRef::NONE.raw(),
                final_descr.clone(),
            ),
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(5)],
                OpRef::NONE.raw(),
                final_descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(5)], OpRef::NONE.raw()),
        ];

        let mut root_constants = HashMap::new();
        root_constants.insert(100, 0);
        root_constants.insert(101, 10);
        root_constants.insert(102, 20);
        backend.set_constants(root_constants);

        let mut token = JitCellToken::new(1500_271);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let failed = backend.execute_token(&token, &[Value::Int(0)]);
        let guard_descr =
            get_latest_descr_from_deadframe(&failed).expect("guard should produce a descr");

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(103, 5);
        backend.set_constants(bridge_constants);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(103)],
                1,
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                middle_descr,
            ),
        ];
        backend
            .compile_bridge(guard_descr, &inputargs, &bridge_ops, &token, &[])
            .unwrap();

        backend.set_constants(HashMap::new());
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 25);
    }

    #[test]
    fn test_jump_to_mismatched_local_label_lowers_as_external_jump() {
        let mut backend = CraneliftBackend::new();
        let start_descr = majit_ir::make_loop_target_descr(1500_360, true);
        let body_descr = make_label_descr(1500_361);

        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
                start_descr.clone(),
            ),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                3,
            ),
            mk_op_with_descr(
                OpCode::Label,
                &[
                    OpRef::from_raw(3),
                    OpRef::from_raw(1),
                    OpRef::from_raw(2),
                    OpRef::from_raw(0),
                ],
                OpRef::NONE.raw(),
                body_descr,
            ),
            mk_op_with_descr(
                OpCode::Jump,
                &[
                    OpRef::from_raw(3),
                    OpRef::from_raw(1),
                    OpRef::from_raw(2),
                    OpRef::from_raw(0),
                ],
                OpRef::NONE.raw(),
                start_descr,
            ),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_362);
        backend
            .compile_loop(&inputargs, &ops, &mut token)
            .expect("mismatched local-label jump should compile via external jump lowering");
    }

    #[test]
    fn test_gc_runtime_opcodes_error() {
        let cases = vec![
            (
                OpCode::CallMallocNursery,
                vec![InputArg::new_int(0)],
                vec![OpRef::from_raw(0)],
                1,
                "configured GC runtime",
            ),
            (
                OpCode::CallMallocNurseryVarsize,
                vec![
                    InputArg::new_int(0),
                    InputArg::new_int(1),
                    InputArg::new_int(2),
                ],
                vec![OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                3,
                "configured GC runtime",
            ),
            (
                OpCode::CallMallocNurseryVarsizeFrame,
                vec![InputArg::new_int(0)],
                vec![OpRef::from_raw(0)],
                1,
                "configured GC runtime",
            ),
            (
                OpCode::CondCallGcWb,
                vec![InputArg::new_ref(0)],
                vec![OpRef::from_raw(0)],
                OpRef::NONE.raw(),
                "configured GC runtime",
            ),
            (
                OpCode::CondCallGcWbArray,
                vec![InputArg::new_ref(0), InputArg::new_int(1)],
                vec![OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                "configured GC runtime",
            ),
        ];

        for (idx, (opcode, inputargs, op_args, pos, detail)) in cases.into_iter().enumerate() {
            let label_args: Vec<OpRef> = (0..inputargs.len() as u32).map(OpRef::from_raw).collect();
            let ops = vec![
                mk_op(OpCode::Label, &label_args, OpRef::NONE.raw()),
                mk_op(opcode, &op_args, pos),
                mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            ];
            assert_compile_unsupported(inputargs, ops, 1400 + idx as u64, opcode, detail);
        }
    }

    #[test]
    fn test_gc_alloc_and_init_with_configured_runtime() {
        let mut backend = make_gc_backend();
        // Constants: 10000=32(size), 10001=-8(tid_ofs), 10002=7(tid),
        // 10003=8(word), 10004=0(vtable_ofs), 10005=0xDEAD(vtable)
        let mut consts = HashMap::new();
        consts.insert(10000, 32_i64);
        consts.insert(10001, -8_i64);
        consts.insert(10002, 7_i64);
        consts.insert(10003, 8_i64);
        consts.insert(10004, 0_i64);
        consts.insert(10005, 0xDEAD_i64);
        backend.set_constants(consts);

        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::from_raw(10000)], 0),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(10001),
                    OpRef::from_raw(10002),
                    OpRef::from_raw(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(10004),
                    OpRef::from_raw(10005),
                    OpRef::from_raw(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1500);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { (*header_of(obj.0)).type_id() }, 7);
        assert_eq!(unsafe { *(obj.0 as *const u64) }, 0xDEAD);
    }

    #[test]
    fn test_gc_batched_allocation_layout_with_configured_runtime() {
        let mut backend = make_gc_backend();
        // Constants: 10000=56(total_size), 10001=-8(tid_ofs), 10002=1(tid1),
        // 10003=8(word), 10004=24(incr), 10005=2(tid2)
        let mut consts = HashMap::new();
        consts.insert(10000, 56_i64);
        consts.insert(10001, -8_i64);
        consts.insert(10002, 1_i64);
        consts.insert(10003, 8_i64);
        consts.insert(10004, 24_i64);
        consts.insert(10005, 2_i64);
        backend.set_constants(consts);

        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::from_raw(10000)], 0),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(10001),
                    OpRef::from_raw(10002),
                    OpRef::from_raw(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::NurseryPtrIncrement,
                &[OpRef::from_raw(0), OpRef::from_raw(10004)],
                1,
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::from_raw(1),
                    OpRef::from_raw(10001),
                    OpRef::from_raw(10005),
                    OpRef::from_raw(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::Finish,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(1501);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj0 = backend.get_ref_value(&frame, 0);
        let obj1 = backend.get_ref_value(&frame, 1);

        assert!(!obj0.is_null());
        assert!(!obj1.is_null());
        assert_eq!(obj1.0, obj0.0 + 24);
        assert_eq!(unsafe { (*header_of(obj0.0)).type_id() }, 1);
        assert_eq!(unsafe { (*header_of(obj1.0)).type_id() }, 2);
    }

    #[test]
    fn test_gc_varsize_alloc_and_length_init_with_configured_runtime() {
        let mut backend = make_gc_backend();
        // Constants: 10000=0(len_ofs), 10001=8(size), 10002=0(kind), 10003=8(itemsize)
        let mut consts = HashMap::new();
        consts.insert(10000, 0_i64);
        consts.insert(10001, 8_i64);
        consts.insert(10002, 0_i64); // FLAG_ARRAY
        consts.insert(10003, 8_i64); // itemsize
        backend.set_constants(consts);

        let ad = make_array_descr(16, 8, Type::Int, Some(0));
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            // rewrite.py:858: [ConstInt(kind), ConstInt(itemsize), v_length]
            mk_op_with_descr(
                OpCode::CallMallocNurseryVarsize,
                &[
                    OpRef::from_raw(10002),
                    OpRef::from_raw(10003),
                    OpRef::from_raw(0),
                ],
                1,
                ad.clone(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::from_raw(1),
                    OpRef::from_raw(10000),
                    OpRef::from_raw(0),
                    OpRef::from_raw(10001),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1502);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3)]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { *(obj.0 as *const i64) }, 3);
    }

    #[test]
    fn test_gc_call_malloc_array_helper_with_configured_runtime() {
        let mut backend = make_gc_backend();
        let mut consts = HashMap::new();
        consts.insert(10000, gc_malloc_array_helper as *const () as i64);
        consts.insert(10001, 8_i64);
        consts.insert(10002, 7_i64);
        backend.set_constants(consts);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallR,
                &[
                    OpRef::from_raw(10000),
                    OpRef::from_raw(10001),
                    OpRef::from_raw(10002),
                    OpRef::from_raw(0),
                ],
                1,
                majit_ir::make_malloc_array_calldescr(),
            ),
            mk_op(
                OpCode::CheckMemoryError,
                &[OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1503);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3)]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { (*header_of(obj.0)).type_id() }, 7);
        assert_eq!(unsafe { *(obj.0 as *const usize) }, 3);
    }

    #[test]
    fn test_cond_call_gc_wb_executes_with_configured_runtime() {
        let mut backend = make_gc_backend();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::CondCallGcWb,
                &[OpRef::from_raw(0)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1503);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut raw = vec![0u64; 2];
        unsafe {
            *(raw.as_mut_ptr() as *mut GcHeader) = GcHeader::with_flags(9, flags::TRACK_YOUNG_PTRS);
        }
        let obj = GcRef(raw.as_mut_ptr() as usize + GcHeader::SIZE);

        backend.execute_token(&token, &[Value::Ref(obj)]);
        assert!(!unsafe { (*header_of(obj.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
    }

    /// test_gc_integration.py:808 test_malloc_1:
    /// Three call_malloc_nursery(size) in the trace. Nursery sized so
    /// the jitframe + first two allocs fit but the third overflows,
    /// triggering a nursery collection.
    ///
    /// RPython: init_nursery(2 * sizeof.size), three call_malloc_nursery(size).
    /// In majit the jitframe itself lives in the nursery, so we size
    /// the nursery for jitframe + 2 user allocs (third overflows).
    #[test]
    fn test_gc_collecting_alloc_preserves_live_ref_inputs() {
        let payload_size: usize = 16;
        let alloc_size = majit_gc::header::GcHeader::SIZE + payload_size; // 24

        // test_gc_integration.py:820: init_nursery(2 * sizeof.size).
        // RPython nursery holds exactly 2 user objects (jitframe is
        // separate). In majit jitframe is nursery-allocated, so compute:
        //   header_words = JF_FRAME_ITEM0_OFS / 8 = 8
        //   depth = max(max_output=1, fail_args=3, inputs=0, 1) = 3
        //   ref_roots = 3 (one per CallMallocNursery result)
        //   jf_payload = (8 + 3 + 3) * 8 = 112
        //   jf_total = GcHeader::SIZE + 112 = 120
        //   nursery = 120 + 2*24 = 168
        let jf_header_words: usize = (JF_FRAME_ITEM0_OFS as usize) / 8;
        let jf_depth = 3_usize; // max(1 finish output, 3 fail_args, 0 inputs)
        let jf_ref_roots = 3_usize;
        let jf_payload = (jf_header_words + jf_depth + jf_ref_roots) * 8;
        let jf_nursery_alloc = majit_gc::header::GcHeader::SIZE + jf_payload;
        let nursery_size = jf_nursery_alloc + 2 * alloc_size;

        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(payload_size));

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        // test_gc_integration.py:808-817:
        //   []
        //   p0 = call_malloc_nursery(size)
        //   p1 = call_malloc_nursery(size)
        //   p2 = call_malloc_nursery(size)  # this overflows
        //   guard_nonnull(p2, descr=faildescr) [p0, p1, p2]
        //   finish(p2, descr=finaldescr)
        let mut consts = HashMap::new();
        consts.insert(10000, alloc_size as i64);
        backend.set_constants(consts);
        let size_arg = OpRef::from_raw(10000);
        let inputargs = vec![];
        let mut guard = mk_op(
            OpCode::GuardNonnull,
            &[OpRef::from_raw(2)],
            OpRef::NONE.raw(),
        );
        guard.fail_args = Some(smallvec::smallvec![
            OpRef::from_raw(0),
            OpRef::from_raw(1),
            OpRef::from_raw(2)
        ]);
        let ops = vec![
            mk_op(OpCode::Label, &[], OpRef::NONE.raw()),
            mk_op(OpCode::CallMallocNursery, &[size_arg], 0),
            mk_op(OpCode::CallMallocNursery, &[size_arg], 1),
            mk_op(OpCode::CallMallocNursery, &[size_arg], 2), // overflows
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1505);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Snapshot nursery free pointer before execution.
        let nf_before = with_cranelift_gc_required(|gc| gc.nursery_free() as usize);

        let frame = backend.execute_token(&token, &[]);

        // Verify collection occurred: after minor collection the nursery
        // resets, so nursery_free is near the start (only p2 allocated
        // post-collection). Without collection nursery_free would advance
        // by jf + 3*alloc = jf + 72 past nf_before.
        let nf_after = with_cranelift_gc_required(|gc| gc.nursery_free() as usize);
        assert!(
            nf_after < nf_before + 3 * alloc_size,
            "nursery should have been collected (free before={nf_before:#x}, \
             after={nf_after:#x}, expected advance without collection ≥ {})",
            3 * alloc_size
        );

        // test_gc_integration.py:826-828:
        // thing = frame.jf_frame[unpack_gcmap(frame)[0]]
        // assert thing == rffi.cast(lltype.Signed, cpu.gc_ll_descr.nursery)
        let p2 = backend.get_ref_value(&frame, 0);
        assert!(!p2.is_null(), "p2 should be non-null after collection");
    }

    #[test]
    fn test_gc_collecting_alloc_preserves_live_ref_results() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        let fd = make_field_descr(0, 8, Type::Int, true);
        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::Label, &[], OpRef::NONE.raw()),
            mk_op(OpCode::CallMallocNursery, &[OpRef::from_raw(16)], 0),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                OpRef::NONE.raw(),
                fd.clone(),
            ),
            mk_op(OpCode::CallMallocNursery, &[OpRef::from_raw(32)], 1),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef::from_raw(1), OpRef::from_raw(101)],
                OpRef::NONE.raw(),
                fd,
            ),
            mk_op(OpCode::CallMallocNursery, &[OpRef::from_raw(24)], 2),
            mk_op(
                OpCode::Finish,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(1505_1);
        let mut constants = HashMap::new();
        constants.insert(100, 111);
        constants.insert(101, 222);
        backend.set_constants(constants);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj0 = backend.get_ref_value(&frame, 0);
        let obj1 = backend.get_ref_value(&frame, 1);
        let obj2 = backend.get_ref_value(&frame, 2);

        assert!(!obj0.is_null());
        assert!(!obj1.is_null());
        assert!(!obj2.is_null());
        assert_eq!(unsafe { *(obj0.0 as *const i64) }, 111);
        assert_eq!(unsafe { *(obj1.0 as *const i64) }, 222);
    }

    #[test]
    fn test_setfield_gc_from_old_object_keeps_young_ref_alive_across_collection() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let old_root_tid = gc.register_type(TypeInfo::with_gc_ptrs(8, vec![0]));
        let young_node_tid = gc.register_type(TypeInfo::simple(16));

        let mut root = gc.alloc_with_type(old_root_tid, 8);
        unsafe {
            gc.add_root(&mut root as *mut GcRef);
        }
        gc.collect_nursery();
        gc.remove_root(&mut root as *mut GcRef);
        assert!(
            !gc.is_in_nursery(root.0),
            "root must be old before JIT runs"
        );

        let filler = gc.alloc_with_type(young_node_tid, 16);
        assert!(gc.is_in_nursery(filler.0));

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let ref_fd = make_field_descr(0, 8, Type::Ref, false);
        let int_fd = make_field_descr(0, 8, Type::Int, true);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::CallMallocNursery, &[OpRef::from_raw(16)], 1),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef::from_raw(1), OpRef::from_raw(100)],
                OpRef::NONE.raw(),
                int_fd,
            ),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
                ref_fd,
            ),
            mk_op(OpCode::CallMallocNursery, &[OpRef::from_raw(24)], 2),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 777);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1505_2);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let moved_root = backend.get_ref_value(&frame, 0);
        assert!(!moved_root.is_null());

        let child = GcRef(unsafe { *(moved_root.0 as *const usize) });
        assert!(
            !child.is_null(),
            "old root lost its young child across collection"
        );
        assert_eq!(unsafe { *(child.0 as *const i64) }, 777);
    }

    #[test]
    fn test_newstr_without_descr_compiles_via_builtin_layout() {
        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(Arc::new(
            Mutex::new(TrackingGcState::default()),
        ))));

        let ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(101),
                    OpRef::from_raw(200),
                ],
            ),
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(102),
                    OpRef::from_raw(201),
                ],
            ),
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(103),
                    OpRef::from_raw(202),
                ],
            ),
            Op::new(
                OpCode::Strgetitem,
                &[OpRef::from_raw(0), OpRef::from_raw(102)],
            ),
            Op::new(OpCode::Strlen, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Finish, &[OpRef::from_raw(4), OpRef::from_raw(5)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 3);
        constants.insert(101, 0);
        constants.insert(102, 1);
        constants.insert(103, 2);
        constants.insert(200, b'a' as i64);
        constants.insert(201, b'b' as i64);
        constants.insert(202, b'c' as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1512);
        backend.compile_loop(&[], &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert_eq!(backend.get_int_value(&frame, 0), b'b' as i64);
        assert_eq!(backend.get_int_value(&frame, 1), 3);
    }

    #[test]
    fn test_copystrcontent_without_descr_compiles_via_builtin_layout() {
        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(Arc::new(
            Mutex::new(TrackingGcState::default()),
        ))));

        let ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(101),
                    OpRef::from_raw(200),
                ],
            ),
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(102),
                    OpRef::from_raw(201),
                ],
            ),
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::Copystrcontent,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(3),
                    OpRef::from_raw(101),
                    OpRef::from_raw(101),
                    OpRef::from_raw(100),
                ],
            ),
            Op::new(
                OpCode::Strgetitem,
                &[OpRef::from_raw(3), OpRef::from_raw(102)],
            ),
            Op::new(OpCode::Finish, &[OpRef::from_raw(5)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 2);
        constants.insert(101, 0);
        constants.insert(102, 1);
        constants.insert(200, b'x' as i64);
        constants.insert(201, b'y' as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1513);
        backend.compile_loop(&[], &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert_eq!(backend.get_int_value(&frame, 0), b'y' as i64);
    }

    #[test]
    fn test_newunicode_without_descr_uses_wide_items() {
        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(Arc::new(
            Mutex::new(TrackingGcState::default()),
        ))));

        let ops = vec![
            Op::new(OpCode::Newunicode, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::Unicodesetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(101),
                    OpRef::from_raw(200),
                ],
            ),
            Op::new(
                OpCode::Unicodegetitem,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Unicodelen, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Finish, &[OpRef::from_raw(2), OpRef::from_raw(3)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        constants.insert(200, 0x2603);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1514);
        backend.compile_loop(&[], &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert_eq!(backend.get_int_value(&frame, 0), 0x2603);
        assert_eq!(backend.get_int_value(&frame, 1), 1);
    }

    #[test]
    fn test_strhash_without_descr_reads_cached_hash_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            Op::new(OpCode::Label, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Strhash, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Strlen, &[OpRef::from_raw(0)]),
            Op::new(
                OpCode::Strgetitem,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
            ),
            Op::new(
                OpCode::Finish,
                &[OpRef::from_raw(2), OpRef::from_raw(3), OpRef::from_raw(4)],
            ),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1515);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut raw = vec![0usize; 3];
        let ptr = raw.as_mut_ptr() as *mut u8;
        unsafe {
            *(ptr.add(BUILTIN_STRING_HASH_OFFSET) as *mut i64) = -7;
            *(ptr.add(BUILTIN_STRING_LEN_OFFSET) as *mut usize) = 3;
            *ptr.add(BUILTIN_STRING_CHARS_OFFSET) = b'a';
            *ptr.add(BUILTIN_STRING_CHARS_OFFSET + 1) = b'b';
            *ptr.add(BUILTIN_STRING_CHARS_OFFSET + 2) = b'c';
        }

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr as usize))]);
        assert_eq!(backend.get_int_value(&frame, 0), -7);
        assert_eq!(backend.get_int_value(&frame, 1), 3);
        assert_eq!(backend.get_int_value(&frame, 2), b'b' as i64);
    }

    #[test]
    fn test_unicodehash_without_descr_reads_cached_hash_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            Op::new(OpCode::Label, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Unicodehash, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Unicodelen, &[OpRef::from_raw(0)]),
            Op::new(
                OpCode::Unicodegetitem,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
            ),
            Op::new(
                OpCode::Finish,
                &[OpRef::from_raw(2), OpRef::from_raw(3), OpRef::from_raw(4)],
            ),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1516);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut raw = vec![0usize; 3];
        let ptr = raw.as_mut_ptr() as *mut u8;
        unsafe {
            *(ptr.add(BUILTIN_STRING_HASH_OFFSET) as *mut i64) = -11;
            *(ptr.add(BUILTIN_STRING_LEN_OFFSET) as *mut usize) = 1;
            *(ptr.add(BUILTIN_STRING_CHARS_OFFSET) as *mut u32) = 0x2603;
        }

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr as usize))]);
        assert_eq!(backend.get_int_value(&frame, 0), -11);
        assert_eq!(backend.get_int_value(&frame, 1), 1);
        assert_eq!(backend.get_int_value(&frame, 2), 0x2603);
    }

    #[test]
    fn test_call_i_preserves_live_ref_inputs_across_collection() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0xABCDEF01;
        }

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        let descr = make_call_descr(vec![Type::Int], Type::Int);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CallI,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
                1,
                descr,
            ),
            mk_op(
                OpCode::Finish,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            collect_nursery_via_runtime as *const () as usize as i64,
        );
        // Vestigial runtime_id arg ignored by the trampoline since the
        // active GC is now routed through `CRANELIFT_ACTIVE_GC` directly.
        constants.insert(101, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1506);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let moved_root = backend.get_ref_value(&frame, 0);
        let call_result = backend.get_int_value(&frame, 1);

        assert!(!moved_root.is_null());
        assert_ne!(moved_root, root);
        assert_eq!(unsafe { *(moved_root.0 as *const u64) }, 0xABCDEF01);
        assert_eq!(call_result, 123);
    }

    #[test]
    fn test_cond_call_n_preserves_live_ref_inputs_across_collection() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0xFACEB00C;
        }

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        let descr = make_call_descr(vec![Type::Int], Type::Void);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op_with_descr(
                OpCode::CondCallN,
                &[
                    OpRef::from_raw(102),
                    OpRef::from_raw(100),
                    OpRef::from_raw(101),
                ],
                OpRef::NONE.raw(),
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            collect_nursery_via_runtime_void as *const () as usize as i64,
        );
        // Vestigial runtime_id arg, see comment on the previous test.
        constants.insert(101, 0);
        constants.insert(102, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1507);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let moved_root = backend.get_ref_value(&frame, 0);

        assert!(!moved_root.is_null());
        assert_ne!(moved_root, root);
        assert_eq!(unsafe { *(moved_root.0 as *const u64) }, 0xFACEB00C);
    }

    #[test]
    fn test_deadframe_ref_survives_collection_after_execute_token() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0x12345678;
        }

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1508);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        with_cranelift_gc_required(|gc| gc.collect_nursery());

        let moved_root = backend.get_ref_value(&frame, 0);
        assert!(!moved_root.is_null());
        assert_ne!(moved_root, root);
        assert_eq!(unsafe { *(moved_root.0 as *const u64) }, 0x12345678);
    }

    #[test]
    fn test_deadframe_drop_after_backend_drop_is_safe() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1509);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        drop(backend);
        drop(frame);
    }

    #[test]
    fn test_guard_not_invalidated_passes() {
        // GUARD_NOT_INVALIDATED should pass (not side-exit) when the loop
        // has not been invalidated.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            // GUARD_NOT_INVALIDATED takes no data args
            mk_op(OpCode::GuardNotInvalidated, &[], OpRef::NONE.raw()),
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(100);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Not invalidated -> guard passes -> Finish returns 10 + 32 = 42
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(32)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_guard_not_invalidated_fails_after_invalidation() {
        // After calling invalidate(), GUARD_NOT_INVALIDATED should fail and
        // the compiled code should side-exit through the guard's fail path.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        // Build a guard with explicit fail_args so we can inspect them.
        let mut guard_op = Op::new(OpCode::GuardNotInvalidated, &[]);
        guard_op.pos = OpRef::from_raw(OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(0),
            OpRef::from_raw(1),
        ]));

        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            guard_op,
            mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2),
            mk_op(OpCode::Finish, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(101);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Before invalidation: guard passes, Finish is reached.
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(32)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);

        // Invalidate the loop.
        token.invalidate();
        assert!(token.is_invalidated());

        // After invalidation: guard fails, side-exits with the guard's fail args.
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(32)]);
        // The guard's fail_args are [OpRef::from_raw(0), OpRef::from_raw(1)] = the input values.
        let descr = backend.get_latest_descr(&frame);
        // fail_index 0 is the GuardNotInvalidated guard (first guard in the trace).
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 10);
        assert_eq!(backend.get_int_value(&frame, 1), 32);
    }

    #[test]
    fn test_guard_not_invalidated_in_loop() {
        // Verify GUARD_NOT_INVALIDATED works inside a loop that runs
        // multiple iterations before being invalidated.
        let mut backend = CraneliftBackend::new();

        // Loop: i = i + 1; guard_not_invalidated; guard i < limit; jump
        let inputargs = vec![InputArg::new_int(0)];

        let mut guard_inv = Op::new(OpCode::GuardNotInvalidated, &[]);
        guard_inv.pos = OpRef::from_raw(OpRef::NONE.raw());
        guard_inv.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(1)]));

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ), // i = i + 1
            guard_inv, // guard_not_invalidated
            mk_op(
                OpCode::IntLt,
                &[OpRef::from_raw(1), OpRef::from_raw(101)],
                2,
            ), // i < 1000000
            mk_op(OpCode::GuardTrue, &[OpRef::from_raw(2)], OpRef::NONE.raw()),
            mk_op(OpCode::Jump, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 1_000_000i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(102);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Without invalidation: runs to completion (i reaches 1000000,
        // GuardTrue fails).
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        // fail_index 1 = GuardTrue (second guard)
        assert_eq!(descr.fail_index(), 1);
        assert_eq!(backend.get_int_value(&frame, 0), 999_999);

        // Now invalidate and run again: should fail at GuardNotInvalidated
        // on the first iteration.
        token.invalidate();
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        // fail_index 0 = GuardNotInvalidated (first guard)
        assert_eq!(descr.fail_index(), 0);
        // i = 0 + 1 = 1 (only one iteration before guard fails)
        assert_eq!(backend.get_int_value(&frame, 0), 1);
    }

    #[test]
    fn test_invalidate_via_backend_trait() {
        // Verify that Backend::invalidate_loop() properly sets the flag.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];

        let mut guard_inv = Op::new(OpCode::GuardNotInvalidated, &[]);
        guard_inv.pos = OpRef::from_raw(OpRef::NONE.raw());
        guard_inv.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(0)]));

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            guard_inv,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(103);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Before invalidation
        let frame = backend.execute_token(&token, &[Value::Int(99)]);
        // Finish (fail_index 1)
        assert_eq!(backend.get_int_value(&frame, 0), 99);

        // Invalidate via the Backend trait method
        backend.invalidate_loop(&token);
        assert!(token.is_invalidated());

        // After invalidation: guard fails
        let frame = backend.execute_token(&token, &[Value::Int(99)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
    }

    /// Guard-bearing callee: guard passes -> finish result propagated.
    /// Guard-bearing callee: guard fails  -> deadframe propagated.

    /// Guard-bearing callee with multiple guards: first guard passes, second
    /// guard fails -> correct fail_index propagated.

    /// Guard-bearing callee with force_token finish shape:
    /// Callee has ForceToken + GuardNotForced2 + Finish(force_token).
    /// Caller uses CallAssemblerR and gets the force_token result.

    #[test]
    fn test_all_guards_have_recovery_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard1 = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        guard1.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(0),
            OpRef::from_raw(1),
        ]));
        let int_add = mk_op(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)], 2);
        let mut guard2 = mk_op(OpCode::GuardFalse, &[OpRef::from_raw(2)], OpRef::NONE.raw());
        guard2.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(2)]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            guard1,
            int_add,
            guard2,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(500);
        backend.set_next_header_pc(2000);
        let mut token = JitCellToken::new(9001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_fail_descr_layouts(&token)
            .expect("compiled layouts should exist");

        // All fail descriptors (2 guards + 1 finish) should have recovery_layout
        for (idx, layout) in layouts.iter().enumerate() {
            assert!(
                layout.recovery_layout.is_some(),
                "fail_descr[{idx}] should have recovery_layout"
            );
            let recovery = layout.recovery_layout.as_ref().unwrap();
            assert!(!recovery.frames.is_empty());
            assert_eq!(recovery.frames[0].trace_id, Some(500));
            assert_eq!(recovery.frames[0].header_pc, Some(2000));
            // slot_types should always be populated
            assert!(recovery.frames[0].slot_types.is_some());
        }
    }

    #[test]
    fn test_frame_layout_includes_slot_types() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(0),
            OpRef::from_raw(1),
            OpRef::from_raw(2),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                OpRef::NONE.raw(),
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(501);
        backend.set_next_header_pc(3000);
        let mut token = JitCellToken::new(9002);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_fail_descr_layouts(&token)
            .expect("compiled layouts should exist");

        // Guard layout
        let guard_layout = &layouts[0];
        let recovery = guard_layout
            .recovery_layout
            .as_ref()
            .expect("guard should have recovery layout");
        let frame = &recovery.frames[0];
        let slot_types = frame
            .slot_types
            .as_ref()
            .expect("slot_types should always be populated");
        assert_eq!(slot_types, &[Type::Int, Type::Ref, Type::Float]);

        // frame_stack should mirror recovery frames
        let frame_stack = guard_layout
            .frame_stack
            .as_ref()
            .expect("frame_stack should be populated from recovery_layout");
        assert_eq!(frame_stack.len(), 1);
        assert_eq!(
            frame_stack[0].slot_types,
            Some(vec![Type::Int, Type::Ref, Type::Float])
        );
    }

    #[test]
    fn test_compiled_guard_frame_stacks_query() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard1 = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(0)], OpRef::NONE.raw());
        guard1.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::from_raw(0),
            OpRef::from_raw(1),
        ]));
        let mut guard2 = mk_op(OpCode::GuardFalse, &[OpRef::from_raw(1)], OpRef::NONE.raw());
        guard2.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::from_raw(1)]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
                OpRef::NONE.raw(),
            ),
            guard1,
            guard2,
            mk_op(OpCode::Finish, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
        ];

        backend.set_next_trace_id(502);
        backend.set_next_header_pc(4000);
        let mut token = JitCellToken::new(9003);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame_stacks = backend
            .compiled_guard_frame_stacks(&token)
            .expect("compiled_guard_frame_stacks should return Some");

        // 2 guards + 1 finish = 3 entries, all with frame stacks
        assert_eq!(frame_stacks.len(), 3);
        for (fail_index, frames) in &frame_stacks {
            assert!(
                !frames.is_empty(),
                "fail_index={fail_index} should have non-empty frame stack"
            );
            assert_eq!(frames[0].trace_id, Some(502));
            assert_eq!(frames[0].header_pc, Some(4000));
            // slot_types populated in every frame
            for frame in frames {
                assert!(frame.slot_types.is_some());
            }
        }

        // Verify fail indices are sequential
        let indices: Vec<u32> = frame_stacks.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    /// Verify that the main opcode dispatch in compile_loop covers all OpCode
    /// variants. We compile a small representative trace and confirm the backend
    /// doesn't return Unsupported. The compile-time exhaustiveness of the match
    /// is the primary guarantee (enforced by CI warning checks).
    #[test]
    fn test_all_opcodes_covered_in_backend() {
        let mut backend = CraneliftBackend::new();
        let mut constants = HashMap::new();
        constants.insert(100, 42i64);
        constants.insert(101, 7i64);
        backend.set_constants(constants);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::from_raw(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::from_raw(0), OpRef::from_raw(100)],
                1,
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::from_raw(1), OpRef::from_raw(101)],
                2,
            ),
            {
                let mut g = mk_op(OpCode::GuardTrue, &[OpRef::from_raw(2)], OpRef::NONE.raw());
                g.fail_args = Some(smallvec::smallvec![OpRef::from_raw(0)]);
                g
            },
            mk_op(OpCode::Finish, &[OpRef::from_raw(1)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(99_999);
        let result = backend.compile_loop(&inputargs, &ops, &mut token);
        assert!(
            result.is_ok(),
            "representative trace should compile: {:?}",
            result.err()
        );
    }
}
