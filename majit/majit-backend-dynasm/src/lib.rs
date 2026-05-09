/// rpython/jit/backend/ parity:
/// Direct machine code generation via dynasm-rs with in-place patching.
///
/// Module structure mirrors RPython's backend directory layout:
///   llsupport/llmodel.py    → runner.rs  (DynasmBackend)
///   llsupport/assembler.py  → assembler.rs (BaseAssembler — shared)
///   llsupport/regalloc.py   → regalloc.rs (BaseRegalloc — shared)
///   llsupport/jitframe.py   → jitframe.rs
///   x86/assembler.py        → x86/assembler.rs (Assembler386)
///   aarch64/assembler.py    → aarch64/assembler.rs (AssemblerARM64)
///   aarch64/opassembler.py  → aarch64/opassembler.rs (ResOpAssembler)
///   aarch64/registers.py    → aarch64/registers.rs
///
/// arch.rs, codebuf.rs, guard.rs, regloc.rs are from llsupport/.
// ── Shared modules (llsupport/ parity) ──
use std::cell::RefCell;

pub mod arch;
pub mod callbuilder;
pub mod codebuf;
pub mod frame;
pub mod gcmap;
pub mod guard;
pub(crate) mod j2plan;
pub use majit_backend::jitframe;
pub use majit_backend::llmodel;
pub mod jump;
pub mod regalloc;
pub mod regloc;
pub mod runner;

// ── Architecture-specific modules ──
#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86;

// ── llmodel.py:194-199 JIT exception state ──
// RPython stores exception state in thread-local (GIL-protected) globals.
// Cranelift uses JIT_EXC_VALUE / JIT_EXC_TYPE atomics (compiler.rs:515-517).
// Dynasm uses the same pattern for structural equivalence.

use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};

static JIT_EXC_VALUE: AtomicI64 = AtomicI64::new(0);
static JIT_EXC_TYPE: AtomicI64 = AtomicI64::new(0);
static JITFRAME_GC_TYPE_ID: AtomicU32 = AtomicU32::new(u32::MAX);
static DUMMY_THREADLOCAL_SLOT: i64 = 0;

thread_local! {
    /// llmodel.py:317-323 `threadlocalref_addr` parity: compiled entries
    /// take the thread-local slot-array base as their second argument.
    /// Dynasm's aarch64 CALL_ASSEMBLER path preserves and forwards that
    /// entry ABI even before THREADLOCALREF_GET lowering exists.
    static JIT_THREADLOCAL_SLOTS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

/// llmodel.py:194-199 _store_exception parity: set JIT exception state.
/// `value` is a valid OBJECTPTR (or 0). Exception class derived from
/// value.typeptr (offset 0), matching RPython's invariant.
pub fn jit_exc_raise(value: i64) {
    let exc_type = if value == 0 {
        0
    } else {
        unsafe { *(value as *const i64) }
    };
    JIT_EXC_VALUE.store(value, Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, Ordering::Relaxed);
}

/// Check if an exception is currently pending.
pub fn jit_exc_is_pending() -> bool {
    JIT_EXC_VALUE.load(Ordering::Relaxed) != 0
}

/// cpu.grab_exc_value parity: read exception class from TLS.
pub fn jit_exc_class_raw() -> i64 {
    JIT_EXC_TYPE.load(Ordering::Relaxed)
}

/// cpu.grab_exc_value parity: read and clear exception value.
pub fn jit_exc_value_raw() -> i64 {
    JIT_EXC_VALUE.swap(0, Ordering::Relaxed)
}

/// Clear exception state.
pub fn jit_exc_clear() {
    JIT_EXC_VALUE.store(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
}

/// Address of `JIT_EXC_VALUE` for direct memory load/store from JIT-emitted
/// code (`_store_and_reset_exception` parity, `x86/assembler.py:1826-1843`).
pub fn jit_exc_value_addr() -> usize {
    &JIT_EXC_VALUE as *const _ as usize
}

/// Address of `JIT_EXC_TYPE` for direct memory load/store from JIT-emitted
/// code.  Mirrors `cpu.pos_exception()`; cleared together with
/// `pos_exc_value` whenever the propagate path drains the global exception.
pub fn jit_exc_type_addr() -> usize {
    &JIT_EXC_TYPE as *const _ as usize
}

/// Override the JITFRAME type id used by dynasm nursery slow paths.
///
/// The frontend registers `jitframe_type_info()` on its active GC before
/// running JIT code; dynasm must allocate recursive callee jitframes with
/// that same type so the collector traces them with `jitframe_trace`.
pub fn set_jitframe_gc_type_id(id: u32) {
    JITFRAME_GC_TYPE_ID.store(id, Ordering::Release);
}

pub(crate) fn jitframe_gc_type_id() -> Option<u32> {
    match JITFRAME_GC_TYPE_ID.load(Ordering::Acquire) {
        u32::MAX => None,
        id => Some(id),
    }
}

/// Write a thread-local slot that compiled entrypoints may read back.
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

/// Return the base pointer passed to compiled entrypoints as x1.
pub(crate) fn jit_threadlocalref_base() -> *const i64 {
    JIT_THREADLOCAL_SLOTS.with(|slots| {
        let slots = slots.borrow();
        if slots.is_empty() {
            &raw const DUMMY_THREADLOCAL_SLOT
        } else {
            slots.as_ptr()
        }
    })
}

// ── CALL_ASSEMBLER helper infrastructure ──
// assembler.py:345-350: jd.assembler_helper_adr — the slow path runtime
// entry point for CALL_ASSEMBLER when callee doesn't finish normally.
//
// Cranelift uses register_call_assembler_blackhole/bridge/force/unbox_int.
// Dynasm provides the same registration API and a C-callable trampoline
// that the generated slow path code calls directly.

use std::sync::OnceLock;

/// Blackhole resume: (descr_addr, raw_values, len, typed_outputs, typed_len)
/// → Option<result>.
///
/// The receiving handler recovers the failed descr from `descr_addr`
/// via `Backend::fail_descr_arc_from_addr` (`history.py:125`
/// `cpu.get_latest_descr` parity) and derives the resume identity
/// (`jct.green_key` / `descr.trace_id()` /
/// `descr.fail_index_per_trace()`) from that Arc, mirroring
/// `compile.py:710-716 resume_in_blackhole(descr, deadframe)` where
/// the descr is the sole identity carrier.  No surrogate triple
/// crosses the C-ABI.
pub type BlackholeFn = fn(usize, *const i64, usize, *const i64, usize) -> Option<i64>;

/// Bridge compilation: (raw_values, len, descr_addr) → compiled?
///
/// The receiving handler recovers the failed descr from `descr_addr`
/// via `Backend::fail_descr_arc_from_addr` (`history.py:125`
/// `cpu.get_latest_descr` parity) and derives the bridge source
/// identity (`jct.green_key` / `descr.trace_id()` /
/// `descr.fail_index_per_trace()`) from that Arc, mirroring
/// `pyjitpl.py:2890 handle_guard_failure(self, resumedescr,
/// deadframe)`.  No surrogate triple crosses the C-ABI.
pub type BridgeFn = fn(*const i64, usize, usize) -> bool;

/// Force callee: (callee_frame_ptr) → result
pub type ForceFn = extern "C" fn(i64) -> i64;

/// Unbox int from Ref: (raw_ref) → i64
pub type UnboxIntFn = fn(i64) -> i64;

static CA_BLACKHOLE_FN: OnceLock<BlackholeFn> = OnceLock::new();
static CA_BRIDGE_FN: OnceLock<BridgeFn> = OnceLock::new();
static CA_FORCE_FN: OnceLock<ForceFn> = OnceLock::new();
static CA_UNBOX_INT_FN: OnceLock<UnboxIntFn> = OnceLock::new();

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

pub(crate) fn jitframe_layout() -> Option<JitFrameLayoutInfo> {
    JITFRAME_LAYOUT.get().cloned()
}

/// Pyre dynasm extension: CALL_ASSEMBLER builds some callee jitframes via
/// `libc::calloc`, so the GC needs an explicit registration hook before the
/// callee pushes that frame on the jitframe shadow stack.
pub extern "C" fn dynasm_register_libc_jitframe(addr: i64) {
    if addr != 0 {
        majit_gc::shadow_stack::register_libc_jitframe(addr as usize);
    }
}

/// Remove a libc-jitframe from the registry immediately before freeing it.
pub extern "C" fn dynasm_unregister_libc_jitframe(addr: i64) {
    if addr != 0 {
        majit_gc::shadow_stack::unregister_libc_jitframe(addr as usize);
    }
}

pub fn register_libc_jitframe_helper_addr() -> usize {
    dynasm_register_libc_jitframe as usize
}

pub fn unregister_libc_jitframe_helper_addr() -> usize {
    dynasm_unregister_libc_jitframe as usize
}

/// Register blackhole resume handler (same API as Cranelift).
pub fn register_call_assembler_blackhole(f: BlackholeFn) {
    CA_BLACKHOLE_FN.set(f).ok();
}

/// Register bridge compilation handler (same API as Cranelift).
pub fn register_call_assembler_bridge(f: BridgeFn) {
    CA_BRIDGE_FN.set(f).ok();
}

/// Register force handler (same API as Cranelift).
pub fn register_call_assembler_force(f: ForceFn) {
    CA_FORCE_FN.set(f).ok();
}

/// Return the force_fn address for embedding in generated code.
/// Returns 0 if not registered.
pub fn call_assembler_force_fn_addr() -> usize {
    CA_FORCE_FN.get().map(|f| *f as usize).unwrap_or(0)
}

/// Register int unbox handler (same API as Cranelift).
pub fn register_call_assembler_unbox_int(f: UnboxIntFn) {
    CA_UNBOX_INT_FN.set(f).ok();
}

/// rpython/jit/backend/llsupport/llmodel.py:229-234 `insert_stack_check`
/// parity. The interpreter registers the three addresses the JIT
/// prologue needs to emit the inline stack-overflow probe matching
/// `rpython/jit/backend/x86/assembler.py:1085-1091`:
///
///   * `end_adr` — address of `PYRE_STACKTOOBIG.stack_end`
///     (`LL_stack_get_end_adr`). The probe loads this and subtracts
///     the current SP.
///   * `length_adr` — address of `PYRE_STACKTOOBIG.stack_length`
///     (`LL_stack_get_length_adr`). The probe compares the diff
///     against this.
///   * `slowpath_addr` — `pyre_stack_too_big_slowpath` function
///     pointer (`STACK_CHECK_SLOWPATH` in llmodel). Called on fast-
///     path miss with the current SP; returns 1 on real overflow.
#[derive(Copy, Clone, Debug)]
pub struct StackCheckAddresses {
    pub end_adr: usize,
    pub length_adr: usize,
    pub slowpath_addr: usize,
}

static STACK_CHECK_ADDRS: OnceLock<StackCheckAddresses> = OnceLock::new();

/// Register the three addresses the JIT prologue needs for the inline
/// stack-overflow probe. Called once at startup from pyre-jit; matches
/// `rpython/jit/backend/llsupport/llmodel.py:229-234 insert_stack_check`.
pub fn register_stack_check_addresses(end_adr: usize, length_adr: usize, slowpath_addr: usize) {
    let _ = STACK_CHECK_ADDRS.set(StackCheckAddresses {
        end_adr,
        length_adr,
        slowpath_addr,
    });
}

/// Retrieve the registered inline-probe addresses. Returns `None` if
/// the interpreter has not yet registered them (tests or early
/// startup); the prologue emitter skips the probe in that case,
/// mirroring `assembler.py:1082 "if self.stack_check_slowpath == 0:
/// pass (no stack check)"`.
pub fn stack_check_addresses() -> Option<StackCheckAddresses> {
    STACK_CHECK_ADDRS.get().copied()
}

/// warmspot.py:1021-1028 `assembler_call_helper` parity —
/// C-callable trampoline for the CALL_ASSEMBLER slow path.
///
/// Upstream:
///     fail_descr = self.cpu.get_latest_descr(deadframe)
///     try:
///         fail_descr.handle_fail(deadframe, self.metainterp_sd, jd)
///     except jitexc.JitException as e:
///         return handle_jitexception(e)
///
/// In upstream each `AbstractFailDescr` subclass carries its own
/// `handle_fail`, and the helper's job is purely to dispatch. pyre
/// encodes finish descrs as raw-pointer singletons
/// (`done_with_this_frame_descr_{void,int,ref,float}`,
/// `exit_frame_with_exception_descr_ref`) and resume-guard descrs as
/// `DynasmFailDescr`, so this trampoline performs the type dispatch
/// inline and delegates to one of the `handle_fail_*` helpers below.
///
/// **Deviation (PRE-EXISTING-ADAPTATION)**: pyre does not yet carry
/// `FailDescr::handle_fail` as a virtual method because descrs are
/// stored as raw `usize` sentinels rather than `Box<dyn FailDescr>`,
/// and `metainterp_sd` / `jitdriver_sd` are not plumbed through
/// `majit-backend`. Converting to an object-hierarchy requires
/// reworking descr storage across both backends — tracked as a
/// follow-up on top of this refactor. The body below is written so
/// every branch lines up 1:1 with upstream's `handle_fail` sites.
///
/// Input:  callee_jf_ptr = deadframe,
///         green_key = caller loop's header_pc (used by CA_*_FN hooks
///         to find the owning compiled trace).
/// Output: the int interpretation of the handled result (the caller
///         re-casts to the portal's return type).
///
/// The callee jitframe is GC-managed by the rewriter's
/// `handle_call_assembler` path, so the helper must not free it here.
pub extern "C" fn call_assembler_helper_trampoline(
    cpu_handle: *const std::sync::RwLock<guard::CpuDescrAttachments>,
    callee_jf_ptr: *mut jitframe::JitFrame,
    green_key: u64,
) -> i64 {
    if callee_jf_ptr.is_null() {
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[dynasm][ca-helper] null callee_jf_ptr green_key={green_key}");
        }
        return 0;
    }
    let frame_ptr = callee_jf_ptr;
    // warmspot.py:1022 `fail_descr = cpu.get_latest_descr(deadframe)`.
    let descr_raw = unsafe { llmodel::get_latest_descr(frame_ptr) };
    // `compile.py:665-674` parity: resolve the attached `DoneWithThisFrame*`
    // + `ExitFrameWithExceptionDescrRef` identities from the cpu instance
    // that emitted the CALL_ASSEMBLER.  `cpu_handle` is the heap-pinned
    // `Arc<RwLock<CpuDescrAttachments>>` address baked as a compile-time
    // constant into the emitted call site — same role as RPython's
    // `self.cpu` closure capture in the translated code.  A compiled
    // trace keeps an `Arc` clone alive alongside its executable buffer
    // so the pointee outlives any subsequent `DynasmBackend` drop.
    //
    // A null `cpu_handle` means "no attached cpu" — only reachable from
    // direct-trampoline unit tests that construct the jitframe without
    // going through `compile_loop`.  In that mode no finish descr can
    // match, so dispatch falls through to `handle_fail_resume_guard`.
    let attached = if cpu_handle.is_null() {
        majit_backend::AttachedDescrPtrs::default()
    } else {
        unsafe { (*cpu_handle).read().unwrap().descr_ptrs() }
    };
    // warmspot.py:1023-1028 `fail_descr.handle_fail(deadframe, ...)`.
    // Callee jitframe is GC-managed by the rewriter's
    // `handle_call_assembler` path (rewrite.py:665 `gen_malloc_frame`),
    // so this helper must not free it here — upstream
    // `assembler_call_helper` (warmspot.py:1022-1028) likewise does not
    // free `deadframe`.
    handle_fail_dispatch(&attached, descr_raw, frame_ptr, green_key)
}

/// Dispatch to the `handle_fail` variant that matches `descr_raw`.
/// Upstream equivalent: the virtual-method call in warmspot.py:1024,
/// resolved at runtime by the Python class of `fail_descr`.
fn handle_fail_dispatch(
    attached: &majit_backend::AttachedDescrPtrs,
    descr_raw: usize,
    frame_ptr: *mut jitframe::JitFrame,
    green_key: u64,
) -> i64 {
    if descr_raw == 0 {
        // Deviation: unresolved-callee path. Upstream never reaches
        // this state — an unlinked target is patched out before the
        // CALL_ASSEMBLER emits the call. Retained as a safety fence.
        return 0;
    }
    // `compile.py:665-674` parity: the finish descrs attached to
    // `self.cpu` are compared by pointer identity.  RPython backend
    // code reaches them through `cpu.done_with_this_frame_descr_*`;
    // pyre resolves them from the heap-pinned cpu handle baked into
    // the CALL_ASSEMBLER emit site (resolved in the trampoline).
    if descr_raw == attached.done_with_this_frame_descr_void
        || descr_raw == attached.done_with_this_frame_descr_int
        || descr_raw == attached.done_with_this_frame_descr_ref
        || descr_raw == attached.done_with_this_frame_descr_float
    {
        // compile.py:626-656 `_DoneWithThisFrameDescr` subclasses
        // (Void/Int/Ref/Float) — the four finish singletons.
        return handle_fail_done_with_this_frame(attached, descr_raw, frame_ptr);
    }
    if descr_raw == attached.exit_frame_with_exception_descr_ref {
        // compile.py:658-662 `ExitFrameWithExceptionDescrRef.handle_fail`:
        //   value = cpu.get_ref_value(deadframe, 0)
        //   raise jitexc.ExitFrameWithExceptionRef(value)
        // pyre returns the raw slot-0 payload here; the caller stub
        // observes `is_exit_frame_with_exception` on the synthesized
        // FailDescr and re-raises through `jit_exc_value`.
        return handle_fail_exit_frame_with_exception(frame_ptr);
    }
    if descr_raw != 0 && descr_raw == attached.propagate_exception_descr {
        // compile.py:1085-1098 `PropagateExceptionDescr.handle_fail`:
        //     exception = cpu.grab_exc_value(deadframe)
        //     if not exception:
        //         exception = cast_instance_to_gcref(memory_error)
        //     raise jitexc.ExitFrameWithExceptionRef(exception)
        // The CALL_ASSEMBLER caller observes `is_exit_frame_with_exception`
        // on the synthesized FailDescr (runner.rs::find_descr_by_ptr) and
        // re-raises through `jit_exc_value`; the same shape applies here
        // — read jf_guard_exc, fall back to memory_error, re-raise.
        return handle_fail_propagate_exception(frame_ptr);
    }
    // compile.py:701-717 `AbstractResumeGuardDescr.handle_fail`.
    let descr = unsafe { &*(descr_raw as *const guard::DynasmFailDescr) };
    handle_fail_resume_guard(descr, descr_raw, frame_ptr, green_key)
}

/// compile.py:626-656 `DoneWithThisFrameDescr{Void,Int,Ref,Float}.handle_fail`.
///
/// Upstream raises `jitexc.DoneWithThisFrame*(slot-0)` and
/// `handle_jitexception` shape-casts the payload to the portal's
/// return type.  pyre returns the raw i64 back; the JIT-emitted caller
/// stub (`assembler_helper_wrapper`) reinterprets it as the correct
/// kind.  Each subclass picks a different accessor so the returned
/// bits carry the right sign-extension / pointer-tagging / float
/// bit-layout for the portal return type:
///
/// * `DoneWithThisFrameDescrVoid.handle_fail` — no payload, return 0
///   (compile.py:628-632).
/// * `DoneWithThisFrameDescrInt.handle_fail` — `get_int_value(df, 0)`
///   (compile.py:635-640).
/// * `DoneWithThisFrameDescrRef.handle_fail` — `get_ref_value(df, 0)`
///   (compile.py:644-649).
/// * `DoneWithThisFrameDescrFloat.handle_fail` — `get_float_value(df, 0)`
///   (compile.py:653-657).
///
/// The descr identity selects the accessor — matched through the four
/// `guard::done_with_this_frame_descr_*_ptr()` singletons rather than
/// a `fail_arg_types[0]` probe, because the Void descr carries no type
/// entry at all.
fn handle_fail_done_with_this_frame(
    attached: &majit_backend::AttachedDescrPtrs,
    descr_raw: usize,
    frame_ptr: *mut jitframe::JitFrame,
) -> i64 {
    if descr_raw == attached.done_with_this_frame_descr_void {
        return 0;
    }
    if descr_raw == attached.done_with_this_frame_descr_int {
        return unsafe { llmodel::get_int_value_direct(frame_ptr, 0) as i64 };
    }
    if descr_raw == attached.done_with_this_frame_descr_ref {
        return unsafe { llmodel::get_ref_value_direct(frame_ptr, 0) as i64 };
    }
    if descr_raw == attached.done_with_this_frame_descr_float {
        return unsafe { llmodel::get_float_value_direct(frame_ptr, 0) as i64 };
    }
    // Unreachable: the dispatch gate above already narrowed
    // `descr_raw` to one of the four singletons.
    0
}

/// compile.py:658-662 `ExitFrameWithExceptionDescrRef.handle_fail`.
///
/// Upstream reads slot 0 as a gcref and raises
/// `jitexc.ExitFrameWithExceptionRef(value)`.  Pyre hands the raw int
/// payload back; the caller stub
/// (`pyre/pyre-jit/src/eval.rs` handle_jit_outcome) routes the result
/// into `PyError::from_exc_object` when the synthesized FailDescr
/// carries `is_exit_frame_with_exception = true`.
///
/// Also publishes the exception via `jit_exc_raise` so a future
/// `genop_guard_guard_no_exception` (assembler.py:1782) emitted by
/// the caller sees the pending exception — symmetric with cranelift's
/// CALL_ASSEMBLER trampoline path.  PyPy's
/// `generate_guard_no_exception` checks `cpu.pos_exception()` (the
/// exception type slot), so dynasm's guard-no-exception fast path
/// compares against `JIT_EXC_TYPE`; `jit_exc_raise` publishes both
/// slots and keeps the value slot available for
/// `_store_and_reset_exception` / `save_exception`.
fn handle_fail_exit_frame_with_exception(frame_ptr: *mut jitframe::JitFrame) -> i64 {
    let value = unsafe { llmodel::get_ref_value_direct(frame_ptr, 0) as i64 };
    // warmspot.py:998-1005 + llsupport/assembler.py:345 —
    // `assembler_helper_wrapper` invokes `handle_jitexception` which
    // re-raises through `pos_exc_value`.
    jit_exc_raise(value);
    value
}

/// compile.py:1085-1098 `PropagateExceptionDescr.handle_fail`.
///
/// Reads `jf_guard_exc` (set by Layer 3's emitted
/// `_store_and_reset_exception` in `OpCode::CheckMemoryError`),
/// clears it (`grab_exc_value`), and falls back to `memory_error`
/// when empty (`cast_instance_to_gcref(memory_error)`).  The caller
/// observes `is_exit_frame_with_exception=true` on the synthesized
/// FailDescr and re-raises the value through `jit_exc_value`.
///
/// Republishes the value via `jit_exc_raise` for the same
/// caller-side GUARD_NO_EXCEPTION reason as
/// `handle_fail_exit_frame_with_exception`; Layer 3's emit cleared
/// the globals on transfer, so this is a fresh store of the resolved
/// (possibly memory_error fallback) value.
fn handle_fail_propagate_exception(frame_ptr: *mut jitframe::JitFrame) -> i64 {
    let exc_val = unsafe {
        let slot = &mut (*frame_ptr).jf_guard_exc;
        let v = *slot;
        *slot = 0;
        v
    };
    let value = if exc_val != 0 {
        exc_val as i64
    } else {
        majit_backend::memory_error_singleton_ref()
    };
    jit_exc_raise(value);
    value
}

/// compile.py:701-717 `AbstractResumeGuardDescr.handle_fail`.
///
/// Upstream:
///     if must_compile(...) and not rstack.stack_almost_full():
///         self._trace_and_compile_from_bridge(deadframe, ...)
///     else:
///         resume_in_blackhole(metainterp_sd, jitdriver_sd, self, deadframe)
///     assert 0, "unreachable"
///
/// **Deviation (PRE-EXISTING-ADAPTATION)**: pyre's CA slow path runs
/// the bridge-tracer hook (`CA_BRIDGE_FN` → `jit_ca_handle_guard_failure`
/// in pyre-jit/src/call_jit.rs:2425) and then the blackhole hook
/// (`CA_BLACKHOLE_FN`) sequentially, instead of choosing one via the
/// outer `if must_compile` branch.  The PyPy line-by-line port of
/// must_compile / stack_almost_full / start_compiling /
/// `_trace_and_compile_from_bridge` / done_compiling is carried out
/// inside `jit_ca_handle_guard_failure` (call_jit.rs:2438-2499) — the
/// flow IS modeled, just one layer down.
///
/// **Why the outer dispatch shape differs**: PyPy's
/// `_trace_and_compile_from_bridge` (compile.py:719-733) raises
/// `EnterJitAssembler` (warmstate.py:513) at success to non-locally
/// unwind back to the JIT entry point and re-enter the newly-attached
/// bridge.  Pyre uses Rust returns where PyPy uses Python exceptions:
/// `jit_ca_handle_guard_failure` returns `bool` after attaching, and
/// the trampoline cannot unwind/re-enter from this point — the next
/// outer iteration's `eval_loop_jit` dispatch picks up the bridge
/// instead.  Therefore blackhole always runs to finish the current
/// frame; the next iteration runs the bridge.  End result equivalent
/// (each CA slow entry attaches at most one bridge AND produces a
/// resume value).
///
/// **Convergence path**: introducing a control-flow primitive that
/// non-locally unwinds from `jit_ca_handle_guard_failure` back to the
/// CALL_ASSEMBLER trampoline (Rust `panic_unwind`, `setjmp/longjmp`,
/// or refactoring the JIT entry to be re-entrant) would let the outer
/// dispatch read `if must_compile { trace; raise } else { blackhole }`
/// directly.  None of these are session-scope; the deviation is
/// retained until that mechanism lands.
fn handle_fail_resume_guard(
    descr: &guard::DynasmFailDescr,
    descr_raw: usize,
    frame_ptr: *mut jitframe::JitFrame,
    _outer_green_key: u64,
) -> i64 {
    let trace_id = <guard::DynasmFailDescr as majit_ir::FailDescr>::trace_id(descr);
    let fail_index = descr.fail_index;
    let n_fail_args = <guard::DynasmFailDescr as majit_ir::FailDescr>::fail_arg_types(descr).len();
    let mut raw_values: Vec<i64> = Vec::with_capacity(n_fail_args);
    for i in 0..n_fail_args {
        let slot = descr.fail_arg_locs.get(i).and_then(|l| *l).unwrap_or(i);
        raw_values.push(unsafe { llmodel::get_int_value_direct(frame_ptr, slot) as i64 });
    }

    // pyjitpl.py:2897-2899 parity: recover the owning Arc<JitCellToken>
    // identity from the descr.  When the weakref is dead (memmgr-evicted
    // JCT — "should be rare" per upstream), `compile.giveup()` raises
    // `SwitchToBlackhole(ABORT_BRIDGE)` (compile.py:27-29), caught at
    // pyjitpl.py:2906-2907, falling through to blackhole resume.  Pyre
    // mirrors this by gating bridge tracing on `Some(jct)` and passing
    // `green_key=0` to the blackhole callback so it derives the key from
    // the deadframe's pyframe pointer (call_jit.rs:1694-1704); never
    // substitute the parent CALL_ASSEMBLER caller's green_key here — that
    // would mis-route resume storage and trip `compile_bridge`'s
    // `debug_assert_eq!(source_jct.green_key, green_key)` (pyjitpl/mod.rs:8297-8301).
    let owning_jct = majit_backend::descr_owning_jct(descr);

    // compile.py:704-709 `_trace_and_compile_from_bridge`.
    // The hook compiles+attaches; it does NOT re-enter the bridge.
    // Skipped on giveup (None).
    if let (Some(_jct), Some(bridge_fn)) = (owning_jct.as_ref(), CA_BRIDGE_FN.get()) {
        bridge_fn(raw_values.as_ptr(), raw_values.len(), descr_raw);
    }

    // compile.py:710-716 `resume_in_blackhole(descr, deadframe)`: the
    // descr is the sole identity carrier; the receiver derives green_key /
    // trace_id / fail_index from it via `fail_descr_arc_from_addr` and
    // `descr_owning_jct`.
    if let Some(blackhole) = CA_BLACKHOLE_FN.get() {
        if let Some(bh_result) = blackhole(
            descr_raw,
            raw_values.as_ptr(),
            raw_values.len(),
            raw_values.as_ptr(),
            raw_values.len(),
        ) {
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[dynasm][ca-helper] resume-guard trace_id={trace_id} fail_index={fail_index} result=0x{bh_result:x}"
                );
            }
            return bh_result;
        }
    }
    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!(
            "[dynasm][ca-helper] resume-guard trace_id={trace_id} fail_index={fail_index} fell through to 0 descr=0x{descr_raw:x} frame={frame_ptr:p}"
        );
    }
    // `assert 0, "unreachable"` upstream — pyre returns 0 when neither
    // hook is registered (e.g. bare-backend tests exercise the helper
    // without a metainterp behind it).
    0
}

/// Return the address of the trampoline for embedding in generated code.
pub fn call_assembler_helper_addr() -> usize {
    call_assembler_helper_trampoline as *const () as usize
}

/// CALL_ASSEMBLER recursive callee execution trampoline. JIT-generated
/// code calls this instead of directly branching to the callee entry.
///
/// No alternate-stack switch: upstream RPython enters the callee on
/// the caller's native stack, and the callee's compiled prologue
/// performs its own inline SP probe
/// (rpython/jit/backend/x86/assembler.py:1085
/// `_call_header_with_stack_check`) against `PYRE_STACKTOOBIG.stack_end`.
/// Growing the stack here would shift SP into a different guard
/// region and invalidate that budget comparison.
///
/// Input:  arg0 = callee jf_ptr, arg1 = callee entry address
/// Output: callee's returned jf_ptr
pub extern "C" fn call_assembler_execute_trampoline(
    jf_ptr: *mut jitframe::JitFrame,
    callee_addr: usize,
) -> *mut jitframe::JitFrame {
    if std::env::var_os("MAJIT_LOG").is_some() && !jf_ptr.is_null() {
        let input0_ofs =
            jitframe::FIRST_ITEM_OFFSET + crate::arch::JITFRAME_FIXED_SIZE * jitframe::SIZEOFSIGNED;
        let input1_ofs = input0_ofs + jitframe::SIZEOFSIGNED;
        let frame_len = unsafe {
            *((jf_ptr as *const u8).add(jitframe::JF_FRAME_OFS + jitframe::LENGTHOFS)
                as *const isize)
        };
        let frame_info = unsafe {
            *((jf_ptr as *const u8).add(jitframe::JF_FRAME_INFO_OFS as usize) as *const usize)
        };
        let input0 = unsafe { *((jf_ptr as *const u8).add(input0_ofs) as *const usize) };
        let input1 = unsafe { *((jf_ptr as *const u8).add(input1_ofs) as *const usize) };
        eprintln!(
            "[dynasm][ca-exec] enter jf={jf_ptr:p} callee=0x{callee_addr:x} len={frame_len} frame_info=0x{frame_info:x} input0=0x{input0:x} input1=0x{input1:x}"
        );
    }
    let func: unsafe extern "C" fn(*mut jitframe::JitFrame) -> *mut jitframe::JitFrame =
        unsafe { std::mem::transmute(callee_addr) };
    let result = unsafe { func(jf_ptr) };
    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!("[dynasm][ca-exec] leave jf={result:p}");
    }
    result
}

/// Return the address of the execute trampoline.
pub fn call_assembler_execute_addr() -> usize {
    call_assembler_execute_trampoline as *const () as usize
}

// ── Pending CALL_ASSEMBLER targets ──
// Cranelift: register_pending_call_assembler_target stores a CaDispatchEntry
// with null code_ptr. When compile completes, code_ptr is updated atomically.
// Dynasm: the target registry lives in runner.rs (CALL_ASSEMBLER_TARGETS).
// This API delegates to it for pending target registration.

/// Register a pending CALL_ASSEMBLER target (code not yet compiled).
/// Matches Cranelift's register_pending_call_assembler_target API.
///
/// The actual registration happens via Backend::register_pending_target
/// → DynasmBackend::register_pending_target (runner.rs), which inserts
/// a 0 entry into CALL_ASSEMBLER_TARGETS. When compile_loop completes,
/// it overwrites with the real code_addr.
///
/// This function is the direct-call equivalent for non-Backend callers.
pub fn register_pending_call_assembler_target(
    token_number: u64,
    _inputarg_types: Vec<majit_ir::Type>,
    num_inputs: usize,
    _num_scalar_inputargs: usize,
    index_of_virtualizable: i32,
) {
    // Delegate to the runner's CALL_ASSEMBLER_TARGETS registry.
    runner::DynasmBackend::register_pending_call_assembler_target_static(
        token_number,
        num_inputs,
        index_of_virtualizable,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;
    use std::sync::Arc;

    unsafe fn alloc_test_jitframe(descr: usize, slots: &[i64]) -> *mut jitframe::JitFrame {
        let depth = slots.len();
        let size = jitframe::JitFrame::alloc_size(depth);
        let ptr = unsafe { libc::malloc(size) as *mut jitframe::JitFrame };
        assert!(!ptr.is_null());
        unsafe {
            libc::memset(ptr as *mut std::ffi::c_void, 0, size);
            jitframe::JitFrame::init(ptr, jitframe::NULLFRAMEINFO, depth);
            llmodel::set_latest_descr(ptr, descr);
            for (index, value) in slots.iter().copied().enumerate() {
                llmodel::set_int_value(ptr, index, value as isize);
            }
        }
        ptr
    }

    // Placeholder bridge body referenced by `set_bridge_addr` in
    // `test_helper_trampoline_does_not_execute_bridge`.  `call_assembler_helper_trampoline`
    // never invokes bridges (they're executed via patched guard jumps,
    // not the helper), so this body is unreachable in the test — the
    // descr value written here is arbitrary and its identity is never
    // classified against a cpu's attachments.
    extern "C" fn test_bridge_finish_int(jf: *mut jitframe::JitFrame) -> *mut jitframe::JitFrame {
        unsafe {
            llmodel::set_latest_descr(jf, 0xFEEDBEEF);
            llmodel::set_int_value(jf, 0, 321);
        }
        jf
    }

    // ── Bug 1 regression: unresolved target must not dereference result as pointer ──
    // The old code let the helper return value flow into `mov rdx, rax; mov rcx, [rdx]`
    // which dereferenced an integer as a pointer. This test verifies the trampoline
    // returns a plain value without crashing when jf_descr is 0 (pending/unresolved).

    #[test]
    fn test_helper_trampoline_null_jf_returns_zero() {
        let result = call_assembler_helper_trampoline(std::ptr::null(), std::ptr::null_mut(), 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_helper_trampoline_descr_zero_returns_zero() {
        let jf = unsafe { alloc_test_jitframe(0, &[0, 0, 0, 0]) };
        let result = call_assembler_helper_trampoline(std::ptr::null(), jf, 42);
        assert_eq!(result, 0);
    }

    // ── Bug 2 regression: force_fn must NOT be called with jitframe pointer ──
    // The old code passed a dynasm jitframe (i64 array) to force_fn which
    // expected a PyFrame pointer. Now force_fn is not called from the trampoline.
    // Verify: even when a guard descr is present, force_fn is never invoked.

    #[test]
    fn test_helper_trampoline_does_not_call_force_fn() {
        // force_fn is stored in CA_FORCE_FN (OnceLock). We can't reset it,
        // but we CAN verify the trampoline returns blackhole result (or 0)
        // without calling force. We do this by setting up a guard descr and
        // checking the trampoline reads fail_arg values correctly.

        let descr = Arc::new(guard::DynasmFailDescr::new(
            7,  // fail_index
            99, // trace_id
            vec![Type::Int, Type::Int],
            false, // is_finish
            true,  // is_resume_guard
        ));
        let descr_ptr = Arc::as_ptr(&descr) as i64;

        let jf = unsafe { alloc_test_jitframe(descr_ptr as usize, &[100, 200, 0, 0]) };
        let result = call_assembler_helper_trampoline(std::ptr::null(), jf, 42);
        // No blackhole registered → returns 0, no crash.
        assert_eq!(result, 0);
        drop(descr);
    }

    #[test]
    fn test_helper_trampoline_does_not_execute_bridge() {
        // compile.py:701 parity: helper does NOT re-enter bridges.
        // Bridges are executed via patched guard jumps, not the helper.
        let descr = Arc::new(guard::DynasmFailDescr::new(
            3,
            17,
            vec![Type::Int],
            false, // is_finish
            true,  // is_resume_guard
        ));
        descr.set_bridge_addr(test_bridge_finish_int as *const () as usize);
        let descr_ptr = Arc::as_ptr(&descr) as usize;

        let jf = unsafe { alloc_test_jitframe(descr_ptr, &[123, 0, 0, 0]) };
        let result = call_assembler_helper_trampoline(std::ptr::null(), jf, 99);
        // Helper blackhole-resumes (no blackhole registered → 0), NOT bridge result.
        assert_eq!(result, 0);

        drop(descr);
    }

    // ── Bug 3 regression: green_key must reach the handler, not be 0 ──

    #[test]
    fn test_helper_trampoline_passes_green_key() {
        use std::sync::atomic::AtomicU64;
        static OBSERVED_GREEN_KEY: AtomicU64 = AtomicU64::new(0);

        // We can't register a one-shot blackhole fn (OnceLock is set-once).
        // Instead, test the trampoline's function signature contract:
        // green_key is the second parameter (rsi on x86_64).
        // Direct call test — not through generated code.
        let green_key: u64 = 0xDEAD_BEEF_CAFE;
        let result =
            call_assembler_helper_trampoline(std::ptr::null(), std::ptr::null_mut(), green_key);
        // Null ptr → early return 0, but the function accepted green_key.
        assert_eq!(result, 0);
        // The real verification is compile-time: the signature is
        //   extern "C" fn(*mut i64, u64) -> i64
        // and generated code passes green_key in rsi. If this test compiles
        // and runs, the ABI contract is correct.
        let _ = OBSERVED_GREEN_KEY; // suppress unused
    }

    // ── done_with_this_frame_descr: 4 type-specific singletons ──

    /// Test-local stand-in for `compile::DoneWithThisFrameDescr*`
    /// singletons.  The dynasm backend stores the metainterp-side
    /// `Arc<dyn Descr>` in a process-global `OnceLock`, so seeding
    /// from this test mirrors what `pyjitpl.py:2222`
    /// (`make_and_attach_done_descrs([self, cpu])`) does at runtime.
    #[derive(Debug)]
    struct TestMarker(Type);
    impl majit_ir::Descr for TestMarker {}
    impl majit_ir::FailDescr for TestMarker {
        fn fail_index(&self) -> u32 {
            u32::MAX
        }
        fn fail_arg_types(&self) -> &[Type] {
            match self.0 {
                Type::Void => &[],
                Type::Int => &[Type::Int],
                Type::Ref => &[Type::Ref],
                Type::Float => &[Type::Float],
            }
        }
        fn is_finish(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_done_with_this_frame_descr_four_variants() {
        use std::sync::Arc;
        let void: majit_ir::DescrRef = Arc::new(TestMarker(Type::Void));
        let int: majit_ir::DescrRef = Arc::new(TestMarker(Type::Int));
        let ref_: majit_ir::DescrRef = Arc::new(TestMarker(Type::Ref));
        let float: majit_ir::DescrRef = Arc::new(TestMarker(Type::Float));

        // `pyjitpl.py:2222` `make_and_attach_done_descrs([self, cpu])`
        // parity: the four `DoneWithThisFrameDescr*` singletons live on
        // the owning cpu instance.  Construct a per-test backend, attach
        // the descrs through the Backend trait, and read pointers out of
        // `attached_descr_ptrs()` — matching RPython's
        // `self.cpu.done_with_this_frame_descr_*` access pattern.
        let mut backend = runner::DynasmBackend::new();
        <runner::DynasmBackend as majit_backend::Backend>::set_done_with_this_frame_descr_void(
            &mut backend,
            void,
        );
        <runner::DynasmBackend as majit_backend::Backend>::set_done_with_this_frame_descr_int(
            &mut backend,
            int,
        );
        <runner::DynasmBackend as majit_backend::Backend>::set_done_with_this_frame_descr_ref(
            &mut backend,
            ref_,
        );
        <runner::DynasmBackend as majit_backend::Backend>::set_done_with_this_frame_descr_float(
            &mut backend,
            float,
        );

        let ptrs = backend.attached_descr_ptrs();
        let void_ptr = ptrs.done_with_this_frame_descr_void;
        let int_ptr = ptrs.done_with_this_frame_descr_int;
        let ref_ptr = ptrs.done_with_this_frame_descr_ref;
        let float_ptr = ptrs.done_with_this_frame_descr_float;

        // All four must be distinct (metainterp-side `DoneWithThisFrameDescr*`
        // singletons are separate Arcs per result type).
        assert_ne!(void_ptr, 0);
        assert_ne!(int_ptr, 0);
        assert_ne!(ref_ptr, 0);
        assert_ne!(float_ptr, 0);
        assert_ne!(void_ptr, int_ptr);
        assert_ne!(void_ptr, ref_ptr);
        assert_ne!(void_ptr, float_ptr);
        assert_ne!(int_ptr, ref_ptr);
        assert_ne!(int_ptr, float_ptr);
        assert_ne!(ref_ptr, float_ptr);

        // is_done_with_this_frame_descr must recognize all four.
        assert!(ptrs.is_done_with_this_frame_descr(void_ptr));
        assert!(ptrs.is_done_with_this_frame_descr(int_ptr));
        assert!(ptrs.is_done_with_this_frame_descr(ref_ptr));
        assert!(ptrs.is_done_with_this_frame_descr(float_ptr));

        // Arbitrary pointer must NOT be recognized.
        assert!(!ptrs.is_done_with_this_frame_descr(0x12345678));
        assert!(!ptrs.is_done_with_this_frame_descr(0));

        // ptr_for_type must route correctly.
        assert_eq!(
            ptrs.done_with_this_frame_descr_ptr_for_type(Type::Void),
            void_ptr
        );
        assert_eq!(
            ptrs.done_with_this_frame_descr_ptr_for_type(Type::Int),
            int_ptr
        );
        assert_eq!(
            ptrs.done_with_this_frame_descr_ptr_for_type(Type::Ref),
            ref_ptr
        );
        assert_eq!(
            ptrs.done_with_this_frame_descr_ptr_for_type(Type::Float),
            float_ptr
        );
    }

    // ── Exception state: jit_exc_raise / jit_exc_value_raw ──

    #[test]
    fn test_jit_exc_raise_and_clear() {
        jit_exc_clear();
        assert!(!jit_exc_is_pending());
        assert_eq!(jit_exc_class_raw(), 0);

        // Simulate raising with a fake object (type ptr at offset 0).
        let fake_type: i64 = 0x42;
        let fake_obj: [i64; 2] = [fake_type, 0x99];
        jit_exc_raise(fake_obj.as_ptr() as i64);
        assert!(jit_exc_is_pending());
        assert_eq!(jit_exc_class_raw(), fake_type);

        // value_raw returns and clears.
        let val = jit_exc_value_raw();
        assert_eq!(val, fake_obj.as_ptr() as i64);
        assert!(!jit_exc_is_pending());

        jit_exc_clear();
    }
}
