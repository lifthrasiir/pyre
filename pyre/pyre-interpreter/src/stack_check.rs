//! Stack overflow protection — RPython rstack parity.
//!
//! Port of `rpython/rlib/rstack.py:42 stack_check()` +
//! `rpython/translator/c/src/stack.c:25 LL_stack_too_big_slowpath`. The
//! four primary entrypoints match RPython's `LL_stack_*` C ABI so the
//! JIT backends can call them from emitted code:
//!
//!   * [`pyre_stack_get_end`] — captured stack base (high address; stacks
//!     grow downward). Raw read, returns 0 when never captured.
//!   * [`pyre_stack_get_length`] — effective stack budget in bytes.
//!   * [`pyre_stack_set_length_fraction`] — multiply `MAX_STACK_SIZE` by
//!     `frac` and store the result as the new effective length.
//!   * [`pyre_stack_too_big_slowpath`] — four-case slow path: first-time
//!     capture, thread-switch cache refresh, stack-underflow base
//!     revision, real overflow.
//!
//! The JIT-addressable state lives in the global [`PYRE_STACKTOOBIG`]
//! matching RPython's `rpy_stacktoobig_t` struct (`stack.h:23`), so
//! `pyre_stack_get_end_adr` / `pyre_stack_get_length_adr` can hand out
//! stable addresses for backend-emitted inline probes. The source of
//! truth for `stack_end` however is a thread-local mirror
//! [`TL_STACK_END`] matching `pypy_threadlocal_s::stack_end`
//! (`stack.c:30`); the slow path reads the TLS value first and refreshes
//! the global cache from it on a thread switch (`stack.c:49`), then
//! writes both when the base is newly captured or revised
//! (`stack.c:63-64`). Pyre is single-threaded today, but keeping the
//! TLS-mirror + global-cache shape matches upstream byte-for-byte.
//!
//! `sys.setrecursionlimit(N)` calls `pyre_stack_set_length_fraction(N *
//! 0.001)` and stores `N` as the visible recursion limit. The default
//! recursion limit is 1000, matching CPython/PyPy.
//!
//! [`stack_check`] is the Rust-side inline fast path. The JIT prologue
//! emits the equivalent inline sequence directly (see
//! `rpython/jit/backend/x86/assembler.py:1080
//! _call_header_with_stack_check`).

use std::sync::atomic::{AtomicI64, AtomicU8, AtomicUsize, Ordering};

use pyre_object::excobject::{ExcKind, w_exception_new};

use crate::PyError;

/// Default recursion limit, re-exported from `crate::module::sys::state`
/// where the visible limit actually lives. The storage is owned by the
/// sys module (matching `space.sys.recursionlimit` in
/// `pypy/module/sys/moduledef.py:25`); `stack_check` only reads and
/// writes it via the accessor functions.
pub use crate::module::sys::state::{DEFAULT_RECURSION_LIMIT, MAX_RECURSION_LIMIT};

/// rpython/translator/c/src/stack.h:8-19 `MAX_STACK_SIZE` parity.
/// RPython hard-codes `3 << 18 = 768 KB` for most architectures and
/// `11 << 18 = 2.8 MB` for `__powerpc__` / `__PPC__` / `__s390x__`
/// (where stack frames are significantly larger). Pyre inherits that
/// architecture-specific default unchanged so
/// `_stack_set_length_fraction(frac) * MAX_STACK_SIZE` produces a
/// byte budget identical to an equivalent PyPy build.
///
/// Note that Rust interpreter frames are larger than RPython's
/// translated-C frames, so the default recursion budget will exhaust
/// sooner (in terms of Python-level call depth) than CPython or
/// translated PyPy at the same `sys.setrecursionlimit()`. User
/// programs that expect CPython-style depth should raise the
/// recursion limit accordingly; the budget formula itself now
/// matches PyPy byte-for-byte.
#[cfg(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "s390x"
))]
pub const MAX_STACK_SIZE: usize = 11 << 18;
#[cfg(not(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "s390x"
)))]
pub const MAX_STACK_SIZE: usize = 3 << 18;

/// rpython/translator/c/src/stack.h:23-27 `rpy_stacktoobig_t` parity.
///
/// ```c
/// typedef struct {
///     char *stack_end;
///     Signed stack_length;
///     char report_error;
/// } rpy_stacktoobig_t;
/// ```
///
/// The field order and `#[repr(C)]` layout exactly match the C struct,
/// so an external tool that knows `rpy_stacktoobig_t` can reinterpret
/// `PYRE_STACKTOOBIG` byte-for-byte. Individual fields use
/// `AtomicUsize` / `AtomicU8` so safe Rust can load and store without
/// `unsafe`; the generated machine code (with `Ordering::Relaxed`) is
/// identical to a plain memory access on aligned naturally-sized
/// fields, matching the plain C loads/stores `rpy_stacktoobig` sees.
#[repr(C)]
pub struct PyreStackTooBig {
    pub stack_end: AtomicUsize,
    pub stack_length: AtomicUsize,
    pub report_error: AtomicU8,
}

/// rpython/translator/c/src/stack.c:14-18 `rpy_stacktoobig` parity.
/// Globally-addressable backing store for the fast-path stack check.
/// JIT backends take `&raw const PYRE_STACKTOOBIG.stack_end` /
/// `&raw const PYRE_STACKTOOBIG.stack_length` via
/// [`pyre_stack_get_end_adr`] / [`pyre_stack_get_length_adr`] and emit
/// inline `MOV [endaddr]; SUB rsp; CMP [lengthaddr]` probes in the
/// compiled prologue (see `rpython/jit/backend/x86/assembler.py:1085`).
///
/// This is the global cache only; the authoritative `stack_end` lives
/// in the thread-local [`TL_STACK_END`] (stack.c:30-40). The inline
/// probe reads the global to avoid the cost of a TLS load per call,
/// and the slow path reconciles the two (stack.c:49, 63-64).
#[unsafe(no_mangle)]
pub static PYRE_STACKTOOBIG: PyreStackTooBig = PyreStackTooBig {
    stack_end: AtomicUsize::new(0),
    stack_length: AtomicUsize::new(MAX_STACK_SIZE),
    report_error: AtomicU8::new(1),
};

thread_local! {
    /// `stack.c:30` `struct pypy_threadlocal_s *tl1; baseptr = tl1->stack_end;`
    /// parity — per-thread mirror of the captured stack base, source
    /// of truth for `LL_stack_too_big_slowpath`. The global
    /// `PYRE_STACKTOOBIG.stack_end` is a cache refreshed from this
    /// value on a thread switch (stack.c:49) and written alongside
    /// the TLS on first-time capture / underflow revision
    /// (stack.c:63-64).
    static TL_STACK_END: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// The visible recursion limit lives in `crate::module::sys::state`
// (`space.sys.recursionlimit` parity, `pypy/module/sys/vm.py:96`);
// `stack_check` forwards reads/writes through
// `crate::module::sys::state::{recursion_limit, set_recursion_limit}`.

/// Pending Python exception produced by a JIT prologue stack check.
/// When the backend slowpath wrapper detects a real stack overflow
/// it constructs a fresh `RecursionError` instance and stores its
/// pointer here; the JIT glue drains this slot at every backend
/// boundary and raises the exception in the interpreter —
/// `rstack.stack_check_slowpath → _StackOverflow` parity via the
/// `pos_exception()` channel.
///
/// A value of 0 means "no pending overflow". Stored as an
/// `AtomicI64` so it can be read and written from the slowpath
/// wrapper (called from compiled code via `extern "C"`) and the
/// glue without taking locks. `swap(0, Relaxed)` drains atomically.
static JIT_PENDING_EXCEPTION: AtomicI64 = AtomicI64::new(0);

/// Read an approximation of the current stack pointer by taking the
/// address of a stack-allocated local. The exact value depends on the
/// caller's frame layout but is monotonically decreasing as recursion
/// deepens, which is all `stack_check` needs. Equivalent to RPython's
/// `llop.stack_current()`.
#[inline(always)]
fn current_sp() -> usize {
    let probe: usize = 0;
    &probe as *const usize as usize
}

/// Reset the captured stack base to the current SP.
///
/// Call this once at the outermost interpreter entry (e.g. just
/// before running a Python program) so subsequent `stack_check`
/// calls measure depth relative to that point. If the base is never
/// reset explicitly, the first `stack_check` call lets
/// [`pyre_stack_too_big_slowpath`] capture it lazily (first-time path
/// in `stack.c:42-44`).
pub fn reset_stack_base() {
    // stack.c:63-64 parity: update both the per-thread mirror and
    // the global cache. TL_STACK_END is the source of truth; the
    // global is refreshed so the inline fast-path probe sees the
    // new value without a TLS load.
    let sp = current_sp();
    TL_STACK_END.with(|c| c.set(sp));
    PYRE_STACKTOOBIG.stack_end.store(sp, Ordering::Relaxed);
}

/// Test-only helper that plants a synthetic `stack_end` into both the
/// TLS mirror (stack.c:40 source of truth) and the global cache
/// (stack.c:14,49 inline-probe visible). The slowpath reads the TLS,
/// so tests that want to force an overflow decision must write both —
/// writing the global alone is silently overridden by the TLS load
/// inside `pyre_stack_too_big_slowpath`.
pub fn plant_stack_end_for_tests(value: usize) {
    TL_STACK_END.with(|c| c.set(value));
    PYRE_STACKTOOBIG.stack_end.store(value, Ordering::Relaxed);
}

/// rpython/translator/c/src/stack.h:37 `LL_stack_get_end` parity.
/// Raw read of `rpy_stacktoobig.stack_end`. Returns `0` when the base
/// has never been captured — the fast-path compare will miss on the
/// `end.wrapping_sub(current) > length` condition and dispatch to
/// [`pyre_stack_too_big_slowpath`], which performs first-time capture.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_get_end() -> usize {
    PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed)
}

/// rpython/translator/c/src/stack.h:38 `LL_stack_get_length` parity.
/// Raw read of `rpy_stacktoobig.stack_length`.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_get_length() -> usize {
    PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed)
}

/// rpython/translator/c/src/stack.h:39 `LL_stack_get_end_adr` parity.
/// Returns the stable address of `PYRE_STACKTOOBIG.stack_end` for the
/// JIT backend to emit inline `MOV reg, [endaddr]` loads. Mirrors
/// `rpython/rlib/rstack.py:32 _stack_get_end_adr`.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_get_end_adr() -> usize {
    &raw const PYRE_STACKTOOBIG.stack_end as usize
}

/// rpython/translator/c/src/stack.h:40 `LL_stack_get_length_adr` parity.
/// Returns the stable address of `PYRE_STACKTOOBIG.stack_length` for
/// the JIT backend to emit inline `CMP reg, [lengthaddr]` compares.
/// Mirrors `rpython/rlib/rstack.py:33 _stack_get_length_adr`.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_get_length_adr() -> usize {
    &raw const PYRE_STACKTOOBIG.stack_length as usize
}

/// rpython/translator/c/src/stack.c:20-23 `LL_stack_set_length_fraction`.
///
/// ```c
/// void LL_stack_set_length_fraction(double fraction) {
///     rpy_stacktoobig.stack_length = (Signed)(MAX_STACK_SIZE * fraction);
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_set_length_fraction(frac: f64) {
    let length = (MAX_STACK_SIZE as f64 * frac) as usize;
    PYRE_STACKTOOBIG
        .stack_length
        .store(length, Ordering::Relaxed);
}

/// rpython/translator/c/src/stack.c:25-66 `LL_stack_too_big_slowpath`.
///
/// Four-case logic, line-by-line parity with the C source:
///
///   * `baseptr == NULL` — first time we see this thread, fall through
///     to the "update base to current" epilogue (stack.c:42-44).
///   * `0 <= diff <= max_stack_size` — within bounds, probably just had
///     a thread switch. Refresh the global cache from the TLS mirror
///     (stack.c:49) and return `0`.
///   * `-max_stack_size <= diff < 0` — stack underflowed; revise the
///     base upward via the shared "update base to current" epilogue
///     (stack.c:52-55), return `0`.
///   * else — real stack overflow (stack.c:56-58), return
///     `PYRE_STACKTOOBIG.report_error` (which is cleared during
///     critical-code sections by [`pyre_stack_criticalcode_start`]).
///
/// The TLS mirror [`TL_STACK_END`] is the source of truth
/// (stack.c:40 `baseptr = tl1->stack_end;`); the global
/// `PYRE_STACKTOOBIG.stack_end` is a cache refreshed from it.
///
/// Returns `1` on real stack overflow (caller should raise
/// `RecursionError`), `0` otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_too_big_slowpath(current: usize) -> u8 {
    let max_stack_size = PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed);
    // stack.c:38-40 OP_THREADLOCALREF_ADDR + baseptr = tl1->stack_end.
    let baseptr = TL_STACK_END.with(|c| c.get());

    if baseptr != 0 {
        // stack.c:46 diff = baseptr - curptr  (signed). We model the
        // two signs using two unsigned wrapping subtractions — the one
        // that produces a small value indicates the actual sign.
        let diff = baseptr.wrapping_sub(current);
        if diff <= max_stack_size {
            // stack.c:47-50 within bounds, probably thread switch.
            // Refresh the global cache from the TLS source of truth.
            // TLS already holds `baseptr`, only the global cache needs
            // the store here (stack.c:49).
            PYRE_STACKTOOBIG.stack_end.store(baseptr, Ordering::Relaxed);
            return 0;
        }
        let neg_diff = current.wrapping_sub(baseptr);
        if neg_diff <= max_stack_size {
            // stack.c:52-55 stack underflow — fall through to the
            // "update base to current" epilogue below.
        } else {
            // stack.c:56-58 real stack overflow.
            return PYRE_STACKTOOBIG.report_error.load(Ordering::Relaxed);
        }
    }

    // stack.c:61-65 update the stack base pointer to the current
    // value. Write both the TLS mirror (source of truth) and the
    // global cache.
    TL_STACK_END.with(|c| c.set(current));
    PYRE_STACKTOOBIG.stack_end.store(current, Ordering::Relaxed);
    0
}

/// Backend-callable slowpath that wraps [`pyre_stack_too_big_slowpath`]
/// and places a fresh `RecursionError` instance into the pending
/// exception slot when a real overflow is detected. The dynasm
/// x86/aarch64 inline prologue probes call this via the address
/// registered with `register_stack_check_addresses`, so the exception
/// is constructed atomically with the backend's decision to exit the
/// prologue with the initial jf_ptr.
///
/// Matches `rpython/rlib/rstack.py:68-73 stack_check_slowpath`, which
/// constructs `_StackOverflow` and raises it into `pos_exception()`
/// so the assembler's `_build_stack_check_slowpath` wrapper can route
/// to `propagate_exception_path`. In pyre the "propagate" half is
/// performed by the glue via [`drain_jit_pending_exception`] on the
/// way back to the interpreter.
///
/// Returns the same 0/1 that [`pyre_stack_too_big_slowpath`] returns,
/// so the backend can still branch on the result directly.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_check_slowpath_for_backend(current: usize) -> u8 {
    let r = pyre_stack_too_big_slowpath(current);
    if r != 0 {
        let exc_obj = w_exception_new(ExcKind::RecursionError, "maximum recursion depth exceeded");
        JIT_PENDING_EXCEPTION.store(exc_obj as i64, Ordering::Relaxed);
    }
    r
}

/// Cranelift-callable one-shot probe combining the rstack.py:42 fast
/// path and the stack.c:25 slowpath into a single `extern "C"`
/// function. Cranelift's IR does not expose a "read current SP"
/// intrinsic, so it calls this helper instead of emitting the inline
/// `MOV [endaddr]; SUB sp; CMP [lengthaddr]` sequence the dynasm
/// backends use. Semantically equivalent to the inline probe + slow
/// path pair — just paid as one function call.
///
/// Returns `1` on real overflow (and stores a `RecursionError` into
/// [`JIT_PENDING_EXCEPTION`] via
/// [`pyre_stack_check_slowpath_for_backend`]), `0` otherwise. The
/// return type is `i64` rather than `u8` so Cranelift's
/// `emit_host_call(..., Some(I64))` gets an unambiguous 64-bit
/// return value in the machine register (the System V AMD64 ABI does
/// not specify the high bytes of a narrower return).
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_check_for_jit_prologue() -> i64 {
    let current = current_sp();
    let end = PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed);
    let length = PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed);
    // rstack.py:58-60 fast path.
    let ofs = end.wrapping_sub(current);
    if ofs <= length {
        return 0;
    }
    pyre_stack_check_slowpath_for_backend(current) as i64
}

/// Drain the pending JIT-prologue overflow exception, if any, and
/// convert it to `PyError`. Called by the glue at every backend
/// boundary and by the interpreter call dispatcher so a prologue-
/// detected overflow surfaces as the user-visible `RecursionError`.
///
/// Matches RPython `pos_exception()` → `propagate_exception_path`
/// unwind semantics.
#[inline]
pub fn drain_jit_pending_exception() -> Result<(), PyError> {
    let obj = JIT_PENDING_EXCEPTION.swap(0, Ordering::Relaxed);
    if obj != 0 {
        Err(unsafe { PyError::from_exc_object(obj as pyre_object::PyObjectRef) })
    } else {
        Ok(())
    }
}

/// Peek whether an overflow exception is pending without draining it.
/// Used by tests and by codegen helpers that want to know whether to
/// skip work without claiming responsibility for surfacing it.
#[inline]
pub fn is_jit_overflow_pending() -> bool {
    JIT_PENDING_EXCEPTION.load(Ordering::Relaxed) != 0
}

/// rpython/translator/c/src/stack.h:42 `LL_stack_criticalcode_start`.
/// Clears the `report_error` flag so the slowpath will not signal a
/// stack overflow during short critical-code sections.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_criticalcode_start() {
    PYRE_STACKTOOBIG.report_error.store(0, Ordering::Relaxed);
}

/// rpython/translator/c/src/stack.h:43 `LL_stack_criticalcode_stop`.
/// Re-enables the `report_error` flag so subsequent real overflows
/// raise `RecursionError` again.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_criticalcode_stop() {
    PYRE_STACKTOOBIG.report_error.store(1, Ordering::Relaxed);
}

/// pypy/module/sys/vm.py:63 `setrecursionlimit` parity. Tentatively
/// applies the new stack budget, probes `stack_check` at the current
/// depth, and rolls the budget back on failure before raising
/// `RecursionError`. Only on success does the visible recursion
/// limit get committed and the shadow-stack depth grown.
pub fn set_recursion_limit(new_limit: i32) -> Result<(), PyError> {
    if new_limit <= 0 {
        return Err(PyError::value_error("recursion limit must be positive"));
    }
    // pypy/module/sys/vm.py:86-87 silent upper bound.
    let limit = new_limit.min(MAX_RECURSION_LIMIT);
    let old_limit = get_recursion_limit();

    // pypy/module/sys/vm.py:88-90 try: _stack_set_length_fraction + stack_check
    pyre_stack_set_length_fraction(limit as f64 * 0.001);
    if stack_check().is_err() {
        // pypy/module/sys/vm.py:91-95 rollback on StackOverflow.
        pyre_stack_set_length_fraction(old_limit as f64 * 0.001);
        return Err(PyError::recursion_error(format!(
            "cannot set the recursion limit to {limit} at the recursion depth: the limit is too low"
        )));
    }

    // pypy/module/sys/vm.py:96-97 commit + grow shadow stack.
    crate::module::sys::state::set_recursion_limit(limit);
    majit_gc::shadow_stack::increase_root_stack_depth((limit as f64 * 0.001 * 163840.0) as usize);
    Ok(())
}

/// pypy/module/sys/vm.py:72 `getrecursionlimit` parity. Reads
/// `space.sys.recursionlimit` via `module::sys::state`.
pub fn get_recursion_limit() -> i32 {
    crate::module::sys::state::recursion_limit()
}

/// rpython/rlib/rstack.py:42 `stack_check()` parity.
///
/// Fast path: `ofs = r_uint(end - current); if ofs <= r_uint(length):
/// return`. On miss, dispatch to [`pyre_stack_too_big_slowpath`] which
/// applies the 4-case logic (first-time capture, thread switch,
/// underflow revision, or real overflow).
#[inline]
pub fn stack_check() -> Result<(), PyError> {
    let current = current_sp();
    let end = PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed);
    let length = PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed);
    // rstack.py:58-60 ofs = r_uint(end - current); if ofs <= r_uint(length): return
    let ofs = end.wrapping_sub(current);
    if ofs <= length {
        return Ok(());
    }
    // rstack.py:63 stack_check_slowpath(current)
    if pyre_stack_too_big_slowpath(current) != 0 {
        return Err(PyError::recursion_error("maximum recursion depth exceeded"));
    }
    Ok(())
}

/// rpython/rlib/rstack.py:75-90 `stack_almost_full` parity.
///
/// ```python
/// def stack_almost_full():
///     current = llop.stack_current(Signed)
///     end = _stack_get_end()
///     length = 15 * (r_uint(_stack_get_length()) >> 4)
///     ofs = r_uint(end - current)
///     if ofs <= length:
///         return False    # fine
///     else:
///         _stack_too_big_slowpath(current)   # may update stack_end
///         end = _stack_get_end()
///         ofs = r_uint(end - current)
///         return ofs > length
/// ```
///
/// Returns `true` if the stack is more than 15/16ths full, computed
/// against the `sys.setrecursionlimit`-driven [`PYRE_STACKTOOBIG`]
/// budget rather than the OS-level thread stack. Callers use this to
/// gate bridge compilation (`compile.py:702-703`) and GC hooks
/// (`warmstate.py:430`). The slowpath is invoked when the cached
/// `stack_end` is stale so it can refresh, matching the two-step
/// check/slowpath/re-check pattern upstream.
#[inline]
pub fn stack_almost_full() -> bool {
    // rstack.py:80 current = llop.stack_current(Signed)
    let current = current_sp();
    // rstack.py:81 end = _stack_get_end()
    let end = PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed);
    // rstack.py:82 length = 15 * (r_uint(_stack_get_length()) >> 4)
    let length_full = PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed);
    let length = 15 * (length_full >> 4);
    // rstack.py:83-84 ofs = r_uint(end - current); if ofs <= length: return False
    let ofs = end.wrapping_sub(current);
    if ofs <= length {
        return false;
    }
    // rstack.py:86-89 slowpath may update stack_end, re-read and compare.
    let _ = pyre_stack_too_big_slowpath(current);
    let end = PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed);
    let ofs = end.wrapping_sub(current);
    ofs > length
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialize access to the global `PYRE_STACKTOOBIG` so parallel
    /// test threads can't stomp on each other. Previously the state
    /// was per-thread (`thread_local!`) and tests didn't need this
    /// lock; moving to a process-wide global (for JIT addressability)
    /// introduces contention.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Reset every global back to its default — call from each test
    /// under `TEST_LOCK` so the order tests run in cannot poison
    /// shared state.
    fn reset_all() {
        reset_stack_base();
        PYRE_STACKTOOBIG
            .stack_length
            .store(MAX_STACK_SIZE, Ordering::Relaxed);
        PYRE_STACKTOOBIG.report_error.store(1, Ordering::Relaxed);
        crate::module::sys::state::reset_recursion_limit_for_tests();
        JIT_PENDING_EXCEPTION.store(0, Ordering::Relaxed);
    }

    /// Plant a synthetic `stack_end` into both the TLS mirror
    /// (source of truth — stack.c:40) and the global cache
    /// (inline-probe visible — stack.c:14,49). Tests that exercise
    /// the slowpath must write both; the slowpath reads the TLS.
    fn plant_stack_end(value: usize) {
        TL_STACK_END.with(|c| c.set(value));
        PYRE_STACKTOOBIG.stack_end.store(value, Ordering::Relaxed);
    }

    /// Acquire the test mutex, tolerating poisoning so a single
    /// previous-test panic doesn't cascade into every follow-up test.
    fn lock_tests() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn fresh_thread_does_not_overflow() {
        let _g = lock_tests();
        reset_all();
        assert!(stack_check().is_ok());
    }

    #[test]
    fn shallow_check_does_not_overflow() {
        let _g = lock_tests();
        reset_all();
        for _ in 0..1000 {
            assert!(stack_check().is_ok());
        }
    }

    #[test]
    fn forced_high_base_triggers_overflow() {
        let _g = lock_tests();
        reset_all();
        // Plant a base far above the current SP so diff > length AND
        // -diff > length — the slowpath should classify this as a
        // real stack overflow and return RecursionError.
        let above = current_sp().saturating_add(2 * MAX_STACK_SIZE);
        plant_stack_end(above);
        let err = stack_check().expect_err("expected RecursionError");
        assert_eq!(err.kind, crate::PyErrorKind::RecursionError);
        reset_all();
    }

    #[test]
    fn default_recursion_limit_matches_python() {
        let _g = lock_tests();
        reset_all();
        assert_eq!(get_recursion_limit(), DEFAULT_RECURSION_LIMIT);
        assert_eq!(pyre_stack_get_length(), MAX_STACK_SIZE);
    }

    #[test]
    fn set_recursion_limit_scales_stack_budget() {
        let _g = lock_tests();
        reset_all();
        // Halving the recursion limit halves the stack budget.
        set_recursion_limit(500).expect("500 is positive");
        assert_eq!(get_recursion_limit(), 500);
        assert!(pyre_stack_get_length() < MAX_STACK_SIZE);
        assert!(pyre_stack_get_length() >= MAX_STACK_SIZE / 2 - 4096);

        // Doubling it doubles the budget.
        set_recursion_limit(2000).expect("2000 is positive");
        assert_eq!(get_recursion_limit(), 2000);
        assert!(pyre_stack_get_length() > MAX_STACK_SIZE);

        reset_all();
    }

    #[test]
    fn set_recursion_limit_rejects_non_positive() {
        let _g = lock_tests();
        reset_all();
        let err = set_recursion_limit(0).expect_err("0 is not positive");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        let err = set_recursion_limit(-5).expect_err("negative is not positive");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        // The limit must NOT have been mutated.
        assert_eq!(get_recursion_limit(), DEFAULT_RECURSION_LIMIT);
        reset_all();
    }

    #[test]
    fn set_recursion_limit_clamps_to_silent_upper_bound() {
        let _g = lock_tests();
        reset_all();
        // pypy/module/sys/vm.py:67 — values above 1_000_000 are clamped.
        set_recursion_limit(50_000_000).expect("positive");
        assert_eq!(get_recursion_limit(), MAX_RECURSION_LIMIT);
        reset_all();
    }

    #[test]
    fn backend_slowpath_raises_into_pending_exception() {
        let _g = lock_tests();
        reset_all();
        // Plant a synthetic high base so the 4-case slowpath reports
        // a real overflow — verifies the wrapper constructs a fresh
        // RecursionError and stores it in JIT_PENDING_EXCEPTION.
        let above = current_sp().saturating_add(2 * MAX_STACK_SIZE);
        plant_stack_end(above);
        let result = pyre_stack_check_slowpath_for_backend(current_sp());
        assert_eq!(result, 1, "slowpath must signal overflow");
        assert!(
            is_jit_overflow_pending(),
            "slowpath must place exception into JIT_PENDING_EXCEPTION"
        );
        let err = drain_jit_pending_exception().expect_err("pending exception must drain");
        assert_eq!(err.kind, crate::PyErrorKind::RecursionError);
        assert!(!is_jit_overflow_pending(), "drain must clear the slot");
        assert!(
            drain_jit_pending_exception().is_ok(),
            "second drain must be Ok"
        );
        reset_all();
    }

    #[test]
    fn jit_prologue_helper_returns_0_on_fast_path() {
        let _g = lock_tests();
        reset_all();
        // Default state: stack_end = current_sp() (via reset_all), so
        // the fast path sees a small positive offset and returns 0
        // without dispatching to the slowpath.
        assert_eq!(pyre_stack_check_for_jit_prologue(), 0i64);
        assert!(!is_jit_overflow_pending());
        reset_all();
    }

    #[test]
    fn jit_prologue_helper_returns_1_on_real_overflow() {
        let _g = lock_tests();
        reset_all();
        let above = current_sp().saturating_add(2 * MAX_STACK_SIZE);
        plant_stack_end(above);
        assert_eq!(pyre_stack_check_for_jit_prologue(), 1i64);
        assert!(is_jit_overflow_pending());
        let _ = drain_jit_pending_exception();
        reset_all();
    }

    #[test]
    fn small_recursion_limit_triggers_overflow_sooner() {
        let _g = lock_tests();
        reset_all();
        // 1 recursionlimit unit = MAX_STACK_SIZE / 1000 bytes.
        // Setting limit=1 leaves only ~8 KiB of headroom — definitely
        // smaller than the synthetic high base we plant below.
        set_recursion_limit(1).expect("positive");
        let above = current_sp().saturating_add(MAX_STACK_SIZE / 100);
        plant_stack_end(above);
        let err = stack_check().expect_err("expected RecursionError");
        assert_eq!(err.kind, crate::PyErrorKind::RecursionError);
        reset_all();
    }

    #[test]
    fn slowpath_first_time_captures_base() {
        let _g = lock_tests();
        reset_all();
        // Simulate "never captured" state — stack.c:42-44 first-time path.
        plant_stack_end(0);
        let sp = current_sp();
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(overflow, 0, "first-time capture must not signal overflow");
        let captured = pyre_stack_get_end();
        assert_eq!(captured, sp, "slowpath must update stack_end to current");
        reset_all();
    }

    #[test]
    fn slowpath_thread_switch_cache_hit() {
        let _g = lock_tests();
        reset_all();
        // baseptr very close to current — within max_stack_size, slowpath
        // should classify this as stack.c:47-50 "within bounds / thread
        // switch" and return 0 without mutating stack_end.
        let sp = current_sp();
        let base = sp.saturating_add(4096);
        plant_stack_end(base);
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(overflow, 0);
        assert_eq!(
            pyre_stack_get_end(),
            base,
            "within-bounds hit must not mutate base"
        );
        reset_all();
    }

    #[test]
    fn slowpath_underflow_revises_base_upward() {
        let _g = lock_tests();
        reset_all();
        // baseptr BELOW current within max_stack_size — stack.c:52-55
        // underflow case. Slowpath should revise base upward to current
        // and return 0.
        let sp = current_sp();
        let below = sp.wrapping_sub(4096);
        plant_stack_end(below);
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(overflow, 0);
        assert_eq!(
            pyre_stack_get_end(),
            sp,
            "underflow must revise base to current"
        );
        reset_all();
    }

    #[test]
    fn slowpath_real_overflow_returns_report_error() {
        let _g = lock_tests();
        reset_all();
        let sp = current_sp();
        let above = sp.saturating_add(2 * MAX_STACK_SIZE);
        plant_stack_end(above);
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(
            overflow, 1,
            "real overflow must signal with report_error=true"
        );
        reset_all();
    }

    #[test]
    fn slowpath_criticalcode_suppresses_overflow() {
        let _g = lock_tests();
        reset_all();
        let sp = current_sp();
        let above = sp.saturating_add(2 * MAX_STACK_SIZE);
        plant_stack_end(above);
        pyre_stack_criticalcode_start();
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(
            overflow, 0,
            "criticalcode must suppress overflow signalling"
        );
        pyre_stack_criticalcode_stop();
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(overflow, 1, "stop must re-enable overflow signalling");
        reset_all();
    }

    #[test]
    fn end_and_length_addresses_are_stable() {
        let _g = lock_tests();
        // rstack.py:32-33 `_stack_get_end_adr` / `_stack_get_length_adr`
        // parity: addresses must be stable across calls (so the JIT
        // backend can embed them as imm64 operands).
        let end_adr_1 = pyre_stack_get_end_adr();
        let end_adr_2 = pyre_stack_get_end_adr();
        assert_eq!(
            end_adr_1, end_adr_2,
            "pyre_stack_get_end_adr must return a stable address"
        );
        let length_adr_1 = pyre_stack_get_length_adr();
        let length_adr_2 = pyre_stack_get_length_adr();
        assert_eq!(
            length_adr_1, length_adr_2,
            "pyre_stack_get_length_adr must return a stable address"
        );
        assert_ne!(
            end_adr_1, length_adr_1,
            "end and length must live at distinct addresses"
        );
    }

    #[test]
    fn end_and_length_addresses_reflect_layout() {
        let _g = lock_tests();
        // stack.h:23-27 `rpy_stacktoobig_t` layout: stack_end at
        // offset 0, stack_length at offset sizeof(usize). Verify pyre's
        // PyreStackTooBig matches.
        let end_adr = pyre_stack_get_end_adr();
        let length_adr = pyre_stack_get_length_adr();
        let struct_adr = &raw const PYRE_STACKTOOBIG as usize;
        assert_eq!(end_adr, struct_adr, "stack_end at offset 0");
        assert_eq!(
            length_adr,
            struct_adr + std::mem::size_of::<usize>(),
            "stack_length at offset sizeof(usize)"
        );
    }

    #[test]
    fn addresses_observe_stored_values() {
        let _g = lock_tests();
        reset_all();
        // Store via FFI, read via raw pointer — confirms the atomic
        // storage is bit-compatible with a plain usize load.
        pyre_stack_set_length_fraction(0.5);
        let length_adr = pyre_stack_get_length_adr();
        let loaded = unsafe { *(length_adr as *const usize) };
        assert_eq!(
            loaded,
            (MAX_STACK_SIZE as f64 * 0.5) as usize,
            "raw load through adr must observe the FFI store"
        );
        reset_all();
    }
}
