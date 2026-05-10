//! `majit-meta`: Meta-tracing automation layer for the majit JIT framework.
//!
//! Provides [`MetaInterp`] — a high-level JIT engine that handles the full
//! lifecycle: warm counting → tracing → optimization → compilation → execution.
//!
//! Interpreter authors only need to:
//! 1. Call [`MetaInterp::on_back_edge`] at backward jumps
//! 2. Record IR ops via [`TraceCtx`] during tracing
//! 3. Provide state extraction/restoration logic
//!
//! Everything else (constant management, FailDescr/CallDescr creation,
//! optimizer invocation, backend compilation, I/O buffering) is automated.

extern crate self as majit_metainterp;

use majit_ir::{OpRef, Type};

pub mod blackhole;
pub mod r#box;
pub mod call_descr;
pub(crate) mod compile;
mod constant_pool;
pub(crate) mod executor;
pub mod greenfield;
pub mod history;
pub mod io_buffer;
pub mod jit;
mod jit_state;
pub mod jitcode;
mod jitdriver;
pub mod jitexc;
pub mod jitframe;
pub mod jitprof;
pub mod memmgr;
pub mod opencoder;
pub mod optimize;
pub mod optimizeopt;
pub mod parity;
mod pyjitpl;
pub mod quasiimmut;
pub mod recorder;
pub mod resume;
pub mod rvmprof;
mod trace_ctx;
pub mod virtualizable;
pub mod virtualref;
pub mod walkvirtual;
pub mod warmstate;

pub use call_descr::{
    CANNOT_RAISE_NO_HEAP_EFFECT_INFO, ELIDABLE_CANNOT_RAISE_EFFECT_INFO, ELIDABLE_EFFECT_INFO,
    ELIDABLE_OR_MEMERROR_EFFECT_INFO, EffectInfoSlot, INT_PY_DIV_EFFECT_INFO,
    INT_PY_MOD_EFFECT_INFO, LOOPINVARIANT_EFFECT_INFO, cannot_raise_effect_info,
    default_effect_info, effect_info_for_slot, make_call_assembler_descr,
    make_call_assembler_descr_by_number, make_call_assembler_descr_with_vable,
    make_call_assembler_descr_with_vable_by_number, make_call_descr,
    make_call_descr_from_target_slot, make_call_descr_with_effect,
};
pub use compile::{make_fail_descr, make_fail_descr_typed, make_finish_fail_descr_typed};
pub use constant_pool::ConstantPool;
pub use io_buffer::{
    emit_commit_io, encode_decimal_i64, io_buffer_commit, io_buffer_discard, io_buffer_write,
    io_buffer_write_fmt, jit_write_number_i64, jit_write_utf8_codepoint,
};
pub use jit_state::{
    DeoptMaterializationCache, JitState, PendingFieldWriteLayout, ResidualVirtualizableSync,
    ResumeDataResult,
};
pub use jitcode::{
    BC_GOTO, JitArgKind, JitCallArg, JitCode, JitCodeBuilder, LivenessInfo, insns,
    live_slots_for_state_field_jit,
};
pub use jitdriver::{DeclarativeJitDriver, JitDriver, JitDriverStaticData};
pub use majit_backend::CompiledTraceInfo;
// Re-export the canonical translate-side Assembler so macro-emitted
// state-field JIT setup (e.g. `__JitMeta::install_canonical_liveness`)
// can build a fresh Assembler without forcing each user crate to
// declare a `majit-translate` dependency.  The same pattern is used
// for `JitCode` / `BhDescr` re-exports above (`jitcode/mod.rs:4`).
pub use majit_translate::jit_codewriter::assembler::Assembler;
pub use parity::{TraceParityCase, assert_trace_parity, normalize_ops, normalize_trace};
pub use pyjitpl::{
    BackEdgeAction, BlackholeRunResult, BridgeRetraceResult, ClosureRuntime, CompileOutcome,
    CompiledExitLayout, CompiledTerminalExitLayout, CompiledTraceLayout, DeadFrameArtifacts,
    DetailedDriverRunOutcome, DriverRunOutcome, InlineDecision, JitCodeMachine, JitCodeRuntime,
    JitCodeSym, JitHooks, JitStats, MIFrame, MIFrameStack, MetaInterp, MetaInterpGlobalData,
    MetaInterpStaticData, RawCompileResult, StandaloneFrameStack, build_state_field_snapshot,
    consume_observed_float_call, consume_observed_int_call, consume_observed_ref_call,
    consume_observed_void_call, counters, observer_arg_to_i64, observer_i64_to_value,
    trace_jitcode, trace_jitcode_observer, trace_jitcode_observer_with_args,
    trace_jitcode_with_args,
};
pub use quasiimmut::QuasiImmut;
pub use trace_ctx::{MergePoint, TraceCtx};

/// Whether `MAJIT_LOG` is set, cached at first access.
/// Compute green key from code pointer and PC.
/// Must use the same hash as the front-end's make_green_key.
pub fn green_key_from_code_ptr(code_ptr: usize, pc: usize) -> u64 {
    (code_ptr as u64).wrapping_mul(1000003) ^ (pc as u64)
}

pub fn majit_log_enabled() -> bool {
    std::env::var_os("MAJIT_LOG").is_some()
}

/// Result of tracing a single instruction.
///
/// Returned by the interpreter's `trace_instruction()` function
/// to indicate what the framework should do next.
#[derive(Debug)]
pub enum TraceAction {
    /// Continue tracing the next instruction.
    Continue,
    /// reached_loop_header() compiled the current trace into an existing
    /// target and tracing must stop immediately.
    ///
    /// RPython parity: pyjitpl.py says compile_trace() "raises in case it
    /// works". pyre surfaces that control-flow edge explicitly.
    CompileTrace,
    /// Close the loop (back-edge to header detected).
    CloseLoop,
    /// Close the loop with explicit jump arguments supplied by the tracer.
    ///
    /// RPython parity: the tracer can also pass the explicit loop-header PC
    /// (the backward-jump target / reached loop header).  This lets the
    /// tracing context retarget its green key from the true merge point,
    /// instead of trying to recover it later from virtualizable state.
    CloseLoopWithArgs {
        jump_args: Vec<OpRef>,
        loop_header_pc: Option<usize>,
    },
    /// Finish the trace with terminal output values.
    ///
    /// `exit_with_exception = true` maps to
    /// `pyjitpl.py:3238 MetaInterp.compile_exit_frame_with_exception` —
    /// the FINISH uses `sd.exit_frame_with_exception_descr_ref` and the
    /// classifier routes to `JitException::ExitFrameWithExceptionRef`.
    /// `false` maps to
    /// `pyjitpl.py:3198 MetaInterp.compile_done_with_this_frame` —
    /// FINISH uses `sd.done_with_this_frame_descr_<kind>`.
    Finish {
        finish_args: Vec<OpRef>,
        finish_arg_types: Vec<Type>,
        exit_with_exception: bool,
    },
    /// Close and compile a segmented loop (force_finish_trace).
    /// pyjitpl.py:1622 _create_segmented_trace_and_blackhole parity.
    /// The trace has GUARD_ALWAYS_FAILS + unreachable FINISH appended.
    /// compile_simple_loop inserts a LABEL at entry for bridge attachment.
    SegmentedLoop,
    /// Abort the current trace (recoverable — may retry later).
    Abort,
    /// Abort the current trace permanently (never trace this location again).
    AbortPermanent,
}

/// Marker macro for the tracing merge point.
///
/// When used with `#[jit_interp]`, this is replaced with `driver.merge_point(...)`.
/// When used standalone, this is a no-op (interpreter runs without tracing).
#[macro_export]
macro_rules! jit_merge_point {
    () => {};
    ($($tt:tt)*) => {};
}

/// Marker macro for the back-edge entry point.
///
/// When used with `#[jit_interp]`, this is replaced with `driver.back_edge(...)`.
/// When used standalone, this is a no-op.
#[macro_export]
macro_rules! can_enter_jit {
    ($($tt:tt)*) => {};
}

/// Assure the JIT that `func(args...)` will produce `result`.
/// `func` must be an elidable function.
///
/// rlib/jit.py:1224 — `record_known_result(result, func, *args)`
///
/// At runtime (non-JIT), verifies `func(args) == result` (debug builds).
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `record_known_result_{i|r}` opcode with func and args visible
/// as separate operands — matching RPython's rtyper decomposition.
///
/// Usage: `record_known_result!(result, my_elidable_fn, arg1, arg2)`
#[macro_export]
macro_rules! record_known_result {
    ($result:expr, $func:path $(, $arg:expr)*) => {
        // rlib/jit.py:1229-1232 — untranslated consistency check
        debug_assert_eq!(
            $func($($arg),*), $result,
            "record_known_result: func(...) != result"
        );
    };
}

/// rlib/jit.py:1301 — `conditional_call(condition, function, *args)`
///
/// At runtime: `if condition { function(args...) }`.
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `conditional_call_ir_v` opcode with func and args as
/// separate operands — matching RPython's ConditionalCallEntry decomposition.
///
/// Usage: `conditional_call!(cond, my_func, arg1, arg2)`
#[macro_export]
macro_rules! conditional_call {
    ($condition:expr, $func:path $(, $arg:expr)*) => {
        if $condition {
            $func($($arg),*);
        }
    };
}

/// rlib/jit.py:1322 — `conditional_call_elidable(value, function, *args)`
///
/// At runtime: `if value is falsy { value = function(args...) }; return value`.
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `conditional_call_value_ir_{i|r}` opcode with func and args as
/// separate operands.
///
/// Usage: `let v = conditional_call_elidable!(cached, compute_fn, arg1, arg2)`
#[macro_export]
macro_rules! conditional_call_elidable {
    ($value:expr, $func:path $(, $arg:expr)*) => {{
        let __val = $value;
        if __val == 0 {
            $func($($arg),*)
        } else {
            __val
        }
    }};
}

/// Hash a green key from i64 slice values, all-Int convention.
///
/// Uses the same algorithm as [`GreenKey::hash_u64`](majit_ir::GreenKey::hash_u64),
/// so callers can compute a key hash without constructing a full `GreenKey`.
/// warmstate.py:584-593 `JitCell.get_uhash` — Int-only path.
///
/// Callers that have non-Int greens (Float / Ref) must use
/// [`green_key_hash_typed`] instead; the per-type
/// `equal_whatever`/`hash_whatever` differs from the Int default and a
/// bare-i64 hash would collide with an Int-typed key carrying the same bits.
#[inline]
pub fn green_key_hash(values: &[i64]) -> u64 {
    majit_ir::GreenKey::new(values.to_vec()).hash_u64()
}

/// Hash a green key from `(i64 bits, GreenType)` slices.
///
/// `warmstate.py:575 _green_args_spec` keys per-type
/// `equal_whatever`/`hash_whatever` off the green's lltype, so a Float
/// green hashes as `f64::from_bits(bits)`-aware and a Ref green hashes
/// as identity over the pointer bits.  Mirrors the typed schema that
/// `#[jit_interp]` macro-emitted code now produces via
/// `GreenKey::with_types`.
#[inline]
pub fn green_key_hash_typed(values: &[i64], types: &[majit_ir::GreenType]) -> u64 {
    debug_assert_eq!(values.len(), types.len());
    majit_ir::GreenKey::with_types(values.to_vec(), types.to_vec()).hash_u64()
}

// ── we_are_jitted / JIT mode flag ──
// Re-exported from majit-codegen so both meta and backend can access it.
pub use majit_backend::{JittedGuard, set_jitted, we_are_jitted};

// ── rstack criticalcode hooks ──
// rpython/translator/c/src/stack.h:42-43 LL_stack_criticalcode_start/stop.
// Used by blackhole_from_resumedata / handle_async_forcing /
// initialize_state_from_guard_failure to suppress StackOverflow during
// critical sections that would leave virtual references dangling.
//
// The actual implementation lives in pyre-interpreter (the interpreter
// owns the rpy_stacktoobig struct). majit-metainterp cannot depend on
// pyre-interpreter directly — pyre depends on majit, not the other way
// — so the interpreter registers the two hooks at startup.
use std::sync::OnceLock;

static CRITICALCODE_START_FN: OnceLock<fn()> = OnceLock::new();
static CRITICALCODE_STOP_FN: OnceLock<fn()> = OnceLock::new();
static STACK_ALMOST_FULL_FN: OnceLock<fn() -> bool> = OnceLock::new();

/// Register the `_stack_criticalcode_start` / `_stack_criticalcode_stop`
/// hooks the interpreter implements. Called once at JIT install time.
pub fn register_criticalcode_hooks(start: fn(), stop: fn()) {
    let _ = CRITICALCODE_START_FN.set(start);
    let _ = CRITICALCODE_STOP_FN.set(stop);
}

/// Register the `rstack.stack_almost_full` hook the interpreter
/// implements against its `PYRE_STACKTOOBIG` budget. Called once at
/// JIT install time. When no hook is registered, [`stack_almost_full`]
/// returns `false` — matching RPython's untranslated fallback in
/// `rpython/rlib/rstack.py:76-77`.
pub fn register_stack_almost_full_hook(f: fn() -> bool) {
    let _ = STACK_ALMOST_FULL_FN.set(f);
}

/// rpython/rlib/rstack.py:75-90 `stack_almost_full`. Returns `true` if
/// the stack is more than 15/16ths full against the recursion-limit
/// budget. Dispatches to the interpreter-registered hook; in tests or
/// standalone binaries without the interpreter's stack-check layer,
/// returns `false` (rstack.py:76-77 `if not we_are_translated: return
/// False`).
#[inline]
pub fn stack_almost_full() -> bool {
    if let Some(f) = STACK_ALMOST_FULL_FN.get() {
        f()
    } else {
        false
    }
}

/// rpython/translator/c/src/stack.h:42 `LL_stack_criticalcode_start`.
/// No-op if the hook is not registered (tests / standalone binaries
/// that don't install the interpreter's stack-check layer).
#[inline]
pub fn criticalcode_start() {
    if let Some(f) = CRITICALCODE_START_FN.get() {
        f();
    }
}

/// rpython/translator/c/src/stack.h:43 `LL_stack_criticalcode_stop`.
#[inline]
pub fn criticalcode_stop() {
    if let Some(f) = CRITICALCODE_STOP_FN.get() {
        f();
    }
}

/// RAII guard wrapping [`criticalcode_start`] / [`criticalcode_stop`].
///
/// RPython's `rstack._stack_criticalcode_start()` uses try/finally to
/// guarantee the matching `_stop()` runs on every exit path (including
/// exceptions). Rust's equivalent is `Drop`: this guard calls
/// `criticalcode_stop()` in its destructor so ordinary returns,
/// `?`-propagated errors, and `panic!` unwind all re-enable the
/// `report_error` flag. Matches rpython/jit/metainterp/resume.py:1315
/// + rpython/jit/metainterp/pyjitpl.py:3281 +
/// rpython/jit/metainterp/compile.py:976 `try/finally` semantics.
pub struct CriticalCodeGuard {
    _private: (),
}

impl CriticalCodeGuard {
    /// Enter a critical section. The returned guard must be held for
    /// the duration of the section; dropping it re-enables stack-
    /// overflow reporting, even if the drop is triggered by panic
    /// unwinding.
    #[inline]
    #[must_use = "CriticalCodeGuard re-enables stack overflow reporting only on drop — binding it to `_` drops it immediately, defeating the guard"]
    pub fn enter() -> Self {
        criticalcode_start();
        CriticalCodeGuard { _private: () }
    }
}

impl Drop for CriticalCodeGuard {
    #[inline]
    fn drop(&mut self) {
        criticalcode_stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn green_key_hash_deterministic() {
        let a = green_key_hash(&[10, 20]);
        let b = green_key_hash(&[10, 20]);
        assert_eq!(a, b);
    }

    #[test]
    fn green_key_hash_different_values() {
        let a = green_key_hash(&[10, 20]);
        let b = green_key_hash(&[10, 21]);
        assert_ne!(a, b);
    }

    #[test]
    fn green_key_hash_matches_green_key() {
        let hash = green_key_hash(&[42, 7]);
        let gk = majit_ir::GreenKey::new(vec![42, 7]);
        assert_eq!(hash, gk.hash_u64());
    }

    #[test]
    fn green_key_hash_typed_diverges_from_all_int_for_float_greens() {
        let bits = (3.14f64).to_bits() as i64;
        let untyped = green_key_hash(&[bits]);
        let typed = green_key_hash_typed(&[bits], &[majit_ir::GreenType::Float]);
        // hash_whatever(Float, bits) vs hash_whatever(Int, bits) — distinct
        // per `warmstate.py:566 _green_args_spec` per-type lookup.
        assert_ne!(
            untyped, typed,
            "Float-typed hash must not collide with Int-typed hash on the same bits",
        );
    }

    #[test]
    fn green_key_hash_typed_matches_with_types() {
        let bits = (3.14f64).to_bits() as i64;
        let hash = green_key_hash_typed(
            &[bits, 42],
            &[majit_ir::GreenType::Float, majit_ir::GreenType::Int],
        );
        let gk = majit_ir::GreenKey::with_types(
            vec![bits, 42],
            vec![majit_ir::GreenType::Float, majit_ir::GreenType::Int],
        );
        assert_eq!(hash, gk.hash_u64());
    }
}
pub mod resumecode;
