//! `specialize_legacy_graph` cutover wrapper.
//!
//! Drives the real `RPythonTyper::specialize` against a pyre
//! `model::FunctionGraph` by way of the
//! [`crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace`]
//! adapter, then projects each per-`Variable` `LowLevelType` back to a
//! `ConcreteType` keyed by the original pyre `ValueId`.
//!
//! ## Why this file is in `translator/rtyper/`
//!
//! The legacy algorithms live alongside this file as
//! `translator/rtyper/legacy_{annotator,resolve,pipeline}.rs`.  This
//! cutover module is the long-lived bridge: the dual-gate (Slice 4),
//! the default flip (Slice 5), and the prod migration (Slice 6) all
//! call into the entry points defined here.  The `flowspace_adapter`
//! sibling continues bridging pyre's surface-DSL
//! `model::FunctionGraph` to the RPython `flowspace::FunctionGraph`
//! shape the rtyper consumes, until pyre's
//! `parse → front → SemanticProgram` chain learns to emit
//! `flowspace::FunctionGraph` directly.
//!
//! ## Slice 2 scope
//!
//! - Build a `FlowspaceAdapterOutput` via the Slice 1c adapter.
//! - Construct a fresh `RPythonAnnotator`; bypass `build_types`
//!   (pyre's surface DSL has no `HostObject` to feed it) by populating
//!   `annotator.annotated` and `annotator.all_blocks` directly with
//!   the adapter's blocks. The annotation shells from Slice 1a are
//!   already attached to each `Variable.annotation`, which is what
//!   `RPythonTyper.bindingrepr` reads.
//! - Construct an `RPythonTyper` and call `specialize(true)` —
//!   `dont_simplify_again=true` because pyre's legacy graph is already
//!   in simplified SSA shape; running the simplify pass would attempt
//!   to call into bookkeeper machinery that requires
//!   `RPythonAnnotator.translator.entry_point_graph`, which we have
//!   not seeded.
//! - Walk the `value_to_var` map and project each `Variable.concretetype`
//!   to `ConcreteType` (Signed/Float/GcRef/Void/Unknown).
//!
//! ## Slice 4 dual-gate readiness
//!
//! Today this function still fails on graphs containing Slice 1b
//! followups that are not yet ported (`Call`, `FieldRead`,
//! `ArrayRead`, ...). The Slice 4 anchor corpus will surface which
//! followup is the next priority. `Ref`-typed operands now route
//! through `valuetype_to_someshell(Ref) → SomeInstance(classdef=None)`
//! (`jit_codewriter/annotation_state.rs:69`), so the rtyper picks
//! `getinstancerepr(rtyper, None, Gc) → InstanceRepr::new_rootinstance
//! → Ptr(GcStruct(OBJECT))` and the projection collapses to
//! `ConcreteType::GcRef` matching the legacy resolver — the previous
//! `SomeObject` blocker is closed.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::model::SomeValue;
use crate::flowspace::argument::Signature;
use crate::flowspace::model::{ConstValue, Constant, GraphFunc, GraphRef, Variable};
use crate::flowspace::pygraph::PyGraph;
use crate::front;
use crate::jit_codewriter::annotation_state::AnnotationState;
use crate::jit_codewriter::type_state::{ConcreteType, TypeResolutionState};
use crate::model::{FunctionGraph as LegacyGraph, ValueId, ValueType};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::flowspace_adapter::{
    FlowspaceAdapterOutput, ValueIdToVariable, function_graph_to_flowspace,
};
use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType};
use crate::translator::rtyper::pyre_call_registry::{
    FunctionPathKey, PyreCallRegistry, PyreFunctionEntry,
};
use crate::translator::rtyper::rtyper::RPythonTyper;

/// Project a post-`specialize` `LowLevelType` back to the legacy
/// `ConcreteType` bucket the codewriter consumes (Signed / Float /
/// GcRef / Void).
///
/// Mapping follows RPython's JIT `history.getkind(TYPE)`
/// (`rpython/jit/metainterp/history.py:46-71`):
/// `Void -> void`, primitive float -> float, `SingleFloat -> int`,
/// machine-word primitives and raw pointers -> int, GC pointers -> ref.
/// `SignedLongLong` / `UnsignedLongLong` follow PyPy's target-size
/// check: they are float kind only when they are wider than `Signed`
/// (32-bit targets); on 64-bit targets they collapse to int kind.
/// Unsupported container/non-pointer shapes return
/// `Err(TyperError::missing_rtype_operation(..))`, mirroring
/// `history.py:70 raise NotImplementedError("type %s not supported")`
/// while routing the failure through the `specialize_legacy_graph`
/// `Result<...>` channel rather than unwinding direct callers.
pub fn lowleveltype_to_concrete(ll: &LowLevelType) -> Result<ConcreteType, TyperError> {
    match ll {
        LowLevelType::Void => Ok(ConcreteType::Void),
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::Bool
        | LowLevelType::Char
        | LowLevelType::UniChar
        | LowLevelType::SingleFloat
        | LowLevelType::Address => Ok(ConcreteType::Signed),
        LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong => {
            Ok(longlong_getkind_concrete())
        }
        LowLevelType::Float => Ok(ConcreteType::Float),
        LowLevelType::Ptr(ptr) => {
            if ptr._gckind() == GcKind::Raw {
                Ok(ConcreteType::Signed)
            } else {
                Ok(ConcreteType::GcRef)
            }
        }
        LowLevelType::SignedLongLongLong
        | LowLevelType::UnsignedLongLongLong
        | LowLevelType::LongFloat
        | LowLevelType::Func(_)
        | LowLevelType::Struct(_)
        | LowLevelType::Array(_)
        | LowLevelType::FixedSizeArray(_)
        | LowLevelType::Opaque(_)
        | LowLevelType::ForwardReference(_)
        | LowLevelType::InteriorPtr(_) => Err(TyperError::missing_rtype_operation(format!(
            "lowleveltype_to_concrete: type {ll:?} not supported \
             (history.py:70 raise NotImplementedError)"
        ))),
    }
}

fn longlong_getkind_concrete() -> ConcreteType {
    if std::mem::size_of::<i64>() > std::mem::size_of::<isize>() {
        ConcreteType::Float
    } else {
        ConcreteType::Signed
    }
}

/// Seed every cached callee PyGraph's blocks into the annotator's
/// `annotated`/`all_blocks` so the rtyper's specialize walk reaches
/// them.
///
// `seed_callee_blocks` retired at Step 3 (2026-05-07).  The orthodox
// path `pycall -> recursivecall -> addpendingblock` from
// description.py:283-305 / annrpython.py:315-336 is now reachable
// through Step 2's `simple_call_SomeObject` registration; the
// program-wide `compute_at_fixpoint` loop discovers callees from
// subject call sites without any pre-seed.  The prior pre-seed was
// a workaround for the missing dispatch.

/// Run `specialize_legacy_graph` and diff against `legacy_state`.
///
/// Returns `Err(message)` when:
///
/// - the real path errors out (typer error from an unported `OpKind`
///   arm), OR
/// - a legacy-known `ValueId` is missing / `Unknown` / different in
///   the real path, OR
/// - the real path produced a definite kind for a `ValueId` the legacy
///   resolver did not resolve.
///
/// Returns `Ok(())` only when the two `TypeResolutionState` projections
/// agree on every definite kind.
///
/// Today this entry survives only as a test helper: production
/// callers go through [`dual_gate_check_with_registry`], which dropped
/// per-graph divergence comparison once Slice 12.2 narrowed legacy
/// usage to the Skip arm.  The legacy-baseline diff stays here so
/// anchor tests can keep validating the LL→Concrete projection.
#[cfg(test)]
pub(crate) fn dual_gate_check(
    legacy_graph: &LegacyGraph,
    legacy_state: &TypeResolutionState,
) -> Result<(), String> {
    // The real path goes through `RPythonTyper::specialize`, which
    // asserts internal invariants (e.g. `genop`'s "wrong level!"
    // contract that every operand carry `concretetype`). Slice 1b
    // followups occasionally surface those asserts on graphs whose
    // shape exposes an unported pyre-front idiom — those panics are
    // diagnostic for "next blocker", not crashes the dual-gate
    // should propagate. Catch the unwind so the gate uniformly
    // returns a stringified error.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        specialize_legacy_graph(legacy_graph)
    }));
    let real_state = match result {
        Ok(Ok(state)) => state,
        Ok(Err(e)) => return Err(format!("real path failed: {e}")),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<unrecognised panic payload>".to_string()
            };
            return Err(format!("real path panicked: {msg}"));
        }
    };

    let mut divergences: Vec<String> = Vec::new();

    // Diff every legacy-defined value. Once this gate is used to prove
    // cutover parity, real-path `Unknown` for a legacy-known value is a
    // coverage bug, not success.
    for (vid, legacy_kind) in &legacy_state.concrete_types {
        if *legacy_kind == ConcreteType::Unknown {
            continue;
        }
        let real_kind = real_state.get(*vid);
        if real_kind != legacy_kind {
            divergences.push(format!(
                "ValueId({}): legacy={:?}, real={:?}",
                vid.0, legacy_kind, real_kind
            ));
        }
    }
    // Asymmetry direction: real should not produce a definite kind for
    // a ValueId the legacy resolver never resolved.
    for (vid, real_kind) in &real_state.concrete_types {
        if *real_kind == ConcreteType::Unknown {
            continue;
        }
        let legacy_kind = legacy_state.get(*vid);
        if *legacy_kind == ConcreteType::Unknown {
            divergences.push(format!(
                "ValueId({}): legacy={:?}, real={:?}",
                vid.0, legacy_kind, real_kind
            ));
        }
    }

    if divergences.is_empty() {
        Ok(())
    } else {
        Err(divergences.join("; "))
    }
}

/// Outcome of a dual-gate run.  Distinguishes a real-path success
/// (`Match` — production consumes `real_state` authoritatively) from
/// a known-unported feature (`Skip(reason)` — production falls back
/// to the legacy walker for the affected graph).
///
/// `Match` carries the real path's `TypeResolutionState` so
/// production callers consume it directly.  Slice 10A inverted the
/// dependency direction: real path is the authoritative producer;
/// legacy is the fallback for Skip-classified graphs.  Slice 12.2
/// retired the per-graph divergence comparison against legacy
/// (`cutover.rs:312`); test-time anchor invariants still run
/// [`dual_gate_check`] for legacy-baseline regression checks against
/// hand-built fixtures, but production no longer compares per
/// ValueId — `Match` means the real path succeeded, full stop.
/// PyPy `codewriter.py:33` consumes the rtyper-produced graph
/// directly, with no dual-gate equivalent; pyre's `Skip` arm is the
/// transitional scaffolding that retires once every owning epic in
/// `is_known_unported`'s table converges.
#[derive(Debug)]
pub enum DualGateOutcome {
    /// Real path completed without panicking and (when the legacy
    /// walker also succeeded as defensive baseline) every legacy-
    /// known `ValueId` carried the same `ConcreteType` in the real
    /// path's projection.  Production consumes `real_state`
    /// authoritatively.  Per-`ValueId` diff against the legacy
    /// baseline runs whenever the legacy walker can produce a
    /// state — divergence routes the graph through
    /// `Skip("dual-gate divergence: ...")` so the codewriter falls
    /// back to the legacy walker output for the affected graph,
    /// matching main's pre-Slice-12.2 contract.  PyPy
    /// `codewriter.py:33` consumes the rtyper-produced graph
    /// directly with no comparison stage; the legacy-baseline diff
    /// is pyre-only scaffolding that retires once the legacy walker
    /// itself retires (Step 5 / Task #127).
    Match {
        real_state: TypeResolutionState,
        /// Per-session annotator's `Variable.annotation` SomeValue
        /// lattice nodes projected onto a `ValueId`-keyed
        /// `AnnotationState`.  Slice 12.1 reader output — what the
        /// orthodox post-jtransform merge consumes instead of the
        /// per-graph `legacy_annotator::annotate(graph)` recompute.
        real_annotations: AnnotationState,
    },
    /// Real path failed on a known-unported feature — the gate
    /// cannot validate this graph yet but the failure is *not* a
    /// ConcreteType divergence.  Categories include:
    ///
    /// - `OpKind::Call::FunctionPath { segments }` not in the
    ///   registry (cross-crate / primitive paths the production
    ///   walker doesn't reach yet).
    /// - `undefined operand ValueId` from cross-block locals
    ///   threading (Cat 3.2 deferred).
    /// - `unimplemented operation` from a not-yet-ported rtyper op
    ///   (e.g. `direct_call` for graphs the rpbc port doesn't
    ///   cover).
    /// - `checkgraph` / flowspace consistency panics from the
    ///   adapter's downstream stages.
    Skip(String),
}

/// Production dual-gate — registry-aware entry.
///
/// Drives the real path via [`specialize_legacy_graph_with_registry`]
/// against a pre-populated `PyreCallRegistry` so graphs with
/// `OpKind::Call::FunctionPath` callsites resolve through the upstream
/// `Constant(<function>) -> getdesc -> FunctionDesc` chain
/// (`bookkeeper.py:353-409`).  Production callers
/// (codewriter.rs `transform_graph_to_jitcode`) build the registry
/// once per `CallControl` (program-wide) and reuse across every
/// dual-gated graph.
///
/// Returns:
///
/// - `Ok(DualGateOutcome::Match)` when the real path succeeds.  The
///   real path's `TypeResolutionState` and `AnnotationState` are
///   the authoritative source for production consumption (Slice 12.1
///   reader output).
/// - `Ok(DualGateOutcome::Skip(reason))` when the real path failed
///   on a known-unported feature (registry miss / adapter
///   invariant break / unimplemented rtyper op).  Callers fall back
///   to legacy walker output for the affected graph.
/// - `Err(message)` when the real path failed for an unrecognised
///   reason — surfaces upstream so the codewriter can re-classify
///   the panic if the message later turns out to be in the
///   known-unported table.
///
/// Defensive per-`ValueId` diff against the legacy walker baseline
/// runs whenever both paths succeed.  Divergence is reported as
/// `Skip("dual-gate divergence: ...")` so the codewriter falls back
/// to the legacy walker output, matching main's pre-Slice-12.2
/// contract.  Anchor tests use [`dual_gate_check`] directly for
/// hand-built fixtures.
pub fn dual_gate_check_with_registry(
    legacy_graph: &LegacyGraph,
    call_registry: &PyreCallRegistry,
) -> Result<DualGateOutcome, String> {
    // Same panic-catch contract as `dual_gate_check` — the rtyper's
    // internal `genop`/`level` asserts surface as diagnostic panics
    // for unported pyre-front idioms; the gate uniformly returns a
    // stringified error so the env-flag wrapper can decide whether
    // to panic, log, or skip.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        specialize_legacy_graph_with_registry(legacy_graph, call_registry)
    }));
    let (real_state, real_annotations) = match result {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => {
            let msg = format!("{e}");
            if is_known_unported(&msg) {
                return Ok(DualGateOutcome::Skip(msg));
            }
            return Err(format!("real path failed: {msg}"));
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<unrecognised panic payload>".to_string()
            };
            if is_known_unported(&msg) {
                return Ok(DualGateOutcome::Skip(msg));
            }
            return Err(format!("real path panicked: {msg}"));
        }
    };
    // Defensive baseline diff against the legacy walker.  Mirror of
    // main's pre-Slice-12.2 contract: every definite `ValueId` is
    // compared, and the first divergence becomes a `Skip` so the
    // production caller falls back to the legacy walker for that
    // function rather than silently shipping a real-path result that
    // disagrees with the legacy/PyPy projection.
    //
    // A legacy panic disables the comparison entirely — that hole is
    // closed by surfacing the panic as a fail-loud `Err`, so the
    // codewriter caller cannot silently accept a real-path result
    // whose parity has not been validated against the legacy baseline.
    let baseline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let legacy_annotations = super::legacy_annotator::annotate(legacy_graph);
        super::legacy_resolve::resolve_types(legacy_graph, &legacy_annotations)
    }));
    let legacy_state = match baseline {
        Ok(state) => state,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<unrecognised panic payload>".to_string()
            };
            return Err(format!(
                "dual-gate baseline panicked (legacy walker crashed before \
                 comparison could run — comparison cannot validate the real \
                 path's result): {msg}"
            ));
        }
    };
    if let Some(divergence) = compare_real_against_legacy(&real_state, &legacy_state) {
        return Ok(DualGateOutcome::Skip(format!(
            "dual-gate divergence: {divergence}"
        )));
    }
    Ok(DualGateOutcome::Match {
        real_state,
        real_annotations,
    })
}

/// Per-`ValueId` diff between the real path's `TypeResolutionState`
/// and the legacy walker's baseline.  Returns `Some(message)` if the
/// real path differs from a definite legacy kind on any `ValueId`,
/// or assigns a definite kind to a `ValueId` the legacy walker did
/// not resolve.  Mirror of the loop body in [`dual_gate_check`] but
/// surfaces the first divergence as a single string for `Skip`.
fn compare_real_against_legacy(
    real_state: &TypeResolutionState,
    legacy_state: &TypeResolutionState,
) -> Option<String> {
    for (vid, legacy_kind) in &legacy_state.concrete_types {
        if *legacy_kind == ConcreteType::Unknown {
            continue;
        }
        let real_kind = real_state.get(*vid);
        if real_kind != legacy_kind {
            return Some(format!(
                "ValueId({}): legacy={:?}, real={:?}",
                vid.0, legacy_kind, real_kind
            ));
        }
    }
    for (vid, real_kind) in &real_state.concrete_types {
        if *real_kind == ConcreteType::Unknown {
            continue;
        }
        let legacy_kind = legacy_state.get(*vid);
        if *legacy_kind == ConcreteType::Unknown {
            return Some(format!(
                "ValueId({}): legacy={:?}, real={:?}",
                vid.0, legacy_kind, real_kind
            ));
        }
    }
    None
}

/// Return true when `msg` matches one of the known-unported
/// categories the dual-gate currently treats as `Skip` rather than
/// `Err`.  Each category is its own future port; the predicate is
/// the central place to enumerate what's still unimplemented.
///
/// As ports land, entries here move out (the port closes the
/// category) and the surface of "unknown failure" shrinks.  When
/// the predicate matches nothing, the gate is at full coverage.
///
/// **Epic owners** (audit 2.D, 2026-05-05) — TODO: each remaining
/// entry is blocked on a specific multi-session porting epic;
/// retiring an entry early is unsafe until its epic converges:
///
/// | Substring                                  | Owning epic / followup                                                      |
/// |--------------------------------------------|------------------------------------------------------------------------------|
/// | `not registered in PyreCallRegistry`       | M2.5g extern Rust helper registry walker (multi-session, blocked).           |
/// | `undefined operand ValueId`                | body-input-retirement-epic Phase 1+2 (adapter producer correctness audit).   |
/// | `unimplemented operation`                  | per-opname rtyper handler ports (each opname its own slice).                 |
/// | `variable used before definition`          | body-input-retirement-epic Phase 2 (cross-block locals threading).           |
/// | `MissingRTypeAttribute`                    | Cat 3.1 typed-Ref → SomeInstance(ClassDef) (typed-ref-someptr epic sibling). |
/// | `KeyError: no binding for arg`             | annotator gap audit (per-call coverage; future epic).                        |
/// | `compute_at_fixpoint failed`               | PBC dispatch / call-family coverage (per-call).                              |
/// | `post-rtyper jtransform variant`           | per-variant emit-site retracing (rpbc / rclass / front-end).  Includes `OpKind::Abort` (pyre-only marker; retire every `Expr::ForLoop` / `stop_unsupported` / `continue_with_unknown` emit-site at the front-end). |
/// | `adapter cross-block body Input`           | body-input-retirement-epic Phase 5 (final emit-site retirement).             |
///
/// PyPy `bookkeeper.py:108-127` propagates fixpoint failures uncaught;
/// pyre's dual-gate Skip is the cutover-time scaffolding that defers
/// exactly the categories enumerated above.  Once every owning epic
/// converges, every match returns `false`, the legacy walker fallback
/// at `codewriter.rs::dual_gate_type_state` becomes dead code, and
/// the predicate retires entirely (Step 5 / Task #127).
pub(crate) fn is_known_unported(msg: &str) -> bool {
    msg.contains("not registered in PyreCallRegistry")
        || msg.contains("undefined operand ValueId")
        || msg.contains("unimplemented operation")
        || msg.contains("variable ")
            && msg.contains(" used before definition")
        // (Retired 2026-05-05.) `normalize_unary_op_name: pyre UnaryOp`
        // and `normalize_binop_name: pyre BinOp` were Skip-classified
        // when production still emitted `not` / `deref` / `same_as`
        // and `and` / `or`.  Both surfaces are now desugared upstream
        // — `UnOp::Not`/`Deref` at `front/ast.rs::Expr::Unary` +
        // `flowspace/rust_source/build_flow.rs::lower_unary_not`,
        // `&&`/`||` at the matching `Expr::Binary` arms.  `invert`
        // landed in `normalize_unary_op_name` 2026-05-05.  Synthetic
        // graphs that inject these ops (anchor tests in
        // `cutover.rs::tests::anchor_unary_*_surfaces_*`) call
        // `specialize_legacy_graph` directly and never reach
        // `is_known_unported`, so removing these substring matches
        // does not affect them.  Any production reach surfaces as a
        // dual-gate divergence panic — the parity-correct outcome.
        // Field / method dispatch on a `SomeInstance(classdef=None)`
        // — pyre's `Ref` ValueType currently lifts to a classdef-less
        // SomeInstance (Cat 3.1 placeholder), so `find_attribute`
        // (`rclass.py:556+find_attribute_or_None`) cannot route the
        // dispatch.  Cat 3.1.A's `InstanceRepr::rtype_getattr` port
        // (`rclass.py:838-857`) routes through `getclsfield`, which
        // surfaces the upstream-orthodox `MissingRTypeAttribute(attr)`
        // when find_attribute returns None.  The companion
        // `"no method ... on Instance("` substring (`rmodel.rs:828`
        // default `rtype_getattr` find_method failure path) was
        // retired 2026-05-06: post-Cat-3.1.A every SomeInstance
        // dispatch goes through the InstanceRepr override, so the
        // default fn never fires for `Instance(...)`-shaped operands.
        // The MissingRTypeAttribute entry stays until typed-Ref
        // ClassDef projection lands and field/method dispatch starts
        // succeeding.
        || msg.contains("MissingRTypeAttribute")
        // Variable's `.annotation` slot empty at `bindingrepr`
        // lookup time — the AnnotationState produced for some
        // ValueIds carries `Unknown`, and `valuetype_to_someshell`
        // returns `None` for `Unknown` (intentionally fail-loud
        // for "annotation gap" so producers know which ValueId
        // missed seeding, see `seed_variable` at
        // `flowspace_adapter.rs:96-115`).  Closing the gap means
        // tightening the front-end / annotator producers so every
        // ValueId has a non-`Unknown` annotation; the gate skips
        // until then.
        || msg.contains("KeyError: no binding for arg")
        // TODO(annotator-fixpoint-fail-loud) — STRICT-PARITY REGRESSION
        // vs main / PyPy.  `bookkeeper.py:108-127` propagates fixpoint
        // exceptions uncaught and `annrpython.py:643` lets
        // `AnnotatorError` reach the caller; absorbing the four
        // patterns below is a NEW-DEVIATION that hides real
        // annotator/rtyper parity gaps as "known unported".  Direct
        // removal verified 2026-05-12 to break `pyre-jit-trace`
        // `build.rs` at every reachable hitter
        // (`make_green_key`, `Frame::load_fast`, `PyFrame::locals_w_mut`,
        // `<default methods of IterOpcodeHandler>::record_for_iter_guard`,
        // `pyjitpl_step::Cannot find attribute`); production cannot
        // compile until each underlying real-path gap is closed
        // (classdef-less SomeInstance dispatch + `PyreCallRegistry::
        // ensure_session` coverage — the builtin-analyser binding-
        // failure reorder converged 2026-05-12 by routing every
        // analyser body through per-touch `arg_at` instead of the
        // retired `args_s_concrete_or_panic` eager prefix).  Multi-session epic — until each gap-hitter is
        // ported, this Skip stays as documented divergence and the
        // codewriter falls back to the legacy walker for production.
        || msg.contains("compute_at_fixpoint failed")
        || msg.contains("complete_pending_blocks failed")
        || msg.contains("Cannot find attribute ")
        || msg.contains("AnnotatorError:")
        // `dyn Trait` dispatch still enters the real-rtyper flowspace
        // adapter as pyre's pre-rtyper `CallTarget::Indirect` shape in
        // some registry-prefill paths.  The production codewriter's
        // jtransform path already runs `rpbc::lower_indirect_calls`
        // before consuming these graphs; the real-rtyper cutover path
        // cannot yet express the matching rclass/rpbc rewrite before
        // adapter input without leaking post-rtyper `VtableMethodPtr` /
        // `IndirectCall` ops into flowspace.  Treat this as a
        // known-unported real-path gap so the dual gate falls back to
        // the legacy type walker instead of panicking during registry
        // population.
        || (msg.contains("Call with CallTarget::Indirect") && msg.contains("rclass"))
        // (Retired 2026-05-06.) `AnnotatorError: immutablevalue(HostObject`
        // was Skip-classified for `SyntheticTransparentCtor` (Ok/Err/Some/
        // None) — the adapter wrapped the ctor name as
        // `HostObject::new_opaque(name)` which has no `is_class()` /
        // `is_user_function()` classification, so `immutablevalue` fell
        // through to "Don't know how to represent".  The adapter now
        // emits `HostObject::new_class(name, [])` instead
        // (`flowspace_adapter.rs:821 SyntheticTransparentCtor` arm),
        // routing through the existing `is_class()` arm in
        // [`crate::annotator::bookkeeper::Bookkeeper::immutablevalue_hostobject`]
        // (`bookkeeper.py:315-316` parity).  No production emit-site of
        // `HostObject::new_opaque` remains in `front/`/`translator/rtyper/`,
        // so any future `immutablevalue(HostObject` failure is a real
        // parity bug — let it surface as a dual-gate divergence panic.
        // TODO(post-rtyper-jtransform-variant-leak): retire this Skip
        // entry per upstream parity — `jit/codewriter/jtransform.py`
        // raises straight through on unexpected opnames.  Today the
        // most frequent reach is `OpKind::Abort` emitted by
        // `front/ast.rs::stop_unsupported` / `continue_with_unknown`
        // when the surface DSL hits an unsupported expression — pyre
        // source like `execute_opcode_step` / `eval_loop_jit` /
        // `build_blackhole_frames_from_deadframe` all carry such
        // placeholders today.  Convergence path: retire each Abort
        // emit-site at the front-end (per-variant epic — closure
        // body, complex match arms, unsupported literals), then
        // every other post-rtyper variant (IndirectCall / ResidualCall
        // / Vable* / JitMergePoint / LoopHeader / InlineCall /
        // RecursiveCall) emitted from `rpbc.rs` / `rclass.rs` ahead
        // of the rtyper.  Until then the Skip absorbs the placeholder
        // leakage; without it `tests::test_codegen_output` /
        // `tests::test_recognition_report` and the analyse-pyre-source
        // pipeline panic at the first reachable Abort.
        || msg.contains("post-rtyper jtransform variant")
        // Cross-block body `Input` whose `name` was not threaded
        // through `Link.args` / target `inputargs` by the predecessor.
        // RPython flowspace has no body-`Input` op — every cross-block
        // local reference goes via `flowcontext.py:872-884 LOAD_FAST`
        // (which writes into `self.locals_w`) and the target block's
        // pre-allocated `inputargs[]`.  Pyre's body-`Input` emission
        // is itself a TODO (`front/ast.rs:2127-2156`); when Cat 2.1
        // cross-block locals threading misses a shape, the adapter
        // now fails loud with this message instead of silently
        // fabricating a fresh Variable (which would hide an
        // SSA / alias-shape divergence from PyPy's flowspace).  Skip
        // until either Cat 2.1 covers every shape, or the front-end's
        // body-`Input` emission is replaced by a Link.args / inputargs
        // threading pass that mirrors RPython.
        || msg.contains("adapter cross-block body Input")
    // `normalize_unary_op_name: pyre UnaryOp` Skip entry retired
    // 2026-05-06: every cast variant pyre's
    // `front/ast.rs::cast_op_name` emits (`same_as` /
    // `cast_int_to_float` / `cast_float_to_int` / `cast_int_to_ptr`;
    // `cast_ptr_to_int` already had a handler) now has a dedicated
    // `RPythonTyper::translate_operation` arm into
    // `rbuiltin.rs::rtype_*` (verbatim ports of upstream
    // `@typer_for(lltype.cast_*)` bodies).  `normalize_unary_op_name`
    // accepts these opnames straight through; any residual fail-loud
    // surfaces as a dual-gate divergence panic — the parity-correct
    // outcome.
}

/// Slice 6 — populate a `PyreCallRegistry` from a
/// `HashMap<CallPath, FunctionGraph>` (the shape pyre's
/// `CallControl::function_graphs()` returns).
///
/// Mirrors [`populate_call_registry_from_program`] but keyed off
/// `CallPath` (parse layer's path type, used by `transform_graph_to_jitcode`'s
/// callcontrol map) instead of `SemanticProgram.functions`.  Every
/// callee referenced from any `OpKind::Call::FunctionPath` callsite
/// in the program lives in this map under the same path the front
/// end emitted (`canonical_call_target`, `front/ast.rs:3497-3528`),
/// so registry lookups during the lift step always hit.
///
/// Two-pass shape (same as the SemanticProgram walker — see that
/// helper's docstring for the recursivecall parity argument):
///
/// 1. `get_or_register` every entry's signature so HostObjects exist
///    before any callee body lifts.
/// 2. `lift_callee_to_pygraph` + `prefill_default_cache` per entry
///    so `cachedgraph` (`description.rs:1037-1039`) hits at the
///    rtyper's `direct_call`.
pub fn populate_call_registry_from_call_graphs(
    function_graphs: &std::collections::HashMap<crate::parse::CallPath, LegacyGraph>,
    registry: &PyreCallRegistry,
) -> Result<(), TyperError> {
    // Dedupe by canonical path — RPython `Bookkeeper.getdesc(pyobj)`
    // (`bookkeeper.py:353-409`) returns the *same* FunctionDesc for
    // any reference to the same callable, keyed by `Constant(pyobj)`
    // identity.  Pyre's `function_graphs: HashMap<CallPath, FunctionGraph>`
    // can carry the same callable under multiple keys (`lib.rs`
    // registers free functions under both their canonical path AND a
    // `crate::`-prefixed alias for callsite matching); without dedupe,
    // each path key would create a separate HostObject / FunctionDesc
    // pair, violating upstream's "create once, share thereafter"
    // contract.
    //
    // The dedupe key strips a leading `crate::` segment so
    // `crate::a::foo` and `a::foo` collapse to the same canonical key
    // `[a, foo]`, while genuinely distinct functions like `a::foo` and
    // `b::foo` keep distinct keys (`[a, foo]` vs `[b, foo]`) and stay
    // separate.  Bare-name dedupe (`graph.name`) was unsafe because
    // `FunctionGraph::name` is the function's bare identifier (no
    // module qualification) so two functions named `foo` in different
    // modules would have falsely aliased to one HostObject / FunctionDesc.
    fn canonical_dedup_key(path: &crate::parse::CallPath) -> Vec<String> {
        let mut segs: Vec<String> = path.segments.iter().cloned().collect();
        if segs.first().map(String::as_str) == Some("crate") {
            segs.remove(0);
        }
        segs
    }
    let mut pending: Vec<(FunctionPathKey, &LegacyGraph, Rc<PyreFunctionEntry>)> =
        Vec::with_capacity(function_graphs.len());
    // `by_canonical_path` tracks the canonical `FunctionPathKey` of
    // the first-encountered alias for each canonical-stripped key.
    // Subsequent aliases of the same callable register an alias row
    // pointing at that canonical key — the registry's `entries` map
    // gets exactly one row per distinct callable, with `aliases`
    // carrying the indirection.
    let mut by_canonical_path: HashMap<Vec<String>, FunctionPathKey> = HashMap::new();
    for (path, graph) in function_graphs {
        let key = FunctionPathKey::from_segments(path.segments.iter().cloned());
        let canonical_strip = canonical_dedup_key(path);
        let entry = if let Some(canonical_key) = by_canonical_path.get(&canonical_strip) {
            if canonical_key != &key {
                registry.alias(key.clone(), canonical_key);
            }
            registry
                .lookup(canonical_key)
                .expect("canonical entry registered")
        } else {
            let signature = signature_for_graph(graph);
            let entry = registry.get_or_register(key.clone(), signature);
            by_canonical_path.insert(canonical_strip, key.clone());
            entry
        };
        pending.push((key, graph, entry));
    }
    // Pass 2 — prefill the default-cache once per *unique* registry
    // entry.  Aliases already point at the same `Rc<PyreFunctionEntry>`
    // so their `prefill_default_cache` would be redundant; identify
    // unique entries by `Rc::as_ptr` so the dedupe survives any
    // alternate dedup-key shape.  Each unique entry is lifted exactly
    // once (`Rc::as_ptr` identity dedup); aliases share the same
    // pre-filled `FunctionDesc.cache` because they hold the same
    // `Rc<PyreFunctionEntry>`.
    let mut lifted: HashSet<*const PyreFunctionEntry> =
        HashSet::with_capacity(by_canonical_path.len());
    for (_key, graph, entry) in &pending {
        let entry_ptr = Rc::as_ptr(entry);
        if !lifted.insert(entry_ptr) {
            continue;
        }
        let pygraph = lift_callee_to_pygraph(graph, signature_for_graph(graph), registry)?;
        entry.prefill_default_cache(pygraph);
    }
    Ok(())
}

// build_program_annotator / build_program_rtyper / ensure_program_
// specialize were retired at Step 4 first slice (2026-05-07).  They
// existed solely to drive a program-wide `compute_at_fixpoint` +
// `RPythonTyper::specialize` over a pre-seeded annotator (Step 3
// retired the seed; the empty annotator made the pass a no-op).
// The matching `ProgramSpecializeState` flag and its accessors were
// retired alongside.  Per-session
// `specialize_legacy_graph_with_registry_seed` now runs the
// orthodox flow directly: lift subject graph -> seed subject blocks
// -> compute_at_fixpoint (drives pycall->recursivecall to discover
// callees) -> RPythonTyper::specialize.

/// Derive a parameter `Signature` directly from a `FunctionGraph`'s
/// startblock inputargs.  Same shape as
/// [`signature_for`] but takes a graph rather than a
/// `SemanticFunction`, since `CallControl::function_graphs()`
/// carries graphs without their lowered SemanticFunction wrapper.
fn signature_for_graph(graph: &LegacyGraph) -> Signature {
    let startblock = graph.block(graph.startblock);
    let argnames: Vec<String> = startblock
        .inputargs
        .iter()
        .enumerate()
        .map(|(idx, vid)| {
            graph
                .value_name(*vid)
                .map(str::to_string)
                .unwrap_or_else(|| format!("arg{idx}"))
        })
        .collect();
    Signature::new(argnames, None, None)
}

/// Specialize a legacy `model::FunctionGraph` end-to-end through the
/// real `RPythonTyper`.
///
/// Returns a `TypeResolutionState` keyed by the original legacy
/// `ValueId` — drop-in replacement for legacy
/// `translator::rtyper::legacy_resolve::resolve_types` once Slice 4
/// dual-gate validates the projection on the anchor corpus.
/// Lift a pyre `model::FunctionGraph` (a callee that may appear on a
/// `OpKind::Call::FunctionPath` callsite of some other graph) into a
/// `Rc<PyGraph>` suitable for pre-filling the callee's
/// `FunctionDesc.cache` (`description.rs:794`).
///
/// `cachedgraph` (`description.rs:1037-1039`) returns the cached
/// `Rc<PyGraph>` as soon as the lookup key matches, skipping the
/// `buildgraph` path that delegates to
/// `translator.buildflowgraph(pyobj, false)` — that delegation
/// requires a real Python `__code__` body which pyre's surface-DSL
/// callees do not have.  Pyre instead lifts the callee through the
/// adapter and wraps the resulting flowspace `FunctionGraph` in a
/// `PyGraph` here.
///
/// `signature` is the callee's authoritative parameter `Signature`
/// (the same one stored on the `FunctionDesc` in `PyreCallRegistry`).
/// `defaults` is empty — pyre's surface DSL does not yet expose
/// per-parameter default values; when it does, this helper grows a
/// `defaults: Vec<Constant>` parameter mirroring upstream
/// `func.__defaults__`.
///
/// `nested_registry` carries the program-wide `PyreCallRegistry`.
/// For leaf callees (no `OpKind::Call` ops in the body) the registry
/// is unused; nested callees recursively consult it as the adapter
/// processes the callee's own `OpKind::Call` ops.
pub fn lift_callee_to_pygraph(
    callee_graph: &LegacyGraph,
    signature: Signature,
    nested_registry: &PyreCallRegistry,
) -> Result<Rc<PyGraph>, TyperError> {
    lift_callee_to_pygraph_seed(callee_graph, None, signature, nested_registry)
}

/// Slice 12.2 — seeded lift entry mirroring
/// [`function_graph_to_flowspace_with_seed_annotations`].  Production
/// goes through [`lift_callee_to_pygraph`]; test fixtures with
/// minimal SSA shapes pass `Some(&AnnotationState)` so `seed_variable`
/// has type info to attach to each `Variable`.
pub(crate) fn lift_callee_to_pygraph_seed(
    callee_graph: &LegacyGraph,
    seed_annotations: Option<&AnnotationState>,
    signature: Signature,
    nested_registry: &PyreCallRegistry,
) -> Result<Rc<PyGraph>, TyperError> {
    // The adapter also returns `value_to_var` and `constant_concretetypes`
    // side maps, but every legacy consumer of those was a dead-write
    // path into `PyreCallRegistry` (Issue 2.5 retirement, 2026-05-07).
    // RPython parity: `Variable.concretetype` and `Constant.concretetype`
    // already carry the per-variable / per-constant LL type after
    // specialise; downstream readers must consult those fields directly
    // (`history.py:204` `same_constant`, `model.py:438` `Variable.
    // concretetype`).  The ByValueId side map was a pyre-only divergence.
    let FlowspaceAdapterOutput { graph, .. } =
        crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace_with_seed_annotations(
            callee_graph,
            seed_annotations,
            nested_registry,
        )?;
    // Pyre's synthetic `GraphFunc` mirrors `description.py:193-203
    // FunctionDesc.__init__` test fixtures — empty Dict globals,
    // name from the legacy graph.  No HostCode body — the cache
    // pre-fill ensures `cachedgraph` never asks for one.
    let func = GraphFunc::new(
        callee_graph.name.clone(),
        Constant::new(ConstValue::Dict(HashMap::new())),
    );
    let pygraph = Rc::new(PyGraph {
        graph,
        func,
        signature: RefCell::new(signature),
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
    });
    Ok(pygraph)
}

/// Derive the canonical `FunctionPathKey` for a `SemanticFunction`.
///
/// Mirrors the front-end's `canonical_call_target`
/// (`front/ast.rs:3497-3528`) inverted: what an `Expr::Path`
/// callsite emits is what we register the callee under.
///
/// Both `func.name` (e.g. `"a::helper"` after `build_graphs_from_items`
/// prepends module prefix at `front/ast.rs:630-632`) and
/// `func.self_ty_root` (e.g. `"a::Foo"` after `qualify_type_name` at
/// `:638`) carry `::`-joined module-qualified strings.  Each
/// `::`-separated component is one `FunctionPathKey` segment — the
/// callsite at `front/ast.rs:3827-3835` produces `["a", "helper"]`
/// (multi-segment, no `::` in any segment), so the registered key
/// must split each `::` boundary too.  Without the split, registry
/// entry `["a::helper"]` (single segment containing `::`) would never
/// match callsite lookup `["a", "helper"]`.
///
/// PyPy's `Bookkeeper.getdesc(<function object>)` (`bookkeeper.py:353-409`)
/// keys on Python function-object identity rather than path strings,
/// so the upstream chain has no segment-shape divergence to worry
/// about.  Pyre's segment-key approach is a stand-in for that
/// identity until host callable identity is available; aligning the
/// segment shape both sides of the registry is a prerequisite for
/// that stand-in to function correctly.
fn function_path_key_for(func: &front::SemanticFunction) -> FunctionPathKey {
    let mut segments: Vec<String> = Vec::new();
    if let Some(t) = &func.self_ty_root {
        segments.extend(t.split("::").map(str::to_string));
    }
    segments.extend(func.name.split("::").map(str::to_string));
    FunctionPathKey::from_segments(segments)
}

/// Derive the parameter `Signature` for a `SemanticFunction` from
/// the startblock's input arg names.
///
/// Mirrors upstream `bookkeeper.py:418
/// cpython_code_signature(pyfunc.__code__)`: the parameter names
/// come from the function's declared parameter list.  Pyre carries
/// these as `value_name(inputarg)` on the startblock; missing names
/// fall back to `arg{N}` to keep the `FunctionDesc.signature.argnames`
/// length matched to the actual parameter count.
fn signature_for(func: &front::SemanticFunction) -> Signature {
    let graph = &func.graph;
    let startblock = graph.block(graph.startblock);
    let argnames: Vec<String> = startblock
        .inputargs
        .iter()
        .enumerate()
        .map(|(idx, vid)| {
            graph
                .value_name(*vid)
                .map(str::to_string)
                .unwrap_or_else(|| format!("arg{idx}"))
        })
        .collect();
    Signature::new(argnames, None, None)
}

/// Slice A.5 — walk a `SemanticProgram` and pre-register every
/// reachable `SemanticFunction` in the call registry.
///
/// Closes Issue 2.1 from the post-A.4 audit: production callers
/// with `OpKind::Call::FunctionPath` ops cannot proceed through the
/// real-rtyper path without the registry pre-populated, because
/// pyre's surface DSL has no Python callable object for the
/// upstream `Constant(<function>) -> getdesc -> FunctionDesc` chain
/// (`bookkeeper.py:353-409`).  This walker registers each
/// `SemanticFunction` so any `simple_call` the adapter emits later
/// finds a matching entry.
///
/// Two passes are required because the per-function lift
/// (`lift_callee_to_pygraph`) calls
/// `function_graph_to_flowspace`, which runs the
/// `OpKind::Call::FunctionPath` arm in `translate_op` and expects
/// every callee path the body references to already be registered.
/// Pass 1 installs `(HostObject, FunctionDesc)` for every program
/// function (signature only, no PyGraph cache).  Pass 2 lifts each
/// function's body and prefills its `FunctionDesc.cache` so the
/// rtyper's `cachedgraph` (`description.rs:1037-1039`) hits at
/// `direct_call` time.
///
/// Annotation seeding lives inside [`function_graph_to_flowspace`]
/// after Slice 12.2; this walker plumbs only the program + registry.
pub fn populate_call_registry_from_program(
    program: &front::SemanticProgram,
    registry: &PyreCallRegistry,
) -> Result<(), TyperError> {
    // Pass 1 — register every function with its signature; the
    // FunctionDesc.cache stays cold until Pass 2.
    let mut pending: Vec<(
        FunctionPathKey,
        &front::SemanticFunction,
        Rc<PyreFunctionEntry>,
    )> = Vec::with_capacity(program.functions.len());
    for func in &program.functions {
        let key = function_path_key_for(func);
        let signature = signature_for(func);
        let entry = registry.get_or_register(key.clone(), signature);
        pending.push((key, func, entry));
    }
    // Pass 2 — lift each function's body using the now-populated
    // registry, then prefill its FunctionDesc.cache.  Lift order is
    // independent of function topology: every callsite's
    // `simple_call` resolves through `registry.lookup` against the
    // Pass-1-installed HostObjects, regardless of whether the
    // callee's own body has been lifted yet.
    for (_key, func, entry) in pending {
        let pygraph = lift_callee_to_pygraph(&func.graph, signature_for(func), registry)?;
        entry.prefill_default_cache(pygraph);
    }
    Ok(())
}

/// Restricted entry: only graphs without `OpKind::Call::FunctionPath`
/// ops resolve through this path.  An empty `PyreCallRegistry` is
/// constructed internally and shared with the annotator; any
/// `simple_call` op the adapter emits would surface a fail-loud
/// `TyperError` at `flowspace_adapter::translate_op`'s
/// `FunctionPath` arm because there is no `(HostObject, FunctionDesc)`
/// pair to resolve the callable to.
///
/// Issue 2.1 (post-Slice A.4 audit): the upstream
/// `Constant(<function>) -> getdesc -> FunctionDesc` chain
/// (`bookkeeper.py:353-409`) is unreachable from this entry because
/// pyre's surface DSL has no Python callable object to resolve;
/// production callers that need free function calls must build a
/// pre-populated registry (one entry per reachable `FunctionPath`,
/// with `register_callee(key, signature, lifted_pygraph)`) and call
/// [`specialize_legacy_graph_with_registry`] directly.  Until the
/// production walker that traverses `SemanticProgram.functions`
/// lands, this entry remains in-place for anchor tests + Slice 4
/// dual-gate validation against graphs that have no `Call` ops.
pub fn specialize_legacy_graph(legacy: &LegacyGraph) -> Result<TypeResolutionState, TyperError> {
    let registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(Rc::new(
        crate::annotator::bookkeeper::Bookkeeper::new(),
    ));
    specialize_legacy_graph_with_registry(legacy, &registry).map(|(state, _)| state)
}

/// Slice 12.2 — test-only entry that lets fixtures hand-build a
/// minimal SSA graph without `OpKind::Input { ty }` ops and seed the
/// adapter's annotation-state explicitly.
///
/// Production graphs from `front/ast.rs` always carry the typed
/// Input / FieldRead / Call ops the legacy walker can recover types
/// from; minimal anchor-test fixtures construct skeletal SSA shapes
/// where the legacy walker has no shape to work with, so they need
/// to inject `(ValueId, ValueType)` pairs explicitly.
#[cfg(test)]
pub(crate) fn specialize_legacy_graph_with_seed_annotations(
    legacy: &LegacyGraph,
    seed_annotations: &AnnotationState,
) -> Result<TypeResolutionState, TyperError> {
    let registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(Rc::new(
        crate::annotator::bookkeeper::Bookkeeper::new(),
    ));
    specialize_legacy_graph_with_registry_seed(legacy, Some(seed_annotations), &registry)
        .map(|(state, _)| state)
}

#[cfg(test)]
pub(crate) fn dual_gate_check_with_registry_seed(
    legacy_graph: &LegacyGraph,
    seed_annotations: &AnnotationState,
    legacy_state: &TypeResolutionState,
    call_registry: &PyreCallRegistry,
) -> Result<DualGateOutcome, String> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        specialize_legacy_graph_with_registry_seed(
            legacy_graph,
            Some(seed_annotations),
            call_registry,
        )
    }));
    match result {
        Ok(Ok((real_state, real_annotations))) => {
            // Test-side regression diff against legacy — keeps anchor
            // tests honest until the orthodox `RPythonAnnotator::
            // processblock` port produces the same shape.
            let mut divergences: Vec<String> = Vec::new();
            for (vid, legacy_kind) in &legacy_state.concrete_types {
                if *legacy_kind == ConcreteType::Unknown {
                    continue;
                }
                let real_kind = real_state.get(*vid);
                if real_kind != legacy_kind {
                    divergences.push(format!(
                        "ValueId({}): legacy={:?}, real={:?}",
                        vid.0, legacy_kind, real_kind
                    ));
                }
            }
            for (vid, real_kind) in &real_state.concrete_types {
                if *real_kind == ConcreteType::Unknown {
                    continue;
                }
                let legacy_kind = legacy_state.get(*vid);
                if *legacy_kind == ConcreteType::Unknown {
                    divergences.push(format!(
                        "ValueId({}): legacy={:?}, real={:?}",
                        vid.0, legacy_kind, real_kind
                    ));
                }
            }
            if divergences.is_empty() {
                Ok(DualGateOutcome::Match {
                    real_state,
                    real_annotations,
                })
            } else {
                Err(divergences.join("; "))
            }
        }
        Ok(Err(e)) => {
            let msg = format!("{e}");
            if is_known_unported(&msg) {
                Ok(DualGateOutcome::Skip(msg))
            } else {
                Err(format!("real path failed: {msg}"))
            }
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<unrecognised panic payload>".to_string()
            };
            if is_known_unported(&msg) {
                Ok(DualGateOutcome::Skip(msg))
            } else {
                Err(format!("real path panicked: {msg}"))
            }
        }
    }
}

/// Slice A.2 — registry-aware entry point.
///
/// Production callers (`codegen.rs:2301`, `jit_codewriter/codewriter.rs`)
/// will move from [`specialize_legacy_graph`] to this entry once the
/// production builder that walks `SemanticProgram.functions` and
/// registers each `FunctionPath` with its parameter `Signature`
/// lands.  Anchor tests that exercise `OpKind::Call` switch to this
/// entry as Slice A.3 wires the consumer in
/// `flowspace_adapter::translate_op`.
pub fn specialize_legacy_graph_with_registry(
    legacy: &LegacyGraph,
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
) -> Result<(TypeResolutionState, AnnotationState), TyperError> {
    specialize_legacy_graph_with_registry_seed(legacy, None, call_registry)
}

/// Internal driver shared by the production
/// [`specialize_legacy_graph_with_registry`] entry and the
/// test-only seeded variant
/// [`specialize_legacy_graph_with_seed_annotations`].
///
/// `seed_annotations: None` keeps the production path
/// (`function_graph_to_flowspace` runs the internal legacy walker);
/// `Some(state)` lets fixtures override that walker so minimal SSA
/// graphs without `OpKind::Input { ty }` ops still type-check.
pub(crate) fn specialize_legacy_graph_with_registry_seed(
    legacy: &LegacyGraph,
    seed_annotations: Option<&AnnotationState>,
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
) -> Result<(TypeResolutionState, AnnotationState), TyperError> {
    // Slice 3 v2 — RPython parity path.
    //
    // Upstream `RPythonTyper.specialize` runs ONCE per `Translator`,
    // not per graph.  Pyre's per-graph dual-gate enters this function
    // once per graph (one per `transform_graph_to_jitcode`), so the
    // upstream "specialize-once" semantics are reproduced by lazily
    // driving the program-wide pass on first entry and gating
    // `seed_callee_blocks` on the resulting state.  After the
    // program-wide pass succeeds, the cached PyGraphs hold
    // post-specialize LL ops; re-seeding them into the per-session
    // annotator would let `specialize_more_blocks` walk the LL ops a
    // second time and trip on `unimplemented operation: 'int_add'` /
    // `'direct_call'` — see memory
    // `skip_unimplemented_int_add_diagnosis_2026_05_06`.
    // ── Step 1 — Slice 1c adapter ──────────────────────────────────
    let FlowspaceAdapterOutput {
        graph,
        value_to_var,
        constant_concretetypes,
        block_map: _,
    } = crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace_with_seed_annotations(
        legacy,
        seed_annotations,
        call_registry,
    )?;

    // ── Step 2 — annotator surface ────────────────────────────────
    //
    // Step 4 second slice (2026-05-07): the annotator + rtyper are
    // now session-shared through `PyreCallRegistry::ensure_session`,
    // mirroring RPython's "one annotator + one rtyper per Translator"
    // (`translator.py:69-83`).  Each per-graph subject seeds its
    // blocks into the shared annotator; subsequent
    // `specialize_more_blocks` rtype only the newly-added blocks.
    let (annotator, rtyper) = call_registry.ensure_session()?;
    // Queue `graph.startblock` onto the orthodox `addpendingblock`
    // queue, mirroring how callees enter through
    // `pycall -> recursivecall -> addpendingblock`
    // (`description.py:283-305`, `annrpython.py:315-336`).
    // `addpendingblock` (`annrpython.rs:1245-1302`) writes
    // `Variable.annotation` for each inputarg via
    // `bindinputargs.setbinding`, inserts `all_blocks[startblock]`
    // and `annotated[startblock] = None`, then
    // `schedulependingblock` queues the block.
    // `complete_pending_blocks` below drains the queue:
    // `processblock` flips `annotated[block] = Some(graph)`,
    // `flowin` walks every op and recurses into successors via
    // `process_link -> addpendingblock(target, inputs_s)`
    // (`annrpython.rs:1502/1690`).  Transitive blocks register
    // themselves into `all_blocks`/`annotated` through the same
    // bindinputargs path on first arrival.
    let subject_inputcells =
        crate::translator::rtyper::flowspace_adapter::derive_subject_inputcells(
            legacy,
            seed_annotations,
        )?;
    let (startblock, exceptblock) = {
        let g = graph.borrow();
        (g.startblock.clone(), g.exceptblock.clone())
    };
    let subject_inputcells_opt: Vec<Option<crate::annotator::model::SomeValue>> =
        subject_inputcells.into_iter().map(Some).collect();
    annotator.addpendingblock(&graph, &startblock, &subject_inputcells_opt);
    // Pre-register the eagerly-created `exceptblock`
    // (`flowspace/model.py:21-25` parity) — non-raising subjects
    // never reach it through a `Link`, but `specialize_more_blocks`
    // must walk it so its inputargs receive an exception-typed
    // `concretetype`.  The rtyper's `setup_block_entry`
    // exception-block branch (`rtyper.rs:1915-1942`) writes
    // `Variable.concretetype = ExceptionData.lltype_of_exception_*`
    // without reading `Variable.annotation`, so the block needs no
    // `flowin`; it only needs to appear in `annotator.annotated`.
    {
        let bkey = crate::flowspace::model::BlockKey::of(&exceptblock);
        annotator
            .annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(graph.clone()));
        annotator.all_blocks.borrow_mut().insert(bkey, exceptblock);
    }

    // Step 3 (2026-05-07): the prior per-session `seed_callee_blocks`
    // call was a workaround for the missing `simple_call_SomeObject`
    // registration — without that registration, flowin trips on the
    // first `simple_call(host_object_const, args)` op with
    // "no unary spec for SimpleCall(PBC)".  Pre-seeding callee blocks
    // bypassed the trip but skipped the orthodox `pycall ->
    // recursivecall -> addpendingblock` chain entirely.
    //
    // Step 2 added `@op.simple_call.register(SomeObject)` (unaryop.py:
    // 114-118 parity) so flowin now dispatches through `s_func.call(
    // argspec)` -> `SomePBC.call` -> `Bookkeeper.pbc_call` ->
    // `FunctionDesc.pycall` -> `annotator.recursivecall`, which calls
    // `addpendingblock(graph, startblock, inputcells)` for the
    // callee.  The orthodox path now seeds callee blocks naturally;
    // the explicit pre-seed is no longer needed.

    // Phase 2 (addpendingblock conversion) — drain the pending
    // queue so `flowin` walks every block reachable from
    // `graph.startblock` and writes `Variable.annotation` for each
    // op result.  Mirrors upstream `RPythonAnnotator.complete()`
    // (`annrpython.py:226-232`), which loops `complete_pending_blocks`
    // → policy hook → exit-on-empty.  Without this drain, the
    // `addpendingblock(startblock, inputcells)` queued just above
    // stays in `genpendingblocks`, `annotated[block]` remains the
    // `None` sentinel, and `specialize_block` (`rtyper.rs:1656`)
    // panics on "annotator.annotated[block] is False sentinel".
    annotator
        .complete_pending_blocks()
        .map_err(|err| TyperError::message(format!("complete_pending_blocks failed: {err}")))?;

    // Slice A.4 — populate per-callsite call-family / calltable state
    // by walking the seeded blocks' call_ops.  `compute_at_fixpoint`
    // (`bookkeeper.py:108-118`, pyre `bookkeeper.rs:627-648`) drains
    // `annotator.call_sites()` through `consider_call_site`
    // (`bookkeeper.py:152-166`, pyre `bookkeeper.rs:675`); each
    // `simple_call(callable_const, *args)` op resolves the callable
    // to a `SomePBC` (via `immutablevalue_hostobject` for the
    // pre-registered `HostObject::UserFunction`), then records the
    // call site in `bookkeeper.pbc_maximal_call_families` so the
    // rtyper's `FunctionRepr.call(hop)` (`rpbc.py:199`) finds the
    // matching call-family row at `find_row` time.
    //
    // The drain runs in BOTH Done and Failed branches:
    //
    // - Done: program-wide drained the cached graphs' callsites, but
    //   the subject we are about to specialize is brand-new (its
    //   freshly-lifted blocks were just seeded into this annotator)
    //   and its `simple_call` ops have not yet been processed.  The
    //   subject's call_sites entries on the bookkeeper are processed
    //   here so its callsites' rows land in
    //   `pbc_maximal_call_families` before specialize tries to look
    //   them up.
    // - Failed / Pending-skipped: legacy entry — drains every
    //   reachable callsite (subject + seeded callees) for the first
    //   time.
    //
    // Errors propagate verbatim (Issue 3.1) — upstream
    // `bookkeeper.py:108-118` runs the call-site walk without a
    // `try`/`except`, so a failed `consider_call_site` terminates
    // `simplify` and unwinds out of the annotator driver.  Pyre's
    // port surfaces the same condition through `?`-propagation
    // here, replacing the prior `let _ = ...` swallow at
    // `bookkeeper.rs:627-648` that masked PBC dispatch failures.
    call_registry
        .bookkeeper()
        .compute_at_fixpoint()
        .map_err(|err| TyperError::message(format!("compute_at_fixpoint failed: {err}")))?;

    // ── Step 3 — incremental rtyper drive ──────────────────────────
    //
    // Lifecycle mapping vs upstream `RPythonTyper.specialize(self,
    // dont_simplify_again=False)` (rtyper.py:178-189):
    //
    //   1. `if not dont_simplify_again: self.annotator.simplify()`
    //      — pyre per-subject flow does not run annotator-wide
    //        simplify; the subject was lifted via the adapter and
    //        its block list is already in canonical shape.
    //   2. `self.exceptiondata.finish(self)` — hoisted into
    //      `PyreCallRegistry::ensure_session` so it runs **once** at
    //      session start (idempotent — `getclassrepr` is cached on
    //      `rtyper.instance_reprs`).
    //   3. `self.already_seen = {}` — pyre keeps `already_seen`
    //      across per-subject calls so previously-rtyped subjects
    //      are not re-walked.
    //   4. `self.specialize_more_blocks()` — invoked here per
    //      subject; walks annotator.annotated\\already_seen.
    //   5. `self.exceptiondata.make_helpers(self)` — currently
    //      no-op (R4 / `MixLevelHelperAnnotator` not ported), see
    //      module header on `translator/rtyper/rclass.rs`.
    //   6. second `specialize_more_blocks()` for helpers — redundant
    //      while make_helpers is a no-op.
    //
    // Step 4 is the only per-subject step; steps 2/5/6 are
    // session-global one-shots, step 1/3 are pyre-incremental
    // adaptations.  Once R4 lands, `make_helpers` migrates to
    // `ensure_session` after `finish_exceptiondata` (single
    // session-prologue), keeping the per-subject body unchanged.
    rtyper.specialize_more_blocks()?;

    // ── Step 4 — read back per-ValueId ConcreteType ───────────────
    let mut state = TypeResolutionState::new();
    for (&vid, var) in &value_to_var {
        let concretetype = var.concretetype.borrow();
        if let Some(lltype) = concretetype.as_ref() {
            state
                .concrete_types
                .insert(vid, lowleveltype_to_concrete(lltype)?);
        }
    }
    // RPython `Constant.concretetype` is the ground truth for constant
    // operands.  Read it directly from the adapter's per-`ValueId` map
    // rather than reconstructing the kind from `AnnotationState`, so a
    // pyre-side annotation gap (e.g. an `Unknown` slot left by the
    // legacy graph builder) does not silently strip the constant's
    // resolved kind.
    for (vid, lltype) in &constant_concretetypes {
        state
            .concrete_types
            .insert(*vid, lowleveltype_to_concrete(lltype)?);
    }

    // ── Step 5 — read back per-ValueId AnnotationState ────────────
    //
    // Slice 12.1: project the per-session annotator's `Variable.annotation`
    // SomeValue lattice nodes back onto a `ValueId`-keyed `AnnotationState`,
    // mirroring `RPythonAnnotator.bindings` lookup in
    // `annrpython.py:395-426`.  The reader output is what the orthodox
    // post-jtransform merge consumes instead of the per-graph
    // `legacy_annotator::annotate(graph)` recompute — the legacy walker
    // re-derives types from `OpKind` shapes alone, the rtyper-passed
    // annotator carries the result of the bookkeeper-driven annotation
    // fixed-point.
    let real_annotations = read_annotations_from_value_to_var(&value_to_var);
    Ok((state, real_annotations))
}

/// Project per-session `Variable.annotation` SomeValue lattice nodes onto
/// a `ValueId`-keyed `AnnotationState`.  Mirrors RPython's
/// `RPythonAnnotator.bindings[arg]` lookup
/// (`annrpython.py:395-426 annotation()` / `:417-426 binding()`).
///
/// For each (ValueId, Variable) entry:
/// - if the rtyper / annotator attached a `SomeValue`, store both the
///   precise lattice node in `some_values[vid]` and a reduced
///   `ValueType` discriminator in `types[vid]`;
/// - otherwise leave the slot empty so consumers fall through to the
///   `valuetype_to_someshell(Unknown)` fail-loud the same way as the
///   legacy walker's `ValueType::Unknown` slots.
fn read_annotations_from_value_to_var(
    value_to_var: &HashMap<ValueId, Variable>,
) -> AnnotationState {
    let mut state = AnnotationState::new();
    for (&vid, var) in value_to_var {
        let annotation = var.annotation.borrow();
        let Some(rc_some) = annotation.as_ref() else {
            continue;
        };
        let vt = somevalue_to_valuetype(rc_some.as_ref());
        state.types.insert(vid, vt);
        state.some_values.insert(vid, rc_some.clone());
    }
    state
}

/// Reduce a `SomeValue` lattice node to its `ValueType` discriminator.
/// Inverse of `valuetype_to_someshell` (`annotation_state.rs:65`).
///
/// RPython parity: `getkind` family in `rpython/rtyper/lltypesystem/lltype.py`
/// reduces lltypes to backend kinds; the analogue here reduces
/// annotation-stage `SomeValue` to pyre's flat `ValueType` enum used by
/// downstream `jit_codewriter` consumers that haven't been ported to
/// `SomeValue` directly.
fn somevalue_to_valuetype(s: &SomeValue) -> ValueType {
    match s {
        SomeValue::Integer(_) => ValueType::Int,
        SomeValue::Bool(_) => ValueType::Bool,
        SomeValue::Float(_) | SomeValue::SingleFloat(_) | SomeValue::LongFloat(_) => {
            ValueType::Float
        }
        SomeValue::Instance(_) | SomeValue::Ptr(_) | SomeValue::PBC(_) => ValueType::Ref,
        // `SomeImpossibleValue` represents unreachable code (`model.py:627`),
        // which projects to `ValueType::Void` in pyre's flat enum just
        // like upstream `lltype.Void`.
        SomeValue::Impossible => ValueType::Void,
        // Other variants (String / List / Tuple / Dict / Iterator /
        // Exception / None_ / Property / InteriorPtr / LLADTMeth /
        // Builtin / BuiltinMethod / WeakRef / TypeOf / ByteArray /
        // Char / UnicodeCodePoint / UnicodeString / Type / Object) have
        // no direct pyre `ValueType` mapping; downstream consumers that
        // care will read from `some_values` directly.  Project to `Ref`
        // so the rtyper's GC-pointer fallback applies.
        _ => ValueType::Ref,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Block, BlockId, LinkArg, ValueId, ValueType};

    fn link_to_returnblock(args: Vec<LinkArg>, returnblock_id: BlockId) -> crate::model::Link {
        crate::model::Link::new_mixed(args, returnblock_id, None)
    }

    #[test]
    fn lowleveltype_to_concrete_signed_family_collapses_to_signed() {
        for ll in [
            LowLevelType::Signed,
            LowLevelType::Unsigned,
            LowLevelType::Bool,
            LowLevelType::Char,
            LowLevelType::UniChar,
            LowLevelType::SingleFloat,
            LowLevelType::Address,
        ] {
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                ConcreteType::Signed,
                "{ll:?} must project to Signed"
            );
        }
    }

    #[test]
    fn known_unported_classifies_indirect_call_adapter_invariant() {
        let msg = "translate_op: Call with CallTarget::Indirect at result=Some(ValueId(4)) \
                   must be lowered to VtableMethodPtr + IndirectCall by rclass.rs before \
                   reaching the flowspace adapter";
        assert!(
            is_known_unported(msg),
            "registry population must skip/fallback on the current indirect-call cutover gap"
        );
    }

    #[test]
    fn lowleveltype_to_concrete_float_family_collapses_to_float() {
        for ll in [LowLevelType::Float] {
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                ConcreteType::Float,
                "{ll:?} must project to Float"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_longlong_follows_target_word_size() {
        let expected = longlong_getkind_concrete();
        for ll in [LowLevelType::SignedLongLong, LowLevelType::UnsignedLongLong] {
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                expected,
                "{ll:?} must match history.getkind's sizeof(TYPE) > sizeof(Signed) branch"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_signedlonglonglong_returns_missing_rtype_err() {
        let err = lowleveltype_to_concrete(&LowLevelType::SignedLongLongLong)
            .expect_err("SignedLongLongLong has no history.getkind mapping");
        assert!(err.is_missing_rtype_operation());
    }

    #[test]
    fn lowleveltype_to_concrete_unsignedlonglonglong_returns_missing_rtype_err() {
        let err = lowleveltype_to_concrete(&LowLevelType::UnsignedLongLongLong)
            .expect_err("UnsignedLongLongLong has no history.getkind mapping");
        assert!(err.is_missing_rtype_operation());
    }

    #[test]
    fn lowleveltype_to_concrete_longfloat_returns_missing_rtype_err() {
        let err = lowleveltype_to_concrete(&LowLevelType::LongFloat)
            .expect_err("LongFloat has no history.getkind mapping");
        assert!(err.is_missing_rtype_operation());
    }

    #[test]
    fn lowleveltype_to_concrete_void_passes_through() {
        assert_eq!(
            lowleveltype_to_concrete(&LowLevelType::Void).expect("supported lltype"),
            ConcreteType::Void
        );
    }

    #[test]
    fn specialize_legacy_graph_minimal_int_identity_resolves_signed() {
        let _lock = anchor_lock();
        // Smallest validation: identity-return graph carrying a single
        // Int-typed inputarg. Slice 4 dual-gate's first anchor — proves
        // the adapter + annotator-surface seeding + specialize +
        // projection chain works end-to-end on a graph the rtyper can
        // resolve without any unported OpKind variants.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);

        let mut graph = LegacyGraph::new("identity_int");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(1))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let state = specialize_legacy_graph_with_seed_annotations(&graph, &annotations)
            .expect("identity int graph must specialize");

        assert_eq!(
            state.get(ValueId(1)),
            &ConcreteType::Signed,
            "Int-typed inputarg must specialize to Signed via SomeInteger → IntegerRepr"
        );
    }

    #[test]
    fn specialize_legacy_graph_minimal_float_identity_resolves_float() {
        let _lock = anchor_lock();
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Float);

        let mut graph = LegacyGraph::new("identity_float");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(1))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let state = specialize_legacy_graph_with_seed_annotations(&graph, &annotations)
            .expect("identity float graph must specialize");

        assert_eq!(
            state.get(ValueId(1)),
            &ConcreteType::Float,
            "Float-typed inputarg must specialize to Float via SomeFloat → FloatRepr"
        );
    }

    #[test]
    fn specialize_legacy_graph_ref_typed_inputarg_resolves_to_gcref() {
        let _lock = anchor_lock();
        // Cat 3.1 fix: `valuetype_to_someshell(Ref)` now lifts to
        // `SomeInstance(classdef=None)` instead of the illegal
        // `SomeObject` placeholder (`model.py:51-69` `SomeObject` is
        // abstract).  The rtyper routes through `getinstancerepr(rtyper,
        // None, Gc)` -> `InstanceRepr::new_rootinstance` ->
        // `Ptr(GcStruct(OBJECT))` and `lowleveltype_to_concrete`
        // collapses any GC pointer to `ConcreteType::GcRef`, matching
        // legacy `resolve_types(Ref) -> GcRef`.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Ref);

        let mut graph = LegacyGraph::new("identity_ref");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(1))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let state = specialize_legacy_graph_with_seed_annotations(&graph, &annotations)
            .expect("Ref-typed inputarg must specialize via SomeInstance(classdef=None)");
        assert_eq!(
            state.get(ValueId(1)),
            &ConcreteType::GcRef,
            "Ref-typed inputarg must project to GcRef matching legacy"
        );
    }

    // ───── Slice 4 dual-gate anchor tests ─────────────────────────
    //
    // Each anchor parses a small Rust DSL fragment, runs pyre's
    // `parse → front → SemanticProgram → FunctionGraph` chain, then
    // calls the legacy resolver and `dual_gate_check`. The expected
    // outcome — Ok or Err with a named OpKind / blocker — is asserted
    // so future Slice 1b followups (BinOp, FieldRead, ...) flip the
    // assertion automatically when their arm lands.
    //
    // Anchor selection covers (a) the trivially-resolvable identity
    // cases that already pass today, and (b) representative shapes
    // from the existing `pipeline_e2e_*` suite that surface the next
    // priority OpKind ports.

    /// Process-local guard serialising every anchor test that drives
    /// `RPythonTyper::specialize` end-to-end.
    ///
    /// The rtyper's lattice has process-global singletons (`bool_repr`,
    /// `none_repr`, the Repr setup-state machine) that mutate during
    /// `setup()`. Cargo's default parallel test runner can interleave
    /// two specialize-driving tests so one observes the other's
    /// `Setupstate::InProgress` and panics with "recursive invocation
    /// of Repr setup()" (`rmodel.rs:427`).
    ///
    /// Holding this `Mutex` for the entire specialize-driving section
    /// of each anchor serialises the runs without forcing
    /// `--test-threads=1` on the whole crate. If the rtyper's setup
    /// machinery is later reworked to be re-entrant, this lock can go
    /// away.
    fn anchor_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn build_anchor_graph(source: &str, fn_name: &str) -> LegacyGraph {
        let parsed = crate::parse::parse_source(source);
        let program = crate::front::build_semantic_program(&parsed)
            .expect("anchor source must build a semantic program");
        program
            .functions
            .iter()
            .find(|f| f.name == fn_name)
            .unwrap_or_else(|| {
                panic!(
                    "anchor source must define `{fn_name}`; got functions: {:?}",
                    program
                        .functions
                        .iter()
                        .map(|f| &f.name)
                        .collect::<Vec<_>>()
                )
            })
            .graph
            .clone()
    }

    fn run_legacy_resolve(graph: &LegacyGraph) -> (AnnotationState, TypeResolutionState) {
        let annotations = crate::translator::rtyper::legacy_annotator::annotate(graph);
        let state = crate::translator::rtyper::legacy_resolve::resolve_types(graph, &annotations);
        (annotations, state)
    }

    #[test]
    fn anchor_int_identity_dual_gate_agrees() {
        let _lock = anchor_lock();
        let graph = build_anchor_graph("fn id(x: i64) -> i64 { x }\n", "id");
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        // Trivial identity — both paths must produce the same kinds.
        // Currently legacy resolves the inputarg as Signed; real path
        // produces the same via SomeInteger → IntegerRepr → Signed.
        let result = dual_gate_check(&graph, &legacy_state);
        assert!(
            result.is_ok(),
            "Int identity must agree under dual-gate, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_float_identity_dual_gate_agrees() {
        let _lock = anchor_lock();
        let graph = build_anchor_graph("fn id(x: f64) -> f64 { x }\n", "id");
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph, &legacy_state);
        assert!(
            result.is_ok(),
            "Float identity must agree under dual-gate, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_int_addition_dual_gate_agrees() {
        let _lock = anchor_lock();
        // From `pipeline_e2e_simple_function`: `fn add(a, b) { a + b }`
        // produces `OpKind::BinOp{op:"add", ...}`. The Slice 1b BinOp
        // followup ports the arm as a pre-rtyper opname pass-through
        // (`add` → flowspace `SpaceOperation("add", ...)`); the real
        // rtyper then rewrites `add` → `int_add` for `Signed` operands
        // via `pair_int_int`, agreeing with the legacy resolver.
        let graph = build_anchor_graph("fn add(a: i64, b: i64) -> i64 { a + b }\n", "add");
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph, &legacy_state);
        assert!(
            result.is_ok(),
            "Int addition must agree under dual-gate post-BinOp port, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_load_fast_same_block_dedups_path_reads() {
        let _lock = anchor_lock();
        // Cat 3.2 first slice: a body `Expr::Path` whose name resolves
        // to a same-block local definition reuses the existing
        // `ValueId` instead of emitting a fresh `OpKind::Input`.
        // RPython `flowspace/flowcontext.py:835 LOAD_FAST` reads the
        // existing locals-stack entry; pyre's analogue forwards the
        // bound `ValueId`.
        //
        // The graph for `fn id(x: &Foo) -> &Foo { x }` lifts the
        // parameter as a single `OpKind::Input { name: "x", .. }` in
        // the entry block.  The body's `x` reference is in the same
        // block as the parameter binding, so the lookup returns the
        // existing `ValueId` and the graph carries exactly one
        // `OpKind::Input` named `"x"` — not two.
        let graph = build_anchor_graph(
            r#"
struct Foo {}
fn id(x: &Foo) -> &Foo { x }
"#,
            "id",
        );
        let input_x_count: usize = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    crate::model::OpKind::Input { name, .. } if name == "x"
                )
            })
            .count();
        assert_eq!(
            input_x_count, 1,
            "Cat 3.2 same-block dedup must collapse the body Path read \
             into the parameter's existing ValueId — expected exactly \
             one OpKind::Input{{name:\"x\"}}, got {input_x_count}"
        );
    }

    #[test]
    fn anchor_ref_identity_dual_gate_agrees() {
        let _lock = anchor_lock();
        // Cat 3.1 anchor: a typed-`Ref` identity graph must agree under
        // dual-gate now that `valuetype_to_someshell(Ref)` projects to
        // `SomeInstance(classdef=None)` instead of the illegal
        // `SomeObject` placeholder.  The rtyper routes through
        // `getinstancerepr(rtyper, None, Gc)` ->
        // `InstanceRepr::new_rootinstance` -> `Ptr(GcStruct(OBJECT))`,
        // which `lowleveltype_to_concrete` collapses to GcRef matching
        // legacy `resolve_types(Ref) -> GcRef`.
        let graph = build_anchor_graph(
            r#"
struct Foo {}
fn id(x: &Foo) -> &Foo { x }
"#,
            "id",
        );
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph, &legacy_state);
        assert!(
            result.is_ok(),
            "Ref-typed identity graph must agree under dual-gate via \
             SomeInstance(classdef=None) -> GcRef, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_int_negation_dual_gate_agrees() {
        let _lock = anchor_lock();
        // Cat 3.3 follow-on: `OpKind::UnaryOp{op:"neg",..}` ports as
        // a pre-rtyper opname pass-through (`neg` is registered
        // upstream by `operation.py:466 add_operator('neg', 1, ...)`)
        // and the rtyper rewrites `neg` -> `int_neg` for `Signed`
        // operands via the unary `pair_int_*` dispatch, agreeing with
        // the legacy resolver.
        let graph = build_anchor_graph("fn negate(x: i64) -> i64 { -x }\n", "negate");
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph, &legacy_state);
        assert!(
            result.is_ok(),
            "Int negation must agree under dual-gate post-UnaryOp port, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_unary_not_surfaces_failloud_no_flowspace_peer() {
        let _lock = anchor_lock();
        // RPython `UNARY_NOT` (`flowspace/flowcontext.py:531-538`)
        // expands to `op.bool(x).eval(self);
        // const(not self.guessbool(w_bool))` — emit `bool`, branch
        // via `guessbool`, then push the negated boolean constant.
        // Rust `!int` (bitwise) corresponds to UNARY_INVERT →
        // `op.invert` (`operation.py:474`).  Pyre's surface DSL
        // lacks a `Bool` ValueType so the bool-vs-invert dispatch
        // cannot be made faithfully here, and mapping `not` to a
        // bare `bool` op alone loses the negation.  Fail-loud
        // preserves pyre IR's `!x` semantics (consumed verbatim
        // downstream) until the frontend desugars `!cond` to
        // `bool` + branch and bitwise `!int` to `invert`.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("not_int");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![crate::model::SpaceOperation {
                result: Some(ValueId(2)),
                kind: crate::model::OpKind::UnaryOp {
                    op: "not".into(),
                    operand: ValueId(1),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(2))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(2)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let err = specialize_legacy_graph(&graph)
            .expect_err("UnaryOp `not` must surface a TyperError until frontend desugaring lands");
        let msg = format!("{err}");
        assert!(
            msg.contains("normalize_unary_op_name") && msg.contains("not"),
            "fail-loud must name the unsupported opname, got: {msg}"
        );
    }

    #[test]
    fn anchor_unary_deref_surfaces_failloud_no_flowspace_peer() {
        let _lock = anchor_lock();
        // `OpKind::UnaryOp{op:"deref",..}` (Rust `*x`) has no
        // RPython peer — the operand table at
        // `flowspace/operation.py:465-474` registers no `deref`,
        // and pyre carries no global invariant proving Rust `*x` is
        // type/reference-transparent.  Aliasing the result to the
        // operand could silently fold a real value load.  Until
        // the frontend either removes `deref` ops or proves the
        // invariant, the adapter must surface fail-loud.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("deref_int");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![crate::model::SpaceOperation {
                result: Some(ValueId(2)),
                kind: crate::model::OpKind::UnaryOp {
                    op: "deref".into(),
                    operand: ValueId(1),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(2))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(2)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let err = specialize_legacy_graph(&graph)
            .expect_err("UnaryOp `deref` must surface a TyperError until frontend invariant lands");
        let msg = format!("{err}");
        assert!(
            msg.contains("normalize_unary_op_name") && msg.contains("deref"),
            "fail-loud must name the unsupported opname, got: {msg}"
        );
    }

    #[test]
    fn anchor_field_read_surfaces_followup() {
        let _lock = anchor_lock();
        // From `pipeline_e2e_with_virtualizable`. Frame.next_instr is a
        // FieldRead. Slice 1b-core does not implement FieldRead →
        // fail-loud expected.
        let graph = build_anchor_graph(
            r#"
struct Frame { next_instr: usize, locals_w: Vec<i64> }
impl Frame {
    fn load_fast(&mut self) -> i64 {
        let idx = self.next_instr;
        self.locals_w[idx]
    }
}
"#,
            "load_fast",
        );
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        let err = dual_gate_check(&graph, &legacy_state)
            .expect_err("FieldRead surfaces a Slice 1b followup");
        assert!(
            err.contains("Slice 1b followup")
                || err.contains("FieldRead")
                || err.contains("real path failed"),
            "anchor must surface a named Slice 1b followup pending, got: {err}"
        );
    }

    #[test]
    fn anchor_control_flow_fib_surfaces_followup() {
        let _lock = anchor_lock();
        // fib has if/else, comparison, and arithmetic.  Earlier this
        // anchor was an expected-failure case because the per-block
        // `Input` op shape and `setup_block_entry` produced a
        // "wrong level!" assertion in the rtyper.  The framestate-
        // based cross-block local merge replaces the body-`Input`
        // emission with explicit phi inputargs, so the dual-gate now
        // agrees end-to-end; flip the assertion to a positive check.
        let graph = build_anchor_graph(
            r#"
fn fib(n: i64) -> i64 {
    if n <= 1 { return n; }
    let a = n - 1;
    let b = n - 2;
    a + b
}
"#,
            "fib",
        );
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        dual_gate_check(&graph, &legacy_state)
            .expect("fib must dual-gate-agree under the framestate cross-block merge");
    }

    #[test]
    fn specialize_legacy_graph_guard_value_skips_translation_keeps_kind() {
        let _lock = anchor_lock();
        // Cat 3.3 first slice: pyre JIT trace markers
        // (`GuardTrue` / `GuardFalse` / `GuardValue`) have no peer in
        // RPython flowspace's high-level operator set
        // (`operation.py:475-510`).  The adapter now skips them
        // (`Ok(Vec::new())`); the operand they read is defined
        // elsewhere and the absence of a result keeps the SSA chain
        // intact.  Specialize must succeed and project the Int operand
        // to `Signed`.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);

        let mut graph = LegacyGraph::new("guard_passthrough");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![crate::model::SpaceOperation {
                result: None,
                kind: crate::model::OpKind::GuardValue {
                    value: ValueId(1),
                    kind_char: 'i',
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(1))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let state = specialize_legacy_graph_with_seed_annotations(&graph, &annotations)
            .expect("GuardValue must be a no-op for the rtyper adapter");
        assert_eq!(
            state.get(ValueId(1)),
            &ConcreteType::Signed,
            "Int operand passing through GuardValue must specialize to Signed"
        );
    }

    #[test]
    fn specialize_legacy_graph_unported_opkind_propagates_failloud() {
        let _lock = anchor_lock();
        // Graph carrying a still-fail-loud OpKind (Call::Indirect —
        // requires rclass.rs lowering) must surface the variant's
        // fail-loud message — confirms the adapter's TyperError flows
        // through the full specialize pipeline.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("unported_call");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![crate::model::SpaceOperation {
                result: Some(ValueId(2)),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::Indirect {
                        trait_root: "MyTrait".into(),
                        method_name: "do_it".into(),
                    },
                    args: vec![ValueId(1)],
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(2))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(2)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let err = specialize_legacy_graph(&graph)
            .expect_err("unported OpKind must surface as TyperError");
        let msg = format!("{err}");
        // The empty-registry default entry surfaces an unregistered
        // fail-loud naming the segments and producer slice (A.4) that
        // should populate the registry; this graph uses Indirect to
        // exercise the rclass-rewrite invariant break instead.
        assert!(
            msg.contains("Indirect") && msg.contains("rclass"),
            "fail-loud must propagate the variant + rclass tag, got: {msg}"
        );
    }

    #[test]
    fn anchor_call_function_path_registered_emits_simple_call() {
        let _lock = anchor_lock();
        // Slice A.3c — when the registry is pre-populated with the
        // callee's FunctionPath, the adapter emits flowspace
        // `simple_call(callable_const, *args)` where `callable_const`
        // wraps the registry entry's HostObject.  The rtyper's
        // `bookkeeper.getdesc(host)` short-circuits at the cache
        // (Slice A.3a) returning the registry's pre-built
        // FunctionDesc.
        //
        // This anchor verifies the plumbing reaches at least the
        // rtyper's simple_call dispatch — whether the rtyper succeeds
        // or surfaces a downstream `pair_simple_call` followup
        // depends on FunctionDesc.cache wiring (graph build via the
        // bookkeeper's `specialize` pass).  The anchor accepts either
        // outcome and locks the failure mode so future slices can
        // flip the assertion.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("call_resolved");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![crate::model::SpaceOperation {
                result: Some(ValueId(2)),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: vec!["foo".into()],
                    },
                    args: vec![ValueId(1)],
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(2))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(2)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        // Pre-populate the registry with the callee's FunctionPath +
        // Signature + lifted leaf PyGraph in a single
        // `register_callee` call (Issue 2.2 atomic API).  Without
        // the cache pre-fill, `cachedgraph`
        // (description.rs:1037-1039) would miss and `pair_simple_call`
        // would invoke `buildgraph` on the synthetic `GraphFunc`,
        // which has no `HostCode` and would fail at
        // `translator.buildflowgraph`.
        let bookkeeper = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let registry =
            crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(bookkeeper);

        // Build a leaf callee `fn foo(x: i64) -> i64 { x }` —
        // identity returns the inputarg, no nested Calls so a
        // child `PyreCallRegistry` can stay empty during the lift.
        let mut callee_annotations = AnnotationState::new();
        callee_annotations.set(ValueId(10), ValueType::Int);
        let mut callee_graph = LegacyGraph::new("foo");
        let foo_start = Block {
            id: callee_graph.startblock,
            inputargs: vec![ValueId(10)],
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(10))],
                callee_graph.returnblock,
            )],
            framestate: None,
        };
        let foo_return = Block {
            id: callee_graph.returnblock,
            inputargs: vec![ValueId(10)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        callee_graph.blocks = vec![foo_start, foo_return];
        let leaf_registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(
            std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new()),
        );
        let pygraph = lift_callee_to_pygraph_seed(
            &callee_graph,
            Some(&callee_annotations),
            crate::flowspace::argument::Signature::new(vec!["x".to_string()], None, None),
            &leaf_registry,
        )
        .expect("leaf callee must lift to PyGraph");
        registry.register_callee(
            crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments(["foo"]),
            crate::flowspace::argument::Signature::new(vec!["x".to_string()], None, None),
            pygraph,
        );

        let (state, _annotations) =
            specialize_legacy_graph_with_registry_seed(&graph, Some(&annotations), &registry)
                .expect("Slice A.4 cache pre-fill must let the leaf Call resolve end-to-end");
        // Slice A.4 closes the loop:
        //   1. Slice A.3c emits `simple_call(host_obj_const, *args)`.
        //   2. Slice A.4's `prefill_default_cache` fills
        //      `FunctionDesc.cache[GraphCacheKey::None]` with the
        //      lifted leaf `PyGraph`, so `cachedgraph`
        //      (description.rs:1037-1039) returns the cached
        //      graph instead of calling `buildflowgraph` on the
        //      synthetic `GraphFunc`.
        //   3. `compute_at_fixpoint` populates
        //      `bookkeeper.pbc_maximal_call_families` /
        //      `calltable` from the seeded blocks' `simple_call` ops.
        //   4. `FunctionRepr.call(hop)` (rpbc.py:199) finds the
        //      call-family row, the rtyper completes
        //      `direct_call`, and the result Variable's
        //      `concretetype` is set to `Signed`
        //      (`int_is_true → Bool` style projection).
        assert_eq!(
            state.get(ValueId(2)),
            &ConcreteType::Signed,
            "leaf Call (fn foo(x: i64) -> i64 {{ x }}) result must project to Signed \
             through the rtyper's full simple_call → FunctionRepr.call chain"
        );
        assert_eq!(
            state.get(ValueId(1)),
            &ConcreteType::Signed,
            "Int inputarg must project to Signed (independent of the Slice A.4 \
             call resolution path)"
        );
    }

    #[test]
    fn populate_call_graphs_mirrors_value_to_var_to_alias_keys() {
        let _lock = anchor_lock();
        // Slice 3b — when two `CallPath` keys collapse to the same
        // canonical `PyreFunctionEntry` (the dedupe drops a leading
        // `crate::` segment), `populate_call_registry_from_call_graphs`
        // must mirror the lifted `value_to_var` and
        // `constant_concretetypes` snapshots to BOTH keys, so the
        // per-graph dual-gate's lookup by the subject's exact path
        // succeeds regardless of which alias of the canonical entry
        // got registered first.
        let graph = build_anchor_graph("fn helper(x: i64) -> i64 { x }\n", "helper");
        let canonical_path = crate::parse::CallPath::from_segments(["helper"]);
        let alias_path = crate::parse::CallPath::from_segments(["crate", "helper"]);
        let mut function_graphs: std::collections::HashMap<crate::parse::CallPath, LegacyGraph> =
            std::collections::HashMap::new();
        function_graphs.insert(canonical_path.clone(), graph.clone());
        function_graphs.insert(alias_path.clone(), graph);

        let registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(
            std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new()),
        );
        populate_call_registry_from_call_graphs(&function_graphs, &registry)
            .expect("populate must succeed for a simple identity function");

        // Both keys resolve to the same `Rc<PyreFunctionEntry>` (alias
        // dedupe contract).
        let canonical_key =
            crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments([
                "helper",
            ]);
        let alias_key =
            crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments([
                "crate", "helper",
            ]);
        let canonical_entry = registry
            .lookup(&canonical_key)
            .expect("canonical registered");
        let alias_entry = registry.lookup(&alias_key).expect("alias registered");
        assert!(
            std::rc::Rc::ptr_eq(&canonical_entry, &alias_entry),
            "alias must point at the same Rc<PyreFunctionEntry> as the canonical key"
        );

        // Issue 2.5 (2026-05-07): the canonical/alias mirror used to be
        // verified against per-`FunctionPathKey` `value_to_var` /
        // `constant_concretetypes` snapshot side maps.  Those side maps
        // were dead-write — no production reader — and were retired.
        // The shared-Rc identity assertion above now subsumes the alias
        // mirror invariant: any future reader of `Variable.concretetype`
        // walks the single canonical `PyGraph.graph` reachable through
        // either alias's `function_desc.cache`, so there is nothing to
        // mirror per-key.
    }

    #[test]
    fn anchor_populate_call_registry_from_program_registers_every_function() {
        let _lock = anchor_lock();
        // Slice A.5 — `populate_call_registry_from_program` must
        // register every `SemanticFunction` in `program.functions`
        // (Pass 1) and prefill each FunctionDesc.cache with a lifted
        // PyGraph (Pass 2).  After the walker runs, a caller graph
        // referencing any of those functions through
        // `OpKind::Call::FunctionPath` resolves end-to-end through
        // `specialize_legacy_graph_with_registry`.
        let parsed = crate::parse::parse_source(
            r#"
            fn helper(x: i64) -> i64 { x }
            fn caller(y: i64) -> i64 {
                helper(y)
            }
        "#,
        );
        let program = crate::front::build_semantic_program(&parsed).expect("source must lower");

        let registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(
            std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new()),
        );
        populate_call_registry_from_program(&program, &registry)
            .expect("walker must register every program function without error");

        // Both functions registered.
        assert_eq!(
            registry.len(),
            program.functions.len(),
            "registry must hold one entry per SemanticFunction in the program"
        );
        let helper_entry = registry
            .lookup(
                &crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments([
                    "helper",
                ]),
            )
            .expect("helper must be registered under its function name");
        let caller_entry = registry
            .lookup(
                &crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments([
                    "caller",
                ]),
            )
            .expect("caller must be registered too — walker covers every function");

        // Both FunctionDescs carry their lifted PyGraph.
        assert!(
            helper_entry
                .function_desc
                .borrow()
                .cache
                .borrow()
                .contains_key(&crate::annotator::description::GraphCacheKey::None),
            "helper's FunctionDesc.cache must be prefilled with the lifted PyGraph"
        );
        assert!(
            caller_entry
                .function_desc
                .borrow()
                .cache
                .borrow()
                .contains_key(&crate::annotator::description::GraphCacheKey::None),
            "caller's FunctionDesc.cache must be prefilled too"
        );
        assert_eq!(
            helper_entry.function_desc.borrow().signature.argnames,
            vec!["x".to_string()],
            "helper's signature must come from its declared parameter list"
        );
        assert_eq!(
            caller_entry.function_desc.borrow().signature.argnames,
            vec!["y".to_string()],
            "caller's signature must come from its declared parameter list"
        );
    }

    #[test]
    fn anchor_load_fast_repeated_cross_block_read_dedups_within_block() {
        let _lock = anchor_lock();
        // Cat 3.2 same-block dedup invariant — orthogonal to Cat 2.1
        // Slice 2's lazy cross-block install.  RPython parity:
        // `flowspace/flowcontext.py:835 LOAD_FAST` reads the existing
        // locals slot rather than introducing a fresh Variable on
        // every read, so `x + x` inside one block must collapse to a
        // single `OpKind::Input { name: "x" }` followed by a single
        // BinOp consuming that op's result twice.
        //
        // Pre-Slice-2: the post-`if` block emitted a naked
        // `OpKind::Input{name:"x"}` directly (no inputarg, no
        // `Link.args` thread-back) and the dedup invariant held
        // trivially because there was only one cross-block reading
        // block.  Post-Slice-2: lazy thread-back installs an
        // `OpKind::Input{name:"x"}` per block on the predecessor
        // chain (entry parameter + each empty pass-through merge +
        // the actual reading block), and the dedup invariant becomes
        // a per-block assertion: no single block emits more than one
        // `Input{name:"x"}`.
        let graph = build_anchor_graph(
            r#"
fn cross_block(x: i64, cond: bool) -> i64 {
    if cond { return 0; }
    x + x
}
"#,
            "cross_block",
        );
        for block in &graph.blocks {
            let count: usize = block
                .operations
                .iter()
                .filter(|op| {
                    matches!(
                        &op.kind,
                        crate::model::OpKind::Input { name, .. } if name == "x"
                    )
                })
                .count();
            assert!(
                count <= 1,
                "Cat 3.2 same-block dedup: B{} emitted {} `Input{{name:\"x\"}}` \
                 ops but `x + x` must collapse to one within a single block",
                block.id.0,
                count,
            );
        }
        let total_input_x: usize = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    crate::model::OpKind::Input { name, .. } if name == "x"
                )
            })
            .count();
        assert!(
            total_input_x >= 1,
            "Cat 3.2 cross-block read: at least the entry block must define \
             `OpKind::Input{{name:\"x\"}}` for the function parameter",
        );
    }

    #[test]
    fn anchor_cat_2_1_both_open_arm_rebind_post_merge_read_phi_threads() {
        let _lock = anchor_lock();
        // Cat 2.1 Stage C — both-open-arm + per-arm rebind merge.
        // RPython `flowspace/flowcontext.py` parity: at the merge
        // point of `if cond { x = 1 } else { x = 2 }; x`, the post-
        // merge `LOAD_FAST x` resolves to a fresh phi inputarg in
        // the merge block whose `Link.args` from each arm carry the
        // arm-rebound `ValueId`.
        //
        // Stage A1-A3 + B1-B2 + C1 minimal land the foundations: per-
        // block `framestate`, first-bind positional slot order, union
        // / getoutputargs, ctx restore between arms, relaxed lazy
        // installer fence, None-kill of one-arm-only locals.  The
        // lazy installer (Slice 2) handles the cross-block read of
        // the rebound `x` at the post-merge use site, allocating a
        // single merge-block inputarg + threading each arm's rebound
        // `ValueId` back through that arm's `Link.args`.
        //
        // This anchor pins the both-open-arm + rebind shape end-to-
        // end via `dual_gate_check`.  Per-block `Link.args` arity =
        // target `inputargs` arity is checked separately to
        // surface any predecessor / target mismatch the lazy install
        // might leave behind.
        let graph = build_anchor_graph(
            r#"
fn rebind_both_arms(cond: bool) -> i64 {
    let x: i64;
    if cond { x = 1; } else { x = 2; }
    x + x
}
"#,
            "rebind_both_arms",
        );
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        dual_gate_check(&graph, &legacy_state).expect(
            "Cat 2.1 Stage C — both-open-arm rebind merge resolves via lazy Link.args + phi inputarg",
        );
        for block in &graph.blocks {
            for link in &block.exits {
                let target_arity = graph.block(link.target).inputargs.len();
                assert_eq!(
                    link.args.len(),
                    target_arity,
                    "Stage C invariant: Link from B{} to B{} arity {} != target inputargs {}",
                    block.id.0,
                    link.target.0,
                    link.args.len(),
                    target_arity,
                );
            }
        }
    }

    #[test]
    fn anchor_cat_2_1_cross_block_reads_pass_dual_gate_after_link_args_thread() {
        let _lock = anchor_lock();
        // Cat 2.1 Slice 2 — cross-block locals threading via lazy
        // `Link.args` / target-`inputarg` install at the actual
        // cross-block read site.  RPython
        // `flowspace/flowcontext.py:835 LOAD_FAST` parity.
        //
        // Slice 2 wires `front/ast.rs::lower_expr`'s `Expr::If` arm
        // through `lazy_install_local_at_current_block`: when a
        // post-`if` block performs a `LOAD_FAST` of a local whose
        // definition lives in a dominating block, the read site
        // allocates a fresh `OpKind::Input` in the merge block,
        // promotes it to `inputargs`, and walks back to every
        // predecessor edge whose `set_branch` / `set_goto` recorded a
        // `LocalsFrameState` snapshot, appending the predecessor-side
        // `ValueId` to that link's `args` so `len(link.args) ==
        // len(target.inputargs)` per `flowspace/model.py:114
        // Link.__init__`.
        //
        // This shape (single-open-arm-no-rebind: `if cond { return; }
        // x + x`) is the simplest case — only one predecessor, no
        // rebinding inside the open arm.  Slices 3-6 extend the
        // recording to the both-open-arm + match + loop + break /
        // continue join sites.
        //
        // The previous incarnation of this anchor pinned the failure
        // mode via `expect_err`; Slice 2 closes that gap so the
        // assertion flips to `expect`.  The graph-shape checks below
        // pin the post-Slice-2 invariants — the merge block has its
        // own inputargs and the predecessor's `link.args` arity now
        // matches it.
        let graph = build_anchor_graph(
            r#"
fn cross_block(x: i64, cond: bool) -> i64 {
    if cond { return 0; }
    x + x
}
"#,
            "cross_block",
        );
        let (annotations, legacy_state) = run_legacy_resolve(&graph);
        dual_gate_check(&graph, &legacy_state)
            .expect("Cat 2.1 — Slice 2 closes cross-block locals via lazy Link.args threading");

        // Per-block invariant: every link from `block` to its target
        // carries `args` whose arity matches the target's inputargs.
        // Equivalent to `flowspace/model.py:114 Link.__init__` +
        // `:checkgraph`.
        for block in &graph.blocks {
            for link in &block.exits {
                let target_arity = graph.block(link.target).inputargs.len();
                assert_eq!(
                    link.args.len(),
                    target_arity,
                    "Slice 2 invariant: Link from B{} to B{} arity {} != target inputargs {}",
                    block.id.0,
                    link.target.0,
                    link.args.len(),
                    target_arity,
                );
            }
        }
    }

    // `build_program_annotator_starts_empty_after_step3` retired at
    // Step 4 first slice (2026-05-07).  The function being tested
    // (`build_program_annotator`) was retired together with the
    // dead-pass `build_program_rtyper` and `ensure_program_specialize`
    // helpers, since after Step 3 they ran a no-op against an empty
    // annotator.  Per-session annotator construction inside
    // `specialize_legacy_graph_with_registry_seed` is the only
    // remaining production lifecycle.
}
