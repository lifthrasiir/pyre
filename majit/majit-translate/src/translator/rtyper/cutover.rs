//! P8.3b — `specialize_legacy_graph` cutover wrapper.
//!
//! Drives the real `RPythonTyper::specialize` against a legacy
//! `model::FunctionGraph` by way of the
//! [`crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace`]
//! adapter, then projects each per-`Variable` `LowLevelType` back to a
//! `ConcreteType` keyed by the original legacy `ValueId`.
//!
//! ## Why this file is in `translator/rtyper/` and not `translate_legacy/`
//!
//! `translate_legacy/` is a documented PRE-EXISTING-ADAPTATION queued
//! for deletion at Slice 10. The Slice 4 dual-gate, the Slice 5 default
//! flip, and the Slice 6 prod migration all call into the entry point
//! defined here, so the entry point must survive Slice 10 — it lives in
//! `translator/` (the RPython-orthodox future home). The
//! `flowspace_adapter` sibling module is similarly long-lived: it
//! bridges pyre's surface-DSL `model::FunctionGraph` to the RPython
//! `flowspace::FunctionGraph` shape the rtyper consumes, until pyre's
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
//! followup is the next priority. `Ref`-typed operands additionally
//! fail in `bindingrepr` per the documented Slice 1a blocker
//! (`SomeObject.rtyper_makerepr` returns
//! `TyperError::missing_rtype_operation` from `rmodel.rs:2475`).

use std::rc::Rc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::flowspace::model::{BlockKey, BlockRef, GraphRef};
use crate::jit_codewriter::annotation_state::AnnotationState;
use crate::jit_codewriter::type_state::{ConcreteType, TypeResolutionState};
use crate::model::{FunctionGraph as LegacyGraph, ValueType};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::flowspace_adapter::{
    FlowspaceAdapterOutput, function_graph_to_flowspace,
};
use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType};
use crate::translator::rtyper::rtyper::RPythonTyper;

/// Project a post-`specialize` `LowLevelType` back to the legacy
/// `ConcreteType` bucket the codewriter consumes (Signed / Float /
/// GcRef / Void). Returns `ConcreteType::Unknown` only when the
/// `Variable.concretetype` slot was never populated (i.e. the rtyper
/// silently skipped the variable — Slice 4 anchor tests will surface
/// any silent skip as a divergence).
///
/// Mapping follows RPython's JIT `history.getkind(TYPE)`:
/// `Void -> void`, primitive float -> float, `SingleFloat -> int`,
/// machine-word primitives and raw pointers -> int, GC pointers -> ref.
/// `SignedLongLong` / `UnsignedLongLong` follow PyPy's target-size
/// check: they are float kind only when they are wider than `Signed`
/// (32-bit targets); on 64-bit targets they collapse to int kind.
/// Unsupported container/non-pointer shapes stay `Unknown`.
pub fn lowleveltype_to_concrete(ll: &LowLevelType) -> ConcreteType {
    match ll {
        LowLevelType::Void => ConcreteType::Void,
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::Bool
        | LowLevelType::Char
        | LowLevelType::UniChar
        | LowLevelType::SingleFloat
        | LowLevelType::Address => ConcreteType::Signed,
        LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong => {
            longlong_getkind_concrete()
        }
        LowLevelType::Float => ConcreteType::Float,
        LowLevelType::Ptr(ptr) => {
            if ptr._gckind() == GcKind::Raw {
                ConcreteType::Signed
            } else {
                ConcreteType::GcRef
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
        | LowLevelType::InteriorPtr(_) => ConcreteType::Unknown,
    }
}

fn longlong_getkind_concrete() -> ConcreteType {
    if std::mem::size_of::<i64>() > std::mem::size_of::<isize>() {
        ConcreteType::Float
    } else {
        ConcreteType::Signed
    }
}

fn valuetype_to_concrete(vt: &ValueType) -> ConcreteType {
    match vt {
        ValueType::Int => ConcreteType::Signed,
        ValueType::Float => ConcreteType::Float,
        ValueType::Ref => ConcreteType::GcRef,
        ValueType::Void => ConcreteType::Void,
        ValueType::State | ValueType::Unknown => ConcreteType::Unknown,
    }
}

/// Seed `RPythonAnnotator.annotated` and `RPythonAnnotator.all_blocks`
/// from the adapter output. `RPythonTyper::specialize_more_blocks`
/// (`rtyper.rs:1791-1813`) iterates `annotated.keys()` and resolves
/// each key through `all_blocks.get(...)`, so both maps must contain
/// every block the rtyper should specialize.
///
/// `annotated`'s value is `Option<GraphRef>`. `Some(graph)` (the third
/// state in upstream's `Some(Some(graph))` chain — see
/// annrpython.py:265-267) attaches the block to a graph.
/// `specialize_block` (`rtyper.rs:1610-1625`) panics on the
/// `Some(None)` "False sentinel" branch — that case represents
/// upstream's stripped-graph state, which pyre's legacy adapter does
/// not produce. The adapter's single `flowspace::FunctionGraph` is the
/// owning graph for every translated block, so we register `Some(graph)`
/// uniformly across the seeded entries.
fn seed_annotator_blocks(
    annotator: &Rc<RPythonAnnotator>,
    graph: &GraphRef,
    block_map: &std::collections::HashMap<crate::model::BlockId, BlockRef>,
) {
    let mut annotated = annotator.annotated.borrow_mut();
    let mut all_blocks = annotator.all_blocks.borrow_mut();
    for block_ref in block_map.values() {
        let key = BlockKey::of(block_ref);
        annotated.insert(key.clone(), Some(graph.clone()));
        all_blocks.insert(key, block_ref.clone());
    }
}

/// Slice 4 dual-gate validation against the legacy resolver.
///
/// Re-runs `specialize_legacy_graph` against the same `legacy_graph` +
/// `annotations` the legacy `resolve_types` ran on, then diffs the
/// per-`ValueId` `ConcreteType` projection.
///
/// Returns `Err(message)` when:
///
/// - the real path errors out (typer error from an unported `OpKind`
///   arm or a documented Slice 4 blocker like the `Ref → SomeObject`
///   arm), OR
/// - a legacy-known `ValueId` is missing / `Unknown` / different in
///   the real path, OR
/// - the real path produced a definite kind for a `ValueId` the legacy
///   resolver did not resolve.
///
/// Returns `Ok(())` only when the two `TypeResolutionState` projections
/// agree on every definite kind. A real-path `Unknown` for a
/// legacy-known value is a coverage bug, not success.
///
/// Slice 5's default flip will turn this from a validation gate into
/// the production-call replacement; Slice 6 migrates prod callsites;
/// Slice 10 deletes the legacy half. Until then, this function is the
/// single audit point for "does the cutover preserve every kind?".
pub fn dual_gate_check(
    legacy_graph: &LegacyGraph,
    annotations: &AnnotationState,
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
        specialize_legacy_graph(legacy_graph, annotations)
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

/// Specialize a legacy `model::FunctionGraph` end-to-end through the
/// real `RPythonTyper`.
///
/// Returns a `TypeResolutionState` keyed by the original legacy
/// `ValueId` — drop-in replacement for legacy
/// `translate_legacy::rtyper::rtyper::resolve_types` once Slice 4
/// dual-gate validates the projection on the anchor corpus.
pub fn specialize_legacy_graph(
    legacy: &LegacyGraph,
    annotations: &AnnotationState,
) -> Result<TypeResolutionState, TyperError> {
    // ── Step 1 — Slice 1c adapter ──────────────────────────────────
    let FlowspaceAdapterOutput {
        graph,
        value_to_var,
        constant_value_ids,
        block_map,
    } = function_graph_to_flowspace(legacy, annotations)?;

    // ── Step 2 — annotator surface ────────────────────────────────
    let annotator = RPythonAnnotator::new(None, None, None, false);
    seed_annotator_blocks(&annotator, &graph, &block_map);

    // ── Step 3 — rtyper construction + specialize ─────────────────
    let rtyper = Rc::new(RPythonTyper::new(&annotator));
    rtyper.initialize_exceptiondata()?;
    // dont_simplify_again=true: pyre's legacy graph is already in
    // simplified SSA shape and the simplify pass requires a populated
    // `translator.entry_point_graph` that this entry point cannot
    // supply.
    rtyper.specialize(true)?;

    // ── Step 4 — read back per-ValueId ConcreteType ───────────────
    let mut state = TypeResolutionState::new();
    for (&vid, var) in &value_to_var {
        let concretetype = var.concretetype.borrow();
        if let Some(lltype) = concretetype.as_ref() {
            state
                .concrete_types
                .insert(vid, lowleveltype_to_concrete(lltype));
        }
    }
    for vid in constant_value_ids {
        let concrete = valuetype_to_concrete(annotations.get(vid));
        if concrete != ConcreteType::Unknown {
            state.concrete_types.insert(vid, concrete);
        }
    }
    Ok(state)
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
                lowleveltype_to_concrete(&ll),
                ConcreteType::Signed,
                "{ll:?} must project to Signed"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_float_family_collapses_to_float() {
        for ll in [LowLevelType::Float] {
            assert_eq!(
                lowleveltype_to_concrete(&ll),
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
                lowleveltype_to_concrete(&ll),
                expected,
                "{ll:?} must match history.getkind's sizeof(TYPE) > sizeof(Signed) branch"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_unsupported_shapes_stay_unknown() {
        for ll in [
            LowLevelType::SignedLongLongLong,
            LowLevelType::UnsignedLongLongLong,
            LowLevelType::LongFloat,
        ] {
            assert_eq!(
                lowleveltype_to_concrete(&ll),
                ConcreteType::Unknown,
                "{ll:?} must stay Unknown like getkind's unsupported branch"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_void_passes_through() {
        assert_eq!(
            lowleveltype_to_concrete(&LowLevelType::Void),
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
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        graph.blocks = vec![startblock, returnblock];

        let state = specialize_legacy_graph(&graph, &annotations)
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
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        graph.blocks = vec![startblock, returnblock];

        let state = specialize_legacy_graph(&graph, &annotations)
            .expect("identity float graph must specialize");

        assert_eq!(
            state.get(ValueId(1)),
            &ConcreteType::Float,
            "Float-typed inputarg must specialize to Float via SomeFloat → FloatRepr"
        );
    }

    #[test]
    fn specialize_legacy_graph_ref_typed_inputarg_surfaces_slice4_blocker() {
        let _lock = anchor_lock();
        // Slice 1a's documented blocker: ValueType::Ref → SomeObject,
        // and `rmodel.rs:2475` returns
        // `TyperError::missing_rtype_operation` for SomeObject. This
        // test pins the failure mode so any future fix (robject port
        // or AnnotationState enrichment) flips this assertion.
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
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        graph.blocks = vec![startblock, returnblock];

        let err = specialize_legacy_graph(&graph, &annotations)
            .expect_err("Ref-typed inputarg must surface the documented Slice 4 blocker");
        let msg = format!("{err}");
        assert!(
            msg.contains("SomeObject") || msg.contains("rtype"),
            "Slice 4 blocker should be a SomeObject / rtyper-related error, got: {msg}"
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
        let annotations = crate::translate_legacy::annotator::annrpython::annotate(graph);
        let state = crate::translate_legacy::rtyper::rtyper::resolve_types(graph, &annotations);
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
        let result = dual_gate_check(&graph, &annotations, &legacy_state);
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
        let result = dual_gate_check(&graph, &annotations, &legacy_state);
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
        let result = dual_gate_check(&graph, &annotations, &legacy_state);
        assert!(
            result.is_ok(),
            "Int addition must agree under dual-gate post-BinOp port, got: {:?}",
            result
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
        let err = dual_gate_check(&graph, &annotations, &legacy_state)
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
        // From `pipeline_e2e_with_control_flow`. fib has if/else (so
        // multi-block topology Slice 1c handles), comparison (BinOp),
        // and arithmetic (BinOp). Post-BinOp port the BinOp arm
        // resolves; the deeper blocker is now multi-block link/phi
        // shape (e.g. how pyre-front's per-block Input ops interact
        // with `setup_block_entry` for non-startblock inputargs), which
        // surfaces via the rtyper's "wrong level!" assertion that
        // `dual_gate_check` catches and stringifies as "real path
        // panicked".
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
        let err = dual_gate_check(&graph, &annotations, &legacy_state)
            .expect_err("fib surfaces a Slice 1b followup");
        assert!(
            err.contains("Slice 1b followup")
                || err.contains("real path panicked")
                || err.contains("BinOp")
                || err.contains("real path failed"),
            "anchor must surface a named Slice 1b followup pending, got: {err}"
        );
    }

    #[test]
    fn specialize_legacy_graph_unported_opkind_propagates_failloud() {
        let _lock = anchor_lock();
        // Graph carrying an unported OpKind (Call) must surface the
        // Slice 1b followup pending message — confirms the adapter's
        // fail-loud flows through the full specialize pipeline.
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
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(2)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        graph.blocks = vec![startblock, returnblock];

        let err = specialize_legacy_graph(&graph, &annotations)
            .expect_err("unported OpKind must surface as TyperError");
        let msg = format!("{err}");
        assert!(
            msg.contains("Call") && msg.contains("Slice 1b followup"),
            "fail-loud must propagate the variant + slice tag, got: {msg}"
        );
    }
}
