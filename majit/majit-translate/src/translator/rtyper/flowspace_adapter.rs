//! P8.3a — `model::FunctionGraph` (legacy) → `flowspace::FunctionGraph`
//! (real) adapter scaffolding.
//!
//! This file is a **PRE-EXISTING-ADAPTATION** with no upstream RPython
//! counterpart. RPython's pipeline never has a "legacy graph model" to
//! convert from — the annotator builds its `FunctionGraph` (from
//! `rpython/flowspace/model.py`) directly, and the rtyper consumes it
//! in place. Pyre carries two graph models in parallel during the
//! Phase 8 cutover (`crate::model::FunctionGraph` for the legacy
//! `translate_legacy/` pipeline, and
//! `crate::flowspace::model::FunctionGraph` for the real
//! `translator/rtyper/` pipeline) — this adapter exists solely to
//! bridge the gap until P8.8 deletes the legacy graph and its callers.
//!
//! ## Why not `PyGraph`?
//!
//! `PyGraph` (`flowspace::pygraph::PyGraph`) wraps a `FunctionGraph`
//! with `GraphFunc` / `HostCode` / `Signature` / `defaults` —
//! Python-runtime function metadata that pyre's surface DSL does not
//! produce (`parse → front → SemanticProgram` operates on Rust source,
//! not CPython callables). `RPythonTyper::specialize`
//! (`rtyper.rs:1743`) does NOT consume `PyGraph` directly — it iterates
//! `RPythonAnnotator.annotated` / `all_blocks`, which Slice 2's
//! `specialize_legacy_graph` will populate with the
//! [`FlowspaceAdapterOutput`] this adapter returns. Skipping the PyGraph
//! wrapping avoids fabricating fake `GraphFunc` / `HostCode` instances.
//!
//! ## Slice progression
//!
//! Per the local rtyper cutover plan:
//!
//! - **Slice 1a:** annotation lift — project legacy
//!   `AnnotationState.types` (`ValueId → ValueType`) to `SomeValue`
//!   shells on freshly-allocated `flowspace::Variable`s. Variable
//!   identity remains block-local per `flowspace/model.py:checkgraph`;
//!   the adapter only keeps `ValueId → Variable` representatives for
//!   post-specialize readback.
//! - **Slice 1b:** per-OpKind translation table — Slice 1b-core landed
//!   the dispatcher framework + skip arms (Input / ConstInt /
//!   ConstFloat). The first followup ports the `BinOp` arm as a
//!   pre-rtyper opname pass-through (`add`/`sub`/`lt`/... → flowspace
//!   `SpaceOperation` of the same name). Remaining per-variant
//!   lowering arms (Call / FieldRead / ArrayRead / ...) land as later
//!   Slice 1b-followup commits.
//! - **Slice 1c (current):** block topology — wire `flowspace::Block`
//!   instances per legacy `Block`, translate `exits` / `exitcase` /
//!   `exitswitch`, designate `startblock` / `returnblock` /
//!   `exceptblock`, and assemble a `flowspace::FunctionGraph`.
//!   `getreturnvar` (`rtyper.rs:1633-1638`) becomes non-degenerate
//!   because the returnblock's inputarg is materialised as the
//!   canonical flowspace return `Variable`.
//! - **Slice 2:** `specialize_legacy_graph` wrapper (in
//!   `translator/rtyper/cutover.rs`) drives this adapter, runs
//!   `RPythonTyper::specialize`, projects `LowLevelType → ConcreteType`,
//!   and returns a [`TypeResolutionState`] keyed by the original
//!   legacy `ValueId`.
//! - **Slices 4–10:** dual-gate validation, default flip, prod
//!   migration, test fixture migration, legacy deletion.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::annotator::model::{SomeFloat, SomeInteger, SomeObjectBase, SomeValue};
use crate::flowspace::model::{
    self as flowspace_model, Block as FlowspaceBlock, BlockRef, ConstValue, Constant,
    FunctionGraph as FlowspaceGraph, Hlvalue, Link as FlowspaceLink, SpaceOperation as FlowspaceOp,
    Variable, c_last_exception,
};
use crate::jit_codewriter::annotation_state::AnnotationState;
use crate::model::{
    BlockId, ExitCase, ExitSwitch, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueId,
    ValueType,
};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

/// Map from legacy `ValueId` to a representative `flowspace::Variable`
/// the adapter created for readback.
///
/// This is not the graph's identity model. RPython `checkgraph` requires
/// block inputargs and operation results to be defined in exactly one
/// block, so [`function_graph_to_flowspace`] uses block-local Variables
/// while translating the actual graph. The representative map lets
/// Slice 2 project `Variable.concretetype` back into pyre's legacy
/// `ValueId` keyed `TypeResolutionState`.
pub type ValueIdToVariable = HashMap<ValueId, Variable>;

/// RPython `SomeValue` lattice projection of the legacy `ValueType`.
///
/// `RPythonTyper.bindingrepr` (rtyper.rs:961) dispatches purely on the
/// `SomeValue` shape via `rtyper_makekey` / `rtyper_makerepr`. `Int`
/// and `Float` resolve cleanly through `rint::IntegerRepr` /
/// `rfloat::FloatRepr`. The `Ref` and `Unknown` arms have **known
/// blockers** in Slice 4 — see the table below.
///
/// Mapping:
///
/// | legacy `ValueType` | `SomeValue` shell      | RPython source / status |
/// |--------------------|------------------------|--------------------------|
/// | `Int`              | `Integer(SomeInteger)` | model.py:206-224 — resolves via [`crate::translator::rtyper::rmodel::rtyper_makerepr`] `SomeValue::Integer` arm. |
/// | `Float`            | `Float(SomeFloat)`     | model.py:164-183 — resolves via the `SomeValue::Float` arm. |
/// | `Ref`              | `Object(SomeObjectBase)` | model.py:51-125 lattice top. **Slice 4 blocker:** `rmodel.rs:2475` returns `TyperError::missing_rtype_operation("SomeObject.rtyper_makerepr — port rpython/rtyper/robject.py")`. The dual-gate cannot resolve any `Ref`-typed operand until either (a) `rpython/rtyper/robject.py` lands or (b) pyre's legacy `AnnotationState` is enriched to carry classdef metadata so this projection can return `SomeInstance` instead. |
/// | `Void`             | `Impossible`           | model.py:627 — resolves via `SomeValue::Impossible` arm to `impossible_repr`. |
/// | `State`/`Unknown`  | `Object(SomeObjectBase)` | Same Slice 4 blocker as `Ref`. `State`/`Unknown` are pyre-only `ValueType` variants with no RPython lattice point; the fallback keeps Slice 1a self-consistent but inherits the `Ref` failure mode. |
///
/// `SomeInteger::default()` returns `nonneg=false, unsigned=false,
/// knowntype=Int` (model.rs:502-505) — the bottom of the integer
/// sub-lattice. `SomeFloat::default()` analogously returns
/// `knowntype=Float, immutable=true` (model.rs:352-355).
///
/// **Slice 1a scope deliberately excludes the fix** for the `Ref`
/// blocker. Discovery of the failure mode is what Slice 4's dual-gate is
/// designed for — the anchor test corpus will pin down which legacy
/// graphs actually flow Ref-typed operands through `bindingrepr`, so
/// the fix can be sized accurately (full robject port vs. minimal
/// Instance shell vs. AnnotationState enrichment).
pub fn valuetype_to_someshell(vt: &ValueType) -> SomeValue {
    match vt {
        ValueType::Int => SomeValue::Integer(SomeInteger::default()),
        ValueType::Float => SomeValue::Float(SomeFloat::default()),
        ValueType::Ref => SomeValue::Object(SomeObjectBase::default()),
        ValueType::Void => SomeValue::Impossible,
        ValueType::State | ValueType::Unknown => SomeValue::Object(SomeObjectBase::default()),
    }
}

/// Allocate a fresh `flowspace::Variable` and attach the projected
/// `SomeValue` shell to its `annotation` slot.
///
/// The legacy `ValueId` does NOT carry over to `Variable.id` —
/// `Variable::new` allocates a fresh process-wide identity
/// (`flowspace/model.rs:2042`). Identity correspondence is preserved
/// out-of-band by [`ValueIdToVariable`].
fn seed_variable(vid: ValueId, annotations: &AnnotationState) -> Variable {
    let var = Variable::new();
    let shell = valuetype_to_someshell(annotations.get(vid));
    *var.annotation.borrow_mut() = Some(Rc::new(shell));
    let _ = vid; // suppress unused-warning until Slice 1b consumes it via debug naming
    var
}

/// Build the `ValueId → flowspace::Variable` map for every value
/// reachable from `legacy.blocks`.
///
/// Three reference-site classes seed the map:
///
/// 1. **Definitions** — `block.inputargs` (RPython-orthodox phi nodes)
///    and `op.result`. Every operand referenced via `op.args` /
///    `link.args` / `exitswitch` resolves to a definition site in the
///    same graph (legacy `FunctionGraph` is mostly SSA), so seeding
///    definitions covers most of the value set.
///
/// 2. **Link-side sentinels** — `link.args` / `link.last_exception` /
///    `link.last_exc_value`. RPython `flowspace/model.py:114` and
///    pyre's front (`front/ast.rs:5320-5331`) allow a `Link.args` slot
///    to carry a *fresh* prevblock-side `Variable` whose only "defining
///    site" is the link itself — the value flows into the target
///    block's inputarg via this synthetic Variable. The adapter must
///    seed a `Variable` for each such `ValueId` so Slice 1c's link
///    translation can resolve the operand without tripping the
///    "undefined operand" invariant in `lookup_operand`.
///
/// 3. **Exitswitch values** — `block.exitswitch = Some(ExitSwitch::Value(vid))`
///    sometimes references a `ValueId` defined in a successor block's
///    inputarg context (rarely but legitimately in legacy graphs).
///    Seeded for the same reason.
///
/// Each `ValueId` is seeded exactly once via `entry().or_insert_with`,
/// preserving operand identity across multiple readers — Slice 1b's op
/// translator looks up the same Variable instance for every reader of a
/// given `ValueId`, matching upstream Python's reference semantics where
/// `op.args[i]` and `op2.args[j]` may be the same `Variable` object.
pub fn build_value_to_variable_map(
    legacy: &FunctionGraph,
    annotations: &AnnotationState,
) -> ValueIdToVariable {
    let mut map: ValueIdToVariable = HashMap::new();
    for block in &legacy.blocks {
        // Class 1a — block-inputarg definitions.
        for &vid in &block.inputargs {
            map.entry(vid)
                .or_insert_with(|| seed_variable(vid, annotations));
        }

        // Per-block name → inputarg-Variable lookup for `OpKind::Input`
        // rebind-aliasing. Pyre's surface front (`front/ast.rs`) emits a
        // *leading* `Input{name, ty}` op for each named parameter whose
        // `op.result` matches a `block.inputargs` entry, and may emit
        // *additional* `Input{same name}` ops with fresh `op.result`
        // ValueIds for body-side rebinds. RPython's flowspace has no
        // such Input op machinery — the parameter Variable IS the
        // inputarg. Without aliasing the rebind result to the
        // canonical inputarg Variable, `setup_block_entry` writes
        // `concretetype` only on the inputarg's Rc<RefCell> and the
        // body's BinOp lookup hits a fresh Variable with `None`
        // concretetype, tripping `genop`'s "wrong level!" assertion.
        let mut name_to_inputarg_var: HashMap<&str, Variable> = HashMap::new();
        for op in &block.operations {
            if let OpKind::Input { name, .. } = &op.kind {
                if let Some(result) = op.result {
                    if block.inputargs.contains(&result) {
                        if let Some(var) = map.get(&result) {
                            name_to_inputarg_var
                                .entry(name.as_str())
                                .or_insert_with(|| var.clone());
                        }
                    }
                }
            }
        }

        // Class 1b — op-result definitions, with Input rebind aliasing.
        for op in &block.operations {
            let Some(result) = op.result else { continue };
            if map.contains_key(&result) {
                continue;
            }
            let var = if let OpKind::Input { name, .. } = &op.kind {
                name_to_inputarg_var
                    .get(name.as_str())
                    .cloned()
                    .unwrap_or_else(|| seed_variable(result, annotations))
            } else {
                seed_variable(result, annotations)
            };
            map.insert(result, var);
        }
        // Class 3 — exitswitch-referenced values.
        if let Some(crate::model::ExitSwitch::Value(vid)) = &block.exitswitch {
            map.entry(*vid)
                .or_insert_with(|| seed_variable(*vid, annotations));
        }
        // Class 2 — link-side sentinels.
        for link in &block.exits {
            for arg in &link.args {
                if let LinkArg::Value(vid) = arg {
                    map.entry(*vid)
                        .or_insert_with(|| seed_variable(*vid, annotations));
                }
            }
            if let Some(LinkArg::Value(vid)) = &link.last_exception {
                map.entry(*vid)
                    .or_insert_with(|| seed_variable(*vid, annotations));
            }
            if let Some(LinkArg::Value(vid)) = &link.last_exc_value {
                map.entry(*vid)
                    .or_insert_with(|| seed_variable(*vid, annotations));
            }
        }
    }
    map
}

/// `ValueId → Hlvalue` map combining the [`ValueIdToVariable`] map with
/// constant-inlining of `OpKind::ConstInt` / `ConstFloat` define-ops.
///
/// RPython's flowspace inlines constants natively as `Hlvalue::Constant`
/// in `op.args` (`flowspace/operation.py:152` `simple_call(target,
/// *args)` — `target` and each `arg` is either a `Variable` or
/// `Constant`). Pyre's legacy graph splits constants into define-ops
/// (`OpKind::ConstInt(n)` produces a fresh `ValueId` consumed
/// elsewhere). The adapter must recombine: every `ValueId` defined as a
/// const becomes a `Hlvalue::Constant`; every other defined `ValueId`
/// remains a `Hlvalue::Variable` from the Slice 1a map.
///
/// Constants are wrapped with their low-level concretetype attached,
/// matching RPython's `Constant.concretetype` shape. The legacy graph
/// used a separate `ValueId` for the define-op; after inlining, that
/// `ValueId` is tracked separately for readback.
pub fn build_value_to_hlvalue_map(
    legacy: &FunctionGraph,
    value_to_var: &ValueIdToVariable,
) -> HashMap<ValueId, Hlvalue> {
    let mut map: HashMap<ValueId, Hlvalue> = value_to_var
        .iter()
        .map(|(&vid, var)| (vid, Hlvalue::Variable(var.clone())))
        .collect();

    for block in &legacy.blocks {
        for op in &block.operations {
            let Some(result) = op.result else {
                continue;
            };
            match &op.kind {
                OpKind::ConstInt(n) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Int(*n),
                            LowLevelType::Signed,
                        )),
                    );
                }
                OpKind::ConstFloat(bits) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Float(*bits),
                            LowLevelType::Float,
                        )),
                    );
                }
                _ => {}
            }
        }
    }
    map
}

/// Look up the `Hlvalue` for a `ValueId` operand. Surfaces a
/// fail-loud `TyperError` when the operand is undefined (every
/// referenced `ValueId` must have been seeded by Slice 1a's
/// [`build_value_to_variable_map`] or shadowed by
/// [`build_value_to_hlvalue_map`]'s const inlining).
fn lookup_operand(
    value_map: &HashMap<ValueId, Hlvalue>,
    vid: ValueId,
) -> Result<Hlvalue, TyperError> {
    value_map.get(&vid).cloned().ok_or_else(|| {
        TyperError::message(format!(
            "translate_op: undefined operand {vid:?} — adapter invariant \
             broken (every referenced ValueId must be defined as a block \
             inputarg or op result)"
        ))
    })
}

/// Resolve the `Hlvalue` result slot for a legacy op. When the op has
/// no result (`Option::None`), allocate a fresh anonymous Variable per
/// RPython convention (every `SpaceOperation.result` slot is non-None
/// upstream — model.py:432-438; void-result ops use a throwaway
/// `Variable()`).
fn resolve_result_hlvalue(
    op: &SpaceOperation,
    value_map: &HashMap<ValueId, Hlvalue>,
) -> Result<Hlvalue, TyperError> {
    match op.result {
        Some(vid) => lookup_operand(value_map, vid),
        None => Ok(Hlvalue::Variable(Variable::new())),
    }
}

/// Translate a single legacy `model::SpaceOperation` into zero or more
/// `flowspace::SpaceOperation`s.
///
/// Returns `Ok(Vec::new())` when the op is **fully consumed by other
/// adapter infrastructure** — `OpKind::Input` (handled by Slice 1c
/// block topology, where the result `ValueId` becomes a
/// `block.inputargs` entry) and `OpKind::ConstInt` / `ConstFloat`
/// (handled by [`build_value_to_hlvalue_map`], which inlines the
/// constant at every consuming op's args site).
///
/// Returns `Err(TyperError)` for variants whose lowering is deferred to
/// a Slice 1b followup commit. The error message names the specific
/// variant so Slice 4's dual-gate failure cleanly identifies which
/// followup needs to land.
pub fn translate_op(
    op: &SpaceOperation,
    value_map: &HashMap<ValueId, Hlvalue>,
) -> Result<Vec<FlowspaceOp>, TyperError> {
    match &op.kind {
        // ─── Skipped: fully consumed by other adapter infrastructure ───
        OpKind::Input { .. } => Ok(Vec::new()),
        OpKind::ConstInt(_) | OpKind::ConstFloat(_) => Ok(Vec::new()),

        // ─── Pre-rtyper opname pass-through ───
        // `binary_op_name` (front/ast.rs:2959-2978) emits the same
        // pre-rtyper opnames RPython's flowspace registers via
        // `add_operator('add', 2, ...)` etc. (operation.py:475-507):
        // `add`, `sub`, `mul`, `mod`, `lt`, `eq`, ... So the legacy
        // `op` string passes straight through into the flowspace
        // SpaceOperation; the real rtyper will rewrite `add` →
        // `int_add` per the operand types via the same
        // `pair_int_int` machinery used for upstream graphs. Compound
        // forms like `add_assign` (front/ast.rs:2980+) have no
        // RPython counterpart and will surface as TyperError when
        // the rtyper fails to find a `rtype_add_assign`; that's a
        // deferred Slice 1b sub-followup, not a regression.
        OpKind::BinOp {
            op: opname,
            lhs,
            rhs,
            ..
        } => {
            let l = lookup_operand(value_map, *lhs)?;
            let r = lookup_operand(value_map, *rhs)?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(opname.clone(), vec![l, r], result)])
        }

        // ─── Slice 1b followups: deferred per-variant ports ───
        // `Call` deliberately falls through to fail-loud here. The
        // upstream `simple_call` opname (operation.py:152) needs a
        // proper `Hlvalue::Constant` target argument, which requires
        // resolving `CallTarget` variants (FunctionPath / Method /
        // Indirect / SyntheticTransparentCtor) to their RPython
        // equivalents. Each variant lands in its own followup so
        // Slice 4's anchor tests can exercise them in isolation.
        // Each fail-loud message names the specific OpKind so Slice 4's
        // anchor tests pinpoint exactly which followup to land next.
        other => Err(TyperError::message(format!(
            "translate_op: OpKind variant `{}` not yet ported \
             (Slice 1b followup pending) — reachable from legacy graph \
             at result={:?}",
            opkind_variant_name(other),
            op.result,
        ))),
    }
}

/// Stable variant name for fail-loud messages. Matches the RPython
/// convention of identifying ops by their opname stem so Slice 4
/// dual-gate failures are immediately greppable.
fn opkind_variant_name(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::Input { .. } => "Input",
        OpKind::ConstInt(_) => "ConstInt",
        OpKind::ConstFloat(_) => "ConstFloat",
        OpKind::FieldRead { .. } => "FieldRead",
        OpKind::FieldWrite { .. } => "FieldWrite",
        OpKind::ArrayRead { .. } => "ArrayRead",
        OpKind::ArrayWrite { .. } => "ArrayWrite",
        OpKind::InteriorFieldRead { .. } => "InteriorFieldRead",
        OpKind::InteriorFieldWrite { .. } => "InteriorFieldWrite",
        OpKind::Call { .. } => "Call",
        OpKind::GuardTrue { .. } => "GuardTrue",
        OpKind::GuardFalse { .. } => "GuardFalse",
        OpKind::GuardValue { .. } => "GuardValue",
        // Catch-all for variants pyre may add without bumping this
        // table — surfaces as `<unknown>` in the fail-loud message
        // rather than a misleading variant tag.
        _ => "<unknown OpKind variant>",
    }
}

/// Output of [`function_graph_to_flowspace`] — the assembled
/// `flowspace::FunctionGraph` plus enough side tables for Slice 2's
/// `specialize_legacy_graph` wrapper to drive `RPythonTyper::specialize`
/// against pyre's annotator surface and read back per-`ValueId`
/// concretetypes.
#[derive(Debug)]
pub struct FlowspaceAdapterOutput {
    /// Assembled `flowspace::FunctionGraph` carrying every legacy block
    /// translated to a `flowspace::Block` over `Hlvalue` operands.
    /// Wrapped in `Rc<RefCell<_>>` to match RPython's
    /// `FunctionDesc.cache` ownership shape — Slice 2 hands this to
    /// `RPythonAnnotator` directly.
    pub graph: Rc<RefCell<FlowspaceGraph>>,
    /// `ValueId → flowspace::Variable` (Slice 1a output) — Slice 2 reads
    /// `Variable.concretetype` per `ValueId` after `specialize` returns.
    pub value_to_var: ValueIdToVariable,
    /// Legacy constant define ValueIds that were intentionally
    /// materialised as `flowspace::Constant`s instead of graph
    /// `Variable` definitions. RPython stores concretetype on
    /// Constants directly; pyre's legacy `TypeResolutionState` is
    /// ValueId-keyed, so Slice 2 uses this adapter-local set to project
    /// constant ValueIds back without inventing a graph Variable.
    pub constant_value_ids: HashSet<ValueId>,
    /// `BlockId → flowspace::BlockRef` mapping. Includes the canonical
    /// `returnblock` and `exceptblock` (mapped to the
    /// `FunctionGraph::with_return_var`-allocated final blocks) so any
    /// legacy Link targeting them resolves correctly.
    pub block_map: HashMap<BlockId, BlockRef>,
}

/// Translate a legacy `ExitCase` into the `Hlvalue` slot expected by
/// `flowspace::Link.exitcase`. RPython encodes the discriminating value
/// as a `Hlvalue::Constant` carrying the matched bool / Python value
/// (`flowspace/model.py:114-120`).
fn exitcase_to_hlvalue(exitcase: Option<&ExitCase>) -> Option<Hlvalue> {
    match exitcase {
        None => None,
        Some(ExitCase::Bool(b)) => Some(Hlvalue::Constant(constant_from_constvalue(
            ConstValue::Bool(*b),
        ))),
        Some(ExitCase::Const(cv)) => Some(Hlvalue::Constant(constant_from_constvalue(cv.clone()))),
    }
}

fn constant_from_constvalue(value: ConstValue) -> Constant {
    match value {
        ConstValue::Int(n) => Constant::with_concretetype(ConstValue::Int(n), LowLevelType::Signed),
        ConstValue::Bool(b) => Constant::with_concretetype(ConstValue::Bool(b), LowLevelType::Bool),
        ConstValue::Float(bits) => {
            Constant::with_concretetype(ConstValue::Float(bits), LowLevelType::Float)
        }
        other => Constant::new(other),
    }
}

fn legacy_const_define_hlvalue(op: &SpaceOperation) -> Option<(ValueId, Hlvalue)> {
    let result = op.result?;
    match &op.kind {
        OpKind::ConstInt(n) => Some((
            result,
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(*n),
                LowLevelType::Signed,
            )),
        )),
        OpKind::ConstFloat(bits) => Some((
            result,
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Float(*bits),
                LowLevelType::Float,
            )),
        )),
        _ => None,
    }
}

/// Translate a single legacy `LinkArg` into a `Hlvalue`. `LinkArg::Value`
/// resolves through `value_map` (which carries Variable identities for
/// regular operands and inlined constants for `OpKind::ConstInt` /
/// `ConstFloat` defines per Slice 1b-core's
/// [`build_value_to_hlvalue_map`]). `LinkArg::Const` materialises a
/// fresh `Hlvalue::Constant`.
fn link_arg_to_hlvalue(
    arg: &LinkArg,
    value_map: &HashMap<ValueId, Hlvalue>,
) -> Result<Hlvalue, TyperError> {
    match arg {
        LinkArg::Value(vid) => lookup_operand(value_map, *vid),
        LinkArg::Const(cv) => Ok(Hlvalue::Constant(constant_from_constvalue(cv.clone()))),
    }
}

/// Translate an exception-link extra variable.
///
/// RPython `flowspace/model.py:636-642` defines `link.last_exception`
/// and `link.last_exc_value` in the link scope before checking
/// `link.args`; those same Variables may then appear in `link.args`.
/// Pyre's legacy graph represents them as fresh `ValueId`s whose only
/// definition site is the link, so the adapter must materialise them in
/// a per-link map instead of requiring a block-local definition.
fn link_extravar_to_hlvalue(
    arg: &LinkArg,
    value_map: &mut HashMap<ValueId, Hlvalue>,
    value_to_var: &mut ValueIdToVariable,
    annotations: &AnnotationState,
) -> Result<Hlvalue, TyperError> {
    match arg {
        LinkArg::Value(vid) => {
            if let Some(existing) = value_map.get(vid).cloned() {
                return Ok(existing);
            }
            let var = seed_variable(*vid, annotations);
            value_to_var.entry(*vid).or_insert_with(|| var.clone());
            let hlvalue = Hlvalue::Variable(var);
            value_map.insert(*vid, hlvalue.clone());
            Ok(hlvalue)
        }
        LinkArg::Const(cv) => Ok(Hlvalue::Constant(constant_from_constvalue(cv.clone()))),
    }
}

/// One-way conversion from the legacy `crate::model::FunctionGraph` +
/// `AnnotationState` pair into a `flowspace::FunctionGraph` whose
/// blocks carry `Hlvalue` operands and per-value `SomeValue`
/// annotations on its `Variable`s.
///
/// Two-pass topology assembly:
///
/// 1. **Pass 1** — allocate one `flowspace::BlockRef` per legacy
///    non-final block, allocating fresh `Variable`s for each block's
///    inputargs. Assemble the `flowspace::FunctionGraph` via
///    `FunctionGraph::with_return_var`, supplying the canonical
///    returnblock inputarg so the rtyper's `getreturnvar`
///    (`rtyper.rs:1633-1638`) finds a real return `Variable`.
/// 2. **Pass 2** — for each non-final block, translate `operations` via
///    [`translate_op`], translate `exits` (link args + targets +
///    exitcase) via [`link_arg_to_hlvalue`] / [`exitcase_to_hlvalue`],
///    and translate `exitswitch` via the `value_map`.
///
/// Slice 1c lands the topology assembly. Per-OpKind operation
/// translation depends on Slice 1b followups; Slice 1c uses
/// [`translate_op`] as-is, which means any legacy graph carrying an
/// unported OpKind variant surfaces a fail-loud `TyperError` from this
/// function. Trivial graphs (only `Input` / `ConstInt` / `ConstFloat`
/// op definitions) flow through cleanly.
pub fn function_graph_to_flowspace(
    legacy: &FunctionGraph,
    annotations: &AnnotationState,
) -> Result<FlowspaceAdapterOutput, TyperError> {
    let mut value_to_var: ValueIdToVariable = HashMap::new();
    let mut constant_hlvalues: HashMap<ValueId, Hlvalue> = HashMap::new();
    let mut constant_value_ids: HashSet<ValueId> = HashSet::new();

    for legacy_block in &legacy.blocks {
        for legacy_op in &legacy_block.operations {
            if let Some((vid, hlvalue)) = legacy_const_define_hlvalue(legacy_op) {
                constant_hlvalues.insert(vid, hlvalue);
                constant_value_ids.insert(vid);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Pass 1 — allocate fresh `flowspace::BlockRef` for every legacy
    // non-final block. The legacy `returnblock` and `exceptblock` are
    // skipped here; `FunctionGraph::with_return_var` allocates the
    // canonical flowspace finals, and the block_map is populated with
    // those after graph construction.
    // ──────────────────────────────────────────────────────────────

    let mut block_map: HashMap<BlockId, BlockRef> = HashMap::new();
    let mut block_inputarg_vars: HashMap<BlockId, HashMap<ValueId, Variable>> = HashMap::new();

    for legacy_block in &legacy.blocks {
        if legacy_block.id == legacy.returnblock || legacy_block.id == legacy.exceptblock {
            continue;
        }
        let mut local_inputs: HashMap<ValueId, Variable> = HashMap::new();
        let mut inputargs: Vec<Hlvalue> = Vec::with_capacity(legacy_block.inputargs.len());
        for &vid in &legacy_block.inputargs {
            let var = seed_variable(vid, annotations);
            value_to_var.entry(vid).or_insert_with(|| var.clone());
            local_inputs.insert(vid, var.clone());
            inputargs.push(Hlvalue::Variable(var));
        }
        block_inputarg_vars.insert(legacy_block.id, local_inputs);
        block_map.insert(legacy_block.id, FlowspaceBlock::shared(inputargs));
    }

    let startblock = block_map.get(&legacy.startblock).cloned().ok_or_else(|| {
        TyperError::message(format!(
            "function_graph_to_flowspace: legacy.startblock {:?} not in legacy.blocks",
            legacy.startblock
        ))
    })?;

    // Resolve the returnblock's inputarg as a fresh final-block
    // Variable. Even when the legacy graph reuses the source ValueId
    // here, RPython's checkgraph treats target inputargs as definitions
    // in the target block, not as the predecessor's Variable object.
    let return_var = legacy
        .blocks
        .iter()
        .find(|b| b.id == legacy.returnblock)
        .and_then(|b| b.inputargs.first().copied())
        .map(|vid| {
            let var = seed_variable(vid, annotations);
            value_to_var.entry(vid).or_insert_with(|| var.clone());
            Hlvalue::Variable(var)
        })
        .unwrap_or_else(|| Hlvalue::Variable(Variable::new()));

    let graph = FlowspaceGraph::with_return_var(legacy.name.clone(), startblock, return_var);
    let returnblock_ref = graph.returnblock.clone();
    let exceptblock_ref = graph.exceptblock.clone();

    if let Some(legacy_exceptblock) = legacy.blocks.iter().find(|b| b.id == legacy.exceptblock) {
        if legacy_exceptblock.inputargs.len() == 2 {
            let mut except_inputargs = Vec::with_capacity(2);
            for &vid in &legacy_exceptblock.inputargs {
                let var = seed_variable(vid, annotations);
                value_to_var.entry(vid).or_insert_with(|| var.clone());
                except_inputargs.push(Hlvalue::Variable(var));
            }
            exceptblock_ref.borrow_mut().inputargs = except_inputargs;
        }
    }

    let graph_ref = Rc::new(RefCell::new(graph));

    // Map the canonical finals so any legacy Link targeting them
    // resolves to the flowspace finals constructed above.
    block_map.insert(legacy.returnblock, returnblock_ref);
    block_map.insert(legacy.exceptblock, exceptblock_ref);

    // ──────────────────────────────────────────────────────────────
    // Pass 2 — fill operations + exits + exitswitch for each non-final
    // legacy block. Final blocks (returnblock / exceptblock) are
    // already terminal in flowspace — `mark_final()` was set by
    // `FunctionGraph::with_return_var`.
    // ──────────────────────────────────────────────────────────────

    for legacy_block in &legacy.blocks {
        if legacy_block.id == legacy.returnblock || legacy_block.id == legacy.exceptblock {
            continue;
        }
        let block_ref = block_map[&legacy_block.id].clone();
        let mut value_map = constant_hlvalues.clone();
        let mut name_to_value: HashMap<String, Hlvalue> = HashMap::new();

        if let Some(inputs) = block_inputarg_vars.get(&legacy_block.id) {
            for (&vid, var) in inputs {
                let hlvalue = Hlvalue::Variable(var.clone());
                value_map.insert(vid, hlvalue.clone());
                if let Some(name) = legacy.value_name(vid) {
                    name_to_value.entry(name.to_string()).or_insert(hlvalue);
                }
            }
        }
        for legacy_op in &legacy_block.operations {
            if let (Some(result), OpKind::Input { name, ty: _ }) =
                (legacy_op.result, &legacy_op.kind)
            {
                if legacy_block.inputargs.contains(&result) {
                    if let Some(existing) = value_map.get(&result).cloned() {
                        name_to_value.entry(name.clone()).or_insert(existing);
                    }
                }
            }
        }

        // Translate operations.
        let mut translated_ops: Vec<FlowspaceOp> = Vec::new();
        for legacy_op in &legacy_block.operations {
            if let Some((vid, hlvalue)) = legacy_const_define_hlvalue(legacy_op) {
                value_map.insert(vid, hlvalue.clone());
                if let Some(name) = legacy.value_name(vid) {
                    name_to_value.insert(name.to_string(), hlvalue);
                }
                translated_ops.extend(translate_op(legacy_op, &value_map)?);
                continue;
            }

            if let (Some(result), OpKind::Input { name, ty: _ }) =
                (legacy_op.result, &legacy_op.kind)
            {
                if !value_map.contains_key(&result) {
                    let Some(alias) = name_to_value.get(name).cloned() else {
                        return Err(TyperError::message(format!(
                            "function_graph_to_flowspace: Input `{name}` at result={result:?} \
                             has no in-scope flowspace Variable/Constant; RPython flowspace has \
                             no standalone Input operation"
                        )));
                    };
                    if let Hlvalue::Variable(var) = &alias {
                        value_to_var.entry(result).or_insert_with(|| var.clone());
                    }
                    value_map.insert(result, alias);
                }
                translated_ops.extend(translate_op(legacy_op, &value_map)?);
                continue;
            }

            if let Some(result) = legacy_op.result {
                if !value_map.contains_key(&result) {
                    let var = seed_variable(result, annotations);
                    value_to_var.entry(result).or_insert_with(|| var.clone());
                    value_map.insert(result, Hlvalue::Variable(var));
                }
            }
            translated_ops.extend(translate_op(legacy_op, &value_map)?);
            if let Some(result) = legacy_op.result {
                if let Some(name) = legacy.value_name(result) {
                    if let Some(value) = value_map.get(&result).cloned() {
                        name_to_value.insert(name.to_string(), value);
                    }
                }
            }
        }

        // Translate exits.
        let mut translated_exits: Vec<flowspace_model::LinkRef> =
            Vec::with_capacity(legacy_block.exits.len());
        for legacy_link in &legacy_block.exits {
            let mut link_value_map = value_map.clone();
            let last_exception = legacy_link
                .last_exception
                .as_ref()
                .map(|arg| {
                    link_extravar_to_hlvalue(
                        arg,
                        &mut link_value_map,
                        &mut value_to_var,
                        annotations,
                    )
                })
                .transpose()?;
            let last_exc_value = legacy_link
                .last_exc_value
                .as_ref()
                .map(|arg| {
                    link_extravar_to_hlvalue(
                        arg,
                        &mut link_value_map,
                        &mut value_to_var,
                        annotations,
                    )
                })
                .transpose()?;
            let target = block_map.get(&legacy_link.target).cloned().ok_or_else(|| {
                TyperError::message(format!(
                    "function_graph_to_flowspace: legacy link target {:?} not found in \
                     legacy.blocks (block id={:?})",
                    legacy_link.target, legacy_block.id,
                ))
            })?;
            let args: Vec<Hlvalue> = legacy_link
                .args
                .iter()
                .map(|arg| link_arg_to_hlvalue(arg, &link_value_map))
                .collect::<Result<Vec<_>, _>>()?;
            let exitcase = exitcase_to_hlvalue(legacy_link.exitcase.as_ref());
            let mut link = FlowspaceLink::new(args, Some(target), exitcase);
            if let Some(llexitcase) = &legacy_link.llexitcase {
                link.llexitcase = Some(Hlvalue::Constant(constant_from_constvalue(
                    llexitcase.clone(),
                )));
            }
            link.extravars(last_exception, last_exc_value);
            link.prevblock = Some(Rc::downgrade(&block_ref));
            translated_exits.push(link.into_ref());
        }

        // Translate exitswitch.
        let translated_exitswitch = match &legacy_block.exitswitch {
            None => None,
            Some(ExitSwitch::Value(vid)) => Some(lookup_operand(&value_map, *vid)?),
            Some(ExitSwitch::LastException) => Some(Hlvalue::Constant(c_last_exception())),
        };

        // Commit to the flowspace::Block. Borrow scope is tight to
        // avoid alias-clash with link.prevblock's Weak above.
        {
            let mut block = block_ref.borrow_mut();
            block.operations = translated_ops;
            block.exits = translated_exits;
            block.exitswitch = translated_exitswitch;
        }
    }

    flowspace_model::checkgraph(&graph_ref.borrow());

    Ok(FlowspaceAdapterOutput {
        graph: graph_ref,
        value_to_var,
        constant_value_ids,
        block_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{KnownType, SomeObjectTrait};
    use crate::model::{Block, BlockId, FunctionGraph as LegacyGraph, OpKind, SpaceOperation};

    #[test]
    fn valuetype_int_lifts_to_someinteger_default() {
        let s = valuetype_to_someshell(&ValueType::Int);
        match s {
            SomeValue::Integer(i) => {
                assert_eq!(i.knowntype(), KnownType::Int);
                assert!(!i.nonneg, "default SomeInteger.nonneg must be false");
                assert!(!i.unsigned, "default SomeInteger.unsigned must be false");
            }
            other => panic!("ValueType::Int must lift to SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn valuetype_float_lifts_to_somefloat_default() {
        let s = valuetype_to_someshell(&ValueType::Float);
        match s {
            SomeValue::Float(f) => {
                assert_eq!(f.knowntype(), KnownType::Float);
                assert!(f.immutable(), "SomeFloat is immutable per model.py:164-183");
            }
            other => panic!("ValueType::Float must lift to SomeFloat, got {other:?}"),
        }
    }

    #[test]
    fn valuetype_ref_lifts_to_someobject_lattice_top() {
        let s = valuetype_to_someshell(&ValueType::Ref);
        match s {
            SomeValue::Object(b) => {
                assert_eq!(b.knowntype, KnownType::Object);
                assert!(!b.immutable, "default SomeObjectBase is mutable");
            }
            other => panic!("ValueType::Ref must lift to SomeValue::Object, got {other:?}"),
        }
    }

    #[test]
    fn valuetype_void_lifts_to_impossible_lattice_bottom() {
        let s = valuetype_to_someshell(&ValueType::Void);
        assert!(
            matches!(s, SomeValue::Impossible),
            "ValueType::Void must lift to SomeValue::Impossible, got {s:?}"
        );
    }

    #[test]
    fn valuetype_state_and_unknown_lift_to_someobject_fallback() {
        for vt in [ValueType::State, ValueType::Unknown] {
            let s = valuetype_to_someshell(&vt);
            assert!(
                matches!(s, SomeValue::Object(_)),
                "{vt:?} must fall back to SomeValue::Object, got {s:?}"
            );
        }
    }

    #[test]
    fn seed_variable_attaches_lifted_annotation_observable_via_clone() {
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(7), ValueType::Int);
        let var = seed_variable(ValueId(7), &annotations);

        // Reference semantics: the annotation Rc-shares across clones
        // (flowspace/model.rs:2010-2018), so a clone observes the same
        // shell instance.
        let clone = var.clone();
        let clone_ann = clone.annotation.borrow();
        let shell = clone_ann
            .as_ref()
            .expect("seed_variable must populate annotation slot");
        assert!(
            matches!(shell.as_ref(), SomeValue::Integer(_)),
            "annotation must round-trip the lifted SomeInteger shell"
        );
    }

    #[test]
    fn seed_variable_unknown_value_id_lifts_to_object_fallback() {
        // Missing entries in AnnotationState.types resolve to
        // ValueType::Unknown via AnnotationState::get
        // (annotation_state.rs:30-32). The adapter must still produce a
        // Variable with a non-None annotation so bindingrepr can resolve.
        let annotations = AnnotationState::new();
        let var = seed_variable(ValueId(42), &annotations);
        let ann = var.annotation.borrow();
        let shell = ann.as_ref().expect("annotation slot must be populated");
        assert!(
            matches!(shell.as_ref(), SomeValue::Object(_)),
            "missing annotation must fall back to SomeValue::Object, got {shell:?}"
        );
    }

    fn legacy_graph_with_inputarg_and_result(input: ValueId, result: ValueId) -> LegacyGraph {
        let mut graph = LegacyGraph::new("test");
        let mut block = Block {
            id: BlockId(0),
            inputargs: vec![input],
            operations: vec![SpaceOperation {
                result: Some(result),
                kind: OpKind::ConstInt(0),
            }],
            exitswitch: None,
            exits: Vec::new(),
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];
        graph
    }

    #[test]
    fn build_value_to_variable_map_seeds_inputargs_and_op_results() {
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Ref);
        let graph = legacy_graph_with_inputarg_and_result(ValueId(1), ValueId(2));

        let map = build_value_to_variable_map(&graph, &annotations);

        assert_eq!(
            map.len(),
            2,
            "map must seed both the inputarg and the op result"
        );
        assert!(
            matches!(
                map[&ValueId(1)]
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| s.as_ref().clone()),
                Some(SomeValue::Integer(_))
            ),
            "inputarg ValueId(1) (Int) must be seeded with SomeInteger"
        );
        assert!(
            matches!(
                map[&ValueId(2)]
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| s.as_ref().clone()),
                Some(SomeValue::Object(_))
            ),
            "op-result ValueId(2) (Ref) must be seeded with SomeObject"
        );
    }

    #[test]
    fn build_value_to_variable_map_dedupes_by_value_id() {
        // Two ops both reading the same inputarg (legacy graphs are SSA
        // — every ValueId has one definition, but multiple readers).
        // Slice 1a must produce one Variable identity per ValueId.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);
        annotations.set(ValueId(3), ValueType::Int);

        let mut graph = LegacyGraph::new("dedup_test");
        let mut block = Block {
            id: BlockId(0),
            inputargs: vec![ValueId(1)],
            operations: vec![
                SpaceOperation {
                    result: Some(ValueId(2)),
                    kind: OpKind::ConstInt(7),
                },
                SpaceOperation {
                    result: Some(ValueId(3)),
                    kind: OpKind::ConstInt(11),
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        let map = build_value_to_variable_map(&graph, &annotations);

        assert_eq!(map.len(), 3, "three distinct ValueIds → three Variables");
        // The identity invariant: the inputarg's Variable is one fresh
        // identity, the two op results are two more fresh identities, and
        // they don't collide.
        assert_ne!(map[&ValueId(1)], map[&ValueId(2)]);
        assert_ne!(map[&ValueId(1)], map[&ValueId(3)]);
        assert_ne!(map[&ValueId(2)], map[&ValueId(3)]);
    }

    #[test]
    fn build_value_to_variable_map_aliases_input_rebind_to_inputarg() {
        // Pyre's surface front emits a leading `Input{name}` op whose
        // result IS a block.inputarg, plus follow-up `Input{same name}`
        // ops with FRESH result ValueIds for body-side rebinds. The
        // adapter must alias the rebind result to the canonical
        // inputarg Variable so `setup_block_entry`'s
        // `concretetype` write reaches both — otherwise the body's
        // BinOp lookup hits a fresh Variable with no concretetype and
        // trips genop's "wrong level!" assertion.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int); // rebind of same name

        let mut graph = LegacyGraph::new("rebind_alias");
        let mut block = Block {
            id: BlockId(0),
            inputargs: vec![ValueId(1)],
            operations: vec![
                // Leading definition: result IS the inputarg.
                SpaceOperation {
                    result: Some(ValueId(1)),
                    kind: OpKind::Input {
                        name: "x".to_string(),
                        ty: ValueType::Int,
                    },
                },
                // Rebind: result is fresh; same name → alias to ValueId(1)'s Variable.
                SpaceOperation {
                    result: Some(ValueId(2)),
                    kind: OpKind::Input {
                        name: "x".to_string(),
                        ty: ValueType::Int,
                    },
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        let map = build_value_to_variable_map(&graph, &annotations);
        assert_eq!(
            map[&ValueId(1)],
            map[&ValueId(2)],
            "Input rebind result must alias to inputarg Variable identity"
        );
    }

    // ───── Slice 1b-core tests: dispatcher + skip arms + fail-loud ─────

    #[test]
    fn build_value_to_hlvalue_map_inlines_const_defines() {
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);
        annotations.set(ValueId(3), ValueType::Float);

        let mut graph = LegacyGraph::new("const_inline");
        let mut block = Block {
            id: BlockId(0),
            inputargs: vec![ValueId(1)],
            operations: vec![
                SpaceOperation {
                    result: Some(ValueId(2)),
                    kind: OpKind::ConstInt(42),
                },
                SpaceOperation {
                    result: Some(ValueId(3)),
                    kind: OpKind::ConstFloat(0xC000_0000_0000_0000), // f64::from_bits → -2.0
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        let var_map = build_value_to_variable_map(&graph, &annotations);
        let hl_map = build_value_to_hlvalue_map(&graph, &var_map);

        // Inputarg keeps its Variable identity.
        assert!(matches!(hl_map[&ValueId(1)], Hlvalue::Variable(_)));

        // ConstInt define is inlined as Hlvalue::Constant(Int).
        match &hl_map[&ValueId(2)] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(n) => assert_eq!(*n, 42),
                other => panic!("ValueId(2) must be ConstValue::Int, got {other:?}"),
            },
            other => panic!("ValueId(2) must be inlined as Hlvalue::Constant, got {other:?}"),
        }

        // ConstFloat define is inlined as Hlvalue::Constant(Float).
        match &hl_map[&ValueId(3)] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Float(bits) => assert_eq!(*bits, 0xC000_0000_0000_0000),
                other => panic!("ValueId(3) must be ConstValue::Float, got {other:?}"),
            },
            other => panic!("ValueId(3) must be inlined as Hlvalue::Constant, got {other:?}"),
        }
    }

    #[test]
    fn translate_op_skips_input_define() {
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let op = SpaceOperation {
            result: Some(ValueId(1)),
            kind: OpKind::Input {
                name: "x".to_string(),
                ty: ValueType::Int,
            },
        };
        let result = translate_op(&op, &value_map).expect("Input must translate to skip");
        assert!(
            result.is_empty(),
            "Input define has no SpaceOperation analogue (handled by Slice 1c \
             via block.inputargs); translate_op must yield empty Vec"
        );
    }

    #[test]
    fn translate_op_skips_const_int_define() {
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let op = SpaceOperation {
            result: Some(ValueId(1)),
            kind: OpKind::ConstInt(7),
        };
        let result = translate_op(&op, &value_map).expect("ConstInt must translate to skip");
        assert!(
            result.is_empty(),
            "ConstInt define is inlined by build_value_to_hlvalue_map; \
             translate_op must yield empty Vec"
        );
    }

    #[test]
    fn translate_op_skips_const_float_define() {
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let op = SpaceOperation {
            result: Some(ValueId(1)),
            kind: OpKind::ConstFloat(0),
        };
        let result = translate_op(&op, &value_map).expect("ConstFloat must translate to skip");
        assert!(result.is_empty());
    }

    #[test]
    fn translate_op_binop_lowers_to_passthrough_spaceop() {
        // BinOp arm: `add` / `sub` / `lt` / ... pass through to a
        // flowspace SpaceOperation with the same opname; lhs/rhs args
        // get resolved via lookup_operand and the result Hlvalue via
        // resolve_result_hlvalue.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let lhs_var = Hlvalue::Variable(Variable::new());
        let rhs_var = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        value_map.insert(ValueId(1), lhs_var.clone());
        value_map.insert(ValueId(2), rhs_var.clone());
        value_map.insert(ValueId(3), result_var.clone());

        let op = SpaceOperation {
            result: Some(ValueId(3)),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: ValueId(1),
                rhs: ValueId(2),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map).expect("BinOp arm must lower");
        assert_eq!(translated.len(), 1, "BinOp lowers to exactly one SpaceOp");
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "add", "opname passes through unchanged");
        assert_eq!(lowered.args.len(), 2);
    }

    #[test]
    fn translate_op_binop_undefined_lhs_surfaces_invariant_break() {
        // BinOp arm threads operand lookups through lookup_operand, so a
        // missing lhs surfaces the "adapter invariant broken" message
        // rather than a silent panic.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(3), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(3)),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: ValueId(99), // not in value_map
                rhs: ValueId(2),
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map)
            .expect_err("undefined BinOp operand must surface invariant break");
        let msg = format!("{err}");
        assert!(msg.contains("undefined operand"));
    }

    #[test]
    fn translate_op_call_surfaces_followup_pending() {
        // Slice 1b-core leaves Call unimplemented. The fail-loud message
        // names the variant so Slice 4 dual-gate failures pinpoint the
        // followup commit.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(2)),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["a".into(), "b".into()],
                },
                args: vec![ValueId(1)],
                result_ty: ValueType::Int,
            },
        };
        // Ensure the result ValueId is in the map so resolve_result_hlvalue
        // wouldn't trip first.
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));

        let err = translate_op(&op, &value_map)
            .expect_err("Call lowering is a Slice 1b followup — must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("Call") && msg.contains("Slice 1b followup"),
            "fail-loud message must name the variant + slice tag, got: {msg}"
        );
    }

    #[test]
    fn translate_op_field_read_surfaces_followup_pending() {
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let op = SpaceOperation {
            result: Some(ValueId(2)),
            kind: OpKind::FieldRead {
                base: ValueId(1),
                field: crate::model::FieldDescriptor::new("f", Some("Owner".into())),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let err =
            translate_op(&op, &value_map).expect_err("FieldRead lowering is a Slice 1b followup");
        let msg = format!("{err}");
        assert!(
            msg.contains("FieldRead"),
            "fail-loud message must name the variant, got: {msg}"
        );
    }

    #[test]
    fn translate_op_undefined_operand_surfaces_invariant_break() {
        // Although Slice 1b-core's only implemented arm with operands is
        // gone (Call → followup), the lookup_operand helper is shared
        // with future arms. Validate it surfaces a clear "adapter
        // invariant broken" message.
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let err = lookup_operand(&value_map, ValueId(99))
            .expect_err("undefined operand lookup must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("undefined operand") && msg.contains("invariant"),
            "fail-loud message must explain the invariant, got: {msg}"
        );
    }

    // ───── Slice 1c tests: topology assembly ─────

    fn link_to_returnblock(args: Vec<LinkArg>, returnblock_id: BlockId) -> crate::model::Link {
        let mut link = crate::model::Link::new_mixed(args, returnblock_id, None);
        link.prevblock = None;
        link
    }

    fn legacy_minimal_identity_return_graph() -> LegacyGraph {
        // Smallest valid legacy graph: one inputarg, returns it
        // directly. Slice 1c must produce a flowspace::FunctionGraph
        // whose startblock has the single inputarg Variable,
        // exits→returnblock, and the returnblock's inputarg is the same
        // Variable identity (so RPythonTyper.getreturnvar resolves
        // correctly).
        //
        // RPython convention: returnblock canonically has one inputarg
        // (`flowspace/model.py:13-18`). True void returns use a
        // `SomeNone` / `Void`-typed argument; pyre's legacy graph
        // mirrors that by always emitting a single ValueId in the
        // returnblock's inputargs.
        let mut graph = LegacyGraph::new("identity_return");
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
        graph
    }

    #[test]
    fn function_graph_to_flowspace_minimal_identity_return_assembles_graph() {
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        let legacy = legacy_minimal_identity_return_graph();

        let output = function_graph_to_flowspace(&legacy, &annotations)
            .expect("minimal graph must assemble");

        // value_to_var must contain the inputarg.
        assert!(
            output.value_to_var.contains_key(&ValueId(1)),
            "value_to_var must seed the legacy inputarg"
        );

        // block_map must contain startblock + returnblock + exceptblock.
        assert!(output.block_map.contains_key(&legacy.startblock));
        assert!(output.block_map.contains_key(&legacy.returnblock));
        assert!(output.block_map.contains_key(&legacy.exceptblock));

        // The flowspace graph's startblock is the same BlockRef as the
        // mapped legacy.startblock.
        let graph = output.graph.borrow();
        assert!(Rc::ptr_eq(
            &graph.startblock,
            &output.block_map[&legacy.startblock],
        ));

        // The flowspace graph's returnblock is the mapped legacy.returnblock.
        assert!(Rc::ptr_eq(
            &graph.returnblock,
            &output.block_map[&legacy.returnblock],
        ));

        // The startblock has exactly one exit, targeting the returnblock.
        let startblock = graph.startblock.borrow();
        assert_eq!(startblock.exits.len(), 1);
        let exit = startblock.exits[0].borrow();
        assert!(Rc::ptr_eq(
            exit.target.as_ref().expect("link must have target"),
            &graph.returnblock,
        ));
    }

    #[test]
    fn function_graph_to_flowspace_returnvar_identity_preserved() {
        // When the returnblock has an inputarg ValueId, the flowspace
        // graph's returnblock must use the SAME Variable identity (so
        // RPythonTyper.getreturnvar finds the right Variable —
        // rtyper.rs:1633-1638).
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("with_return_var");
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
            inputargs: vec![ValueId(2)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        graph.blocks = vec![startblock, returnblock];

        let output =
            function_graph_to_flowspace(&graph, &annotations).expect("graph must assemble");

        // The flowspace returnblock's single inputarg must be the
        // Variable we seeded for ValueId(2).
        let flowspace_graph = output.graph.borrow();
        let returnblock_ref = &flowspace_graph.returnblock;
        let returnblock = returnblock_ref.borrow();
        assert_eq!(
            returnblock.inputargs.len(),
            1,
            "returnblock must carry the single return-value inputarg"
        );
        match &returnblock.inputargs[0] {
            Hlvalue::Variable(v) => {
                let expected = &output.value_to_var[&ValueId(2)];
                assert_eq!(
                    v, expected,
                    "returnblock inputarg must preserve Variable identity from value_to_var"
                );
            }
            other => panic!("returnblock inputarg must be a Variable, got {other:?}"),
        }
    }

    #[test]
    fn function_graph_to_flowspace_const_define_op_inlined_in_link_args() {
        // ConstInt(7) defines ValueId(2). Slice 1b's
        // build_value_to_hlvalue_map inlines it into Link.args as
        // Hlvalue::Constant — Slice 1c's link translation must use that
        // mapping rather than wrapping the unused Variable.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("const_link_arg");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![SpaceOperation {
                result: Some(ValueId(2)),
                kind: OpKind::ConstInt(7),
            }],
            exitswitch: None,
            // Return ValueId(2), the ConstInt define.
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(2))],
                graph.returnblock,
            )],
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(3)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        annotations.set(ValueId(3), ValueType::Int);
        graph.blocks = vec![startblock, returnblock];

        let output =
            function_graph_to_flowspace(&graph, &annotations).expect("graph must assemble");

        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        // ConstInt define is a skip arm — operations must be empty.
        assert!(
            startblock.operations.is_empty(),
            "ConstInt define has no flowspace::SpaceOperation analogue"
        );
        // The exit's link arg must be the inlined Constant.
        let exit = startblock.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        match exit.args[0].as_ref().expect("link arg is Some") {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(n) => assert_eq!(*n, 7),
                other => panic!("link arg constant must be Int, got {other:?}"),
            },
            other => panic!("link arg must be Hlvalue::Constant, got {other:?}"),
        }
    }

    #[test]
    fn function_graph_to_flowspace_exception_link_materialises_extravars_before_args() {
        // RPython checkgraph defines exception-link extravars before
        // validating link.args. Pyre's legacy graph represents those
        // as fresh ValueIds whose only definition site is the link.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);
        annotations.set(ValueId(3), ValueType::Int);
        annotations.set(ValueId(4), ValueType::Int);
        annotations.set(ValueId(10), ValueType::Int);
        annotations.set(ValueId(11), ValueType::Ref);

        let mut graph = LegacyGraph::new("canraise_with_extravars");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1), ValueId(2)],
            operations: vec![SpaceOperation {
                result: Some(ValueId(3)),
                kind: OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: ValueId(1),
                    rhs: ValueId(2),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: Some(crate::model::ExitSwitch::LastException),
            exits: vec![
                link_to_returnblock(vec![LinkArg::Value(ValueId(3))], graph.returnblock),
                crate::model::Link::new_mixed(
                    vec![LinkArg::Value(ValueId(10)), LinkArg::Value(ValueId(11))],
                    graph.exceptblock,
                    Some(crate::model::exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(ValueId(10))),
                    Some(LinkArg::Value(ValueId(11))),
                ),
            ],
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(4)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        let exceptblock = Block {
            id: graph.exceptblock,
            inputargs: vec![ValueId(10), ValueId(11)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        graph.blocks = vec![startblock, returnblock, exceptblock];

        let output =
            function_graph_to_flowspace(&graph, &annotations).expect("exception graph assembles");
        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        let exc_link = startblock.exits[1].borrow();
        assert!(exc_link.last_exception.is_some());
        assert!(exc_link.last_exc_value.is_some());
        assert_eq!(
            exc_link.args[0].as_ref(),
            exc_link.last_exception.as_ref(),
            "exception type arg must reuse link.last_exception Variable"
        );
        assert_eq!(
            exc_link.args[1].as_ref(),
            exc_link.last_exc_value.as_ref(),
            "exception value arg must reuse link.last_exc_value Variable"
        );
    }

    #[test]
    fn function_graph_to_flowspace_unported_opkind_surfaces_failloud() {
        // A graph carrying a Slice 1b-followup-pending OpKind (Call)
        // must surface that op's translate_op error from inside Pass 2,
        // not silently emit a partial graph.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("unported_op");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![SpaceOperation {
                result: Some(ValueId(2)),
                kind: OpKind::Call {
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
            inputargs: vec![ValueId(3)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
        };
        annotations.set(ValueId(3), ValueType::Int);
        graph.blocks = vec![startblock, returnblock];

        let err = function_graph_to_flowspace(&graph, &annotations)
            .expect_err("unported OpKind must surface as TyperError");
        let msg = format!("{err}");
        assert!(
            msg.contains("Call") && msg.contains("Slice 1b followup"),
            "unported-op error must propagate translate_op's fail-loud message, got: {msg}"
        );
    }

    #[test]
    fn opkind_variant_name_covers_core_variants() {
        // Sanity: variant_name maps each core OpKind to a stable string.
        // Any new OpKind variant added without a corresponding arm here
        // surfaces as "<unknown OpKind variant>" in fail-loud messages,
        // which prompts the developer to extend this table.
        assert_eq!(opkind_variant_name(&OpKind::ConstInt(0)), "ConstInt");
        assert_eq!(opkind_variant_name(&OpKind::ConstFloat(0)), "ConstFloat");
        assert_eq!(
            opkind_variant_name(&OpKind::Input {
                name: "x".into(),
                ty: ValueType::Int
            }),
            "Input"
        );
    }
}
