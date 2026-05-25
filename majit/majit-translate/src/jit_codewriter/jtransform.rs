//! Graph-based jtransform: semantic rewrite pass.
#![allow(non_snake_case)]
//!
//! RPython equivalent: jtransform.py Transformer.optimize_block()
//!
//! Transforms a FunctionGraph by rewriting operations:
//! - FieldRead on virtualizable fields → VableFieldRead marker
//! - FieldWrite on virtualizable fields → VableFieldWrite marker
//! - ArrayRead on virtualizable arrays → VableArrayRead marker
//! - Call classification → elidable/residual/may_force tagging

use serde::{Deserialize, Serialize};

use crate::call::CallDescriptor;
use crate::jit_codewriter::support::{NormalizedArg, decode_builtin_call};
use crate::model::{
    CallFuncPtr, CallTarget, FieldDescriptor, FunctionGraph, OpKind, SpaceOperation, ValueType,
    remap_control_flow_metadata_var,
};
use majit_ir::descr::{EffectInfo, ExtraEffect, OopSpecIndex};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VirtualizableFieldDescriptor {
    pub name: String,
    pub owner_root: Option<String>,
    pub index: usize,
    /// RPython: cpu.arraydescrof(ARRAY.TO).itemsize — item byte size for
    /// vable arrays. None for scalar fields.
    pub array_itemsize: Option<usize>,
    /// RPython: arraydescr.is_item_signed() — FLAG_SIGNED for vable arrays.
    pub array_is_signed: Option<bool>,
}

impl VirtualizableFieldDescriptor {
    pub fn new(name: impl Into<String>, owner_root: Option<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            owner_root,
            index,
            array_itemsize: None,
            array_is_signed: None,
        }
    }

    /// Create a descriptor with arraydescr info (for vable array fields).
    /// RPython: `VirtualizableInfo.__init__` stores `cpu.arraydescrof(ARRAY.TO)`.
    pub fn new_with_arraydescr(
        name: impl Into<String>,
        owner_root: Option<String>,
        index: usize,
        itemsize: usize,
        is_signed: bool,
    ) -> Self {
        Self {
            name: name.into(),
            owner_root,
            index,
            array_itemsize: Some(itemsize),
            array_is_signed: Some(is_signed),
        }
    }

    fn matches(&self, field: &FieldDescriptor) -> bool {
        if self.name != field.name {
            return false;
        }
        let Some(cfg_owner) = self.owner_root.as_ref() else {
            return true;
        };
        let Some(field_owner) = field.owner_root.as_ref() else {
            return false;
        };
        // RPython `jtransform.py:982 self.callcontrol.get_vinfo(
        // v_virtualizable.concretetype)` compares lltype object IDENTITY
        // — a single `Ptr(GcStruct(...))` instance per type, looked up
        // by the call control's `vinfos` dict.  Pyre carries the same
        // identity as a string `owner_root`; the canonical form is
        // `"<module_path>::<bare_name>"` (see
        // `descr.rs canonical_struct_name`).  Helpers built from
        // `impl OpcodeStepExecutor for PyFrame` in `pyre-interpreter`
        // resolve `self` to `"pyframe::PyFrame"`; the vable config
        // (`virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT`) ships the
        // bare `"PyFrame"` for symmetry with `lib.rs`'s fixture-side
        // vable_fields registrations.  Compare against the canonical
        // form so both shapes match without mutating either source of
        // truth.
        if cfg_owner == field_owner {
            return true;
        }
        let cfg_canonical = majit_ir::descr::canonical_struct_name(cfg_owner);
        let field_canonical = majit_ir::descr::canonical_struct_name(field_owner);
        cfg_canonical == field_canonical
    }
}

/// Configuration for the graph rewrite pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformConfig {
    /// Whether to rewrite virtualizable field/array accesses.
    pub lower_virtualizable: bool,
    /// Whether to classify function calls by effect.
    pub classify_calls: bool,
    /// Virtualizable scalar field descriptors.
    #[serde(default)]
    pub vable_fields: Vec<VirtualizableFieldDescriptor>,
    /// Virtualizable array field descriptors.
    #[serde(default)]
    pub vable_arrays: Vec<VirtualizableFieldDescriptor>,
    /// Explicit call effect overrides.
    ///
    /// RPython equivalent: effect classification travels on call descriptors
    /// rather than being rediscovered from source text.
    #[serde(default)]
    pub call_effects: Vec<CallEffectOverride>,
}

impl Default for GraphTransformConfig {
    fn default() -> Self {
        Self {
            lower_virtualizable: true,
            classify_calls: true,
            vable_fields: Vec::new(),
            vable_arrays: Vec::new(),
            call_effects: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallEffectKind {
    Elidable,
    Residual,
    MayForce,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEffectOverride {
    /// `op.args[0]`-equivalent funcptr identity used to match the
    /// override against a call site.
    pub target: CallTarget,
    /// `calldescr`-equivalent EffectInfo wrapper attached to the call.
    pub descriptor: CallDescriptor,
}

impl CallEffectOverride {
    pub fn new(target: CallTarget, effect: CallEffectKind) -> Self {
        Self {
            target,
            descriptor: CallDescriptor::override_effect(effect_info_for_kind(effect)),
        }
    }
}

fn effect_info_for_kind(effect: CallEffectKind) -> EffectInfo {
    match effect {
        CallEffectKind::Elidable => {
            EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None)
        }
        CallEffectKind::Residual => EffectInfo::new(ExtraEffect::CanRaise, OopSpecIndex::None),
        CallEffectKind::MayForce => EffectInfo::new(
            ExtraEffect::ForcesVirtualOrVirtualizable,
            OopSpecIndex::None,
        ),
    }
}

/// A note about a transformation applied to the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTransformNote {
    pub function: String,
    pub detail: String,
}

/// Result of a graph transformation pass.
#[derive(Debug, Clone)]
pub struct GraphTransformResult {
    pub graph: FunctionGraph,
    pub notes: Vec<GraphTransformNote>,
    /// Number of ops rewritten by virtualizable lowering.
    pub vable_rewrites: usize,
    /// Number of calls classified.
    pub calls_classified: usize,
}

/// Rewrite a semantic graph with JIT-specific transformations.
///
/// Convenience wrapper that creates a `Transformer` and runs it.
/// RPython equivalent: jtransform.py `transform_graph()`.
///
/// This wrapper does NOT run `lower_indirect_calls` (the
/// rtyper-equivalent pass lives in
/// `translator/rtyper/rpbc.rs`).  Callers that can produce
/// `CallTarget::Indirect` must go through
/// `codewriter::transform_graph_to_jitcode` instead, which threads
/// `&CallControl` and runs the lowering pass before jtransform.
/// This debug-assertion catches missed lowering sites at the
/// remaining entries (test fixtures + the cfg(test)-gated
/// `translator/rtyper/legacy_pipeline.rs::analyze_function`) —
/// plan Rev 2 §Phase B3.
pub fn rewrite_graph(graph: &FunctionGraph, config: &GraphTransformConfig) -> GraphTransformResult {
    #[cfg(debug_assertions)]
    crate::translator::rtyper::rpbc::assert_no_indirect_call_targets(graph);
    let mut transformer = Transformer::new(config);
    transformer.transform(graph)
}

/// Variant of `rewrite_graph` that threads a `&mut CallControl` into the
/// `Transformer`. Required when the graph contains `OpKind::IndirectCall`
/// (emitted by `translator/rtyper/rpbc.rs::lower_indirect_calls`), since
/// `lower_indirect_call_op` needs `CallControl::getcalldescr` +
/// `guess_call_kind` + `graphs_from`. Used by the cfg(test)-gated
/// `translator::rtyper::legacy_pipeline::analyze_function` after it
/// calls `lower_indirect_calls` upstream.
pub fn rewrite_graph_with_callcontrol(
    graph: &FunctionGraph,
    config: &GraphTransformConfig,
    callcontrol: &mut crate::call::CallControl,
) -> GraphTransformResult {
    #[cfg(debug_assertions)]
    crate::translator::rtyper::rpbc::assert_no_indirect_call_targets(graph);
    let mut transformer = Transformer::new(config).with_callcontrol(callcontrol);
    transformer.transform(graph)
}

/// JIT graph transformer.
///
/// RPython equivalent: `jtransform.py` class `Transformer`.
///
/// Rewrites operations in a FunctionGraph to JIT-specific instructions:
/// - Virtualizable field/array access → VableFieldRead/VableArrayRead
/// - Hint calls → identity/VableForce
/// - Call classification → CallElidable/CallResidual/CallMayForce
pub struct Transformer<'a> {
    /// RPython: `Transformer.callcontrol`.
    callcontrol: Option<&'a mut crate::call::CallControl>,
    /// RPython: `Transformer.portal_jd` (`jtransform.py:65`) —
    /// "non-None only for the portal graph(s)". Consulted by
    /// `handle_jit_marker__jit_merge_point` (`jtransform.py:1690-1712`)
    /// to stamp `portal_jd.index` onto the rewritten op and to assert
    /// the marker's jitdriver matches this portal's. Pyre stores the
    /// index alone; the full `JitDriverStaticData` is owned by
    /// `CallControl::jitdrivers_sd` and can be looked up by index there.
    portal_jd_index: Option<usize>,
    /// RPython: `Transformer.__init__` config for virtualizable lowering.
    config: &'a GraphTransformConfig,
    /// Type resolution state from the rtype pass.
    /// Used by `make_three_lists()` to split args by kind.
    /// RPython: types are on `Variable.concretetype` — we pass them explicitly.
    /// RPython: `Transformer.vable_array_vars`.
    /// Stores (vable_base, array_index, itemsize, is_signed) per vable array variable.
    vable_array_vars: std::collections::HashMap<
        crate::flowspace::model::Variable,
        (crate::flowspace::model::Variable, usize, usize, bool),
    >,
    /// RPython: `Transformer.vable_flags`. Keyed by Variable identity
    /// matching upstream `self.vable_flags[op.args[0]] = ...`
    /// (`jtransform.py` populates with `Variable` objects).
    #[allow(dead_code)]
    vable_flags: std::collections::HashMap<crate::flowspace::model::Variable, VableFlag>,
    /// Value aliases from identity rewrites (same_as / hint rewriting).
    aliases: std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
    notes: Vec<GraphTransformNote>,
    vable_rewrites: usize,
    calls_classified: usize,
    /// RPython: DependencyTracker — caches transitive analysis results.
    /// Shared across all getcalldescr() calls within this transform pass.
    analysis_cache: crate::call::AnalysisCache,
}

/// RPython: jtransform.py vable_flags values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VableFlag {
    FreshVirtualizable,
}

/// `support.py:723 result.append(Constant(obj, lltype.typeOf(obj)))`
/// — prepend the constant-injection ops that `decode_builtin_call`'s
/// `NormalizedArg::ConstInt(v)` slots required, in front of the
/// downstream rewrite result.  No-op when no ConstInt slot was
/// materialised.  `Identity` / `Keep` are passed through unchanged
/// (downstream consumers of those variants do not observe the
/// materialised Variables — only `Replace` branches receive the
/// `effective_args` list).
fn prepend_const_prefix(prefix: &mut Vec<SpaceOperation>, result: RewriteResult) -> RewriteResult {
    if prefix.is_empty() {
        return result;
    }
    match result {
        RewriteResult::Replace(ops) => {
            let mut combined = std::mem::take(prefix);
            combined.extend(ops);
            RewriteResult::Replace(combined)
        }
        other => other,
    }
}

/// Result of rewriting a single operation.
///
/// RPython: `rewrite_operation()` returns SpaceOperation, list, None, or Constant.
enum RewriteResult {
    /// Replace with these ops
    Replace(Vec<SpaceOperation>),
    /// Remove the op (identity/alias: result remaps to the given Variable).
    /// RPython `jtransform.py:236 rewrite_op_same_as` returns `op.args[0]`
    /// directly; pyre wraps it in this enum so the caller can fold the
    /// alias into the rename map.
    Identity(crate::flowspace::model::Variable),
    /// Keep the op unchanged
    Keep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedCallResult {
    kind: char,
    ir_type: majit_ir::value::Type,
}

/// RPython: the `key` value stored as `op.args[0]` of a
/// `SpaceOperation('jit_marker', [key, jitdriver, *args])` operation
/// (`jtransform.py:1658-1663`). pyre's front-end does not carry the
/// `jit_marker` opname explicitly — the markers reach the codewriter as
/// `direct_call` s to `PyPyJitDriver::{jit_merge_point, can_enter_jit,
/// loop_header}`. This enum keeps the upstream key distinction inside
/// the dispatch hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JitMarkerKey {
    JitMergePoint,
    /// `can_enter_jit` aliases to `handle_jit_marker__loop_header`
    /// (jtransform.py:1723).
    CanEnterJit,
    LoopHeader,
}

fn jit_marker_key_from_target(target: &CallTarget) -> Option<JitMarkerKey> {
    let CallTarget::Method {
        name,
        receiver_root: Some(receiver_root),
        ..
    } = target
    else {
        return None;
    };
    if receiver_root != "PyPyJitDriver" {
        return None;
    }
    match name.as_str() {
        "jit_merge_point" => Some(JitMarkerKey::JitMergePoint),
        "can_enter_jit" => Some(JitMarkerKey::CanEnterJit),
        "loop_header" => Some(JitMarkerKey::LoopHeader),
        _ => None,
    }
}

/// Split a run of [`Variable`]s into (ints, refs, floats) per upstream
/// `make_three_lists` (`jtransform.py:1616-1627`). Void values are
/// dropped, matching the upstream filter; Unknown defaults to `Ref`.
/// Reads kinds via `FunctionGraph::concretetype_of(var)` — the same
/// `getkind(v.concretetype)` source as RPython's `flatten.py:382
/// getcolor` and `rtyper.py:258 v.concretetype = ...`.
fn split_args_by_kind(
    args: &[crate::flowspace::model::Variable],
) -> (
    Vec<crate::flowspace::model::Variable>,
    Vec<crate::flowspace::model::Variable>,
    Vec<crate::flowspace::model::Variable>,
) {
    let mut ints = Vec::new();
    let mut refs = Vec::new();
    let mut floats = Vec::new();
    for v in args {
        let kind = match FunctionGraph::concretetype_of(v) {
            crate::jit_codewriter::type_state::ConcreteType::Signed => 'i',
            crate::jit_codewriter::type_state::ConcreteType::Float => 'f',
            crate::jit_codewriter::type_state::ConcreteType::Void => 'v',
            // RPython: GcRef or Unknown → 'r'
            crate::jit_codewriter::type_state::ConcreteType::GcRef
            | crate::jit_codewriter::type_state::ConcreteType::Unknown => 'r',
        };
        match kind {
            'i' => ints.push(v.clone()),
            'f' => floats.push(v.clone()),
            'v' => {}
            _ => refs.push(v.clone()),
        }
    }
    (ints, refs, floats)
}

impl<'a> Transformer<'a> {
    /// RPython: `Transformer.__init__(cpu=None, callcontrol=None, portal_jd=None)`
    /// (`jtransform.py:62-66`). Pyre keeps `cpu` / `callcontrol` behind
    /// builder setters because the borrow checker demands a late binding
    /// against the enclosing `CallControl`; `portal_jd` follows the same
    /// pattern. All three fields start `None`, matching upstream class
    /// defaults.
    pub fn new(config: &'a GraphTransformConfig) -> Self {
        Self {
            callcontrol: None,
            portal_jd_index: None,
            config,
            vable_array_vars: std::collections::HashMap::new(),
            vable_flags: std::collections::HashMap::new(),
            aliases: std::collections::HashMap::new(),
            notes: Vec::new(),
            vable_rewrites: 0,
            calls_classified: 0,
            analysis_cache: crate::call::AnalysisCache::default(),
        }
    }

    /// Set the CallControl for call kind dispatch.
    /// RPython: `Transformer.__init__(callcontrol=...)`.
    pub fn with_callcontrol(mut self, cc: &'a mut crate::call::CallControl) -> Self {
        self.callcontrol = Some(cc);
        self
    }

    /// Attach the portal JitDriverStaticData index for the current
    /// graph. RPython `jtransform.py:65 self.portal_jd = portal_jd`
    /// — "non-None only for the portal graph(s)". Pyre stores the
    /// index into `CallControl::jitdrivers_sd` rather than a direct
    /// reference so the builder does not force a second borrow of
    /// `CallControl`. `handle_jit_marker__jit_merge_point`
    /// (`jtransform.py:1690-1712`) uses this for both identity checks
    /// and `Constant(portal_jd.index, lltype.Signed)` synthesis.
    pub fn with_portal_jd(mut self, jd_index: Option<usize>) -> Self {
        self.portal_jd_index = jd_index;
        self
    }

    /// Accessor for the portal jitdriver index, matching upstream
    /// `self.portal_jd` reads inside Transformer methods.
    pub fn portal_jd_index(&self) -> Option<usize> {
        self.portal_jd_index
    }

    /// RPython: Transformer.transform() — process all blocks in the graph.
    ///
    /// Reads operand kinds via `FunctionGraph::concretetype_of(&v)`
    /// (RPython `getkind(v.concretetype)`).  Callers commit kinds to each
    /// backing `Variable.concretetype` cell upstream — through
    /// `legacy_resolve::resolve_types` (which writes through
    /// `FunctionGraph::set_concretetype_of_inline(&var, ct)` per-set)
    /// or `dual_gate_publish_concretetypes`.  Test fixtures that need
    /// to hand-set kinds call `FunctionGraph::set_concretetype_of_inline(&var, ct)`
    /// directly.
    pub fn transform(&mut self, graph: &FunctionGraph) -> GraphTransformResult {
        let mut rewritten = graph.clone();

        let exceptblock = rewritten.exceptblock;
        let graph_name = rewritten.name.clone();
        for block_idx in 0..rewritten.blocks.len() {
            self.optimize_block(&mut rewritten, block_idx, &graph_name, exceptblock);
        }

        GraphTransformResult {
            graph: rewritten,
            notes: std::mem::take(&mut self.notes),
            vable_rewrites: self.vable_rewrites,
            calls_classified: self.calls_classified,
        }
    }

    /// RPython: Transformer.optimize_block()
    fn optimize_block(
        &mut self,
        graph: &mut crate::model::FunctionGraph,
        block_idx: usize,
        graph_name: &str,
        exceptblock: crate::model::BlockId,
    ) {
        let mut new_ops = Vec::with_capacity(graph.blocks[block_idx].operations.len());

        let original_ops = graph.blocks[block_idx].operations.clone();
        for original_op in &original_ops {
            let op = remap_op(original_op, &self.aliases);
            match self.rewrite_operation(&op, graph_name, graph) {
                RewriteResult::Replace(ops) => {
                    new_ops.extend(ops);
                }
                RewriteResult::Identity(alias_target) => {
                    if let Some(result) = op.result.clone() {
                        self.aliases
                            .insert(result, resolve_alias(&alias_target, &self.aliases));
                    }
                }
                RewriteResult::Keep => {
                    new_ops.push(op);
                }
            }
        }

        let (exitswitch, exits) = {
            let block = &graph.blocks[block_idx];
            remap_control_flow_metadata_var(
                &block.exitswitch,
                &block.exits,
                |var| remap_value(var, &self.aliases),
                |b| b,
            )
        };
        let block = &mut graph.blocks[block_idx];
        block.operations = new_ops;
        block.exitswitch = exitswitch;
        block.exits = exits;

        // Upstream `rpython/translator/backendopt/canraise.py:25-47
        // analyze_exceptblock_in_graph` identifies raising blocks by the
        // presence of a Link in `Block.exits` whose target is
        // `graph.exceptblock`.  pyre records a `GraphTransformNote` for
        // such blocks so later phases (e.g. reporting) can surface
        // unconditional raise sites — the note mirrors the upstream signal
        // without the pyre-specific Terminator::Abort variant.
        if block.exits.iter().any(|link| link.target == exceptblock) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: "abort: raises to exceptblock".to_string(),
            });
        }
    }

    /// Variable-returning helper —
    /// allocates a fresh synthetic slot, stamps its `concretetype` cell
    /// to `ty`, and hands back the backing
    /// [`crate::flowspace::model::Variable`].  Call sites that hold the
    /// fresh slot only as a Variable handle (`op.result.clone()`
    /// downstream) use this to skip the local `must_variable`
    /// projection.  RPython parity:
    /// `Variable(concretetype=T)` (`flowspace/model.py:Variable.__init__`).
    fn fresh_synthetic_variable_typed(
        &mut self,
        graph: &mut FunctionGraph,
        ty: crate::jit_codewriter::type_state::ConcreteType,
    ) -> crate::flowspace::model::Variable {
        graph.alloc_value_var_with_type(ty)
    }

    /// RPython parity: `Variable.concretetype = ty` (`flowmodel.py
    /// Variable.__init__`).  Updates the backing Variable's
    /// `concretetype` cell in-place; the optional shape lets the
    /// jtransform rewrite arms call this with `op.result.clone()`
    /// for result-less ops without an extra guard.
    fn stamp_value_kind(
        &mut self,
        _graph: &FunctionGraph,
        value: Option<crate::flowspace::model::Variable>,
        ty: crate::jit_codewriter::type_state::ConcreteType,
    ) {
        if let Some(var) = value {
            FunctionGraph::set_concretetype_of_inline(&var, ty);
        }
    }

    /// Stamp the synthetic kind from a `ValueType` source, skipping
    /// `ValueType::Unknown` so the fallback Unknown→Ref defaulting in
    /// `value_type_to_kind` does not clobber a real `concretetype`
    /// already computed by the rtyper for an existing Variable.
    fn stamp_value_kind_from_value_type(
        &mut self,
        graph: &FunctionGraph,
        value: Option<crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) {
        let ty = match result_ty {
            ValueType::Int | ValueType::Unsigned | ValueType::Bool | ValueType::State => {
                crate::jit_codewriter::type_state::ConcreteType::Signed
            }
            ValueType::Ref => crate::jit_codewriter::type_state::ConcreteType::GcRef,
            ValueType::Float => crate::jit_codewriter::type_state::ConcreteType::Float,
            ValueType::Void => crate::jit_codewriter::type_state::ConcreteType::Void,
            ValueType::Unknown => return,
        };
        self.stamp_value_kind(graph, value, ty);
    }

    /// RPython `op.result.concretetype` is set by the rtyper, so jtransform
    /// reads it directly. Pyre's front-end can leave a callee return type
    /// as `ValueType::Unknown` when the callee path is unresolved (re-export
    /// shadowing, cross-crate paths). When that happens, the rtyper's
    /// backward-inference pass classifies `op.result` from its consumer-op
    /// constraints, and that classification is stamped on the result
    /// Variable's `concretetype` cell via `apply_to_graph` before this
    /// pass runs.  Reading `FunctionGraph::concretetype_of(&v)` therefore
    /// propagates the same `result_kind` the rtyper already chose,
    /// instead of
    /// falling back to `value_type_to_kind(Unknown) == 'r'`.
    fn resolve_call_result_kind(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) -> Option<char> {
        if !matches!(result_ty, ValueType::Unknown) {
            return None;
        }
        let var = result?;
        match FunctionGraph::concretetype_of(var) {
            crate::jit_codewriter::type_state::ConcreteType::Signed => Some('i'),
            crate::jit_codewriter::type_state::ConcreteType::GcRef => Some('r'),
            crate::jit_codewriter::type_state::ConcreteType::Float => Some('f'),
            crate::jit_codewriter::type_state::ConcreteType::Void => Some('v'),
            crate::jit_codewriter::type_state::ConcreteType::Unknown => None,
        }
    }

    /// RPython uses one `op.result.concretetype` for both
    /// `getkind(...)[0]` (opcode suffix) and `getcalldescr(..., RESULT)`.
    /// Keep the Rust port on that same source of truth so `Unknown`
    /// results resolved by `type_state` cannot produce `_i` opnames with
    /// Ref-return calldescrs.
    fn resolve_call_result(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) -> ResolvedCallResult {
        let kind = self
            .resolve_call_result_kind(result, result_ty)
            .unwrap_or_else(|| value_type_to_kind(result_ty));
        let ir_type = match kind {
            'i' => majit_ir::value::Type::Int,
            'r' => majit_ir::value::Type::Ref,
            'f' => majit_ir::value::Type::Float,
            'v' => majit_ir::value::Type::Void,
            other => panic!("unsupported call result kind '{other}'"),
        };
        ResolvedCallResult { kind, ir_type }
    }

    fn direct_funcptr_value(
        &mut self,
        graph: &mut FunctionGraph,
        target: &CallTarget,
    ) -> (crate::flowspace::model::Variable, SpaceOperation) {
        let fnaddr = self
            .callcontrol
            .as_deref()
            .map(|cc| cc.fnaddr_for_target(target))
            .unwrap_or_else(|| crate::call::symbolic_fnaddr_for_target(target));
        // Function pointer materialized as ConstInt — assembler emits it
        // through the `'i'` argcode so the kind is Signed.
        let var = self.fresh_synthetic_variable_typed(
            graph,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );
        (
            var.clone(),
            SpaceOperation {
                result: Some(var),
                kind: OpKind::ConstInt(fnaddr),
            },
        )
    }

    /// RPython: Transformer.rewrite_operation() — dispatch to rewrite_op_*.
    fn rewrite_operation(
        &mut self,
        op: &SpaceOperation,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        match &op.kind {
            // ── rewrite_op_hint ──
            OpKind::Call { target, args, .. } if classify_vable_hint(target).is_some() => {
                self.rewrite_op_hint(op, target, args, graph_name)
            }
            // ── rewrite_op_getfield ──
            //
            // Unlike the setfield/getarrayitem dispatch this runs whether or
            // not `lower_virtualizable` is enabled: the quasi-immutable
            // `-live-` + `record_quasiimmut_field` pair from
            // `rpython/jit/codewriter/jtransform.py:895-903` is independent
            // of virtualizable lowering.  `rewrite_op_getfield` internally
            // falls through to `RewriteResult::Keep` for mutable fields and
            // plain immutables (their purity is carried on the descriptor).
            OpKind::FieldRead { field, ty, .. } => {
                self.rewrite_op_getfield(op, field, ty, graph_name)
            }
            // ── rewrite_op_setfield ──
            OpKind::FieldWrite {
                field, value, ty, ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_setfield(op, field, value, ty, graph_name)
            }
            // ── rewrite_op_getarrayitem ──
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
                ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_getarrayitem(op, base, index, item_ty, graph_name)
            }
            // ── rewrite_op_setarrayitem ──
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
                ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_setarrayitem(op, base, index, value, item_ty, graph_name)
            }
            // ── rewrite_op_direct_call ──
            OpKind::Call {
                target,
                args,
                result_ty,
            } if self.config.classify_calls => {
                self.rewrite_op_direct_call(op, target, args, result_ty, graph_name, graph)
            }
            // ── rewrite_op_indirect_call ──
            // RPython jtransform.py:410-412. Pyre's rtyper-equivalent
            // (`translator/rtyper/rpbc.rs`) lowers
            // `OpKind::Call { target: CallTarget::Indirect, .. }` into
            // `OpKind::IndirectCall { funcptr, args, graphs, .. }`
            // before jtransform runs, so by this point the funcptr is
            // already a regular Variable and `args` still carry the full
            // call argument list, including the receiver.
            OpKind::IndirectCall {
                funcptr,
                args,
                graphs,
                result_ty,
            } if self.config.classify_calls => self.lower_indirect_call_op(
                op,
                funcptr,
                args,
                graphs.as_deref(),
                result_ty,
                graph_name,
                graph,
            ),
            // ── abort placeholders ──
            OpKind::Abort { kind } => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("abort placeholder: {:?}", kind),
                });
                RewriteResult::Keep
            }
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if matches!(binop_name.as_str(), "bitand" | "bitor" | "bitxor") => {
                let canonical = match binop_name.as_str() {
                    "bitand" => "and",
                    "bitor" => "or",
                    "bitxor" => "xor",
                    _ => unreachable!("matched bit op names above"),
                };
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: canonical.into(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "same_as"
                || unop_name == "deref"
                || unop_name == "cast_bool_to_int"
                || unop_name == "cast_bool_to_uint"
                || unop_name == "cast_int_to_uint"
                || unop_name == "cast_uint_to_int" =>
            {
                // RPython `jtransform.py:246-248 rewrite_op_same_as`:
                //
                //     def rewrite_op_same_as(self, op):
                //         if op.args[0] in self.vable_array_vars:
                //             self.vable_array_vars[op.result] = \
                //                 self.vable_array_vars[op.args[0]]
                //
                // `rewrite_op_same_as` returns `None` implicitly.  In
                // `optimize_block` that means "remove the op and rename
                // the result to args[0]" (`jtransform.py:106-111`).
                //
                // `cast_bool_to_int` / `cast_bool_to_uint` /
                // `cast_int_to_uint` / `cast_uint_to_int` follow the
                // same drop-and-alias shape — RPython
                // `jtransform.py:330,331,336,337`:
                //
                //     def rewrite_op_cast_bool_to_int(self, op): pass
                //     def rewrite_op_cast_bool_to_uint(self, op): pass
                //     def rewrite_op_cast_int_to_uint(self, op): pass
                //     def rewrite_op_cast_uint_to_int(self, op): pass
                //
                // are explicit no-ops.  Each cast is identity at LL
                // level because `getkind(lltype.Bool) == getkind(
                // lltype.Signed) == getkind(lltype.Unsigned) == 'int'`
                // (`history.py:45-63`), so backend opcodes need not
                // grow separate handlers.
                //
                // Pyre instead drops the op (`RewriteResult::Identity`)
                // and records `op.result -> *operand` in `self.aliases`
                // (line 446-450). Subsequent ops in the same block
                // (and `block.exitswitch` / `link.args` via
                // `remap_control_flow_metadata`) go through `remap_op`
                // at line 441 before dispatch, so a consumer that
                // originally referenced `op.result` is rewritten to
                // reference `*operand` directly. The later
                // `vable_array_vars.get(&base)` lookup in
                // `rewrite_op_getarrayitem`/`_setarrayitem` then hits
                // the original `(*operand)` entry — same outcome as
                // upstream's explicit propagation, without keeping a
                // redundant alias key.
                RewriteResult::Identity(operand.clone())
            }
            // RPython `jtransform.py:1592` rename pass:
            //   ('cast_bool_to_float', 'cast_int_to_float'),
            // The Bool register class is 'int' at LL, so the same
            // `cast_int_to_float` machine op handles the conversion;
            // backend opcodes don't need a separate `cast_bool_to_float`.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "cast_bool_to_float" => {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::UnaryOp {
                        op: "cast_int_to_float".into(),
                        operand: operand.clone(),
                        result_ty: ValueType::Float,
                    },
                }])
            }
            // RPython `jtransform.py:1243-1255` `rewrite_op_ptr_eq`/`rewrite_op_ptr_ne`
            // + `_rewrite_cmp_ptrs`: equality/inequality of two Ref operands is
            // `ptr_eq`/`ptr_ne` (wired at `blackhole.py:585-590`), not `int_eq`/
            // `int_ne`. Pyre's front-end emits a unified `BinOp { op: "eq"/"ne" }`
            // because Rust's `==`/`!=` is one AST node regardless of operand type;
            // the jtransform layer is where RPython branches on operand kind.
            // Both operands Ref → rewrite to `ptr_eq`/`ptr_ne`. Mixed/Int operands
            // stay as `int_eq`/`int_ne`.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if (binop_name == "eq" || binop_name == "ne")
                && self.get_value_kind_var(lhs) == 'r'
                && self.get_value_kind_var(rhs) == 'r' =>
            {
                let new_op = if binop_name == "eq" {
                    "ptr_eq"
                } else {
                    "ptr_ne"
                };
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: new_op.into(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            // jtransform.py:1227-1235 `_rewrite_cmp_ptrs`: non-GC ptr
            // equality becomes int_eq/int_ne.  Mixed i+r operands from
            // Pyre's unified BinOp need cast_ptr_to_int on the Ref side
            // (blackhole.py:603-606 `cast_ptr_to_int/r>i` is wired).
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if (binop_name == "eq" || binop_name == "ne")
                && (self.get_value_kind_var(lhs) == 'r' || self.get_value_kind_var(rhs) == 'r') =>
            {
                let (lhs_coerced, mut ops) = self.coerce_operand_to_int(graph, lhs);
                let (rhs_coerced, rhs_ops) = self.coerce_operand_to_int(graph, rhs);
                ops.extend(rhs_ops);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: binop_name.clone(),
                        lhs: lhs_coerced,
                        rhs: rhs_coerced,
                        result_ty: result_ty.clone(),
                    },
                });
                RewriteResult::Replace(ops)
            }
            // RPython rtyper produces `float_*` opnames directly when
            // operand `concretetype` is `lltype.Float` — there is no
            // `int_*` op with float operands in RPython's IR
            // (`rpython/rtyper/rfloat.py` rtype_add etc. emit
            // `float_add` / `float_sub` / `float_mul` / `float_truediv`
            // / `float_lt` / `float_eq` etc.).  Pyre's front-end emits
            // a unified `OpKind::BinOp { op: "add" }` because Rust's
            // `+` is one AST node regardless of operand type; this
            // jtransform pass mirrors RPython's rtyper-level
            // distinction by rewriting `int_*` over Float operands to
            // `float_*`.  Both operands Float → arithmetic returns
            // Float; comparisons return Int (bool).  RPython's rtyper
            // inserts `cast_int_to_float` for mixed int/float pairs
            // before emitting the `float_*` op; pyre's lighter rtyper
            // leaves the generic BinOp in place, so jtransform performs
            // that local coercion here.
            //
            // `mod` is handled separately: RPython does not provide
            // `float_mod` (`rpython/rtyper/lltypesystem/lloperation.py:260`
            // "don't implement float_mod, use math.fmod instead"), so
            // `%` over floats lowers to a residual `ll_math_fmod` call.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if canonical_float_arith_binop(binop_name).is_some()
                && is_float_rewrite_domain(self.get_value_kind_var(lhs))
                && is_float_rewrite_domain(self.get_value_kind_var(rhs))
                && (self.get_value_kind_var(lhs) == 'f'
                    || self.get_value_kind_var(rhs) == 'f'
                    || *result_ty == ValueType::Float) =>
            {
                let canonical = match canonical_float_arith_binop(binop_name)
                    .expect("guard checked float arithmetic op")
                {
                    "div" => "truediv", // RPython lltype: `float_truediv`
                    other => other,
                };
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(graph, lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(graph, rhs);
                ops.extend(rhs_ops);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: format!("float_{canonical}"),
                        lhs,
                        rhs,
                        result_ty: ValueType::Float,
                    },
                });
                RewriteResult::Replace(ops)
            }
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                ..
            } if matches!(binop_name.as_str(), "lt" | "le" | "gt" | "ge" | "eq" | "ne")
                && is_float_rewrite_domain(self.get_value_kind_var(lhs))
                && is_float_rewrite_domain(self.get_value_kind_var(rhs))
                && (self.get_value_kind_var(lhs) == 'f' || self.get_value_kind_var(rhs) == 'f') =>
            {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Signed,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(graph, lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(graph, rhs);
                ops.extend(rhs_ops);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: format!("float_{binop_name}"),
                        lhs,
                        rhs,
                        result_ty: ValueType::Int,
                    },
                });
                RewriteResult::Replace(ops)
            }
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if canonical_float_mod_binop(binop_name).is_some()
                && is_float_rewrite_domain(self.get_value_kind_var(lhs))
                && is_float_rewrite_domain(self.get_value_kind_var(rhs))
                && (self.get_value_kind_var(lhs) == 'f'
                    || self.get_value_kind_var(rhs) == 'f'
                    || *result_ty == ValueType::Float) =>
            {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(graph, lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(graph, rhs);
                ops.extend(rhs_ops);
                let target = CallTarget::function_path(["ll_math_fmod"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
                ops.push(funcptr_op);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr),
                        descriptor: CallDescriptor::from_signature(
                            &[majit_ir::value::Type::Float, majit_ir::value::Type::Float],
                            majit_ir::value::Type::Float,
                            EffectInfo::new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None),
                        ),
                        args_i: vec![],
                        args_r: vec![],
                        args_f: vec![lhs, rhs],
                        result_kind: 'f',
                        indirect_targets: None,
                    },
                });
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                });
                RewriteResult::Replace(ops)
            }
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "neg" && self.get_value_kind_var(operand) == 'f' => {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::UnaryOp {
                        op: "float_neg".into(),
                        operand: operand.clone(),
                        result_ty: ValueType::Float,
                    },
                }])
            }
            // RPython hits Python-`%` / `//` semantics through TWO
            // distinct routes upstream:
            //
            //   (a) the bare-op `rewrite_op_int_floordiv =
            //       _do_builtin_call` / `rewrite_op_int_mod =
            //       _do_builtin_call` rewrite at
            //       `jtransform.py:576-577`, which replaces the
            //       SpaceOperation with `direct_call(_ll_2_int_floordiv,
            //       ...)` / `direct_call(_ll_2_int_mod, ...)` BEFORE
            //       jitcode emission.  `_do_builtin_call` resolves the
            //       helper through `support.py:266 _ll_2_int_mod` /
            //       `:255 _ll_2_int_floordiv` and binds the function
            //       pointer via `support.builtin_func_for_spec`.  The
            //       post-rewrite call carries NO oopspec markup
            //       (`_ll_2_int_*` have no `@oopspec` decorator) and
            //       the helper output is C-TRUNCATING (the no-branch
            //       reverse of `ll_int_py_div`); any Python-floor
            //       correction must come from the bytecode emitter at
            //       the BinOp callsite.
            //
            //   (b) the OS_INT_PY_MOD / OS_INT_PY_DIV oopspec at
            //       `jtransform.py:2043`, reached when the rtyper
            //       directly emits `int_py_mod` / `int_py_div` ops
            //       (typically via `objspace/std/intobject.py` Python-
            //       semantic `%` / `//`).  This route stamps
            //       `int.py_mod` / `int.py_div` oopspec on the
            //       residual call so optimisations
            //       (`rewrite.py:713-766 optimize_call_int_py_div`,
            //       `intbounds.py:1654 postprocess_int_floordiv`)
            //       recognise it, and the helper output is
            //       PYTHON-FLOOR (`rint.py:399-500 ll_int_py_div` /
            //       `ll_int_py_mod`).
            //
            // Pyre's front-end emits `BinOp { op: "mod" | "floordiv" }`
            // over `i64` operands from Rust's `%` / `/` — the C-style
            // truncating primitive of the Rust language, lowered from
            // helpers like `pyre-interpreter/src/baseobjspace.rs::int_mod`
            // whose body computes `let r = va % vb; if r != 0 &&
            // (r ^ vb) < 0 { r + vb } else { r }` to convert the
            // C-trunc step into Python-floor.  The route-(a) match —
            // C-truncating input semantic, no oopspec markup — is the
            // structural fit, so the rewrite emits a `CallResidual` to
            // `_ll_2_int_mod` / `_ll_2_int_floordiv` (registered at
            // `pyre/jit_fnaddr.rs` with the canonical RPython helper
            // names; bodies in `majit_metainterp::blackhole::_ll_2_int_*`
            // reduce to `wrapping_rem` / `wrapping_div`) with
            // `OopSpecIndex::None` and `ExtraEffect::CannotRaise`.
            //
            // Effect parity: upstream `_do_builtin_call` does NOT
            // grant `EF_ELIDABLE_*` to helpers that lack the
            // `@elidable` annotator decoration — the residual call's
            // effect family is inherited from the function's actual
            // RPython annotation, not synthesised by `_do_builtin_call`.
            // `support.py:255-271 _ll_2_int_floordiv` / `_ll_2_int_mod`
            // carry no such decorator (compare `rint.py:496
            // @jit.oopspec("int.py_mod")` which DOES decorate the
            // Python-floor sibling).  Pyre therefore stamps
            // `CannotRaise`, not `ElidableCannotRaise`: the C-trunc
            // helpers do not raise (panics on `y == 0` are unreachable
            // from the trace path, gated by the wrapper's pre-check at
            // the BinOp callsite), but the JIT must NOT assume the
            // call is pure (would license CSE / DCE the upstream effect
            // family does not).
            //
            // Pre-existing optimisation passes
            // (`optimize_call_int_py_mod` /
            // `optimize_call_int_py_div` at `optimizeopt/rewrite.rs:1788,1848`)
            // stay parked for the future route-(b) path: pyre has no
            // Python-bytecode emitter that produces `int.py_mod` /
            // `int.py_div` oopspec calls today, so those passes are
            // dormant.  Performance recovery for the BinOp{mod,Int}
            // path lands when (and only when) a route-(a) optimization
            // pass is ported on top of the C-trunc helper.
            //
            // Without this rewrite the assembler encoder
            // (`jit_codewriter/assembler.rs:2778-2789
            // `format!("int_{op}")``) would emit the bare opname,
            // leaking `int_mod/ii>i` / `int_floordiv/ii>i` into
            // `pipeline.insns` where no blackhole handler exists.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if matches!(binop_name.as_str(), "mod" | "floordiv" | "div")
                && matches!(self.get_value_kind_var(lhs), 'i' | 'r')
                && matches!(self.get_value_kind_var(rhs), 'i' | 'r')
                && self.binop_result_is_int(op.result.as_ref(), result_ty) =>
            {
                // **Rust low-level → RPython low-level.**  Pyre's
                // `BinOp { op: "mod" | "floordiv" | "div" }` over i64
                // operands comes from Rust's `%` / `/` operators,
                // which are C-truncating primitives.  At the IR level
                // these match RPython's low-level
                // `llop.int_mod` / `llop.int_floordiv` — the C-trunc
                // ops that the rtyper emits and that
                // `support.py:255-271 _ll_2_int_floordiv` /
                // `_ll_2_int_mod` are the no-branch helpers for
                // (`_ll_2_*` are used "only if the RPython program
                // uses `llop.int_floordiv()` explicitly", per the
                // upstream comment).
                //
                // **NOT** Python `%` / `//` / `/` parity.  High-level
                // Python ops are handled at the rtyper layer:
                // `rint.py:246-262 rtype_mod` / `rtype_floordiv` (and
                // their `rtype_inplace_*` siblings, plus
                // `rtype_div = rtype_floordiv` for integer `/`) call
                // `_rtype_call_helper(hop, 'py_mod'/'py_div', [...])`
                // which invokes the *Python-floor* helpers
                // `ll_int_py_mod` / `ll_int_py_div`
                // (`rint.py:399-500`).  Pyre is NOT porting that
                // path — it is mapping Rust's C-trunc primitives to
                // the RPython C-trunc primitives.
                //
                // Pyre carries the same `int_div`-as-`int_floordiv`
                // collapse at this layer: the rtyper-equivalent never
                // produces an `int_div` op for integer operands
                // (there is no such llop), so pyre routes
                // `BinOp { op:"div" }` through the same
                // `_ll_2_int_floordiv` residual as the `floordiv`
                // canonical.
                //
                // The gate checks the result's proven concretetype
                // explicitly so a Ref/Float/State-typed BinOp does not
                // slip through and emit an `int_mod/ii>i` opname onto
                // a non-int SSA result; when AST lowering left
                // `result_ty == Unknown`, the rtyper-equivalent
                // `type_state` is the carrier for `op.result.concretetype`.
                let _ = result_ty;
                let helper_key = if binop_name == "mod" {
                    "mod"
                } else {
                    "floordiv"
                };
                // PRE-EXISTING-ADAPTATION (no direct RPython precedent):
                // pyre-side recovery when an explicit
                // `lltype.cast_ptr_to_int` (`rbuiltin.py:543-548
                // genop('cast_ptr_to_int', vlist, resulttype=Signed)`)
                // emitted by the front-end is elided from the SSA chain
                // before reaching jtransform.  The gate accepts a
                // Ref-typed LHS/RHS and rebuilds the missing cast here
                // so the residual call sees Signed operands.  RPython
                // `rint.py:246-262 rtype_mod` does NOT auto-cast
                // arbitrary Ref operands; its `hop.inputargs(repr,
                // repr)` assumes the rtyper has already inserted
                // explicit casts at lltype / rbuiltin boundaries.  The
                // wider tolerance here is therefore strictly broader
                // than the RPython contract and exists only to keep the
                // dispatch table closed while the upstream cast-
                // elision is traced and fixed (the convergence path is
                // to find which simplify / inline pass drops the cast
                // and preserve it instead, then narrow this gate back
                // to `'i' && 'i'`).
                let (lhs_var, lhs_pre_ops) = self.coerce_operand_to_int(graph, lhs);
                let (rhs_var, rhs_pre_ops) = self.coerce_operand_to_int(graph, rhs);
                let mut ops = Vec::with_capacity(lhs_pre_ops.len() + rhs_pre_ops.len() + 2);
                ops.extend(lhs_pre_ops);
                ops.extend(rhs_pre_ops);
                ops.extend(self.emit_int_mod_or_floordiv_residual(
                    graph,
                    helper_key,
                    &lhs_var,
                    &rhs_var,
                    op.result.clone(),
                ));
                RewriteResult::Replace(ops)
            }
            // RPython `Transformer.rewrite_op_float_is_true(self, op)`
            // (`jtransform.py:1627-1631`):
            //
            //     def rewrite_op_float_is_true(self, op):
            //         op1 = SpaceOperation('float_ne',
            //                              [op.args[0], Constant(0.0, lltype.Float)],
            //                              op.result)
            //         return self.rewrite_operation(op1)
            //
            // Two upstream surfaces both lower to this rewrite in pyre:
            //
            //   1. The front-end emits the un-rtyped `OpKind::UnaryOp
            //      { op: "bool", .. }` over a float-kind operand
            //      (RPython `op.bool` before the rtyper).
            //   2. The rtyper itself emits `OpKind::UnaryOp { op:
            //      "float_is_true", .. }` from `FloatRepr.rtype_bool`
            //      (`rfloat.rs:191-198`, mirror of upstream
            //      `rfloat.py:rtype_bool`).
            //
            // Both shapes must be rewritten here.  If neither is
            // caught the assembler emits a literal `float_is_true`
            // opname, but downstream backends only register
            // `float_ne` — RPython jtransform.py:1627 collapses both
            // surfaces to the same canonical shape.  Pyre's rewriter
            // does not chain back into `rewrite_operation` the way
            // upstream does (the loop at `jtransform.rs:446-462`
            // consumes `Replace(ops)` without re-dispatch), so emit
            // the canonical `float_ne` opname here rather than
            // leaving an intermediate op for the float-comparison
            // arm at `jtransform.rs:827-854`.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if (unop_name == "bool" && self.get_value_kind_var(operand) == 'f')
                || unop_name == "float_is_true" =>
            {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Signed,
                );
                let zero_var = self.fresh_synthetic_variable_typed(
                    graph,
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                let zero_op = SpaceOperation {
                    result: Some(zero_var.clone()),
                    kind: OpKind::ConstFloat(0.0_f64.to_bits()),
                };
                let ne_op = SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: "float_ne".into(),
                        lhs: operand.clone(),
                        rhs: zero_var,
                        result_ty: ValueType::Int,
                    },
                };
                RewriteResult::Replace(vec![zero_op, ne_op])
            }
            // RPython `jtransform.py:587-588`:
            //   rewrite_op_cast_float_to_uint  = _do_builtin_call
            //   rewrite_op_cast_uint_to_float  = _do_builtin_call
            // `_do_builtin_call` (`jtransform.py:587-588`) re-routes
            // through support helpers
            // (`rpython/jit/codewriter/support.py:274 _ll_1_cast_*`)
            // so blackhole never sees a bare `cast_*_to_uint*` opname
            // — instead a `direct_call(<helper-funcptr>)` residual
            // call carries unsigned-domain semantics (e.g.
            // `r_uint(long(f))` mod 2^64 wrap, `float(u64_value)`
            // u64-domain rounding) preserved at runtime.
            //
            // Pyre's rtyper-side `rtype_cast_uint_to_float` /
            // `rtype_cast_float_to_uint`
            // (`rbuiltin.rs::rtype_cast_uint_to_float` line 1217 and
            // `rtype_cast_float_to_uint` line 1233) emit the literal
            // opname.  The signed-domain `cast_int_to_float` /
            // `cast_float_to_int` backend instructions do NOT
            // preserve the unsigned semantics: `f as i64 as f64`
            // rounds for the signed range, not the unsigned range
            // (e.g. `u64::MAX` becomes `-1.0` instead of `~1.84e19`),
            // and `f as i64` saturates outside `[-2^63, 2^63)` rather
            // than wrapping mod 2^64.
            //
            // The arms below synthesise the `direct_call` shape
            // matching `_do_builtin_call`:
            //   - `direct_funcptr_value` produces a `ConstInt` of the
            //     helper's runtime fnaddr (registered by
            //     `pyre/pyre-interpreter/src/jit_fnaddr.rs` →
            //     `majit_metainterp::blackhole::cast_*_to_*`).
            //   - `OpKind::CallResidual` mirrors
            //     `residual_call_irf_<f|i>` (`handle_residual_call` at
            //     `jtransform.py:439-470` for the integer / float
            //     register classes).
            //   - `ElidableCannotRaise` + `OopSpecIndex::None` keeps
            //     the call out of the `may_call_jitcodes` /
            //     `calldescr_canraise` set, so no `-live-` is
            //     appended (`test_flatten.py:1007-1023`).
            // Const-fold path lives in `opimpl.rs::op_cast_*`; the
            // runtime helpers reproduce the same IEEE-754 mantissa
            // decomposition so runtime and const-fold agree.
            //
            // Coverage: today these arms are unreachable from pyre
            // source — `front/ast.rs:classify_fn_arg_ty` folds u*
            // into `ValueType::Int`, so the rtyper never emits
            // `cast_uint_to_float` / `cast_float_to_uint` opnames in
            // practice.  The wiring is staged for the eventual
            // producer flip — see
            // `front/ast.rs:classify_fn_arg_ty`'s
            // TODO(unsigned-producer-flip) for the cascade list.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "cast_uint_to_float" => {
                // `jtransform.py:587 _do_builtin_call` routes through
                // `support.py:274 _ll_1_cast_uint_to_float`.  Helper
                // `blackhole.rs::cast_uint_to_float` carries u64-domain
                // rounding matching `opimpl.rs::op_cast_uint_to_float`.
                // `handle_residual_call` (`jtransform.py:456-470`) only
                // appends `-live-` when `may_call_jitcodes` or the
                // descriptor can raise; this helper is
                // `ElidableCannotRaise` with `OopSpecIndex::None` so the
                // flatten is `residual_call_irf_f ... -> %f0` /
                // `float_return %f0` with no intervening `-live-`
                // (`test_flatten.py:1021-1023`).
                let target = CallTarget::function_path(["cast_uint_to_float"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                let mut ops = Vec::with_capacity(2);
                ops.push(funcptr_op);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr),
                        descriptor: CallDescriptor::from_signature(
                            &[majit_ir::value::Type::Int],
                            majit_ir::value::Type::Float,
                            EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None),
                        ),
                        args_i: vec![operand.clone()],
                        args_r: vec![],
                        args_f: vec![],
                        result_kind: 'f',
                        indirect_targets: None,
                    },
                });
                RewriteResult::Replace(ops)
            }
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "cast_float_to_uint" => {
                // `jtransform.py:588 _do_builtin_call` →
                // `support.py:274 _ll_1_cast_float_to_uint`.  Helper
                // mirrors `opimpl.rs::op_cast_float_to_uint` (mod 2^64
                // wrap via mantissa/exponent decomposition).
                // `ElidableCannotRaise`+`OopSpecIndex::None` → no
                // `-live-` (`test_flatten.py:1007-1009`).
                let target = CallTarget::function_path(["cast_float_to_uint"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Signed,
                );
                let mut ops = Vec::with_capacity(2);
                ops.push(funcptr_op);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr),
                        descriptor: CallDescriptor::from_signature(
                            &[majit_ir::value::Type::Float],
                            majit_ir::value::Type::Int,
                            EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None),
                        ),
                        args_i: vec![],
                        args_r: vec![],
                        args_f: vec![operand.clone()],
                        result_kind: 'i',
                        indirect_targets: None,
                    },
                });
                RewriteResult::Replace(ops)
            }
            // RPython `jtransform.py:1606` rename pass:
            //   ('uint_is_true', 'int_is_true'),
            // The Unsigned register class is 'int' at LL, so `uint_is_true`
            // is a textual alias for `int_is_true`; the backend opcode
            // table only registers `int_is_true`.  Pyre's rtyper-side
            // `rtype_uint_is_true` (`rbuiltin.rs::rtype_uint_is_true`)
            // emits the literal `uint_is_true` opname for typing
            // `r_uint(...)` truthiness; without this rename the
            // assembler emits an unmapped opname.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "uint_is_true" => {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::jit_codewriter::type_state::ConcreteType::Signed,
                );
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::UnaryOp {
                        op: "int_is_true".into(),
                        operand: operand.clone(),
                        result_ty: ValueType::Int,
                    },
                }])
            }
            // pyre front-end emits assign-form binops (`add_assign`,
            // `mod_assign`, `div_assign`, ...) from Rust's `+=` / `%=`
            // / `/=` operators.  RPython has no analogue at this layer:
            // Python bytecode expands `+=` to BINARY_ADD + STORE_FAST
            // before the rtyper runs, so flow-space never sees an
            // in-place op.  Pyre canonicalises the assign form here,
            // dropping the `_assign` suffix; when the canonical name
            // matches an integer op pyre routes through a residual
            // helper above, emit the same residual directly so the
            // result does not leak past the rewrite as a bare `mod` /
            // `floordiv` / `div` op (this pass does not re-traverse
            // its own Replace output).
            //
            // **Rust low-level → RPython low-level.**  Pyre's
            // `mod_assign` / `div_assign` come from Rust's `%=` / `/=`
            // on i64 (C-trunc primitive), matching RPython's
            // `llop.int_mod` / `llop.int_floordiv` low-level llops.
            // This is **not** a port of upstream's `rtype_inplace_mod
            // = rtype_mod` / `rtype_inplace_div = rtype_inplace_floordiv`
            // — those route the *Python-level* `%=` / `/=` through
            // `_rtype_call_helper(hop, 'py_mod'/'py_div', ...)` to the
            // Python-floor `ll_int_py_mod` / `ll_int_py_div` helpers
            // (`rint.py:399-500`).  Pyre carries no Python-level
            // `%=` / `/=` at this layer.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if canonical_assign_binop(binop_name).is_some() => {
                let canonical =
                    canonical_assign_binop(binop_name).expect("guard checked assign binop");
                // `rint.py:253-255`: `rtype_div = rtype_floordiv` (and
                // `rtype_inplace_div = rtype_inplace_floordiv`) — integer
                // `div` collapses to `floordiv` at the rtyper layer.
                // `div_assign → "div"` from `canonical_assign_binop`,
                // then this branch treats `"div"` as `floordiv` for the
                // residual route, mirroring the plain `BinOp { op:"div" }`
                // arm above.
                let residual_key: Option<&'static str> = match canonical {
                    "mod" => Some("mod"),
                    "floordiv" | "div" => Some("floordiv"),
                    _ => None,
                };
                if let Some(key) = residual_key {
                    if self.get_value_kind_var(lhs) == 'i'
                        && self.get_value_kind_var(rhs) == 'i'
                        && self.binop_result_is_int(op.result.as_ref(), result_ty)
                    {
                        return RewriteResult::Replace(self.emit_int_mod_or_floordiv_residual(
                            graph,
                            key,
                            lhs,
                            rhs,
                            op.result.clone(),
                        ));
                    }
                }
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: canonical.to_string(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            _ => RewriteResult::Keep,
        }
    }

    // ── helpers ──────────────────────────────────────────────

    /// RPython: `Transformer.make_three_lists(vars)` (jtransform.py:437-445).
    /// Split args into three lists by kind (int, ref, float) keyed on
    /// the backing [`crate::flowspace::model::Variable`] (orthodox per
    /// `flowspace/model.py:Variable`).
    ///
    /// RPython: `add_in_correct_list(v, lst_i, lst_r, lst_f)` checks
    /// `getkind(v.concretetype)` and appends to the matching list.
    /// Void args are skipped.
    fn make_three_lists_from_vars(
        &self,
        args: &[crate::flowspace::model::Variable],
    ) -> (
        Vec<crate::flowspace::model::Variable>,
        Vec<crate::flowspace::model::Variable>,
        Vec<crate::flowspace::model::Variable>,
    ) {
        let mut args_i = Vec::new();
        let mut args_r = Vec::new();
        let mut args_f = Vec::new();
        for var in args {
            let kind = self.get_value_kind_var(var);
            match kind {
                'i' => args_i.push(var.clone()),
                'r' => args_r.push(var.clone()),
                'f' => args_f.push(var.clone()),
                'v' => {}                      // void — skip (RPython jtransform.py:449)
                _ => args_r.push(var.clone()), // unknown → ref
            }
        }
        (args_i, args_r, args_f)
    }

    /// RPython: `getkind(v.concretetype)` — get the kind of a value.
    ///
    /// Mirrors `rpython/jit/metainterp/history.py:45-71 getkind`: GC
    /// pointers (`TYPE.TO._gckind != 'raw'`, history.py:66-67) collapse
    /// to `"ref"`, raw pointers (`gckind == 'raw'`, history.py:64-65)
    /// collapse to `"int"`, primitives map to `"int"` / `"float"` /
    /// `"void"`.  The function therefore returns ONLY one of `'i'` /
    /// `'r'` / `'f'` / `'v'`; sub-pointer distinctions like
    /// `Ptr(rstr.STR)` vs `Ptr(rstr.UNICODE)` are NOT folded into the
    /// kind char.
    ///
    /// Pyre currently has no `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
    /// concrete-type channel because pyre-object lacks those GC layouts.
    /// Re-introducing that distinction must happen at the hint dispatch
    /// site, mirroring upstream's explicit
    /// `op.args[0].concretetype == lltype.Ptr(rstr.STR)` checks, not by
    /// overloading `getkind`.
    ///
    /// Refining this getter to return `'s'` / `'u'` would propagate
    /// the refinement into unrelated rewrite arms
    /// (`ptr_eq`/`ptr_ne` synthesis at line 775 above,
    /// `kind_char_to_name` opname formation in `assembler.rs`,
    /// `jit.assert_green` / `jit.isconstant` / `jit.isvirtual` etc.)
    /// where RPython expects a plain `'r'`, breaking parity at every
    /// such call site.
    ///
    /// Reads the Variable's `.concretetype` cell directly per RPython
    /// `rtyper.py:258 v.concretetype = ...` parity, falling back to
    /// `'r'` when the cell is `Unknown`.
    fn get_value_kind_var(&self, var: &crate::flowspace::model::Variable) -> char {
        match FunctionGraph::concretetype_of(var) {
            crate::jit_codewriter::type_state::ConcreteType::Signed => 'i',
            crate::jit_codewriter::type_state::ConcreteType::GcRef => 'r',
            crate::jit_codewriter::type_state::ConcreteType::Float => 'f',
            crate::jit_codewriter::type_state::ConcreteType::Void => 'v',
            crate::jit_codewriter::type_state::ConcreteType::Unknown => 'r',
        }
    }

    fn get_value_type(&self, var: &crate::flowspace::model::Variable) -> Option<ValueType> {
        // RPython `jit/codewriter/jtransform.py`: `getkind(v.concretetype)`
        // — read kind off the Variable.concretetype slot directly.
        match FunctionGraph::concretetype_of(var) {
            crate::jit_codewriter::type_state::ConcreteType::Signed => Some(ValueType::Int),
            crate::jit_codewriter::type_state::ConcreteType::GcRef => Some(ValueType::Ref),
            crate::jit_codewriter::type_state::ConcreteType::Float => Some(ValueType::Float),
            crate::jit_codewriter::type_state::ConcreteType::Void => Some(ValueType::Void),
            crate::jit_codewriter::type_state::ConcreteType::Unknown => None,
        }
    }

    fn binop_result_is_int(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) -> bool {
        if *result_ty == ValueType::Int {
            return true;
        }
        let Some(var) = result else {
            return false;
        };
        self.get_value_type(var) == Some(ValueType::Int)
    }

    /// Emit the `_ll_2_int_mod` / `_ll_2_int_floordiv` residual call
    /// pair (funcptr-materialisation op + `CallResidual`) shared
    /// between the plain `mod` / `floordiv` / `div` arm and the
    /// `canonical_assign_binop` arm.  `helper_key` is `"mod"` for
    /// the int-mod residual or anything else (`"floordiv"`, `"div"`)
    /// for the int-floordiv residual — `rint.py:253-255 rtype_div =
    /// rtype_floordiv` aliases integer `div` to `floordiv` at the
    /// rtyper layer, and pyre carries that alias through here.
    ///
    /// `jtransform.py:469-470`'s `-live-` gate fires only on
    /// `may_call_jitcodes or calldescr_canraise`; this residual is
    /// neither (the helper is an `extern "C"` C-truncating arithmetic
    /// primitive flagged `LLOp(canfold=True)` upstream —
    /// `lloperation.py:203-204`), so no `OpKind::Live` follows.
    fn emit_int_mod_or_floordiv_residual(
        &mut self,
        graph: &mut FunctionGraph,
        helper_key: &str,
        lhs: &crate::flowspace::model::Variable,
        rhs: &crate::flowspace::model::Variable,
        result: Option<crate::flowspace::model::Variable>,
    ) -> Vec<SpaceOperation> {
        let helper_name = if helper_key == "mod" {
            "_ll_2_int_mod"
        } else {
            "_ll_2_int_floordiv"
        };
        let target = CallTarget::function_path([helper_name]);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
        let lhs_var = lhs.clone();
        let rhs_var = rhs.clone();
        let mut ops = Vec::with_capacity(2);
        ops.push(funcptr_op);
        ops.push(SpaceOperation {
            result,
            kind: OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(funcptr),
                descriptor: CallDescriptor::from_signature(
                    &[majit_ir::value::Type::Int, majit_ir::value::Type::Int],
                    majit_ir::value::Type::Int,
                    EffectInfo::new(ExtraEffect::CannotRaise, OopSpecIndex::None),
                ),
                args_i: vec![lhs_var, rhs_var],
                args_r: vec![],
                args_f: vec![],
                result_kind: 'i',
                indirect_targets: None,
            },
        });
        ops
    }

    /// RPython's float rtyper calls `hop.inputargs(Float, Float)`, which
    /// inserts `cast_int_to_float` for mixed int/float operands before
    /// emitting `float_*` or the `math.fmod` helper call.  Pyre's float
    /// rewrite arms only enter when both operands are
    /// `is_float_rewrite_domain` (i.e. `'i'` or `'f'`); Ref operands do
    /// not reach this helper.
    fn coerce_operand_to_float_domain(
        &mut self,
        graph: &mut FunctionGraph,
        value: &crate::flowspace::model::Variable,
    ) -> (crate::flowspace::model::Variable, Vec<SpaceOperation>) {
        match self.get_value_kind_var(value) {
            'f' => (value.clone(), Vec::new()),
            'i' => self.coerce_operand_to_float(graph, value),
            _ => (value.clone(), Vec::new()),
        }
    }

    /// RPython's float rtyper calls `hop.inputargs(Float, Float)`, which
    /// inserts `cast_int_to_float` for mixed int/float operands before
    /// emitting `float_*` or the `math.fmod` helper call.
    fn coerce_operand_to_float(
        &mut self,
        graph: &mut FunctionGraph,
        value: &crate::flowspace::model::Variable,
    ) -> (crate::flowspace::model::Variable, Vec<SpaceOperation>) {
        if self.get_value_kind_var(value) != 'i' {
            return (value.clone(), Vec::new());
        }
        let coerced = self.fresh_synthetic_variable_typed(
            graph,
            crate::jit_codewriter::type_state::ConcreteType::Float,
        );
        (
            coerced.clone(),
            vec![SpaceOperation {
                result: Some(coerced),
                kind: OpKind::UnaryOp {
                    op: "cast_int_to_float".into(),
                    operand: value.clone(),
                    result_ty: ValueType::Float,
                },
            }],
        )
    }

    /// PRE-EXISTING-ADAPTATION recovery helper — no direct RPython
    /// precedent.  Inserts an explicit `cast_ptr_to_int` op for a
    /// Ref-typed operand reaching an arithmetic site that requires
    /// Int operands.  Upstream RPython does NOT auto-cast arbitrary
    /// Ref to Signed at the rtyper boundary; ptr→int conversions come
    /// from explicit `lltype.cast_ptr_to_int` builtins emitted at the
    /// rbuiltin layer.  Pyre's analyzer/simplify chain may elide an
    /// emitted cast before jtransform sees it, leaving a bare Ref at
    /// the binop callsite; this helper rebuilds the missing cast so
    /// the residual call sees Signed operands.  The convergence path
    /// is to fix the cast elision upstream and retire this helper.
    /// Non-Ref operands are returned unchanged.
    fn coerce_operand_to_int(
        &mut self,
        graph: &mut FunctionGraph,
        value: &crate::flowspace::model::Variable,
    ) -> (crate::flowspace::model::Variable, Vec<SpaceOperation>) {
        if self.get_value_kind_var(value) != 'r' {
            return (value.clone(), Vec::new());
        }
        let coerced = self.fresh_synthetic_variable_typed(
            graph,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );
        (
            coerced.clone(),
            vec![SpaceOperation {
                result: Some(coerced),
                kind: OpKind::UnaryOp {
                    op: "cast_ptr_to_int".into(),
                    operand: value.clone(),
                    result_ty: ValueType::Int,
                },
            }],
        )
    }

    // ── rewrite_op_* methods ──────────────────────────────────

    /// RPython: `Transformer.rewrite_op_hint(op)`.
    /// Dispatches based on the hint kind (access_directly, force_virtualizable,
    /// fresh_virtualizable, promote, etc.)
    fn rewrite_op_hint(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        graph_name: &str,
    ) -> RewriteResult {
        let hint_kind = match classify_vable_hint(target) {
            Some(k) => k,
            None => return RewriteResult::Keep,
        };
        match hint_kind {
            crate::hints::VirtualizableHintKind::AccessDirectly
            | crate::hints::VirtualizableHintKind::FreshVirtualizable => {
                // RPython: consume as identity (same_as)
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("rewrite: {target}(...) → identity"),
                });
                if let Some(arg) = args.first() {
                    RewriteResult::Identity(arg.clone())
                } else {
                    RewriteResult::Keep
                }
            }
            crate::hints::VirtualizableHintKind::ForceVirtualizable => {
                // RPython: emit hint_force_virtualizable, preserve value as identity
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("rewrite: {target}(...) → VableForce"),
                });
                self.vable_rewrites += 1;
                if let Some(arg) = args.first() {
                    let base = resolve_alias(arg, &self.aliases);
                    if let Some(result) = op.result.clone() {
                        self.aliases.insert(result, base.clone());
                    }
                    RewriteResult::Replace(vec![SpaceOperation {
                        result: None,
                        kind: OpKind::VableForce { base },
                    }])
                } else {
                    RewriteResult::Keep
                }
            }
            crate::hints::VirtualizableHintKind::Promote => {
                // `rpython/jit/codewriter/jtransform.py:608-614`:
                //     if hints.get('promote') and op.args[0].concretetype is not lltype.Void:
                //         assert op.args[0].concretetype != lltype.Ptr(rstr.STR)
                //         kind = getkind(op.args[0].concretetype)
                //         op0 = SpaceOperation('-live-', [], None)
                //         op1 = SpaceOperation('%s_guard_value' % kind, [op.args[0]], None)
                //         # the special return value None forces op.result
                //         # to be considered equal to op.args[0]
                //         return [op0, op1, None]
                //
                // Skip void args (`concretetype is not lltype.Void` guard).
                // The string-pointer special case (`promote_string` /
                // `str_guard_value`) is a separate hint kind; this arm only
                // handles plain `int/ref/float_guard_value` shapes.  The
                // `None` sentinel that aliases the result back to the input
                // is realized in pyre by `self.aliases.insert(result, base)`
                // before emitting the replacement ops.
                self.rewrite_op_hint_guard_value_family(op, target, args, graph_name)
            }
            crate::hints::VirtualizableHintKind::PromoteString => {
                // `rpython/jit/codewriter/jtransform.py:615-631 promote_string`:
                //     S = lltype.Ptr(rstr.STR)
                //     assert op.args[0].concretetype == S
                //     ...register OS_STREQ_NONNULL + emit str_guard_value...
                //
                // Pyre cannot satisfy `concretetype == Ptr(rstr.STR)`
                // because pyre-object has no `rstr.STR`-equivalent GC
                // struct (`rpython/rtyper/lltypesystem/rstr.py:1226-1237
                // STR.become({hash, chars: Array(Char)})`).  Fail loud
                // — same shape as upstream's `assert` failure — until
                // the layout is ported and the helper body
                // (`rpython/jit/codewriter/support.py:526-538
                // _ll_2_str_eq_nonnull`) lands.
                panic!(
                    "hint_promote_string reached `{target}` in `{graph_name}` \
                     but pyre-object has no `rstr.STR`-equivalent GC \
                     layout to satisfy `jtransform.py:619 assert \
                     op.args[0].concretetype == lltype.Ptr(rstr.STR)`. \
                     Port `rstr.py:1226-1237 STR` into pyre-object and \
                     `support.py:526-538 _ll_2_str_eq_nonnull` into \
                     `majit-metainterp::blackhole` before re-enabling \
                     this rewrite arm."
                )
            }
            crate::hints::VirtualizableHintKind::PromoteUnicode => {
                // `rpython/jit/codewriter/jtransform.py:632-648 promote_unicode`:
                //     U = lltype.Ptr(rstr.UNICODE)
                //     assert op.args[0].concretetype == U
                //     ...register OS_UNIEQ_NONNULL + emit str_guard_value...
                //
                // Same parity blocker as `PromoteString`: pyre-object
                // has no `rstr.UNICODE`-equivalent GC struct
                // (`rpython/rtyper/lltypesystem/rstr.py:1238-1246
                // UNICODE.become({hash, chars: Array(UniChar)})`).
                panic!(
                    "hint_promote_unicode reached `{target}` in `{graph_name}` \
                     but pyre-object has no `rstr.UNICODE`-equivalent \
                     GC layout to satisfy `jtransform.py:636 assert \
                     op.args[0].concretetype == lltype.Ptr(rstr.UNICODE)`."
                )
            }
            crate::hints::VirtualizableHintKind::PromoteOrString => {
                // `rpython/jit/codewriter/jtransform.py:599-606` —
                // when a `hint(arg, ...)` carries both `promote=True`
                // and `promote_string=True`, jtransform discards one
                // based on `op.args[0].concretetype`:
                //
                //     if hints.get('promote_string') and hints.get('promote'):
                //         hints = hints.copy()
                //         if op.args[0].concretetype == lltype.Ptr(rstr.STR):
                //             del hints['promote']
                //         else:
                //             del hints['promote_string']
                //
                // Pyre has no `Ptr(rstr.STR)` layout (see
                // `PromoteString` arm above), so the upstream `if`
                // branch is structurally unreachable — every dual-flag
                // hint takes the `else` branch
                // (`del hints['promote_string']`) and falls through
                // to the plain `promote` arm, emitting
                // `<kind>_guard_value` per `jit.py:608-614` +
                // `getkind(Ptr) == "ref"` (`rpython/jit/metainterp/
                // history.py:64`).
                self.rewrite_op_hint_guard_value_family(op, target, args, graph_name)
            }
        }
    }

    /// Shared body for the `promote=True` arm of
    /// `rpython/jit/codewriter/jtransform.py:608-614 rewrite_op_hint`.
    ///
    /// Returns `RewriteResult::Replace([-live-, <kind>_guard_value(arg)])`
    /// after seeding `self.aliases.insert(result, arg)` to realize the
    /// upstream `None` sentinel (`# the special return value None forces
    /// op.result to be considered equal to op.args[0]`).  The
    /// `<kind>` char is the upstream `getkind()` of `op.args[0]`
    /// (`'i'`/`'r'`/`'f'`); void args fall through.
    ///
    /// `promote_string` / `promote_unicode` (jit.py:615-648) emit the
    /// 3-input `str_guard_value/rid>r` op which requires extras the
    /// IR does not yet carry — those arms panic in the caller above
    /// rather than route through this helper, which only knows how to
    /// emit the 1-input `<kind>_guard_value` family.
    fn rewrite_op_hint_guard_value_family(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        graph_name: &str,
    ) -> RewriteResult {
        let Some(arg) = args.first() else {
            return RewriteResult::Keep;
        };
        let base = resolve_alias(arg, &self.aliases);
        let kind_char = self.get_value_kind_var(&base);
        // jtransform.py:608 `op.args[0].concretetype is not lltype.Void`
        // guard — void args fall through the rewrite (caller may want
        // to keep the original op).
        if kind_char == 'v' {
            return RewriteResult::Keep;
        }
        // jtransform.py:609 `assert op.args[0].concretetype !=
        // lltype.Ptr(rstr.STR)` — pyre has no `Ptr(rstr.STR)` GC
        // layout (`rpython/rtyper/lltypesystem/rstr.py:1226-1237`), so the
        // upstream assertion is structurally satisfied by absence: no
        // pyre value can carry that concretetype, hence no `Ptr(STR)`
        // operand can reach this arm.  Re-introduce the assertion
        // once pyre-object grows the layout and the `PromoteString` /
        // `PromoteUnicode` arms can satisfy their upstream asserts.
        if let Some(result) = op.result.clone() {
            self.aliases.insert(result, base.clone());
        }
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("rewrite: {target}(...) → {kind_char}_guard_value"),
        });
        RewriteResult::Replace(vec![
            SpaceOperation {
                result: None,
                kind: OpKind::Live,
            },
            SpaceOperation {
                result: None,
                kind: OpKind::GuardValue {
                    value: base,
                    kind_char,
                },
            },
        ])
    }

    /// RPython `rpython/jit/codewriter/jtransform.py:830-906 rewrite_op_getfield`.
    ///
    /// Virtualizable lowering takes precedence (RPython `self.vable_array_vars`
    /// tracking + immediate return).  Otherwise the field's immutability rank
    /// drives the emit shape:
    ///
    /// * `IR_IMMUTABLE`           → rewrite the read to
    ///   `getfield_*_pure` (`jtransform.py:875-877`).
    /// * `IR_QUASIIMMUTABLE[_ARRAY]` → emit `[-live-, record_quasiimmut_field,
    ///   getfield_*_pure]` — `jtransform.py:895-903`.
    /// * mutable                  → keep as-is.
    fn rewrite_op_getfield(
        &mut self,
        op: &SpaceOperation,
        field: &FieldDescriptor,
        ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let typed_ty = op
            .result
            .as_ref()
            .and_then(|result| self.get_value_type(result))
            .unwrap_or_else(|| ty.clone());
        // Track virtualizable array field reads
        if let Some(array_field) = self.config.vable_arrays.iter().find(|c| c.matches(field)) {
            if let Some(result) = op.result.clone() {
                // RPython: vable_array_vars[result] = (v_base, arrayfielddescr, arraydescr)
                // We store the vable base plus the arraydescr properties.
                let base_var = match &op.kind {
                    OpKind::FieldRead { base, .. } => base.clone(),
                    _ => unreachable!("rewrite_op_getfield called on non-FieldRead op"),
                };
                let itemsize = array_field.array_itemsize.unwrap_or(8);
                let is_signed = array_field.array_is_signed.unwrap_or(false);
                self.vable_array_vars
                    .insert(result, (base_var, array_field.index, itemsize, is_signed));
            }
        }
        // Virtualizable scalar field → VableFieldRead
        if let Some(vable_field) = self.config.vable_fields.iter().find(|c| c.matches(field)) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!(
                    "rewrite: {} → VableFieldRead[{}]",
                    field.name, vable_field.index
                ),
            });
            self.vable_rewrites += 1;
            let base_var = match &op.kind {
                OpKind::FieldRead { base, .. } => base.clone(),
                _ => unreachable!("rewrite_op_getfield called on non-FieldRead op"),
            };
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::VableFieldRead {
                    base: base_var,
                    field_index: vable_field.index,
                    ty: typed_ty.clone(),
                },
            }]);
        }
        // `jtransform.py:867-903` — immutable and quasi-immutable
        // field reads both become `getfield_*_pure`; the quasi variant
        // additionally prepends `-live-` + `record_quasiimmut_field`.
        //    return [SpaceOperation('-live-', [], None),
        //            SpaceOperation('record_quasiimmut_field',
        //                           [v_inst, descr, descr1], None),
        //            op1]       # op1 = getfield_*_pure
        // Mutable fields stay as plain `getfield_gc_*`.
        let rank = self
            .callcontrol
            .as_deref()
            .and_then(|cc| cc.field_immutability(field.owner_root.as_deref(), &field.name));
        if let Some(rank) = rank {
            let OpKind::FieldRead {
                base,
                field: _,
                ty: _,
                pure: _,
            } = &op.kind
            else {
                return RewriteResult::Keep;
            };
            let base = base.clone();
            if rank.is_quasi_immutable() {
                // PRE-EXISTING-ADAPTATION: RPython
                // `quasiimmut.get_mutate_field_name(fieldname)` —
                // `rpython/jit/metainterp/quasiimmut.py:11-15` — strips the
                // lltype `inst_` prefix before prepending `mutate_`.  Rust
                // structs carry no such prefix, so we prepend `mutate_`
                // directly.
                let mutate_field = FieldDescriptor::new(
                    format!("mutate_{}", field.name),
                    field.owner_root.clone(),
                );
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!(
                        "rewrite: getfield({owner}.{name}) → -live- + record_quasiimmut_field + pure read",
                        owner = field.owner_root.as_deref().unwrap_or("<?>"),
                        name = field.name,
                    ),
                });
                return RewriteResult::Replace(vec![
                    SpaceOperation {
                        result: None,
                        kind: OpKind::Live,
                    },
                    SpaceOperation {
                        result: None,
                        kind: OpKind::RecordQuasiImmutField {
                            base: base.clone(),
                            field: field.clone(),
                            mutate_field,
                        },
                    },
                    SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::FieldRead {
                            base: base.clone(),
                            field: field.clone(),
                            ty: typed_ty.clone(),
                            pure: true,
                        },
                    },
                ]);
            }
            if rank.is_immutable() {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!(
                        "rewrite: getfield({owner}.{name}) → pure read",
                        owner = field.owner_root.as_deref().unwrap_or("<?>"),
                        name = field.name,
                    ),
                });
                return RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::FieldRead {
                        base: base.clone(),
                        field: field.clone(),
                        ty: typed_ty.clone(),
                        pure: true,
                    },
                }]);
            }
        }
        if &typed_ty != ty {
            let OpKind::FieldRead { base, pure, .. } = &op.kind else {
                return RewriteResult::Keep;
            };
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::FieldRead {
                    base: base.clone(),
                    field: field.clone(),
                    ty: typed_ty,
                    pure: *pure,
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: rewrite_op_setfield
    fn rewrite_op_setfield(
        &mut self,
        op: &SpaceOperation,
        field: &FieldDescriptor,
        value: &crate::flowspace::model::Variable,
        ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let typed_ty = self.get_value_type(value).unwrap_or_else(|| ty.clone());
        if let Some(vable_field) = self.config.vable_fields.iter().find(|c| c.matches(field)) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!(
                    "rewrite: {} = ... → VableFieldWrite[{}]",
                    field.name, vable_field.index
                ),
            });
            self.vable_rewrites += 1;
            let base_var = match &op.kind {
                OpKind::FieldWrite { base, .. } => base.clone(),
                _ => unreachable!("rewrite_op_setfield called on non-FieldWrite op"),
            };
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::VableFieldWrite {
                    base: base_var,
                    field_index: vable_field.index,
                    value: value.clone(),
                    ty: typed_ty,
                },
            }]);
        }
        if &typed_ty != ty {
            let base_var = match &op.kind {
                OpKind::FieldWrite { base, .. } => base.clone(),
                _ => unreachable!("rewrite_op_setfield called on non-FieldWrite op"),
            };
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::FieldWrite {
                    base: base_var,
                    field: field.clone(),
                    value: value.clone(),
                    ty: typed_ty,
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: rewrite_op_getarrayitem
    fn rewrite_op_getarrayitem(
        &mut self,
        op: &SpaceOperation,
        base: &crate::flowspace::model::Variable,
        index: &crate::flowspace::model::Variable,
        item_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let typed_item_ty = op
            .result
            .as_ref()
            .and_then(|result| self.get_value_type(result))
            .unwrap_or_else(|| item_ty.clone());
        if let Some((vable_base, arr_idx, itemsize, is_signed)) =
            self.vable_array_vars.get(base).cloned()
        {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("rewrite: array[idx] → VableArrayRead[{arr_idx}]"),
            });
            self.vable_rewrites += 1;
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::VableArrayRead {
                    base: vable_base,
                    array_index: arr_idx,
                    elem_index: index.clone(),
                    item_ty: typed_item_ty,
                    array_itemsize: itemsize,
                    array_is_signed: is_signed,
                },
            }]);
        }
        if &typed_item_ty != item_ty {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::ArrayRead {
                    base: base.clone(),
                    index: index.clone(),
                    item_ty: typed_item_ty,
                    array_type_id: match &op.kind {
                        OpKind::ArrayRead { array_type_id, .. } => array_type_id.clone(),
                        _ => unreachable!("rewrite_op_getarrayitem called on non-ArrayRead op"),
                    },
                    nolength: match &op.kind {
                        OpKind::ArrayRead { nolength, .. } => *nolength,
                        _ => unreachable!("rewrite_op_getarrayitem called on non-ArrayRead op"),
                    },
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: rewrite_op_setarrayitem
    fn rewrite_op_setarrayitem(
        &mut self,
        op: &SpaceOperation,
        base: &crate::flowspace::model::Variable,
        index: &crate::flowspace::model::Variable,
        value: &crate::flowspace::model::Variable,
        item_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let typed_item_ty = self
            .get_value_type(value)
            .unwrap_or_else(|| item_ty.clone());
        if let Some((vable_base, arr_idx, itemsize, is_signed)) =
            self.vable_array_vars.get(base).cloned()
        {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("rewrite: array[idx] = v → VableArrayWrite[{arr_idx}]"),
            });
            self.vable_rewrites += 1;
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::VableArrayWrite {
                    base: vable_base,
                    array_index: arr_idx,
                    elem_index: index.clone(),
                    value: value.clone(),
                    item_ty: typed_item_ty,
                    array_itemsize: itemsize,
                    array_is_signed: is_signed,
                },
            }]);
        }
        if &typed_item_ty != item_ty {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::ArrayWrite {
                    base: base.clone(),
                    index: index.clone(),
                    value: value.clone(),
                    item_ty: typed_item_ty,
                    array_type_id: match &op.kind {
                        OpKind::ArrayWrite { array_type_id, .. } => array_type_id.clone(),
                        _ => unreachable!("rewrite_op_setarrayitem called on non-ArrayWrite op"),
                    },
                    nolength: match &op.kind {
                        OpKind::ArrayWrite { nolength, .. } => *nolength,
                        _ => unreachable!("rewrite_op_setarrayitem called on non-ArrayWrite op"),
                    },
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: `Transformer.rewrite_op_direct_call(op)`.
    ///
    /// RPython jtransform.py:406-410:
    /// ```python
    /// def rewrite_op_direct_call(self, op):
    ///     kind = self.callcontrol.guess_call_kind(op)
    ///     return getattr(self, 'handle_%s_call' % kind)(op)
    /// ```
    fn rewrite_op_direct_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        // RPython `jtransform.py:406-408`:
        //   def rewrite_op_direct_call(op): ... handle_%s_call
        //
        // The indirect path (`jtransform.py:410-412 rewrite_op_indirect_call`)
        // is reached via `OpKind::IndirectCall` which the rtyper-equivalent
        // layer (`translator/rtyper/rpbc.rs::lower_indirect_calls`) emits
        // before jtransform runs, dispatched from `rewrite_operation` to
        // `lower_indirect_call_op`. By this point, no `CallTarget::Indirect`
        // can survive — that invariant is asserted in debug builds by
        // `assert_no_indirect_call_targets`.
        debug_assert!(
            !matches!(target, CallTarget::Indirect { .. }),
            "CallTarget::Indirect must be lowered by translator/rtyper/rpbc.rs \
             before reaching rewrite_op_direct_call",
        );
        // RPython `jtransform.py:1658-1663 rewrite_op_jit_marker`:
        // marker calls never reach `guess_call_kind` — they dispatch straight
        // to `handle_jit_marker__*`. Upstream keys on `op.args[0].value`;
        // pyre keys on the direct_call callee identity since the front-end
        // lowers `driver.jit_merge_point(...)` etc. to `CallTarget::Method`.
        if let Some(key) = jit_marker_key_from_target(target) {
            if let Some(ops) = self.try_handle_jit_marker(key, args) {
                return RewriteResult::Replace(ops);
            }
        }
        // STRUCTURAL-ADAPTATION (Rust-frontend → RPython rtyper gap).
        // RPython's rtyper resolves `Result`/`Option`-style tagged-union
        // construction to malloc + setfield at lowering time, so by the
        // time a graph reaches jtransform there are no `Ok(_)`-wrapper
        // SpaceOperations left to classify. Pyre's Rust frontend
        // (`front/ast.rs:1471 syn::Expr::Call`) lowers every `Path(args)`
        // expression to `OpKind::Call` uniformly — `Ok(StepResult::Continue)`
        // becomes a residual call to a `(r) → r` funcptr, and the trace
        // recorder emits a `CallR` op for it. The trait-dispatch path
        // executes the same constructor as zero-cost native code and
        // emits NO IR ops, so shadow-walker validation diverges by one
        // synthetic `CallR` per opcode arm. Recognise that single known
        // family of transparent Rust wrappers here and elide it via the
        // existing alias mechanism — same orthodoxy as RPython
        // `_noop_rewrite` (jtransform.py:399-401), just at the call-shape
        // surface that pyre still needs because its frontend skips the
        // rtyper step.
        //
        // Identity check is delegated to
        // [`Self::is_synthetic_result_option_ctor`]. The frontend must
        // already have resolved this to `CallTarget::SyntheticTransparentCtor`;
        // jtransform does not perform name-only matching.
        if self.is_synthetic_result_option_ctor(target, args, result_ty) {
            return RewriteResult::Identity(args[0].clone());
        }
        // RPython: guess_call_kind(op) → dispatch to handle_*_call
        if let Some(cc) = self.callcontrol.as_mut() {
            let kind = cc.guess_call_kind(op);
            return match kind {
                crate::call::CallKind::Regular => {
                    self.handle_regular_call(op, target, args, result_ty, graph_name, graph)
                }
                crate::call::CallKind::Residual => {
                    // RPython jtransform.py:456-471:
                    //   calldescr = self.callcontrol.getcalldescr(op, ...)
                    //   op1 = self.rewrite_call(op, 'residual_call', ...)
                    //
                    // RPython ALWAYS produces residual_call_* for residual
                    // calls — the effect is only in the calldescr, NOT in
                    // the opcode name. No dispatch_by_effect.
                    // RPython call.py:220-222: NON_VOID_ARGS + RESULT. Even
                    // for a configured effect override, keep the signature from
                    // getcalldescr() instead of accepting an effect-only descr.
                    let non_void_args = resolve_non_void_arg_types_from_vars(args);
                    let result_ir_type = self
                        .resolve_call_result(op.result.as_ref(), result_ty)
                        .ir_type;
                    let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
                    let extraeffect =
                        classify_call(target, &self.config.call_effects, Some(cc_ref))
                            .map(|(d, _)| d.extra_info.extraeffect);
                    let descriptor = cc_ref.getcalldescr(
                        op,
                        non_void_args,
                        result_ir_type,
                        OopSpecIndex::None,
                        extraeffect,
                        &mut self.analysis_cache,
                        None,
                    );
                    self.handle_residual_call(
                        graph, op, target, descriptor, args, result_ty, graph_name,
                    )
                }
                crate::call::CallKind::Builtin => {
                    self.handle_builtin_call(op, target, args, result_ty, graph_name, graph)
                }
                crate::call::CallKind::Recursive => {
                    self.handle_recursive_call(op, target, args, result_ty, graph_name, graph)
                }
            };
        }

        // Fallback when no CallControl: effect-only classification (legacy path).
        // RPython: always residual_call_*, effect only in calldescr.
        if let Some((descriptor, _effect)) = classify_call(
            target,
            &self.config.call_effects,
            self.callcontrol.as_deref(),
        ) {
            let non_void_args = resolve_non_void_arg_types_from_vars(args);
            let descriptor = descriptor.with_signature(
                &non_void_args,
                self.resolve_call_result(op.result.as_ref(), result_ty)
                    .ir_type,
            );
            self.handle_residual_call(graph, op, target, descriptor, args, result_ty, graph_name)
        } else {
            RewriteResult::Keep
        }
    }

    /// RPython: `Transformer.handle_builtin_call(op)`.
    /// Builtin operations with oopspec semantics — dispatched to
    /// specific lowering based on the oopspec name.
    ///
    /// RPython jtransform.py:484-520.
    ///
    /// Currently: look up effect from describe_call / call_effects
    /// and produce the matching typed call op. Future: oopspec-specific
    /// lowering (list_getitem → getarrayitem_gc, etc.)
    fn handle_builtin_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        // RPython `jtransform.py:484-485`:
        //   oopspec_name, args = support.decode_builtin_call(op)
        //
        // Run the strict-parity decode here so the per-prefix dispatch
        // below sees the normalized (permuted / constant-injected) arg
        // list, not the raw call args.  When the target carries no
        // oopspec registration, `decode_builtin_call` is not invoked —
        // pyre treats that as the unclassified-builtin path and falls
        // through to the `describe_call` / `Keep` branches below.
        //
        // When `oopspec_argnames` is registered for the target,
        // `decode_builtin_call` runs the full `parse_oopspec` +
        // `normalize_opargs` pipeline.  `NormalizedArg::ConstInt(v)`
        // slots are materialised here as fresh `OpKind::ConstInt(v)`
        // ops prepended to whatever the downstream rewrite emits —
        // mirroring upstream's `Constant(obj, ...)` Variable
        // introduction.  Without an argname registry the normalized
        // list is just `Pass(vid)` for each raw arg, so the
        // effective_args == args.to_vec() and no prefix ops are
        // emitted.
        let (decoded_oopspec, effective_args, mut const_prefix_ops) =
            if let Some(cc) = self.callcontrol.as_deref() {
                if cc.get_oopspec(target).is_some() {
                    let (name, normalized) = decode_builtin_call(op, cc);
                    let mut eff: Vec<crate::flowspace::model::Variable> =
                        Vec::with_capacity(normalized.len());
                    let mut prefix: Vec<SpaceOperation> = Vec::new();
                    for slot in normalized {
                        match slot {
                            NormalizedArg::Pass(var) => eff.push(var),
                            // `support.py:723 Constant(obj, lltype.Signed)`.
                            // Only integer literals (`lltype.Signed`-tagged)
                            // reach this arm — `parse_literal_slot` panics
                            // on char literals (`lltype.Char` has no pyre
                            // `ConcreteType` analogue) and routes float
                            // literals to `ConstFloat` below.
                            NormalizedArg::ConstInt(v) => {
                                let var = graph.alloc_value_var_with_type(
                                    crate::jit_codewriter::type_state::ConcreteType::Signed,
                                );
                                prefix.push(SpaceOperation {
                                    result: Some(var.clone()),
                                    kind: OpKind::ConstInt(v),
                                });
                                eff.push(var);
                            }
                            // `support.py:723 Constant(obj, lltype.Float)`
                            NormalizedArg::ConstFloat(bits) => {
                                let var = graph.alloc_value_var_with_type(
                                    crate::jit_codewriter::type_state::ConcreteType::Float,
                                );
                                prefix.push(SpaceOperation {
                                    result: Some(var.clone()),
                                    kind: OpKind::ConstFloat(bits),
                                });
                                eff.push(var);
                            }
                        }
                    }
                    (Some(name), eff, prefix)
                } else {
                    (None, args.to_vec(), Vec::new())
                }
            } else {
                (None, args.to_vec(), Vec::new())
            };
        let args: &[crate::flowspace::model::Variable] = &effective_args;
        let user_oopspec: Option<String> = decoded_oopspec.clone();

        // jtransform.py:487-511 — oopspec dispatch by prefix.
        if let Some(base) = user_oopspec.as_deref() {
            // jtransform.py:497 — jit.* oopspecs → __handle_jit_call
            if base.starts_with("jit.") {
                let result =
                    self._handle_jit_call(base, op, target, args, result_ty, graph_name, graph);
                return prepend_const_prefix(&mut const_prefix_ops, result);
            }
            // NOTE: conditional_call!/conditional_call_elidable!/record_known_result!
            // are handled by jitcode_lower (proc-macro level), NOT here.
            // The codewriter AST parser does not expand macro_rules!, so these
            // macros appear as Stmt::Macro → UnknownKind::MacroStmt.
            // The jitcode_lower proc-macro intercepts the macros directly and
            // emits BC_COND_CALL_* / BC_RECORD_KNOWN_RESULT_* bytecodes.
        }
        let (oopspecindex, extraeffect_override) = if let Some((descriptor, _)) = classify_call(
            target,
            &self.config.call_effects,
            self.callcontrol.as_deref(),
        ) {
            (
                descriptor.extra_info.oopspecindex,
                Some(descriptor.extra_info.extraeffect),
            )
        } else if let Some(descriptor) = crate::call::describe_call(target) {
            (
                descriptor.extra_info.oopspecindex,
                Some(descriptor.extra_info.extraeffect),
            )
        } else if let Some(spec) = user_oopspec.as_deref() {
            // rlib/jit.py:250 — map user oopspec string to OopSpecIndex.
            // jtransform.py:1731-1755 — jit.* oopspecs.
            let idx = map_user_oopspec_to_index(spec);
            (idx, None)
        } else {
            // Unknown builtin — keep as unclassified Call.
            return RewriteResult::Keep;
        };

        // RPython jtransform.py:1990-2002:
        //   calldescr = self.callcontrol.getcalldescr(op, oopspecindex, extraeffect)
        //
        // RPython reuses the same calldescr for both the op and callinfocollection.
        // We compute arg types once and clone for the collection.
        //
        // jtransform.py:2186 _handle_dict_lookup_call passes
        // `extradescr=[cpu.fielddescrof(STRUCT, 'entries'), cpu.arraydescrof(STRUCT.entries.TO)]`
        // derived from `op.args[1].concretetype.TO`. pyre's
        // `FunctionGraph::concretetype_of(&v)` collapses lltype to four kinds, so
        // the dict struct is not recoverable here — extradescrs stays
        // None until full lltype propagation lands.
        // OptHeap::_optimize_call_dict_lookup returns false on None extradescrs
        // and the call falls through emit_residual_call (heap.py:472-475 emit
        // → force_from_effectinfo on the call's own effectinfo).
        let non_void_args = resolve_non_void_arg_types_from_vars(args);
        let result_ir_type = self
            .resolve_call_result(op.result.as_ref(), result_ty)
            .ir_type;
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                oopspecindex,
                extraeffect_override,
                &mut self.analysis_cache,
                None,
            )
        };

        let effect_str = format!("{:?}", descriptor.extra_info.extraeffect);
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("builtin {target} → {effect_str}"),
        });
        self.calls_classified += 1;

        // RPython jtransform.py:2000-2002:
        //   func = ptr2int(op.args[0].value)
        //   self.callcontrol.callinfocollection.add(oopspecindex, calldescr, func)
        //
        // RPython reuses the SAME calldescr returned by getcalldescr() —
        // it carries the real NON_VOID_ARGS and RESULT types from call.py:334.
        if oopspecindex != OopSpecIndex::None {
            if let Some(cc) = self.callcontrol.as_mut() {
                let func_as_int = cc.fnaddr_for_target(target) as u64;

                cc.callinfocollection
                    .add(oopspecindex, descriptor.to_descr_ref(), func_as_int);
                cc.callinfocollection
                    .register_func_name(func_as_int, format!("{target}"));
            }
        }

        // RPython jtransform.py:2003-2007: __handle_oopspec_call always
        // produces residual_call_*, appends -live- if calldescr_canraise.
        // Effect is only in the calldescr, never in the opcode name.
        let result =
            self.handle_residual_call(graph, op, target, descriptor, args, result_ty, graph_name);
        prepend_const_prefix(&mut const_prefix_ops, result)
    }

    /// RPython: `Transformer.handle_regular_call(op)`.
    /// Callee is a candidate graph — emit `inline_call_*` referencing
    /// the callee's JitCode. The meta-interpreter will descend into
    /// the callee JitCode at runtime.
    ///
    /// RPython jtransform.py:473-482.
    fn handle_regular_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        // RPython jtransform.py:477-478: get_jitcode(targetgraph)
        //
        // Route through the CallControl-aware qualified path so inherent
        // methods like `PyFrame::pop_top` resolve to their
        // `function_graphs[["PyFrame", "pop_top"]]` entry. The stateless
        // `target_to_call_path` fallback only yields the bare method name,
        // which never matches `CallControl::register_inherent_method`'s
        // qualified key (`call.rs:941`) and leaves the shell body-less in
        // `drain_pending_graphs`. Phase I3 of the eval-loop automation
        // plan (`producer_side_jitcode_shell_leaks_2026_04_20.md`).
        let jitcode = if let Some(cc) = self.callcontrol.as_mut() {
            let path = cc
                .target_to_path(target)
                .unwrap_or_else(|| target_to_call_path(target));
            crate::jitcode::JitCodeHandle::new(cc.get_jitcode(&path))
        } else {
            crate::jitcode::JitCodeHandle::new(std::sync::Arc::new(crate::jitcode::JitCode::new(
                "<missing-callcontrol>",
            )))
        };
        // RPython jtransform.py:480: rewrite_call(op, 'inline_call', [jitcode])
        // Split args by kind (RPython make_three_lists)
        let (args_i, args_r, args_f) = self.make_three_lists_from_vars(args);
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);

        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → inline_call[jitcode={:?}]", jitcode.name),
        });
        self.calls_classified += 1;
        // RPython jtransform.py:480-481: inline_call always followed by -live-
        RewriteResult::Replace(vec![
            SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::InlineCall {
                    jitcode,
                    args_i,
                    args_r,
                    args_f,
                    result_kind,
                },
            },
            SpaceOperation {
                result: None,
                kind: OpKind::Live,
            },
        ])
    }

    /// RPython: `Transformer.handle_recursive_call(op)`.
    /// Recursive call back to the portal — emit `recursive_call_*`.
    ///
    /// RPython jtransform.py:522-534.
    fn handle_recursive_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        // RPython jtransform.py:522-534:
        //   jitdriver_sd = callcontrol.jitdriver_sd_from_portal_runner_ptr(funcptr)
        //   num_green_args = len(jitdriver_sd.jitdriver.greens)
        //   greens = args[1:1+num_green_args]
        //   reds = args[1+num_green_args:]
        //   recursive_call_{kind}(jd_index, G_I, G_R, G_F, R_I, R_R, R_F)
        let path = target_to_call_path(target);
        let (jd_index, num_green_args) = self
            .callcontrol
            .as_ref()
            .and_then(|cc| cc.jitdriver_sd_from_portal_graph(&path))
            .map(|sd| (sd.index, sd.greens.len()))
            .unwrap_or((0, 0));

        // RPython: skip funcptr (args[0]), split rest into green/red.
        // In our AST, args don't include funcptr, so split directly.
        let green_args = if num_green_args <= args.len() {
            &args[..num_green_args]
        } else {
            args
        };
        let red_args = if num_green_args <= args.len() {
            &args[num_green_args..]
        } else {
            &[]
        };
        let (greens_i, greens_r, greens_f) = self.make_three_lists_from_vars(green_args);
        let (reds_i, reds_r, reds_f) = self.make_three_lists_from_vars(red_args);
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);

        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!(
                "call {target} → recursive_call[jd={jd_index}, greens={num_green_args}]"
            ),
        });
        self.calls_classified += 1;

        // RPython jtransform.py:526: promote_greens emits guard_value
        // for each non-void green arg before the recursive_call.
        let mut ops = self.promote_greens(green_args);

        // RPython jtransform.py:532-533: recursive_call + -live-
        ops.push(SpaceOperation {
            result: op.result.clone(),
            kind: OpKind::RecursiveCall {
                jd_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
                result_kind,
            },
        });
        ops.push(SpaceOperation {
            result: None,
            kind: OpKind::Live,
        });
        RewriteResult::Replace(ops)
    }

    /// RPython: `Transformer.promote_greens(args, jitdriver)`.
    ///
    /// Emits `-live-` + `{kind}_guard_value` for each non-void green arg.
    /// This ensures green values are constant before the recursive call.
    ///
    /// RPython jtransform.py:1646-1656.
    fn promote_greens(
        &self,
        green_args: &[crate::flowspace::model::Variable],
    ) -> Vec<SpaceOperation> {
        let mut ops = Vec::new();
        for var in green_args {
            let kind = self.get_value_kind_var(var);
            if kind == 'v' {
                continue; // skip void
            }
            // RPython: -live- then {kind}_guard_value
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::GuardValue {
                    value: var.clone(),
                    kind_char: kind,
                },
            });
        }
        ops
    }

    /// RPython: `Transformer.__handle_jit_call(op, oopspec_name, args)` (jtransform.py:1730-1757).
    /// Dispatches jit.* oopspec calls to dedicated opcodes or __handle_oopspec_call.
    fn _handle_jit_call(
        &mut self,
        oopspec_name: &str,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        match oopspec_name {
            // jtransform.py:1731-1732
            "jit.debug" => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: "jit.debug → jit_debug".to_string(),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: None,
                    kind: OpKind::JitDebug {
                        args: args.to_vec(),
                    },
                }])
            }
            // jtransform.py:1733-1735
            "jit.assert_green" => {
                let value_var = args[0].clone();
                let kind_char = self.get_value_kind_var(&value_var);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.assert_green → {kind_char}_assert_green"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: None,
                    kind: OpKind::AssertGreen {
                        value: value_var,
                        kind_char,
                    },
                }])
            }
            // jtransform.py:1736-1737
            "jit.current_trace_length" => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: "jit.current_trace_length → current_trace_length".to_string(),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CurrentTraceLength,
                }])
            }
            // jtransform.py:1738-1740
            "jit.isconstant" => {
                let value_var = args[0].clone();
                let kind_char = self.get_value_kind_var(&value_var);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.isconstant → {kind_char}_isconstant"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::IsConstant {
                        value: value_var,
                        kind_char,
                    },
                }])
            }
            // jtransform.py:1741-1743
            "jit.isvirtual" => {
                let value_var = args[0].clone();
                let kind_char = self.get_value_kind_var(&value_var);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.isvirtual → {kind_char}_isvirtual"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::IsVirtual {
                        value: value_var,
                        kind_char,
                    },
                }])
            }
            // jtransform.py:1744-1747
            "jit.force_virtual" => self._handle_oopspec_call(
                graph,
                op,
                target,
                args,
                result_ty,
                graph_name,
                OopSpecIndex::JitForceVirtual,
                Some(majit_ir::descr::ExtraEffect::ForcesVirtualOrVirtualizable),
                None,
            ),
            // jtransform.py:1748-1755
            "jit.not_in_trace" => {
                // jtransform.py:1750-1753: not_in_trace must return void
                assert!(
                    *result_ty == ValueType::Void,
                    "jit.not_in_trace() function must return None"
                );
                self._handle_oopspec_call(
                    graph,
                    op,
                    target,
                    args,
                    result_ty,
                    graph_name,
                    OopSpecIndex::NotInTrace,
                    None,
                    None,
                )
            }
            // jtransform.py:1756-1757
            _ => {
                // jtransform.py:1757
                panic!("missing support for jit.* oopspec: {oopspec_name}");
            }
        }
    }

    /// RPython: `Transformer.__handle_oopspec_call(op, args, oopspecindex, extraeffect)`
    /// (jtransform.py:1988-2008).
    /// Produces a residual_call with the given oopspecindex embedded in the calldescr,
    /// and registers the function in the callinfocollection.
    fn _handle_oopspec_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        oopspecindex: OopSpecIndex,
        extraeffect: Option<majit_ir::descr::ExtraEffect>,
        extradescrs: Option<Vec<majit_ir::DescrRef>>,
    ) -> RewriteResult {
        // jtransform.py:1990-1993
        let non_void_args = resolve_non_void_arg_types_from_vars(args);
        let result_ir_type = self
            .resolve_call_result(op.result.as_ref(), result_ty)
            .ir_type;
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                oopspecindex,
                extraeffect,
                &mut self.analysis_cache,
                extradescrs,
            )
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("oopspec {oopspecindex:?} → residual_call"),
        });
        self.calls_classified += 1;
        // jtransform.py:1999-2002: callinfocollection.add
        if oopspecindex != OopSpecIndex::None {
            if let Some(cc) = self.callcontrol.as_mut() {
                let func_as_int = cc.fnaddr_for_target(target) as u64;
                cc.callinfocollection
                    .add(oopspecindex, descriptor.to_descr_ref(), func_as_int);
                cc.callinfocollection
                    .register_func_name(func_as_int, format!("{target}"));
            }
        }
        // jtransform.py:2003-2008: residual_call + optional -live-
        self.handle_residual_call(graph, op, target, descriptor, args, result_ty, graph_name)
    }

    // NOTE: rewrite_op_jit_conditional_call, _rewrite_op_cond_call, and
    // rewrite_op_jit_record_known_result are handled by jitcode_lower
    // (proc-macro level), not jtransform. The codewriter AST parser does
    // not expand macro_rules!, so these macros never reach jtransform.
    // See jitcode_lower.rs: lower_conditional_call, lower_conditional_call_elidable,
    // lower_record_known_result.
    //
    // `_rewrite_op_cond_call` below is a structural mirror of
    // `rpython/jit/codewriter/jtransform.py:1665-1683`. pyre dispatches
    // conditional_call via the proc-macro path (see above), so this
    // function is never reached at runtime; the Rust #[allow(dead_code)]
    // is a deliberate PRE-EXISTING-ADAPTATION marker. Keeping the body
    // here lets future porters cross-reference our conditional_call
    // lowering against the upstream flow line-by-line.

    /// RPython: `Transformer._rewrite_op_cond_call(op, rewritten_opname)`
    /// (jtransform.py:1665-1683).
    ///
    /// Called by upstream `rewrite_op_jit_conditional_call` and
    /// `rewrite_op_jit_conditional_call_value`; in pyre those two
    /// lower through `jitcode_lower::lower_conditional_call` /
    /// `lower_conditional_call_elidable` instead. This body is kept as
    /// structural documentation so the two code paths stay aligned.
    #[allow(dead_code)]
    fn _rewrite_op_cond_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        is_value: bool,
    ) -> RewriteResult {
        // jtransform.py:1666-1672: validate no floats, ≤4+2 args
        for arg in args {
            if self.get_value_kind_var(arg) == 'f' {
                panic!("Conditional call does not support floats");
            }
        }
        if args.len() > 4 + 2 {
            panic!("Conditional call does not support more than 4 arguments");
        }
        // jtransform.py:1673-1676: calldescr from function call (args[1:] → result)
        let condition_or_value_var = args[0].clone();
        let func_args: &[crate::flowspace::model::Variable] =
            if args.len() > 1 { &args[1..] } else { &[] };
        let non_void_args = resolve_non_void_arg_types_from_vars(func_args);
        let resolved_result = self.resolve_call_result(op.result.as_ref(), result_ty);
        let result_ir_type = resolved_result.ir_type;
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                OopSpecIndex::None,
                None,
                &mut self.analysis_cache,
                None,
            )
        };
        // jtransform.py:1677: assert not forces_virtual_or_virtualizable
        assert!(
            !descriptor
                .extra_info
                .check_forces_virtual_or_virtualizable(),
            "conditional_call target must not force virtualizable"
        );
        // jtransform.py:1678-1680: rewrite_call with force_ir=True
        let (args_i, args_r, args_f) = self.make_three_lists_from_vars(func_args);
        assert!(
            args_f.is_empty(),
            "force_ir: no float args in conditional_call"
        );
        let result_kind = resolved_result.kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let rewritten_opname = if is_value {
            "conditional_call_value"
        } else {
            "conditional_call"
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("{rewritten_opname} → {rewritten_opname}_ir_{result_kind}"),
        });
        self.calls_classified += 1;
        let call_kind = if is_value {
            OpKind::ConditionalCallValue {
                value: condition_or_value_var,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
                result_kind,
            }
        } else {
            OpKind::ConditionalCall {
                condition: condition_or_value_var,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
            }
        };
        // jtransform.py:1681-1682: -live- if calldescr_canraise
        let mut ops = vec![SpaceOperation {
            result: op.result.clone(),
            kind: call_kind,
        }];
        if descriptor.extra_info.check_can_raise(false) {
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
        }
        RewriteResult::Replace(ops)
    }

    /// RPython: `Transformer.rewrite_op_jit_conditional_call(op)`
    /// (jtransform.py:1685-1686). Dispatch wrapper kept for structural
    /// parity; pyre's `rewrite_operation` match does not reach it.
    #[allow(dead_code)]
    fn rewrite_op_jit_conditional_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self._rewrite_op_cond_call(graph, op, target, args, result_ty, graph_name, false)
    }

    /// RPython: `Transformer.rewrite_op_jit_conditional_call_value(op)`
    /// (jtransform.py:1687-1688). Dispatch wrapper kept for structural
    /// parity; pyre's `rewrite_operation` match does not reach it.
    #[allow(dead_code)]
    fn rewrite_op_jit_conditional_call_value(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self._rewrite_op_cond_call(graph, op, target, args, result_ty, graph_name, true)
    }

    /// RPython: `Transformer.rewrite_op_jit_marker(op)` (jtransform.py:1658-1663)
    /// — dispatch portion only. Upstream keys on `op.args[0].value`; pyre
    /// already matched the callee identity in `rewrite_op_direct_call` via
    /// `jit_marker_key_from_target`. Returns `None` when marker state is not
    /// yet wired (no `portal_jd_index`, no `CallControl`, or not enough args
    /// to separate greens from reds) so the caller can fall through to the
    /// regular direct-call handling.
    ///
    /// Upstream also honours `jitdriver.active` (jtransform.py:1661-1662);
    /// when `active=False` the marker is dropped. pyre has no `active` flag
    /// yet — once annotator-level JitDriver config lands it can emit
    /// `Some(Vec::new())` from this hook to match the `return []` shape.
    fn try_handle_jit_marker(
        &mut self,
        key: JitMarkerKey,
        args: &[crate::flowspace::model::Variable],
    ) -> Option<Vec<SpaceOperation>> {
        match key {
            JitMarkerKey::LoopHeader | JitMarkerKey::CanEnterJit => {
                // jtransform.py:1723 `handle_jit_marker__can_enter_jit =
                // handle_jit_marker__loop_header`.
                let jitdriver_index = self.portal_jd_index?;
                Some(self.handle_jit_marker__loop_header(jitdriver_index))
            }
            JitMarkerKey::JitMergePoint => {
                let jitdriver_index = self.portal_jd_index?;
                let cc = self.callcontrol.as_deref()?;
                let jd = cc.jitdriver_sd_from_jitdriver(jitdriver_index)?;
                let num_greens = jd.greens.len();
                // Skip the receiver: pyre lowers `driver.jit_merge_point(...)`
                // (`front/ast.rs::lower_expr:1181-1209`) as a method call whose
                // `Call.args[0]` is the `PyPyJitDriver` receiver; the user-facing
                // green/red arguments start at index 1. Upstream's equivalent
                // `jit_marker` op has `args[0]=marker_name_const` and
                // `args[1]=driver_const`, so `op.args[2:]` is the user payload.
                // `jit_marker_key_from_target` already consumed the method name
                // before reaching this branch, leaving only the driver receiver
                // to strip.
                let user_args = match args.split_first() {
                    Some((_receiver, rest)) if rest.len() >= num_greens => rest,
                    _ => return None,
                };
                // jtransform.py:1695 `ops = self.promote_greens(...)` —
                // prepends per-green `-live-` + `{kind}_guard_value` pairs.
                let greens_raw = &user_args[..num_greens];
                let mut ops = self.promote_greens(greens_raw);
                let (greens_i, greens_r, greens_f) = split_args_by_kind(greens_raw);
                let (reds_i, reds_r, reds_f) = split_args_by_kind(&user_args[num_greens..]);
                // jtransform.py:1712 final shape is `ops + [op3, op1, op2]`.
                ops.extend(self.handle_jit_marker__jit_merge_point(
                    greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
                ));
                Some(ops)
            }
        }
    }

    /// RPython: `Transformer.handle_jit_marker__jit_merge_point(op, jitdriver)`
    /// (jtransform.py:1690-1712). Called from `rewrite_op_jit_marker` when the
    /// marker key is `'jit_merge_point'`.
    ///
    /// Upstream takes a `SpaceOperation('jit_marker', [key, jitdriver, *args])`
    /// and `make_three_lists` both green and red args inside. pyre's
    /// `rewrite_op_direct_call` already splits call args by kind, so this port
    /// accepts already-split vectors — the caller feeds `args_i/args_r/args_f`
    /// partitioned at the green/red boundary.
    ///
    /// Returns `[live_preamble, jit_merge_point, live_recursive]`, matching
    /// upstream's `ops + [op3, op1, op2]` shape. The leading `promote_greens`
    /// prefix (`ops`) is empty until a follow-up slice ports `promote_greens`;
    /// until then greens arrive as Variables/Constants and are forwarded to
    /// the marker unchanged.
    fn handle_jit_marker__jit_merge_point(
        &mut self,
        greens_i: Vec<crate::flowspace::model::Variable>,
        greens_r: Vec<crate::flowspace::model::Variable>,
        greens_f: Vec<crate::flowspace::model::Variable>,
        reds_i: Vec<crate::flowspace::model::Variable>,
        reds_r: Vec<crate::flowspace::model::Variable>,
        reds_f: Vec<crate::flowspace::model::Variable>,
    ) -> Vec<SpaceOperation> {
        // jtransform.py:1691-1692 `assert self.portal_jd is not None`
        let jitdriver_index = self
            .portal_jd_index
            .expect("'jit_merge_point' in non-portal graph!");
        let merge = SpaceOperation {
            result: None,
            kind: OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
            },
        };
        // jtransform.py:1708-1712 — `op2` live for `do_recursive_call()`,
        // `op3` live for inlined short preambles. Final shape is
        // `ops + [op3, op1, op2]`.
        let live_preamble = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        let live_recursive = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        vec![live_preamble, merge, live_recursive]
    }

    /// RPython: `Transformer.handle_jit_marker__loop_header(op, jitdriver)`
    /// (jtransform.py:1714-1718). `handle_jit_marker__can_enter_jit` aliases
    /// to the same function (jtransform.py:1723); pyre keeps the alias at the
    /// `try_handle_jit_marker` dispatch layer rather than inside this method.
    fn handle_jit_marker__loop_header(&mut self, jitdriver_index: usize) -> Vec<SpaceOperation> {
        vec![SpaceOperation {
            result: None,
            kind: OpKind::LoopHeader { jitdriver_index },
        }]
    }

    /// RPython: `Transformer.rewrite_op_jit_record_known_result(op)`
    /// (jtransform.py:292-313).
    #[allow(dead_code)]
    fn rewrite_op_jit_record_known_result(
        &mut self,
        _graph: &FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        _result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // jtransform.py:293-295: validate no floats
        for arg in args {
            if self.get_value_kind_var(arg) == 'f' {
                panic!("record_known_result does not support floats");
            }
        }
        // jtransform.py:298-300: calldescr from function (args[1:] → args[0])
        // args[0] = known result, args[1..] = function args
        let result_value = args[0].clone();
        let func_args: &[crate::flowspace::model::Variable] =
            if args.len() > 1 { &args[1..] } else { &[] };
        let result_kind = self.get_value_kind_var(&result_value);
        let result_ir_type = match result_kind {
            'i' => majit_ir::value::Type::Int,
            'r' => majit_ir::value::Type::Ref,
            _ => {
                panic!("record_known_result: unsupported result kind '{result_kind}'");
            }
        };
        let non_void_args = resolve_non_void_arg_types_from_vars(func_args);
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                OopSpecIndex::None,
                None,
                &mut self.analysis_cache,
                None,
            )
        };
        // jtransform.py:301: assert calldescr.get_extra_info().check_is_elidable()
        assert!(
            descriptor.extra_info.check_is_elidable(),
            "record_known_result: function must be elidable"
        );
        // jtransform.py:302-307: record_known_result_{i|r}
        let opname = format!("record_known_result_{result_kind}");
        // jtransform.py:308-310: rewrite_call with force_ir=True
        let (args_i, args_r, args_f) = self.make_three_lists_from_vars(func_args);
        assert!(
            args_f.is_empty(),
            "force_ir: no float args in record_known_result"
        );
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("{opname} → {opname}_ir_v"),
        });
        self.calls_classified += 1;
        // jtransform.py:311-313: -live- if calldescr_canraise
        let mut ops = vec![SpaceOperation {
            result: None, // record_known_result produces void
            kind: OpKind::RecordKnownResult {
                result_value,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
                result_kind,
            },
        }];
        if descriptor.extra_info.check_can_raise(false) {
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
        }
        RewriteResult::Replace(ops)
    }

    /// RPython: `Transformer.handle_residual_call(op)` (jtransform.py:456-471).
    /// Call that the JIT should NOT look inside — emit residual_call_*.
    /// Args are split by kind via `rewrite_call()` → `make_three_lists()`.
    /// `target` is the funcptr identity (mirrors `op.args[0]` upstream),
    /// kept separate from `descriptor` per jtransform.py:457.
    fn handle_residual_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.handle_residual_call_with_targets(
            graph, op, target, descriptor, args, result_ty, graph_name, None,
        )
    }

    /// RPython `jtransform.py:456-471` + `jtransform.py:547` sidecar:
    /// the `IndirectCallTargets(lst)` passed via `extraargs` rides along
    /// with the residual_call opcode.  This variant exposes the
    /// `indirect_targets` parameter so `handle_regular_indirect_call`
    /// can attach the candidate jitcode list without having to build the
    /// `OpKind::CallResidual` twice.
    fn handle_residual_call_with_targets(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        indirect_targets: Option<crate::model::IndirectCallTargets>,
    ) -> RewriteResult {
        let note_detail = match &indirect_targets {
            Some(t) => format!(
                "call {target} → residual indirect ({} candidates)",
                t.lst.len()
            ),
            None => format!("call {target} → residual"),
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: note_detail,
        });
        self.calls_classified += 1;
        // RPython jtransform.py:467: rewrite_call(op, 'residual_call', ...)
        let (args_i, args_r, args_f) = self.make_three_lists_from_vars(args);
        // RPython reads `op.result.concretetype` directly because rtyper
        // has typed every Variable. Pyre's front-end can leave a callee's
        // declared return as `ValueType::Unknown` (re-export shadowing,
        // unresolved cross-crate path); the rtyper's backward-inference
        // pass then assigns a definitive kind via the consumer-op
        // constraint. Honour that kind so the residual_call's
        // `result_kind` matches what every downstream consumer sees,
        // instead of falling back to `'r'` from the Unknown default.
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, target);
        // RPython jtransform.py:469-470: residual_call followed by -live-
        // if the call can raise or may call jitcodes.
        // jtransform.py:547: `handle_regular_indirect_call` passes
        // `may_call_jitcodes=True`, which forces a trailing `-live-`.
        let can_raise = descriptor.extra_info.check_can_raise(false) || indirect_targets.is_some();
        let mut ops = vec![
            funcptr_op,
            SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::CallResidual {
                    funcptr: CallFuncPtr::Value(funcptr),
                    descriptor,
                    args_i,
                    args_r,
                    args_f,
                    result_kind,
                    indirect_targets,
                },
            },
        ];
        if can_raise {
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
        }
        RewriteResult::Replace(ops)
    }

    /// RPython orthodox dispatch for `OpKind::IndirectCall` — line-by-line
    /// port of `jtransform.py:410-412 rewrite_op_indirect_call` +
    /// `jtransform.py:538-553 handle_regular_indirect_call`.
    ///
    /// `funcptr` is the runtime Variable already produced by the
    /// rtyper-equivalent layer (`translator/rtyper/rpbc.rs::lower_indirect_calls`),
    /// so this method does NOT synthesize anything — it emits exactly:
    ///
    /// ```text
    /// [Live, IntGuardValue(funcptr, 'i'),
    ///  CallResidual { funcptr: Value(funcptr),
    ///                 indirect_targets: Some(IndirectCallTargets { lst }), .. },
    ///  Live]    // trailing -live- because may_call_jitcodes=true
    /// ```
    ///
    /// `lst` is built from `candidates` via `cc.get_jitcode(p)` which
    /// returns the `Arc<JitCode>` shell from `CallControl::jitcodes`.
    fn lower_indirect_call_op(
        &mut self,
        op: &SpaceOperation,
        funcptr: &crate::flowspace::model::Variable,
        args: &[crate::flowspace::model::Variable],
        graphs: Option<&[crate::parse::CallPath]>,
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        let (args_i, args_r, args_f) = self.make_three_lists_from_vars(args);
        let resolved_result = self.resolve_call_result(op.result.as_ref(), result_ty);
        let result_kind = resolved_result.kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let non_void_args = resolve_non_void_arg_types_from_vars(args);
        let result_ir_type = resolved_result.ir_type;
        let cc_mut = self
            .callcontrol
            .as_mut()
            .expect("rewrite_op_indirect_call requires &mut CallControl");
        let descriptor = cc_mut.getcalldescr(
            op,
            non_void_args,
            result_ir_type,
            OopSpecIndex::None,
            None,
            &mut self.analysis_cache,
            None,
        );
        match cc_mut.guess_call_kind(op) {
            crate::call::CallKind::Regular => {
                let candidates = cc_mut
                    .graphs_from(op)
                    .expect("regular indirect call must have candidate graphs");
                let lst: Vec<crate::jitcode::JitCodeHandle> = candidates
                    .iter()
                    .map(|p| crate::jitcode::JitCodeHandle::new(cc_mut.get_jitcode(p)))
                    .collect();

                // jtransform.py:545-552 emit sequence:
                //   op0 = SpaceOperation('-live-', [], None)
                //   op1 = SpaceOperation('int_guard_value', [op.args[0]], None)
                //   op2 = self.handle_residual_call(op, [IndirectCallTargets(lst)], True)
                // then [op0, op1] + op2 (op2 is itself [residual_call, '-live-'])
                let mut ops = Vec::<SpaceOperation>::with_capacity(4);
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                });
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::GuardValue {
                        value: funcptr.clone(),
                        kind_char: 'i',
                    },
                });
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr.clone()),
                        descriptor,
                        args_i,
                        args_r,
                        args_f,
                        result_kind,
                        indirect_targets: Some(crate::model::IndirectCallTargets { lst }),
                    },
                });
                // jtransform.py:469-470: residual_call followed by -live-
                // because `may_call_jitcodes=True` for the regular-indirect path.
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                });

                self.calls_classified += 1;
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("indirect call → {} candidates", candidates.len()),
                });
                RewriteResult::Replace(ops)
            }
            crate::call::CallKind::Residual => {
                let can_raise = descriptor.extra_info.check_can_raise(false);
                self.calls_classified += 1;
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: match graphs {
                        Some(graphs) => {
                            format!("call <indirect> → residual family ({} impls)", graphs.len())
                        }
                        None => "call <indirect> → residual unknown family".to_string(),
                    },
                });
                let mut ops = vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr.clone()),
                        descriptor,
                        args_i,
                        args_r,
                        args_f,
                        result_kind,
                        indirect_targets: None,
                    },
                }];
                if can_raise {
                    ops.push(SpaceOperation {
                        result: None,
                        kind: OpKind::Live,
                    });
                }
                RewriteResult::Replace(ops)
            }
            crate::call::CallKind::Builtin | crate::call::CallKind::Recursive => {
                unreachable!("indirect calls cannot classify as builtin/recursive")
            }
        }
    }

    /// RPython: elidable call — pure function, result depends only on args.
    /// RPython jtransform.py:546-562.
    ///
    /// `target` is the funcptr identity per jtransform.py:457.
    #[allow(dead_code)]
    fn handle_elidable_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → elidable"),
        });
        self.calls_classified += 1;
        let (args_i, args_r, args_f) = self.make_three_lists_from_vars(args);
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, target);
        RewriteResult::Replace(vec![
            funcptr_op,
            SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::CallElidable {
                    funcptr: CallFuncPtr::Value(funcptr),
                    descriptor,
                    args_i,
                    args_r,
                    args_f,
                    result_kind,
                },
            },
        ])
    }

    /// RPython: may-force call — can trigger GC or force virtualizables.
    /// RPython jtransform.py:609-625.
    ///
    /// `target` is the funcptr identity per jtransform.py:457.
    #[allow(dead_code)]
    fn handle_may_force_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → may_force"),
        });
        self.calls_classified += 1;
        let (args_i, args_r, args_f) = self.make_three_lists_from_vars(args);
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, target);
        // RPython: call_may_force always followed by -live-
        RewriteResult::Replace(vec![
            funcptr_op,
            SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::CallMayForce {
                    funcptr: CallFuncPtr::Value(funcptr),
                    descriptor,
                    args_i,
                    args_r,
                    args_f,
                    result_kind,
                },
            },
            SpaceOperation {
                result: None,
                kind: OpKind::Live,
            },
        ])
    }

    /// Decide whether a `direct_call` is a transparent Rust prelude
    /// constructor that the frontend has already proved is not a real
    /// callable. Returns `true` iff every requirement holds, so the caller can
    /// safely emit `RewriteResult::Identity(args[0])`.
    ///
    /// Requirements:
    /// 1. Target is `CallTarget::SyntheticTransparentCtor`.
    /// 2. `args.len() == 1`.
    /// 3. The arg's resolved IR kind equals the result's IR kind. A
    ///    transparent wrapper preserves representation (`r → r`,
    ///    `i → i`, …); a kind mismatch (e.g. `i → r`) means the
    ///    Rust call is doing real boxing and must not be elided.
    fn is_synthetic_result_option_ctor(
        &self,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
    ) -> bool {
        if args.len() != 1 {
            return false;
        }
        let CallTarget::SyntheticTransparentCtor { name } = target else {
            return false;
        };
        if !matches!(name.as_str(), "Ok" | "Err" | "Some") {
            return false;
        }
        // arg/result IR-kind parity. `resolve_non_void_arg_types_from_vars`
        // returns `Type::Ref` when type_state is missing or the
        // value is unknown — that's the same default
        // `value_type_to_kind` applies for an `Unknown` result, so
        // the comparison stays sound under partial type info.
        let arg_types = resolve_non_void_arg_types_from_vars(args);
        let arg_ir = arg_types
            .first()
            .copied()
            .unwrap_or(majit_ir::value::Type::Ref);
        let result_ir = value_type_to_ir_type(result_ty);
        arg_ir == result_ir
    }
}

/// RPython: `getkind(concretetype)[0]` → 'i', 'r', 'f', or 'v'.
///
/// RPython's rtyper resolves all types before jtransform runs, so
/// getkind() never sees an unknown type. In our pipeline, Unknown
/// means the annotate/rtype pass couldn't resolve the type. We map
/// Unknown to 'r' (ref) since most Python-level values are GC refs.
/// RPython: `NON_VOID_ARGS = [x.concretetype for x in op.args[1:]
///                             if x.concretetype is not Void]`
/// (call.py:220-221).
///
/// Resolve the IR types of call arguments, skipping Void.
fn resolve_non_void_arg_types_from_vars(
    args: &[crate::flowspace::model::Variable],
) -> Vec<majit_ir::value::Type> {
    args.iter()
        .filter_map(|var| {
            let kind = match crate::model::FunctionGraph::concretetype_of(var) {
                crate::jit_codewriter::type_state::ConcreteType::Signed => 'i',
                crate::jit_codewriter::type_state::ConcreteType::GcRef => 'r',
                crate::jit_codewriter::type_state::ConcreteType::Float => 'f',
                crate::jit_codewriter::type_state::ConcreteType::Void => 'v',
                crate::jit_codewriter::type_state::ConcreteType::Unknown => 'r',
            };
            match kind {
                'v' => None, // RPython: skip Void args
                'i' => Some(majit_ir::value::Type::Int),
                'r' => Some(majit_ir::value::Type::Ref),
                'f' => Some(majit_ir::value::Type::Float),
                _ => Some(majit_ir::value::Type::Ref),
            }
        })
        .collect()
}

fn value_type_to_kind(ty: &ValueType) -> char {
    match ty {
        // RPython `getkind(BOOL_TYPE)` / `getkind(Unsigned)` both
        // return `'int'` (ll Bool / Unsigned share register class
        // with Signed at the codewriter register-kind layer).
        ValueType::Int | ValueType::Unsigned | ValueType::Bool | ValueType::State => 'i',
        ValueType::Ref | ValueType::Unknown => 'r',
        ValueType::Float => 'f',
        ValueType::Void => 'v',
    }
}

/// RPython `rfloat._rtype_template` calls `hop.inputargs(Float, Float)`,
/// which only coerces `Signed → Float`.  Pyre's float arms enter
/// when both operands fall into this domain.
fn is_float_rewrite_domain(kind: char) -> bool {
    matches!(kind, 'i' | 'f')
}

/// Rust assignment operators have no separate RPython flow op.  Include them
/// here so float-domain rewrites run before the generic assignment collapse.
fn canonical_float_arith_binop(op: &str) -> Option<&'static str> {
    match op {
        "add" | "add_assign" => Some("add"),
        "sub" | "sub_assign" => Some("sub"),
        "mul" | "mul_assign" => Some("mul"),
        "div" | "div_assign" => Some("div"),
        _ => None,
    }
}

fn canonical_float_mod_binop(op: &str) -> Option<&'static str> {
    match op {
        "mod" | "mod_assign" => Some("mod"),
        _ => None,
    }
}

fn canonical_assign_binop(op: &str) -> Option<&'static str> {
    match op {
        "add_assign" => Some("add"),
        "sub_assign" => Some("sub"),
        "mul_assign" => Some("mul"),
        "div_assign" => Some("div"),
        "mod_assign" => Some("mod"),
        "bitand_assign" => Some("and"),
        "bitor_assign" => Some("or"),
        "bitxor_assign" => Some("xor"),
        "rshift_assign" => Some("rshift"),
        "lshift_assign" => Some("lshift"),
        _ => None,
    }
}

/// Convert codewriter ValueType to IR Type.
///
/// RPython: `x.concretetype` → lltype mapping.
/// Used by getcalldescr to build NON_VOID_ARGS and RESULT types.
fn value_type_to_ir_type(ty: &ValueType) -> majit_ir::value::Type {
    match ty {
        ValueType::Int | ValueType::Unsigned | ValueType::Bool | ValueType::State => {
            majit_ir::value::Type::Int
        }
        ValueType::Ref | ValueType::Unknown => majit_ir::value::Type::Ref,
        ValueType::Float => majit_ir::value::Type::Float,
        ValueType::Void => majit_ir::value::Type::Void,
    }
}

/// Convert a CallTarget to a CallPath for jitcode lookup.
fn target_to_call_path(target: &CallTarget) -> crate::parse::CallPath {
    match target {
        CallTarget::FunctionPath { segments } => {
            crate::parse::CallPath::from_segments(segments.iter().map(String::as_str))
        }
        CallTarget::Method { name, .. } => crate::parse::CallPath::from_segments([name.as_str()]),
        CallTarget::SyntheticTransparentCtor { name } => {
            crate::parse::CallPath::from_segments([name.as_str()])
        }
        // RPython: an indirect_call has no single jitcode-lookup path —
        // the family is handled via the op-based `graphs_from(op)` +
        // `IndirectCallTargets` sidecar.  This fallback returns a stub
        // path only reached by callers that don't distinguish; the real
        // consumer (`handle_regular_indirect_call`) uses the family path
        // directly.
        CallTarget::Indirect {
            trait_root,
            method_name,
        } => crate::parse::CallPath::from_segments([trait_root.as_str(), method_name.as_str()]),
        CallTarget::UnsupportedExpr => crate::parse::CallPath::from_segments(["<unsupported>"]),
    }
}

/// RPython `jtransform.py:264-275 _renamings_get(self, v)` — follow the
/// alias chain to the canonical Variable.  Variable-keyed: RPython
/// `self._renamings` is a `{Variable: Variable}` dict
/// (`jtransform.py:71`).
fn resolve_alias(
    value: &crate::flowspace::model::Variable,
    aliases: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
) -> crate::flowspace::model::Variable {
    let mut cur = value.clone();
    while let Some(next) = aliases.get(&cur).cloned() {
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

fn remap_value(
    value: &crate::flowspace::model::Variable,
    aliases: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
) -> crate::flowspace::model::Variable {
    resolve_alias(value, aliases)
}

fn remap_call_funcptr(
    funcptr: &CallFuncPtr,
    aliases: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
) -> CallFuncPtr {
    match funcptr {
        CallFuncPtr::Target(target) => CallFuncPtr::Target(target.clone()),
        CallFuncPtr::Value(var) => CallFuncPtr::Value(remap_value(var, aliases)),
    }
}

fn remap_op(
    op: &SpaceOperation,
    aliases: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
) -> SpaceOperation {
    let kind = match &op.kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstFloat(_)
        | OpKind::CurrentTraceLength
        | OpKind::Live
        | OpKind::LoopHeader { .. }
        | OpKind::Abort { .. } => op.kind.clone(),
        OpKind::GuardValue { value, kind_char } => OpKind::GuardValue {
            value: remap_value(value, aliases),
            kind_char: *kind_char,
        },
        OpKind::VtableMethodPtr {
            receiver,
            trait_root,
            method_name,
        } => OpKind::VtableMethodPtr {
            receiver: remap_value(receiver, aliases),
            trait_root: trait_root.clone(),
            method_name: method_name.clone(),
        },
        OpKind::VableForce { base } => OpKind::VableForce {
            base: remap_value(base, aliases),
        },
        OpKind::JitMergePoint {
            jitdriver_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::JitMergePoint {
                jitdriver_index: *jitdriver_index,
                greens_i: greens_i.iter().map(remap_var).collect(),
                greens_r: greens_r.iter().map(remap_var).collect(),
                greens_f: greens_f.iter().map(remap_var).collect(),
                reds_i: reds_i.iter().map(remap_var).collect(),
                reds_r: reds_r.iter().map(remap_var).collect(),
                reds_f: reds_f.iter().map(remap_var).collect(),
            }
        }
        OpKind::IndirectCall {
            funcptr,
            args,
            graphs,
            result_ty,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::IndirectCall {
                funcptr: remap_var(funcptr),
                args: args.iter().map(remap_var).collect(),
                graphs: graphs.clone(),
                result_ty: result_ty.clone(),
            }
        }
        OpKind::RecordQuasiImmutField {
            base,
            field,
            mutate_field,
        } => OpKind::RecordQuasiImmutField {
            base: remap_value(base, aliases),
            field: field.clone(),
            mutate_field: mutate_field.clone(),
        },
        OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        } => OpKind::FieldRead {
            base: remap_value(base, aliases),
            field: field.clone(),
            ty: ty.clone(),
            pure: *pure,
        },
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } => OpKind::FieldWrite {
            base: remap_value(base, aliases),
            field: field.clone(),
            value: remap_value(value, aliases),
            ty: ty.clone(),
        },
        OpKind::ArrayRead {
            base,
            index,
            item_ty,
            array_type_id,
            nolength,
        } => OpKind::ArrayRead {
            base: remap_value(base, aliases),
            index: remap_value(index, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
            array_type_id,
            nolength,
        } => OpKind::ArrayWrite {
            base: remap_value(base, aliases),
            index: remap_value(index, aliases),
            value: remap_value(value, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
        },
        OpKind::InteriorFieldRead {
            base,
            index,
            field,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldRead {
            base: remap_value(base, aliases),
            index: remap_value(index, aliases),
            field: field.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::InteriorFieldWrite {
            base,
            index,
            field,
            value,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldWrite {
            base: remap_value(base, aliases),
            index: remap_value(index, aliases),
            field: field.clone(),
            value: remap_value(value, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::Call {
            target,
            args,
            result_ty,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::Call {
                target: target.clone(),
                args: args.iter().map(remap_var).collect(),
                result_ty: result_ty.clone(),
            }
        }
        OpKind::GuardTrue { cond } => OpKind::GuardTrue {
            cond: remap_value(cond, aliases),
        },
        OpKind::GuardFalse { cond } => OpKind::GuardFalse {
            cond: remap_value(cond, aliases),
        },
        OpKind::VableFieldRead {
            base,
            field_index,
            ty,
        } => OpKind::VableFieldRead {
            base: remap_value(base, aliases),
            field_index: *field_index,
            ty: ty.clone(),
        },
        OpKind::VableFieldWrite {
            base,
            field_index,
            value,
            ty,
        } => OpKind::VableFieldWrite {
            base: remap_value(base, aliases),
            field_index: *field_index,
            value: remap_value(value, aliases),
            ty: ty.clone(),
        },
        OpKind::VableArrayRead {
            base,
            array_index,
            elem_index,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayRead {
            base: remap_value(base, aliases),
            array_index: *array_index,
            elem_index: remap_value(elem_index, aliases),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::VableArrayWrite {
            base,
            array_index,
            elem_index,
            value,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayWrite {
            base: remap_value(base, aliases),
            array_index: *array_index,
            elem_index: remap_value(elem_index, aliases),
            value: remap_value(value, aliases),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::BinOp {
            op,
            lhs,
            rhs,
            result_ty,
        } => OpKind::BinOp {
            op: op.clone(),
            lhs: remap_value(lhs, aliases),
            rhs: remap_value(rhs, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } => OpKind::UnaryOp {
            op: op.clone(),
            operand: remap_value(operand, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::JitDebug { args } => OpKind::JitDebug {
            args: args.iter().map(|var| remap_value(var, aliases)).collect(),
        },
        OpKind::RecordKnownResult {
            result_value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::RecordKnownResult {
                result_value: remap_var(result_value),
                funcptr: funcptr.clone(),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::AssertGreen { value, kind_char } => OpKind::AssertGreen {
            value: remap_value(value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsConstant { value, kind_char } => OpKind::IsConstant {
            value: remap_value(value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsVirtual { value, kind_char } => OpKind::IsVirtual {
            value: remap_value(value, aliases),
            kind_char: *kind_char,
        },
        OpKind::CallElidable {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::CallElidable {
                funcptr: remap_call_funcptr(funcptr, aliases),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::CallResidual {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            indirect_targets,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::CallResidual {
                funcptr: remap_call_funcptr(funcptr, aliases),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                indirect_targets: indirect_targets.clone(),
                result_kind: *result_kind,
            }
        }
        OpKind::CallMayForce {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::CallMayForce {
                funcptr: remap_call_funcptr(funcptr, aliases),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::InlineCall {
            jitcode,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::InlineCall {
                jitcode: jitcode.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::RecursiveCall {
            jd_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::RecursiveCall {
                jd_index: *jd_index,
                greens_i: greens_i.iter().map(remap_var).collect(),
                greens_r: greens_r.iter().map(remap_var).collect(),
                greens_f: greens_f.iter().map(remap_var).collect(),
                reds_i: reds_i.iter().map(remap_var).collect(),
                reds_r: reds_r.iter().map(remap_var).collect(),
                reds_f: reds_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::ConditionalCall {
            condition,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::ConditionalCall {
                condition: remap_var(condition),
                funcptr: funcptr.clone(),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
            }
        }
        OpKind::ConditionalCallValue {
            value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::ConditionalCallValue {
                value: remap_var(value),
                funcptr: funcptr.clone(),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
    };
    SpaceOperation {
        result: op.result.clone(),
        kind,
    }
}

fn classify_vable_hint(target: &CallTarget) -> Option<crate::hints::VirtualizableHintKind> {
    target
        .path_segments()
        .and_then(|segments| crate::hints::classify_virtualizable_hint_segments(segments))
}

/// Match a `CallEffectOverride` pattern against a call target.
///
/// The match is loose only in the asymmetric receiver-root direction:
/// when the pattern leaves `receiver_root` as `None`, any target
/// receiver matches; otherwise both sides must agree, either by
/// exact source-syntax equality or — when the target carries a
/// `classdef_hint` and `callcontrol` is available — through the
/// classdef-keyed side-table that resolves the concrete receiver
/// per `bookkeeper.py:431-442 getmethoddesc`. The pre-C6
/// `is_generic_receiver` wildcard branch is retired; there is no
/// production path that produces `callcontrol=None` at this site
/// (`Transformer::with_callcontrol` is set by every production
/// entry — `codewriter.rs:382` + `legacy_pipeline.rs:92`).
///
/// PRE-EXISTING-ADAPTATION: the override table itself is pyre-only
/// (no upstream basis). The producer (`codewriter.rs::
/// stamp_classdef_hints_on_graph`) stores the receiver's `ClassDef.name`
/// (RPython `classdef.py:36 self.name`, fully qualified `module.Class`)
/// canonicalised through `register_classdef_impl_type` to `::`
/// separators so distinct classdefs sharing a leaf identifier don't
/// collide. Override patterns such as `PYFRAME_CALL_OWNER_ROOT =
/// "PyFrame"` are written bare in `pyre-jit-trace/src/call_spec.rs`, so
/// the classdef-keyed branch also accepts a leaf-suffix match
/// (`crate::parse::canonical_leaf`) against the qualified side-table
/// value — accepting both `module.Class` and `module::Class` forms.
fn call_target_matches_loose(
    pattern: &CallTarget,
    target: &CallTarget,
    callcontrol: Option<&crate::call::CallControl>,
) -> bool {
    match (pattern, target) {
        (
            CallTarget::Method {
                name: pn,
                receiver_root: pr,
                ..
            },
            CallTarget::Method {
                name: tn,
                receiver_root: tr,
                classdef_hint,
            },
        ) => {
            if pn != tn {
                return false;
            }
            match (pr.as_deref(), tr.as_deref()) {
                (Some(p), Some(t)) => {
                    if p == t {
                        return true;
                    }
                    if let (Some(key), Some(cc)) = (classdef_hint, callcontrol) {
                        if let Some(side) = cc.classdef_impl_type_for_pattern_match(*key) {
                            if side == p {
                                return true;
                            }
                            // Side-table value is the canonicalised
                            // qualified `ClassDef.name` (`::`-joined per
                            // `register_classdef_impl_type`); pattern
                            // receivers in pyre-jit-trace's
                            // PYFRAME_CALL_EFFECTS are bare leaf
                            // spellings. Compare on the leaf after the
                            // final separator (`canonical_leaf` accepts
                            // both `::` and `.`) so the override matcher
                            // fires for bare patterns regardless of how
                            // the producer's classdef was named.
                            if crate::parse::canonical_leaf(side) == p {
                                return true;
                            }
                        }
                    }
                    false
                }
                _ => true,
            }
        }
        _ => pattern == target,
    }
}

/// Map a user-level oopspec string (from `@oopspec(...)`) to an `OopSpecIndex`.
///
/// rlib/jit.py:250 — `@oopspec(spec)` stores a spec string on the function.
/// jtransform.py:1731-1755 `__handle_jit_call` patterns the spec name.
///
/// For the JIT-specific `jit.*` specs, RPython emits SpaceOperations with
/// distinct names (e.g. `jit_debug`, `int_isconstant`); for list/dict/str
/// specs RPython uses dedicated OS_* indices. This helper currently maps
/// the cases that have a direct OopSpecIndex equivalent.
fn map_user_oopspec_to_index(spec: &str) -> majit_ir::descr::OopSpecIndex {
    use majit_ir::descr::OopSpecIndex;
    // Normalize: `jit.isconstant(value)` → `jit.isconstant`
    let base = spec.split('(').next().unwrap_or(spec).trim();
    match base {
        // All jit.* oopspecs are intercepted by _handle_jit_call() before
        // reaching this function. Remaining oopspecs map to OS_* indices.
        "virtual_ref" | "virtual_ref_finish" => OopSpecIndex::JitForceVirtualizable,
        // jtransform.py:507-509: oopspec_name.endswith('dict.lookup')
        _ if base.ends_with("dict.lookup") => OopSpecIndex::DictLookup,
        _ => OopSpecIndex::None,
    }
}

/// Classify a call's side-effect level.
///
/// RPython equivalent: jtransform.py effect classification
/// (EF_ELIDABLE, EF_FORCES_VIRTUAL, etc.)
fn classify_call(
    target: &CallTarget,
    overrides: &[CallEffectOverride],
    callcontrol: Option<&crate::call::CallControl>,
) -> Option<(CallDescriptor, CallEffectKind)> {
    fn classify_effect_info(info: &majit_ir::descr::EffectInfo) -> CallEffectKind {
        if info.check_forces_virtual_or_virtualizable() {
            CallEffectKind::MayForce
        } else if info.check_is_elidable() {
            CallEffectKind::Elidable
        } else {
            CallEffectKind::Residual
        }
    }

    if let Some(descriptor) = overrides
        .iter()
        .find(|override_| call_target_matches_loose(&override_.target, target, callcontrol))
        .map(|override_| override_.descriptor.clone())
    {
        let effect = classify_effect_info(&descriptor.get_extra_info());
        return Some((descriptor, effect));
    }
    let descriptor = crate::call::describe_call(target)?;
    let effect = classify_effect_info(&descriptor.get_extra_info());
    Some((descriptor, effect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_codewriter::type_state::ConcreteType;
    use crate::model::{CallFuncPtr, CallTarget, FunctionGraph, LinkArg, OpKind, ValueType};

    #[test]
    fn rewrite_graph_canonicalizes_frontend_bitops() {
        let mut graph = FunctionGraph::new("bitops");
        let lhs_var = graph.alloc_value_var();
        let rhs_var = graph.alloc_value_var();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "bitxor".to_string(),
                    lhs: lhs_var,
                    rhs: rhs_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var));

        let transformed = rewrite_graph(&graph, &GraphTransformConfig::default());
        match &transformed.graph.block(graph.startblock).operations[0].kind {
            OpKind::BinOp { op, .. } => assert_eq!(op, "xor"),
            other => panic!("expected canonical BinOp, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_removes_same_as_and_remaps_return() {
        let mut graph = FunctionGraph::new("same_as_identity");
        let input_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let alias_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: input_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(alias_var));

        let transformed = rewrite_graph(&graph, &GraphTransformConfig::default());
        let block = transformed.graph.block(graph.startblock);

        assert!(
            !block.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, .. } if op == "same_as"
            )),
            "same_as must be eliminated before assembly: {:?}",
            block.operations
        );
        assert_eq!(block.exits.len(), 1);
        assert_eq!(block.exits[0].target, transformed.graph.returnblock);
        assert_eq!(block.exits[0].args, vec![LinkArg::Value(input_var)]);
    }

    #[test]
    fn rewrite_graph_remaps_guard_value_after_same_as_identity() {
        let mut graph = FunctionGraph::new("same_as_then_guard");
        let input_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let alias_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: input_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::GuardValue {
                value: alias_var,
                kind_char: 'i',
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        let input_slot = graph.slot_of(&input_var).expect("input must be registered");

        let transformed = rewrite_graph(&graph, &GraphTransformConfig::default());
        let guard = transformed
            .graph
            .block(transformed.graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::GuardValue { value, .. } => Some(value),
                _ => None,
            })
            .expect("GuardValue must survive the transform");

        assert_eq!(
            transformed.graph.slot_of(guard),
            Some(input_slot),
            "GuardValue.value must follow PyPy _do_renaming through same_as"
        );
    }

    #[test]
    fn rewrite_graph_remaps_vtable_method_receiver_after_same_as_identity() {
        let mut graph = FunctionGraph::new("same_as_then_vtable_method");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "receiver".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let alias_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: receiver_var.clone(),
                    result_ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::VtableMethodPtr {
                receiver: alias_var,
                trait_root: "Handler".into(),
                method_name: "run".into(),
            },
            true,
        );
        graph.set_return(graph.startblock, None);
        let receiver_slot = graph
            .slot_of(&receiver_var)
            .expect("receiver must be registered");

        let transformed = rewrite_graph(&graph, &GraphTransformConfig::default());
        let vtable_receiver = transformed
            .graph
            .block(transformed.graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::VtableMethodPtr { receiver, .. } => Some(receiver),
                _ => None,
            })
            .expect("VtableMethodPtr must survive the transform");

        assert_eq!(
            transformed.graph.slot_of(vtable_receiver),
            Some(receiver_slot),
            "VtableMethodPtr.receiver must follow PyPy _do_renaming through same_as"
        );
    }

    #[test]
    fn rewrite_graph_coerces_mixed_float_add() {
        let mut graph = FunctionGraph::new("mixed_float_add");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Float);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Float);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config).transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, lhs_var);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.clone().unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        let cast_result_var = cast_result;
        match &ops[3].kind {
            OpKind::BinOp {
                op,
                lhs: rewritten_lhs,
                rhs: rewritten_rhs,
                result_ty,
            } => {
                assert_eq!(op, "float_add");
                assert_eq!(*rewritten_lhs, cast_result_var);
                assert_eq!(*rewritten_rhs, rhs_var);
                assert_eq!(*result_ty, ValueType::Float);
            }
            other => panic!("expected float_add, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_coerces_mixed_float_comparison() {
        let mut graph = FunctionGraph::new("mixed_float_eq");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "eq".into(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Float);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config).transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, rhs_var);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.clone().unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        let cast_result_var = cast_result;
        match &ops[3].kind {
            OpKind::BinOp {
                op,
                lhs: rewritten_lhs,
                rhs: rewritten_rhs,
                result_ty,
            } => {
                assert_eq!(op, "float_eq");
                assert_eq!(*rewritten_lhs, lhs_var);
                assert_eq!(*rewritten_rhs, cast_result_var);
                assert_eq!(*result_ty, ValueType::Int);
            }
            other => panic!("expected float_eq, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_lowers_float_mod_to_ll_math_fmod_residual_call() {
        let mut graph = FunctionGraph::new("float_mod");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "mod".into(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Float);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Float);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config).transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 6, "Input + Input + cast + fnptr + call + Live");
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, lhs_var);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.clone().unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        let expected_fnaddr =
            crate::call::symbolic_fnaddr_for_target(&CallTarget::function_path(["ll_math_fmod"]));
        assert!(matches!(&ops[3].kind, OpKind::ConstInt(fnaddr) if *fnaddr == expected_fnaddr));
        match &ops[4].kind {
            OpKind::CallResidual {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                indirect_targets,
            } => {
                assert!(matches!(funcptr, CallFuncPtr::Value(_)));
                assert_eq!(
                    descriptor.extra_info.extraeffect,
                    ExtraEffect::ElidableCanRaise
                );
                assert!(args_i.is_empty());
                assert!(args_r.is_empty());
                let cast_result_var = cast_result;
                assert_eq!(args_f, &vec![cast_result_var, rhs_var]);
                assert_eq!(*result_kind, 'f');
                assert!(indirect_targets.is_none());
            }
            other => panic!("expected CallResidual, got {other:?}"),
        }
        assert!(matches!(ops[5].kind, OpKind::Live));
    }

    #[test]
    fn rewrite_graph_tags_vable_fields() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let base = graph.slot_of(&base_var).expect("base registered");
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig {
            vable_fields: vec![VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };
        let result = rewrite_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 1);
        // Should be rewritten to VableFieldRead
        let rewritten_op = &result.graph.block(graph.startblock).operations[0];
        let OpKind::VableFieldRead {
            base: rewritten_base,
            field_index,
            ..
        } = &rewritten_op.kind
        else {
            panic!("expected VableFieldRead, got {:?}", rewritten_op.kind);
        };
        assert_eq!(*field_index, 0);
        assert_eq!(result.graph.slot_of(rewritten_base), Some(base));
    }

    #[test]
    fn rewrite_graph_tags_vable_arrays_with_explicit_base() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let index_var = graph.alloc_value_var();
        let base = graph.slot_of(&base_var).expect("base registered");
        let array_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var,
                    field: crate::model::FieldDescriptor::new(
                        "locals_stack_w",
                        Some("Frame".into()),
                    ),
                    ty: ValueType::Ref,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayRead {
                base: array_var,
                index: index_var,
                item_ty: ValueType::Int,
                array_type_id: None,
                nolength: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig {
            vable_arrays: vec![VirtualizableFieldDescriptor::new_with_arraydescr(
                "locals_stack_w",
                Some("Frame".into()),
                0,
                8,
                true,
            )],
            ..Default::default()
        };
        let result = rewrite_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 1);
        let rewritten_op = &result.graph.block(graph.startblock).operations[1];
        let OpKind::VableArrayRead {
            base: rewritten_base,
            array_index,
            ..
        } = &rewritten_op.kind
        else {
            panic!(
                "expected VableArrayRead with explicit base, got {:?}",
                rewritten_op.kind
            );
        };
        assert_eq!(*array_index, 0);
        assert_eq!(result.graph.slot_of(rewritten_base), Some(base));
    }

    #[test]
    fn rewrite_graph_requires_matching_field_owner_root() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("OtherFrame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig {
            vable_fields: vec![VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };
        let result = rewrite_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 0);
        assert!(matches!(
            result.graph.block(graph.startblock).operations[0].kind,
            OpKind::FieldRead { .. }
        ));
    }

    #[test]
    fn rewrite_graph_types_fieldwrite_from_value_kind() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let value_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: base_var,
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                value: value_var.clone(),
                ty: ValueType::Unknown,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        FunctionGraph::set_concretetype_of_inline(
            &value_var,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::FieldWrite { ty, .. } => assert_eq!(*ty, ValueType::Int),
            other => panic!("expected FieldWrite, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_types_arraywrite_from_value_kind() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let index_var = graph.alloc_value_var();
        let value_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayWrite {
                base: base_var,
                index: index_var,
                value: value_var.clone(),
                item_ty: ValueType::Unknown,
                array_type_id: None,
                nolength: false,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        FunctionGraph::set_concretetype_of_inline(
            &value_var,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::ArrayWrite { item_ty, .. } => assert_eq!(*item_ty, ValueType::Int),
            other => panic!("expected ArrayWrite, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_types_fieldread_from_result_kind() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var,
                    field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Unknown,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &result_var,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::FieldRead { ty, .. } => assert_eq!(*ty, ValueType::Int),
            other => panic!("expected FieldRead, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_types_arrayread_from_result_kind() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let index_var = graph.alloc_value_var();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: base_var,
                    index: index_var,
                    item_ty: ValueType::Unknown,
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &result_var,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::ArrayRead { item_ty, .. } => assert_eq!(*item_ty, ValueType::Int),
            other => panic!("expected ArrayRead, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_classifies_calls() {
        let mut graph = FunctionGraph::new("test");
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::method("call_callable", Some("PyFrame".into())),
                args: vec![],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let result = rewrite_graph(
            &graph,
            &crate::test_support::pyre_pipeline_config().transform,
        );
        assert_eq!(result.calls_classified, 1);
        assert!(matches!(
            result.graph.block(graph.startblock).operations[0].kind,
            OpKind::ConstInt(_)
        ));
        assert!(matches!(
            result.graph.block(graph.startblock).operations[1].kind,
            OpKind::CallResidual { .. }
        ));
    }

    #[test]
    fn residual_call_unknown_result_uses_resolved_type_for_opcode_and_descr() {
        let mut graph = FunctionGraph::new("outer");
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::function_path(["unknown_external_int"]),
                    args: vec![],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);

        let mut cc = crate::call::CallControl::new();
        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let (result_kind, descriptor) = transformed
            .graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    result_kind,
                    descriptor,
                    ..
                } => Some((*result_kind, descriptor)),
                _ => None,
            })
            .expect("residual call must be emitted");

        assert_eq!(result_kind, 'i');
        assert_eq!(descriptor.result_ir_type(), majit_ir::Type::Int);
    }

    #[test]
    fn int_mod_unknown_ast_result_rewrites_when_type_state_proves_signed() {
        let mut graph = FunctionGraph::new("int_mod_body");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "mod".to_string(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let ops = &transformed.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "mod")),
            "bare int_mod must not survive jtransform: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    result_kind,
                    args_i,
                    ..
                } => Some((descriptor, *result_kind, args_i)),
                _ => None,
            })
            .expect("mod must rewrite to residual helper call");
        assert_eq!(residual.1, 'i');
        assert_eq!(residual.2, &vec![lhs_var, rhs_var]);
        assert_eq!(residual.0.result_ir_type(), majit_ir::Type::Int);
        assert_eq!(residual.0.get_extra_info().oopspecindex, OopSpecIndex::None);
    }

    #[test]
    fn mod_assign_rewrites_directly_to_int_mod_residual() {
        // Rust low-level → RPython low-level: pyre constructs
        // `mod_assign` from Rust's `%=` operator on i64, which has
        // C-truncating remainder semantics.  That maps to RPython's
        // explicit `llop.int_mod` route (`support.py:266-271
        // _ll_2_int_mod`), not to Python-level `%=` / `rtype_mod`
        // (`rint.py:260-262`, which calls `py_mod`).  The assign arm
        // must emit the `_ll_2_int_mod` residual directly because
        // `transform_op`'s Replace output is not re-traversed within
        // the same pass — leaving a bare `mod` here would leak past
        // the jtransform rewrite gate.
        let mut graph = FunctionGraph::new("mod_assign_body");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "mod_assign".to_string(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let ops = &transformed.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "mod_assign" || op == "mod"
            )),
            "mod_assign must not survive as a bare BinOp: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    result_kind,
                    args_i,
                    ..
                } => Some((descriptor, *result_kind, args_i)),
                _ => None,
            })
            .expect("mod_assign must rewrite to residual _ll_2_int_mod call");
        assert_eq!(residual.1, 'i');
        assert_eq!(residual.2, &vec![lhs_var, rhs_var]);
        assert_eq!(residual.0.result_ir_type(), majit_ir::Type::Int);
    }

    #[test]
    fn div_assign_rewrites_directly_to_int_floordiv_residual() {
        // Rust low-level → RPython low-level: pyre constructs
        // `div_assign` from Rust's `/=` operator on i64, which has
        // C-truncating division semantics.  That maps to RPython's
        // explicit `llop.int_floordiv` route (`support.py:255-264
        // _ll_2_int_floordiv`), not to Python-level `/=` /
        // `rtype_inplace_div` (`rint.py:253-255`, which aliases to
        // `rtype_floordiv` and calls `py_div`).  The assign arm
        // aliases `"div"` to `floordiv` and emits the residual
        // directly so no bare `div` / `floordiv` survives this pass.
        let mut graph = FunctionGraph::new("div_assign_body");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "div_assign".to_string(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let ops = &transformed.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. }
                    if op == "div_assign" || op == "div" || op == "floordiv"
            )),
            "div_assign must not survive as a bare BinOp: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    result_kind,
                    args_i,
                    ..
                } => Some((descriptor, *result_kind, args_i)),
                _ => None,
            })
            .expect("div_assign must rewrite to residual _ll_2_int_floordiv call");
        assert_eq!(residual.1, 'i');
        assert_eq!(residual.2, &vec![lhs_var, rhs_var]);
        assert_eq!(residual.0.result_ir_type(), majit_ir::Type::Int);
    }

    #[test]
    fn plain_int_div_rewrites_directly_to_int_floordiv_residual() {
        // `rint.py:253-255 rtype_div = rtype_floordiv`: a plain
        // `BinOp { op:"div" }` over int operands (Rust `a / b` on
        // i64s) routes through the same `_ll_2_int_floordiv`
        // residual as `floordiv`.  RPython has no `int_div` op; the
        // rtyper aliases `div` to `floordiv` for integer reprs.
        let mut graph = FunctionGraph::new("int_div_body");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "div".to_string(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let ops = &transformed.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "div" || op == "floordiv"
            )),
            "plain int div must not survive as a bare BinOp: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    result_kind,
                    args_i,
                    ..
                } => Some((descriptor, *result_kind, args_i)),
                _ => None,
            })
            .expect("int div must rewrite to residual _ll_2_int_floordiv call");
        assert_eq!(residual.1, 'i');
        assert_eq!(residual.2, &vec![lhs_var, rhs_var]);
        assert_eq!(residual.0.result_ir_type(), majit_ir::Type::Int);
    }

    #[test]
    fn residual_direct_call_materializes_funcptr_const() {
        let mut cc = crate::call::CallControl::new();
        let target = CallTarget::function_path(["custom_reader"]);
        let mut graph = FunctionGraph::new("test");
        let arg_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "arg".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: target.clone(),
                args: vec![arg_var],
                result_ty: ValueType::Ref,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);
        let ops = &result.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 4, "Input + ConstInt + CallResidual + Live");
        match &ops[1].kind {
            OpKind::ConstInt(fnaddr) => {
                assert_eq!(*fnaddr, cc.fnaddr_for_target(&target));
            }
            other => panic!("expected materialized funcptr ConstInt, got {other:?}"),
        }
        match &ops[2].kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                ..
            } => {}
            other => panic!("expected CallResidual with runtime funcptr, got {other:?}"),
        }
        assert!(matches!(ops[3].kind, OpKind::Live));
    }

    #[test]
    fn rewrite_graph_uses_explicit_call_effect_overrides() {
        // RPython: residual calls always produce residual_call_*, regardless
        // of effect. The effect is only in the calldescr (descriptor).
        let mut graph = FunctionGraph::new("test");
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["custom_reader"]),
                args: vec![],
                result_ty: ValueType::Ref,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let result = rewrite_graph(
            &graph,
            &GraphTransformConfig {
                call_effects: vec![CallEffectOverride::new(
                    CallTarget::function_path(["custom_reader"]),
                    CallEffectKind::MayForce,
                )],
                ..Default::default()
            },
        );
        let ops = &result.graph.block(graph.startblock).operations;
        // RPython: always residual_call_*, effect in descriptor only.
        assert!(matches!(ops[0].kind, OpKind::ConstInt(_)));
        assert!(matches!(ops[1].kind, OpKind::CallResidual { .. }));
        // Verify the effect is correctly carried in the descriptor.
        if let OpKind::CallResidual { descriptor, .. } = &ops[1].kind {
            assert!(
                descriptor
                    .extra_info
                    .check_forces_virtual_or_virtualizable()
            );
        } else {
            panic!("expected CallResidual");
        }
    }

    #[test]
    fn rewrite_graph_reports_unknowns() {
        let mut graph = FunctionGraph::new("demo");
        graph.push_op_var(
            graph.startblock,
            OpKind::Abort {
                kind: crate::model::UnknownKind::UnsupportedExpr {
                    variant: crate::model::UnsupportedExprKind::OtherExpr,
                },
            },
            false,
        );
        graph.set_raise(graph.startblock, "not implemented");
        let result = rewrite_graph(&graph, &GraphTransformConfig::default());
        assert_eq!(result.notes.len(), 2); // unknown + abort
    }

    #[test]
    fn rewrite_graph_consumes_identity_virtualizable_hints() {
        let mut graph = FunctionGraph::new("demo");
        let frame_var = graph.alloc_value_var();
        let hinted_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_access_directly"]),
                args: vec![frame_var],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(hinted_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: hinted_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let result = rewrite_graph(
            &graph,
            &GraphTransformConfig {
                vable_fields: vec![VirtualizableFieldDescriptor::new(
                    "next_instr",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            },
        );

        assert_eq!(result.graph.block(graph.startblock).operations.len(), 1);
        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::VableFieldRead { field_index, .. } => assert_eq!(*field_index, 0),
            other => panic!("expected VableFieldRead after hint suppression, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_rewrites_hint_force_virtualizable() {
        let mut graph = FunctionGraph::new("demo");
        let frame_var = graph.alloc_value_var();
        let forced_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_force_virtualizable"]),
                args: vec![frame_var.clone()],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(forced_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: forced_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let result = rewrite_graph(
            &graph,
            &GraphTransformConfig {
                vable_fields: vec![VirtualizableFieldDescriptor::new(
                    "next_instr",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            },
        );

        let ops = &result.graph.block(graph.startblock).operations;
        let OpKind::VableForce { base } = &ops[0].kind else {
            panic!("expected ops[0] to be VableForce, got {:?}", ops[0].kind);
        };
        assert_eq!(base, &frame_var);
        assert!(matches!(
            ops[1].kind,
            OpKind::VableFieldRead { field_index: 0, .. }
        ));
    }

    /// `rpython/jit/codewriter/jtransform.py:608-614 rewrite_op_hint`
    /// `promote=True` branch: emits `[-live-, <kind>_guard_value(x),
    /// None]` where the `None` sentinel aliases the result back to the
    /// input arg.  In pyre's `RewriteResult` model the alias is applied
    /// by `optimize_block` from `self.aliases.insert(result, base)` and
    /// the two emitted ops show up at the call site as
    /// `[OpKind::Live, OpKind::GuardValue { kind_char }]`.
    #[test]
    fn rewrite_graph_rewrites_hint_promote() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        let promoted_var = graph.alloc_value_var();
        let consumed_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        // `hint_promote(v)` — mirrors `rlib/jit.py:101 promote(x)` after
        // lowering to the operator-level helper name.
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote"]),
                args: vec![v_var.clone()],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(promoted_var.clone());
        // A downstream op that names the promote result so we can
        // observe that `optimize_block` aliased it back to `v`.
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: promoted_var,
                field: crate::model::FieldDescriptor::new("payload", Some("Box".into())),
                ty: ValueType::Int,
                pure: false,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(consumed_var);
        graph.set_return(graph.startblock, None);

        let result = rewrite_graph(&graph, &GraphTransformConfig::default());
        let ops = &result.graph.block(graph.startblock).operations;
        // Expected post-rewrite shape: [Live, GuardValue, FieldRead].
        assert_eq!(ops.len(), 3, "got {ops:?}");
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue {
                value, kind_char, ..
            } => {
                assert_eq!(value, &v_var, "guard target must remain the input arg");
                assert_eq!(*kind_char, 'r', "default kind without type-state");
            }
            other => panic!("expected GuardValue, got {other:?}"),
        }
        // `None` sentinel parity: the downstream FieldRead, which named
        // the `promoted` result, must have its base resolved back to `v`.
        match &ops[2].kind {
            OpKind::FieldRead { base, .. } => {
                assert_eq!(base, &v_var, "promote result must alias back to input arg");
            }
            other => panic!("expected FieldRead, got {other:?}"),
        }
    }

    /// `jtransform.py:608` voidness guard — `if hints.get('promote')
    /// and op.args[0].concretetype is not lltype.Void`.  Pyre falls
    /// through (Keep) when `value_kind(arg) == 'v'`.
    #[test]
    fn rewrite_graph_keeps_hint_promote_on_void_arg() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote"]),
                args: vec![v_var.clone()],
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig::default();
        // Mark `v` as void-kind on its backing Variable before
        // rewriting (mirrors RPython's `v.concretetype = lltype.Void`).
        FunctionGraph::set_concretetype_of_inline(
            &v_var,
            crate::jit_codewriter::type_state::ConcreteType::Void,
        );
        let mut transformer = Transformer::new(&config);
        // Direct call to rewrite_operation — without setting up the
        // optimize_block plumbing — verifies the Keep result for the
        // void-kind branch in isolation.
        let op = graph
            .block(graph.startblock)
            .operations
            .last()
            .unwrap()
            .clone();
        assert!(matches!(
            transformer.rewrite_operation(&op, "demo", &mut graph),
            RewriteResult::Keep
        ));
    }

    /// RPython `rpython/jit/codewriter/jtransform.py:895-903` — a
    /// quasi-immutable field read lowers to
    /// `[-live-, record_quasiimmut_field(v, descr, descr1), getfield_*_pure]`.
    /// Covers Issue 5.
    #[test]
    fn getfield_rewrite_emits_record_quasiimmut_for_quasi_immut() {
        use crate::call::CallControl;
        use crate::model::{FieldDescriptor, ImmutableRank};

        let mut cc = CallControl::new();
        cc.immutable_fields_by_struct.insert(
            "Cell".to_string(),
            vec![("value".to_string(), ImmutableRank::QuasiImmutable)],
        );

        let mut graph = FunctionGraph::new("read_cell");
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "cell".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var,
                field: FieldDescriptor::new("value", Some("Cell".to_string())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);
        let ops: Vec<&OpKind> = result
            .graph
            .block(graph.startblock)
            .operations
            .iter()
            .map(|o| &o.kind)
            .collect();

        // Expect the triple [Live, RecordQuasiImmutField, FieldRead] in
        // order, preceded by the Input op.
        let live_idx = ops
            .iter()
            .position(|k| matches!(k, OpKind::Live))
            .expect("Live marker present");
        assert!(matches!(
            ops[live_idx + 1],
            OpKind::RecordQuasiImmutField {
                field, mutate_field, ..
            } if field.name == "value"
                && mutate_field.name == "mutate_value"
                && mutate_field.owner_root.as_deref() == Some("Cell")
        ));
        assert!(matches!(
            ops[live_idx + 2],
            OpKind::FieldRead {
                field, pure: true, ..
            } if field.name == "value"
        ));
    }

    /// A plain-immutable field read lowers directly to a pure read, without
    /// the quasi-immutable bookkeeping pair.  Mirrors the `pure` /
    /// non-`pure` fork at `jtransform.py:867-878`.
    #[test]
    fn getfield_rewrite_preserves_plain_immutable_read() {
        use crate::call::CallControl;
        use crate::model::{FieldDescriptor, ImmutableRank};

        let mut cc = CallControl::new();
        cc.immutable_fields_by_struct.insert(
            "Point".to_string(),
            vec![("x".to_string(), ImmutableRank::Immutable)],
        );

        let mut graph = FunctionGraph::new("read_x");
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "p".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var,
                field: FieldDescriptor::new("x", Some("Point".to_string())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);
        let ops = &result.graph.block(graph.startblock).operations;

        assert!(
            !ops.iter()
                .any(|o| matches!(o.kind, OpKind::RecordQuasiImmutField { .. })),
            "plain immutable must not emit record_quasiimmut_field"
        );
        assert!(
            ops.iter().any(|o| matches!(
                &o.kind,
                OpKind::FieldRead {
                    field, pure: true, ..
                } if field.name == "x"
            )),
            "FieldRead for x should become a pure read"
        );
    }

    #[test]
    fn handle_jit_marker_loop_header_emits_single_loop_header_op() {
        // jtransform.py:1714-1718 `SpaceOperation('loop_header', [c_index], None)`.
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let ops = transformer.handle_jit_marker__loop_header(7);
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::LoopHeader { jitdriver_index } => assert_eq!(*jitdriver_index, 7),
            other => panic!("expected OpKind::LoopHeader, got {other:?}"),
        }
        assert!(ops[0].result.is_none(), "loop_header produces no result");
    }

    #[test]
    fn handle_jit_marker_jit_merge_point_emits_live_merge_live_sequence() {
        // jtransform.py:1707-1712 — return shape is `ops + [op3, op1, op2]`
        // where op3=live_preamble, op1=jit_merge_point, op2=live_recursive.
        use crate::flowspace::model::Variable;
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_portal_jd(Some(3));
        let green_i = Variable::new();
        let red_i_a = Variable::new();
        let red_i_b = Variable::new();
        let red_r = Variable::new();
        let ops = transformer.handle_jit_marker__jit_merge_point(
            vec![green_i.clone()],
            vec![],
            vec![],
            vec![red_i_a.clone(), red_i_b.clone()],
            vec![red_r.clone()],
            vec![],
        );
        assert_eq!(ops.len(), 3, "expect live + merge + live");
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                reds_i,
                reds_r,
                ..
            } => {
                assert_eq!(*jitdriver_index, 3);
                assert_eq!(greens_i, &vec![green_i]);
                assert_eq!(reds_i, &vec![red_i_a, red_i_b]);
                assert_eq!(reds_r, &vec![red_r]);
            }
            other => panic!("expected OpKind::JitMergePoint, got {other:?}"),
        }
        assert!(matches!(ops[2].kind, OpKind::Live));
    }

    #[test]
    fn jit_marker_key_recognises_pypyjitdriver_methods() {
        let merge = CallTarget::method("jit_merge_point", Some("PyPyJitDriver".into()));
        assert_eq!(
            jit_marker_key_from_target(&merge),
            Some(JitMarkerKey::JitMergePoint)
        );
        let cej = CallTarget::method("can_enter_jit", Some("PyPyJitDriver".into()));
        assert_eq!(
            jit_marker_key_from_target(&cej),
            Some(JitMarkerKey::CanEnterJit)
        );
        let lh = CallTarget::method("loop_header", Some("PyPyJitDriver".into()));
        assert_eq!(
            jit_marker_key_from_target(&lh),
            Some(JitMarkerKey::LoopHeader)
        );
        // Other receivers or other methods must not match.
        let other = CallTarget::method("jit_merge_point", Some("OtherDriver".into()));
        assert_eq!(jit_marker_key_from_target(&other), None);
        let other_method = CallTarget::method("something_else", Some("PyPyJitDriver".into()));
        assert_eq!(jit_marker_key_from_target(&other_method), None);
        // Non-method targets are never markers.
        let free_fn = CallTarget::function_path(["module", "jit_merge_point"]);
        assert_eq!(jit_marker_key_from_target(&free_fn), None);
    }

    #[test]
    fn try_handle_jit_marker_can_enter_jit_aliases_to_loop_header() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_portal_jd(Some(2));
        let ops = transformer
            .try_handle_jit_marker(JitMarkerKey::CanEnterJit, &[])
            .expect("can_enter_jit should dispatch when portal_jd is set");
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::LoopHeader { jitdriver_index } => assert_eq!(*jitdriver_index, 2),
            other => panic!("expected LoopHeader, got {other:?}"),
        }
    }

    #[test]
    fn promote_greens_emits_live_guard_value_pair_per_green() {
        // jtransform.py:1646-1656. One `-live-` + `{kind}_guard_value` pair
        // per green, in input order. Without a type_state every green falls
        // back to kind 'r'.
        let config = GraphTransformConfig::default();
        let transformer = Transformer::new(&config);
        let mut graph = crate::model::FunctionGraph::new("test_promote_greens_fixture");
        graph.set_next_value(3);
        let greens: Vec<crate::flowspace::model::Variable> =
            (0..3).map(|i| graph.must_variable_at(i)).collect();
        let ops = transformer.promote_greens(&greens);
        assert_eq!(ops.len(), 6, "expect 2 ops per green");
        for i in 0..greens.len() {
            assert!(
                matches!(ops[i * 2].kind, OpKind::Live),
                "slot {i} should start with Live"
            );
            match &ops[i * 2 + 1].kind {
                OpKind::GuardValue {
                    value, kind_char, ..
                } => {
                    assert_eq!(value, &greens[i]);
                    assert_eq!(*kind_char, 'r');
                }
                other => panic!("slot {i} expected GuardValue, got {other:?}"),
            }
        }
    }

    #[test]
    fn promote_greens_empty_input_yields_empty_output() {
        let config = GraphTransformConfig::default();
        let transformer = Transformer::new(&config);
        assert!(transformer.promote_greens(&[]).is_empty());
    }

    #[test]
    fn try_handle_jit_marker_returns_none_without_portal() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        // No portal_jd set → dispatch is a no-op (caller falls through).
        assert!(
            transformer
                .try_handle_jit_marker(JitMarkerKey::LoopHeader, &[])
                .is_none()
        );
    }

    #[test]
    #[should_panic(expected = "'jit_merge_point' in non-portal graph!")]
    fn handle_jit_marker_jit_merge_point_without_portal_panics() {
        // jtransform.py:1691 `assert self.portal_jd is not None`.
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        transformer.handle_jit_marker__jit_merge_point(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        );
    }

    /// Upstream parity test for the entry-level dispatch shape — a
    /// straight port of `test_jtransform.py:1011-1046 test_jit_merge_point_1`.
    ///
    /// Input: `try_handle_jit_marker(JitMergePoint, [receiver, g1, g2,
    /// r1])` with two greens and one red, all int-typed via direct
    /// `FunctionGraph::set_concretetype_of_inline`. Expected output (`promote_greens` +
    /// `handle_jit_marker__jit_merge_point`):
    ///
    /// ```text
    /// op0 = -live-
    /// op1 = int_guard_value(g1)
    /// op2 = -live-
    /// op3 = int_guard_value(g2)
    /// op4 = -live-                                   ← live_preamble
    /// op5 = jit_merge_point(idx, I[g1,g2], R[], F[], ← merge
    ///                       I[r1],     R[], F[])
    /// op6 = -live-                                   ← live_recursive
    /// ```
    ///
    /// Locks the entry-level `try_handle_jit_marker` ↔
    /// `promote_greens` ↔ `handle_jit_marker__jit_merge_point`
    /// composition shape so a regression that drops the promote_greens
    /// prefix or the trailing live ops fails immediately.
    #[test]
    fn try_handle_jit_marker_jit_merge_point_emits_full_promote_greens_sequence() {
        use crate::jit_codewriter::call::CallControl;
        use crate::jit_codewriter::type_state::ConcreteType;
        use crate::parse::CallPath;

        let g1_slot: usize = 10;
        let g2_slot: usize = 11;
        let r1_slot: usize = 12;
        let receiver_slot: usize = 99;

        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["test", "portal"]),
            vec!["green1".into(), "green2".into()],
            vec!["red1".into()],
            Vec::new(),
            Vec::new(),
        );

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_portal_jd(Some(0));

        // promote_greens reads Variable.concretetype directly to populate
        // the GuardValue.value field — grow the backing Variable table
        // to cover slot 99 (the receiver slot index used below).
        let mut graph = crate::model::FunctionGraph::new("test_jit_merge_point_fixture");
        graph.set_next_value(100);
        // jtransform now reads operand kinds via Variable.concretetype;
        // hydrate it before invoking the handler so the green/red
        // classifier picks `'i'` instead of the Unknown-defaulted `'r'`.
        let receiver_var = graph.must_variable_at(receiver_slot);
        let g1_var = graph.must_variable_at(g1_slot);
        let g2_var = graph.must_variable_at(g2_slot);
        let r1_var = graph.must_variable_at(r1_slot);
        FunctionGraph::set_concretetype_of_inline(&g1_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&g2_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&r1_var, ConcreteType::Signed);
        let args_vars = vec![receiver_var, g1_var.clone(), g2_var.clone(), r1_var.clone()];
        let ops = transformer
            .try_handle_jit_marker(JitMarkerKey::JitMergePoint, &args_vars)
            .expect("portal_jd + cc + 2-greens + 1-red satisfies dispatch preconditions");

        assert_eq!(ops.len(), 7, "promote_greens(2 greens)*2 + merge*3 = 7");

        // promote_greens prefix: -live-, int_guard_value(g1), -live-, int_guard_value(g2)
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue {
                value, kind_char, ..
            } => {
                assert_eq!(value, &g1_var);
                assert_eq!(*kind_char, 'i');
            }
            other => panic!("ops[1] expected GuardValue(g1, 'i'), got {other:?}"),
        }
        assert!(matches!(ops[2].kind, OpKind::Live));
        match &ops[3].kind {
            OpKind::GuardValue {
                value, kind_char, ..
            } => {
                assert_eq!(value, &g2_var);
                assert_eq!(*kind_char, 'i');
            }
            other => panic!("ops[3] expected GuardValue(g2, 'i'), got {other:?}"),
        }
        // live_preamble + merge + live_recursive
        assert!(matches!(ops[4].kind, OpKind::Live));
        match &ops[5].kind {
            OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
            } => {
                assert_eq!(*jitdriver_index, 0);
                assert_eq!(greens_i, &vec![g1_var.clone(), g2_var.clone()]);
                assert!(greens_r.is_empty());
                assert!(greens_f.is_empty());
                assert_eq!(reds_i, &vec![r1_var.clone()]);
                assert!(reds_r.is_empty());
                assert!(reds_f.is_empty());
            }
            other => panic!("ops[5] expected JitMergePoint, got {other:?}"),
        }
        assert!(matches!(ops[6].kind, OpKind::Live));
        assert!(
            ops[5].result.is_none(),
            "jit_merge_point produces no result"
        );
    }

    /// Synthetic Rust-wrapper elision must accept a single-arg `Ok(_)`
    /// whose result type matches the arg type and whose frontend target was
    /// resolved as a synthetic transparent constructor.
    #[test]
    fn synthetic_result_ctor_identity_accepts_prelude_ok() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_ok");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        assert!(transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Ok"),
            &[arg],
            &ValueType::Ref,
        ));
    }

    /// Reject a normal function call named `Ok`; user-function protection now
    /// lives in the frontend resolver, so jtransform only trusts the explicit
    /// synthetic target variant.
    #[test]
    fn synthetic_result_ctor_identity_rejects_function_path_ok() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_fn_path");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::function_path(["Ok"]),
            &[arg],
            &ValueType::Ref,
        ));
    }

    /// Reject when the target is not the explicit synthetic variant even if
    /// the spelling looks like a prelude constructor.
    #[test]
    fn synthetic_result_ctor_identity_rejects_name_only_matching() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_name_only");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::function_path(["Ok"]),
            &[arg],
            &ValueType::Ref,
        ));
    }

    /// Reject when arg/result IR kinds disagree — that means the
    /// callee is doing real boxing work, not a transparent wrapper.
    #[test]
    fn synthetic_result_ctor_identity_rejects_kind_mismatch() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_kind_mismatch");
        let arg = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let transformer = Transformer::new(&config);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Ok"),
            &[arg],
            &ValueType::Ref,
        ));
    }

    /// Reject names not in the narrow allow-list and reject name-only
    /// matching for qualified paths. The frontend maps approved qualified
    /// constructors to `SyntheticTransparentCtor` with the final segment.
    #[test]
    fn synthetic_result_ctor_identity_rejects_other_names() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_other_names");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::function_path(["Result", "Ok"]),
            &[arg.clone()],
            &ValueType::Ref,
        ));
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Foo"),
            &[arg.clone()],
            &ValueType::Ref,
        ));
        assert!(transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Err"),
            &[arg.clone()],
            &ValueType::Ref,
        ));
        assert!(transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Some"),
            &[arg],
            &ValueType::Ref,
        ));
    }

    #[test]
    fn map_user_oopspec_dict_lookup() {
        use majit_ir::descr::OopSpecIndex;
        assert_eq!(
            super::map_user_oopspec_to_index("ordereddict.lookup(d, key, hash, flag)"),
            OopSpecIndex::DictLookup
        );
        assert_eq!(
            super::map_user_oopspec_to_index("ordereddict.lookup"),
            OopSpecIndex::DictLookup
        );
        assert_eq!(
            super::map_user_oopspec_to_index("dict.lookup"),
            OopSpecIndex::DictLookup
        );
        assert_eq!(
            super::map_user_oopspec_to_index("dict.setitem"),
            OopSpecIndex::None
        );
    }

    // ── RPython indirect_call plumbing tests — parity guard ──────────
    //
    // RPython upstream: `jtransform.py:538-553 handle_regular_indirect_
    // call` emits `[-live-, int_guard_value, residual_call +
    // IndirectCallTargets, -live-]`; `assembler.py:208-209` collects
    // the candidate jitcodes into `indirectcalltargets`.  These two
    // tests anchor the RPython-orthodox post-jtransform shape so any
    // future change to `lower_indirect_calls` / `Transformer::transform`
    // surfaces here rather than reaching the assembler with a wrong
    // op sequence.

    /// Build an impl-method graph with a single `Input { ty: Ref }`
    /// representing the receiver `self` — mirrors the one-arg signature
    /// `fn run(&self)` that `FunctionReprBase.call` feeds into
    /// `indirect_call(funcptr, self, c_graphs)` (`rpbc.py:207-217`).
    fn build_handler_run_impl_graph(name: &str) -> FunctionGraph {
        let mut graph = FunctionGraph::new(name);
        graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "self".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, None);
        graph
    }

    /// Build a graph that calls `receiver.run()` on a `dyn Handler`
    /// receiver with two registered impls, run the rtyper-equivalent
    /// `lower_indirect_calls` pass and then `Transformer::transform`,
    /// and assert the post-jtransform sequence is exactly
    /// `[VtableMethodPtr, Live, GuardValue{kind='i'},
    ///   CallResidual{funcptr=Value(_), indirect_targets=Some}, Live]`.
    /// RPython `jtransform.py:410-412 + 538-553` orthodox port parity.
    #[test]
    fn lower_indirect_call_op_emit_order() {
        use crate::call::CallControl;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let mut cc = CallControl::new();
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "A",
            build_handler_run_impl_graph("A::run"),
        );
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "B",
            build_handler_run_impl_graph("B::run"),
        );
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver_var],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let ops = &result.graph.block(graph.startblock).operations;
        let post_input = &ops[1..];
        assert!(
            matches!(post_input[0].kind, OpKind::VtableMethodPtr { .. }),
            "expected VtableMethodPtr first, got {:?}",
            post_input[0].kind
        );
        assert!(
            matches!(post_input[1].kind, OpKind::Live),
            "expected Live second, got {:?}",
            post_input[1].kind
        );
        match &post_input[2].kind {
            OpKind::GuardValue { kind_char: 'i', .. } => {}
            other => panic!("expected GuardValue kind='i', got {other:?}"),
        }
        match &post_input[3].kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                indirect_targets: Some(t),
                ..
            } => assert_eq!(t.lst.len(), 2, "both impls should be candidates"),
            other => panic!(
                "expected CallResidual with runtime funcptr + indirect_targets, got {other:?}"
            ),
        }
        // jtransform.py:547 handle_residual_call(..., may_call_jitcodes=True)
        // forces a trailing `-live-`.
        assert!(
            matches!(post_input[4].kind, OpKind::Live),
            "expected trailing Live, got {:?}",
            post_input[4].kind
        );
    }

    /// End-to-end smoke: after the rtyper-equivalent `lower_indirect_calls`
    /// pass + `Transformer::transform`, the `CallResidual.indirect_targets`
    /// payload carries exactly one `JitCodeHandle` per candidate impl
    /// (each shell allocated by `CallControl::get_jitcode`). The assembler
    /// later merges these handles into `Assembler.indirectcalltargets`
    /// (RPython `assembler.py:208-209`).
    #[test]
    fn indirectcalltargets_reach_call_residual_payload() {
        use crate::call::CallControl;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let mut cc = CallControl::new();
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "A",
            build_handler_run_impl_graph("A::run"),
        );
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "B",
            build_handler_run_impl_graph("B::run"),
        );
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver_var],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let residual = result
            .graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    indirect_targets, ..
                } => indirect_targets.clone(),
                _ => None,
            })
            .expect("residual call with indirect_targets");

        assert_eq!(residual.lst.len(), 2);
        // `IndirectCallTargets.lst` carries shell handles returned by
        // `CallControl::get_jitcode()`. RPython appends/indices jitcodes
        // only after `transform_graph_to_jitcode(...)` completes, so the
        // candidate handles must still be unindexed at this stage.
        assert!(
            residual.lst.iter().all(|h| h.try_index().is_none()),
            "indirect target shells should not have final all_jitcodes indices yet"
        );
    }

    // ── Kind matrix: `indirect_regular_call_{r,ir,irf}_{i,r,f,v}` ────
    //
    // RPython upstream: `test_jtransform.py:340-367` parameterization +
    // `test_jtransform.py:447-484 indirect_regular_call_test`.  Each
    // test builds a `receiver.m(extras...)` site where `extras` covers
    // the arg kind signature (`r`, `ir`, `irf`) and `result_ty` covers
    // the result kind (`i`, `r`, `f`, `v`).  The emitted `CallResidual`
    // must split args into `(args_i, args_r, args_f)` by kind — with
    // the receiver always landing in `args_r` — and carry `result_kind`
    // matching the call's `result_ty`.  Receiver-in-args mirrors RPython
    // `rpbc.py:1195-1208 MethodsPBCRepr.redispatch_call`.

    /// Runs the full rtyper-equivalent + jtransform pipeline for a
    /// `receiver.m(extras...)` dyn-Trait call and asserts the emitted
    /// op sequence is exactly
    /// `[VtableMethodPtr, Live, GuardValue{'i'}, CallResidual, Live]`,
    /// with the CallResidual's arg-kind distribution matching `expect`
    /// and `result_kind` matching `expect_res_kind`.  Two impls are
    /// registered so `CallKind::Regular` is selected.
    #[cfg(test)]
    fn check_indirect_regular_call_kind(
        extras: &[ValueType],
        result_ty: ValueType,
        expect: (usize, usize, usize),
        expect_res_kind: char,
    ) {
        use crate::call::CallControl;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let build_impl = |name: &str| {
            let mut g = FunctionGraph::new(name);
            g.push_op_var(
                g.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
            for (i, ty) in extras.iter().enumerate() {
                g.push_op_var(
                    g.startblock,
                    OpKind::Input {
                        name: format!("a{i}"),
                        ty: ty.clone(),
                    },
                    true,
                )
                .unwrap();
            }
            g.set_return(g.startblock, None);
            g
        };

        let mut cc = CallControl::new();
        cc.register_trait_method("m", Some("T"), "A", build_impl("A::m"));
        cc.register_trait_method("m", Some("T"), "B", build_impl("B::m"));
        // The indirect branch of `getcalldescr(op)` validates that the
        // caller's `result_type` matches the witness impl's declared
        // return type (`call.rs` indirect-arm signature check).  Inject
        // matching strings so the non-void kind-matrix cases resolve
        // without panicking.
        let return_str = match result_ty {
            ValueType::Int | ValueType::State => "i64",
            ValueType::Unsigned => "u64",
            ValueType::Ref | ValueType::Unknown => "String",
            ValueType::Float => "f64",
            ValueType::Void => "",
            ValueType::Bool => "bool",
        };
        if !return_str.is_empty() {
            cc.return_types.insert(
                crate::parse::CallPath::for_impl_method("A", "m"),
                return_str.to_string(),
            );
            cc.return_types.insert(
                crate::parse::CallPath::for_impl_method("B", "m"),
                return_str.to_string(),
            );
        }
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let mut args_vars: Vec<crate::flowspace::model::Variable> = vec![receiver_var];
        for (i, ty) in extras.iter().enumerate() {
            args_vars.push(
                graph
                    .push_op_var(
                        graph.startblock,
                        OpKind::Input {
                            name: format!("a{i}"),
                            ty: ty.clone(),
                        },
                        true,
                    )
                    .unwrap(),
            );
        }
        let has_result = !matches!(result_ty, ValueType::Void);
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("T", "m"),
                args: args_vars,
                result_ty,
            },
            has_result,
        );
        graph.set_return(graph.startblock, None);

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let ops = &result.graph.block(graph.startblock).operations;
        let post_input: Vec<_> = ops
            .iter()
            .filter(|op| !matches!(&op.kind, OpKind::Input { .. }))
            .collect();

        assert!(
            matches!(post_input[0].kind, OpKind::VtableMethodPtr { .. }),
            "expected VtableMethodPtr first, got {:?}",
            post_input[0].kind,
        );
        assert!(
            matches!(post_input[1].kind, OpKind::Live),
            "expected Live second, got {:?}",
            post_input[1].kind,
        );
        assert!(
            matches!(
                post_input[2].kind,
                OpKind::GuardValue { kind_char: 'i', .. }
            ),
            "expected GuardValue kind_char='i', got {:?}",
            post_input[2].kind,
        );
        match &post_input[3].kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                args_i,
                args_r,
                args_f,
                result_kind,
                indirect_targets: Some(t),
                ..
            } => {
                assert_eq!(args_i.len(), expect.0, "args_i count");
                assert_eq!(args_r.len(), expect.1, "args_r count (receiver counted)");
                assert_eq!(args_f.len(), expect.2, "args_f count");
                assert_eq!(*result_kind, expect_res_kind, "result_kind");
                assert_eq!(t.lst.len(), 2, "both impls in family");
            }
            other => {
                panic!("expected CallResidual with Value funcptr + indirect_targets, got {other:?}")
            }
        }
        assert!(
            matches!(post_input[4].kind, OpKind::Live),
            "expected trailing Live (may_call_jitcodes=true), got {:?}",
            post_input[4].kind,
        );
    }

    #[test]
    fn indirect_regular_call_r_i() {
        check_indirect_regular_call_kind(&[], ValueType::Int, (0, 1, 0), 'i');
    }
    #[test]
    fn indirect_regular_call_r_r() {
        check_indirect_regular_call_kind(&[], ValueType::Ref, (0, 1, 0), 'r');
    }
    #[test]
    fn indirect_regular_call_r_f() {
        check_indirect_regular_call_kind(&[], ValueType::Float, (0, 1, 0), 'f');
    }
    #[test]
    fn indirect_regular_call_r_v() {
        check_indirect_regular_call_kind(&[], ValueType::Void, (0, 1, 0), 'v');
    }

    #[test]
    fn indirect_regular_call_ir_i() {
        check_indirect_regular_call_kind(&[ValueType::Int], ValueType::Int, (1, 1, 0), 'i');
    }
    #[test]
    fn indirect_regular_call_ir_r() {
        check_indirect_regular_call_kind(&[ValueType::Int], ValueType::Ref, (1, 1, 0), 'r');
    }
    #[test]
    fn indirect_regular_call_ir_f() {
        check_indirect_regular_call_kind(&[ValueType::Int], ValueType::Float, (1, 1, 0), 'f');
    }
    #[test]
    fn indirect_regular_call_ir_v() {
        check_indirect_regular_call_kind(&[ValueType::Int], ValueType::Void, (1, 1, 0), 'v');
    }

    #[test]
    fn indirect_regular_call_irf_i() {
        check_indirect_regular_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Int,
            (1, 1, 1),
            'i',
        );
    }
    #[test]
    fn indirect_regular_call_irf_r() {
        check_indirect_regular_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Ref,
            (1, 1, 1),
            'r',
        );
    }
    #[test]
    fn indirect_regular_call_irf_f() {
        check_indirect_regular_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Float,
            (1, 1, 1),
            'f',
        );
    }
    #[test]
    fn indirect_regular_call_irf_v() {
        check_indirect_regular_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Void,
            (1, 1, 1),
            'v',
        );
    }

    // ── Kind matrix: `indirect_residual_call_{r,ir,irf}_{i,r,f,v}` ───
    //
    // RPython upstream: `test_jtransform.py:420-445 indirect_residual_call_test`.
    // `handle_residual_indirect_call` is an alias for the regular
    // residual-call path (`jtransform.py:536`): no `int_guard_value` guard
    // and no `IndirectCallTargets` sidecar; only `[residual_call_*, -live-]`.
    // On the pyre side we additionally retain the `VtableMethodPtr` op
    // emitted by `lower_indirect_calls` — the rtyper-equivalent layer runs
    // unconditionally and provides the runtime funcptr `Variable` consumed
    // by the `CallResidual.funcptr` `CallFuncPtr::Value` operand.

    /// Same skeleton as `check_indirect_regular_call_kind` but drops
    /// `find_all_graphs_for_tests`, so `candidate_graphs` stays empty
    /// and `guess_call_kind(op)` falls through to `CallKind::Residual`
    /// via the `graphs_from(op)` call.py:137-139 fall-through.
    /// Emit is `[VtableMethodPtr, CallResidual{indirect_targets: None},
    /// Live?]` — the trailing `-live-` appears only when the descriptor's
    /// `can_raise` is true, which is the default for non-elidable
    /// family calls.
    #[cfg(test)]
    fn check_indirect_residual_call_kind(
        extras: &[ValueType],
        result_ty: ValueType,
        expect: (usize, usize, usize),
        expect_res_kind: char,
    ) {
        use crate::call::CallControl;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let build_impl = |name: &str| {
            let mut g = FunctionGraph::new(name);
            g.push_op_var(
                g.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
            for (i, ty) in extras.iter().enumerate() {
                g.push_op_var(
                    g.startblock,
                    OpKind::Input {
                        name: format!("a{i}"),
                        ty: ty.clone(),
                    },
                    true,
                )
                .unwrap();
            }
            g.set_return(g.startblock, None);
            g
        };

        let mut cc = CallControl::new();
        cc.register_trait_method("m", Some("T"), "A", build_impl("A::m"));
        cc.register_trait_method("m", Some("T"), "B", build_impl("B::m"));
        let return_str = match result_ty {
            ValueType::Int | ValueType::State => "i64",
            ValueType::Unsigned => "u64",
            ValueType::Ref | ValueType::Unknown => "String",
            ValueType::Float => "f64",
            ValueType::Void => "",
            ValueType::Bool => "bool",
        };
        if !return_str.is_empty() {
            cc.return_types.insert(
                crate::parse::CallPath::for_impl_method("A", "m"),
                return_str.to_string(),
            );
            cc.return_types.insert(
                crate::parse::CallPath::for_impl_method("B", "m"),
                return_str.to_string(),
            );
        }
        // NOTE: intentionally *no* `find_all_graphs_for_tests` — that keeps
        // `candidate_graphs` empty so `guess_call_kind(op)` classifies
        // this call as `CallKind::Residual` via the call.py:137-139
        // `graphs_from(op) is None` fall-through.

        let mut graph = FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let mut args_vars: Vec<crate::flowspace::model::Variable> = vec![receiver_var];
        for (i, ty) in extras.iter().enumerate() {
            args_vars.push(
                graph
                    .push_op_var(
                        graph.startblock,
                        OpKind::Input {
                            name: format!("a{i}"),
                            ty: ty.clone(),
                        },
                        true,
                    )
                    .unwrap(),
            );
        }
        let has_result = !matches!(result_ty, ValueType::Void);
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("T", "m"),
                args: args_vars,
                result_ty,
            },
            has_result,
        );
        graph.set_return(graph.startblock, None);

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let ops = &result.graph.block(graph.startblock).operations;
        let post_input: Vec<_> = ops
            .iter()
            .filter(|op| !matches!(&op.kind, OpKind::Input { .. }))
            .collect();

        assert!(
            matches!(post_input[0].kind, OpKind::VtableMethodPtr { .. }),
            "expected VtableMethodPtr first, got {:?}",
            post_input[0].kind,
        );
        // Residual path must NOT emit the [Live, GuardValue] prefix —
        // upstream `jtransform.py:536` aliases
        // `handle_residual_indirect_call = handle_residual_call`.
        assert!(
            !post_input
                .iter()
                .any(|op| matches!(op.kind, OpKind::GuardValue { .. })),
            "residual path must not emit int_guard_value, got ops: {:?}",
            post_input.iter().map(|op| &op.kind).collect::<Vec<_>>(),
        );
        match &post_input[1].kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                args_i,
                args_r,
                args_f,
                result_kind,
                indirect_targets: None,
                ..
            } => {
                assert_eq!(args_i.len(), expect.0, "args_i count");
                assert_eq!(args_r.len(), expect.1, "args_r count (receiver counted)");
                assert_eq!(args_f.len(), expect.2, "args_f count");
                assert_eq!(*result_kind, expect_res_kind, "result_kind");
            }
            other => panic!(
                "expected CallResidual with Value funcptr + indirect_targets=None, got {other:?}"
            ),
        }
    }

    #[test]
    fn indirect_residual_call_r_i() {
        check_indirect_residual_call_kind(&[], ValueType::Int, (0, 1, 0), 'i');
    }
    #[test]
    fn indirect_residual_call_r_r() {
        check_indirect_residual_call_kind(&[], ValueType::Ref, (0, 1, 0), 'r');
    }
    #[test]
    fn indirect_residual_call_r_f() {
        check_indirect_residual_call_kind(&[], ValueType::Float, (0, 1, 0), 'f');
    }
    #[test]
    fn indirect_residual_call_r_v() {
        check_indirect_residual_call_kind(&[], ValueType::Void, (0, 1, 0), 'v');
    }

    #[test]
    fn indirect_residual_call_ir_i() {
        check_indirect_residual_call_kind(&[ValueType::Int], ValueType::Int, (1, 1, 0), 'i');
    }
    #[test]
    fn indirect_residual_call_ir_r() {
        check_indirect_residual_call_kind(&[ValueType::Int], ValueType::Ref, (1, 1, 0), 'r');
    }
    #[test]
    fn indirect_residual_call_ir_f() {
        check_indirect_residual_call_kind(&[ValueType::Int], ValueType::Float, (1, 1, 0), 'f');
    }
    #[test]
    fn indirect_residual_call_ir_v() {
        check_indirect_residual_call_kind(&[ValueType::Int], ValueType::Void, (1, 1, 0), 'v');
    }

    #[test]
    fn indirect_residual_call_irf_i() {
        check_indirect_residual_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Int,
            (1, 1, 1),
            'i',
        );
    }
    #[test]
    fn indirect_residual_call_irf_r() {
        check_indirect_residual_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Ref,
            (1, 1, 1),
            'r',
        );
    }
    #[test]
    fn indirect_residual_call_irf_f() {
        check_indirect_residual_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Float,
            (1, 1, 1),
            'f',
        );
    }
    #[test]
    fn indirect_residual_call_irf_v() {
        check_indirect_residual_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Void,
            (1, 1, 1),
            'v',
        );
    }

    /// `rpython/jit/codewriter/jtransform.py:599-606` — when a
    /// `hint(arg, promote=True, promote_string=True)` carries both
    /// flags, the rewrite drops one based on whether the arg's
    /// `concretetype` is `Ptr(STR)`.  Pyre's `value_kind` is too
    /// coarse to make that distinction (every pointer maps to `'r'`),
    /// so `PromoteOrString` defaults to the plain `<kind>_guard_value`
    /// path — `ref_guard_value` is safe for every Ref, including
    /// non-string pointers.  Users who need the value-equality
    /// `str_guard_value` shape invoke `hint_promote_string(x)`
    /// explicitly.
    #[test]
    fn rewrite_graph_promote_or_string_picks_ref_guard_value_for_ref_arg() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote_or_string"]),
                args: vec![v_var],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let result = rewrite_graph(&graph, &GraphTransformConfig::default());
        let ops = &result.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue { kind_char, .. } => {
                assert_eq!(
                    *kind_char, 'r',
                    "Ref arg must default to ref_guard_value (no Ptr(STR) info)"
                );
            }
            other => panic!("expected GuardValue, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_promote_or_string_picks_int_guard_value_for_int_arg() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote_or_string"]),
                args: vec![v_var.clone()],
                result_ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        // value_kind defaults to 'r' without an entry in type_state;
        // seed `v` as `Signed` so the dual-hint arm sees an Int arg
        // and routes through the plain `<kind>_guard_value` path
        // instead of the str_guard_value helper chain.
        FunctionGraph::set_concretetype_of_inline(&v_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);
        let ops = &result.graph.block(graph.startblock).operations;
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue { kind_char, .. } => {
                assert_eq!(
                    *kind_char, 'i',
                    "Int arg must route through the int_guard_value path"
                );
            }
            other => panic!("expected GuardValue, got {other:?}"),
        }
    }
}
