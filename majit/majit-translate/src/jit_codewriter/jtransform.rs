//! Graph-based jtransform: semantic rewrite pass.
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
use crate::model::{
    CallFuncPtr, CallTarget, FieldDescriptor, FunctionGraph, OpKind, SpaceOperation, ValueId,
    ValueType, remap_control_flow_metadata,
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
        self.name == field.name
            && self
                .owner_root
                .as_ref()
                .is_none_or(|owner| field.owner_root.as_ref() == Some(owner))
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
    /// ConcreteType for every synthetic ValueId emitted by jtransform
    /// (Phase C v2 Slice C-2). Callers merge this into their existing
    /// `TypeResolutionState` so the rewritten graph's kind classification
    /// stays authoritative without requiring a full `resolve_types`
    /// rerun.  See `Transformer::synth_kinds`.
    pub synth_kinds:
        std::collections::HashMap<ValueId, crate::jit_codewriter::type_state::ConcreteType>,
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
/// last remaining entry (`translate_legacy/pipeline.rs::analyze_function`
/// + lib/test helpers) — plan Rev 2 §Phase B3.
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
/// `guess_call_kind` + `graphs_from`. Used by
/// `translate_legacy::pipeline::analyze_function` after it calls
/// `lower_indirect_calls` upstream.
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
    type_state: Option<&'a crate::jit_codewriter::type_state::TypeResolutionState>,
    /// RPython: `Transformer.vable_array_vars`.
    /// Stores (vable_base, array_index, itemsize, is_signed) per vable array variable.
    vable_array_vars: std::collections::HashMap<ValueId, (ValueId, usize, usize, bool)>,
    /// RPython: `Transformer.vable_flags`.
    #[allow(dead_code)]
    vable_flags: std::collections::HashMap<ValueId, VableFlag>,
    /// Value aliases from identity rewrites (same_as / hint rewriting).
    aliases: std::collections::HashMap<ValueId, ValueId>,
    notes: Vec<GraphTransformNote>,
    vable_rewrites: usize,
    calls_classified: usize,
    /// RPython: DependencyTracker — caches transitive analysis results.
    /// Shared across all getcalldescr() calls within this transform pass.
    analysis_cache: crate::call::AnalysisCache,
    /// Cursor into the running graph's `next_value` counter. Kept in
    /// sync with `FunctionGraph::next_value`: seeded from
    /// `graph.next_value` at `transform()` entry and written back at
    /// exit so any future rewrite rule that needs to synthesize a fresh
    /// ValueId can pull one without colliding with the rtyper-equivalent
    /// pass output.
    next_synthetic_value: usize,
    /// ValueId → ConcreteType for synthetic ValueIds allocated via
    /// `fresh_synthetic_value_typed` and for existing result ValueIds
    /// whose kind is changed by a rewrite in this pass.  RPython
    /// attaches `concretetype` directly on every `Variable`
    /// (`flowmodel.py:Variable.__init__`); pyre stamps the equivalent
    /// table entry here so later operations in the same transform pass,
    /// and the post-transform type merge, see the rewritten kind.
    synth_kinds:
        std::collections::HashMap<ValueId, crate::jit_codewriter::type_state::ConcreteType>,
}

/// RPython: jtransform.py vable_flags values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VableFlag {
    FreshVirtualizable,
}

/// Result of rewriting a single operation.
///
/// RPython: `rewrite_operation()` returns SpaceOperation, list, None, or Constant.
enum RewriteResult {
    /// Replace with these ops
    Replace(Vec<SpaceOperation>),
    /// Remove the op (identity/alias: result remaps to the given value)
    Identity(ValueId),
    /// Keep the op unchanged
    Keep,
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

/// Split a run of `ValueId`s into (ints, refs, floats) per upstream
/// `make_three_lists` (`jtransform.py:1616-1627`). Void values are
/// dropped, matching the upstream filter; Unknown defaults to `Ref`.
fn split_args_by_kind(
    args: &[ValueId],
    type_state: Option<&crate::jit_codewriter::type_state::TypeResolutionState>,
) -> (Vec<ValueId>, Vec<ValueId>, Vec<ValueId>) {
    let mut ints = Vec::new();
    let mut refs = Vec::new();
    let mut floats = Vec::new();
    for &v in args {
        let kind = if let Some(ts) = type_state {
            match ts.concrete_types.get(&v) {
                Some(crate::jit_codewriter::type_state::ConcreteType::Signed) => 'i',
                Some(crate::jit_codewriter::type_state::ConcreteType::Float) => 'f',
                Some(crate::jit_codewriter::type_state::ConcreteType::Void) => 'v',
                // RPython: GcRef or Unknown → 'r'
                _ => 'r',
            }
        } else {
            'r'
        };
        match kind {
            'i' => ints.push(v),
            'f' => floats.push(v),
            'v' => {}
            _ => refs.push(v),
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
            type_state: None,
            vable_array_vars: std::collections::HashMap::new(),
            vable_flags: std::collections::HashMap::new(),
            aliases: std::collections::HashMap::new(),
            notes: Vec::new(),
            vable_rewrites: 0,
            calls_classified: 0,
            analysis_cache: crate::call::AnalysisCache::default(),
            next_synthetic_value: 0,
            synth_kinds: std::collections::HashMap::new(),
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

    /// Set the type resolution state for arg kind splitting.
    /// RPython: types live on `Variable.concretetype`.
    pub fn with_type_state(
        mut self,
        ts: &'a crate::jit_codewriter::type_state::TypeResolutionState,
    ) -> Self {
        self.type_state = Some(ts);
        self
    }

    /// RPython: Transformer.transform() — process all blocks in the graph.
    pub fn transform(&mut self, graph: &FunctionGraph) -> GraphTransformResult {
        let mut rewritten = graph.clone();
        // Keep the ValueId allocator in sync with the graph we're
        // rewriting so rules that synthesize values stay unique.
        self.next_synthetic_value = rewritten.next_value();

        let exceptblock = rewritten.exceptblock;
        for block in &mut rewritten.blocks {
            self.optimize_block(block, &graph.name, exceptblock);
        }

        rewritten.set_next_value(self.next_synthetic_value);

        GraphTransformResult {
            graph: rewritten,
            notes: std::mem::take(&mut self.notes),
            vable_rewrites: self.vable_rewrites,
            calls_classified: self.calls_classified,
            synth_kinds: std::mem::take(&mut self.synth_kinds),
        }
    }

    /// RPython: Transformer.optimize_block()
    fn optimize_block(
        &mut self,
        block: &mut crate::model::Block,
        graph_name: &str,
        exceptblock: crate::model::BlockId,
    ) {
        let mut new_ops = Vec::with_capacity(block.operations.len());

        for original_op in &block.operations {
            let op = remap_op(original_op, &self.aliases);
            match self.rewrite_operation(&op, graph_name) {
                RewriteResult::Replace(ops) => {
                    new_ops.extend(ops);
                }
                RewriteResult::Identity(alias_target) => {
                    if let Some(result) = op.result {
                        self.aliases
                            .insert(result, resolve_alias(alias_target, &self.aliases));
                    }
                }
                RewriteResult::Keep => {
                    new_ops.push(op);
                }
            }
        }

        block.operations = new_ops;
        let (exitswitch, exits) = remap_control_flow_metadata(
            &block.exitswitch,
            &block.exits,
            |v| remap_value(v, &self.aliases),
            |b| b,
        );
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

    fn fresh_synthetic_value(&mut self) -> ValueId {
        let value = ValueId(self.next_synthetic_value);
        self.next_synthetic_value += 1;
        value
    }

    /// Allocate a fresh synthetic ValueId AND stamp its `ConcreteType`
    /// into `synth_kinds` (Phase C v2 Slice C-2).  Mirrors RPython's
    /// `Variable(concretetype=T)` (`flowmodel.py:Variable.__init__`).
    fn fresh_synthetic_value_typed(
        &mut self,
        ty: crate::jit_codewriter::type_state::ConcreteType,
    ) -> ValueId {
        let value = self.fresh_synthetic_value();
        self.synth_kinds.insert(value, ty);
        value
    }

    fn stamp_value_kind(
        &mut self,
        value: Option<ValueId>,
        ty: crate::jit_codewriter::type_state::ConcreteType,
    ) {
        if let Some(value) = value {
            self.synth_kinds.insert(value, ty);
        }
    }

    /// Stamp the synthetic kind from a `ValueType` source, skipping
    /// `ValueType::Unknown` so the fallback Unknown→Ref defaulting in
    /// `value_type_to_kind` does not pollute `synth_kinds` with a
    /// GcRef entry that would mask the real `concretetype` already
    /// computed by the rtyper for an existing ValueId.
    fn stamp_value_kind_from_value_type(&mut self, value: Option<ValueId>, result_ty: &ValueType) {
        let ty = match result_ty {
            ValueType::Int | ValueType::State => {
                crate::jit_codewriter::type_state::ConcreteType::Signed
            }
            ValueType::Ref => crate::jit_codewriter::type_state::ConcreteType::GcRef,
            ValueType::Float => crate::jit_codewriter::type_state::ConcreteType::Float,
            ValueType::Void => crate::jit_codewriter::type_state::ConcreteType::Void,
            ValueType::Unknown => return,
        };
        self.stamp_value_kind(value, ty);
    }

    fn direct_funcptr_value(&mut self, target: &CallTarget) -> (ValueId, SpaceOperation) {
        let fnaddr = self
            .callcontrol
            .as_deref()
            .map(|cc| cc.fnaddr_for_target(target))
            .unwrap_or_else(|| crate::call::symbolic_fnaddr_for_target(target));
        // Function pointer materialized as ConstInt — assembler emits it
        // through the `'i'` argcode so the kind is Signed.
        let value = self
            .fresh_synthetic_value_typed(crate::jit_codewriter::type_state::ConcreteType::Signed);
        (
            value,
            SpaceOperation {
                result: Some(value),
                kind: OpKind::ConstInt(fnaddr),
            },
        )
    }

    /// RPython: Transformer.rewrite_operation() — dispatch to rewrite_op_*.
    fn rewrite_operation(&mut self, op: &SpaceOperation, graph_name: &str) -> RewriteResult {
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
                self.rewrite_op_setfield(op, field, *value, ty, graph_name)
            }
            // ── rewrite_op_getarrayitem ──
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
                ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_getarrayitem(op, *base, *index, item_ty, graph_name)
            }
            // ── rewrite_op_setarrayitem ──
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
                ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_setarrayitem(op, *base, *index, *value, item_ty, graph_name)
            }
            // ── rewrite_op_direct_call ──
            OpKind::Call {
                target,
                args,
                result_ty,
            } if self.config.classify_calls => {
                self.rewrite_op_direct_call(op, target, args, result_ty, graph_name)
            }
            // ── rewrite_op_indirect_call ──
            // RPython jtransform.py:410-412. Pyre's rtyper-equivalent
            // (`translator/rtyper/rpbc.rs`) lowers
            // `OpKind::Call { target: CallTarget::Indirect, .. }` into
            // `OpKind::IndirectCall { funcptr, args, graphs, .. }`
            // before jtransform runs, so by this point the funcptr is
            // already a regular ValueId and `args` still carry the full
            // call argument list, including the receiver.
            OpKind::IndirectCall {
                funcptr,
                args,
                graphs,
                result_ty,
            } if self.config.classify_calls => self.lower_indirect_call_op(
                op,
                *funcptr,
                args,
                graphs.as_deref(),
                result_ty,
                graph_name,
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
                    result: op.result,
                    kind: OpKind::BinOp {
                        op: canonical.into(),
                        lhs: *lhs,
                        rhs: *rhs,
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "same_as" || unop_name == "deref" => {
                // RPython `jtransform.py:246 rewrite_op_same_as`
                // returns None: the op is removed and its result is
                // renamed to its argument. Rust reference/deref sugar
                // has the same identity shape at this IR level.
                RewriteResult::Identity(*operand)
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
                && self.get_value_kind(*lhs) == 'r'
                && self.get_value_kind(*rhs) == 'r' =>
            {
                let new_op = if binop_name == "eq" {
                    "ptr_eq"
                } else {
                    "ptr_ne"
                };
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::BinOp {
                        op: new_op.into(),
                        lhs: *lhs,
                        rhs: *rhs,
                        result_ty: result_ty.clone(),
                    },
                }])
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
                && is_float_rewrite_domain(self.get_value_kind(*lhs))
                && is_float_rewrite_domain(self.get_value_kind(*rhs))
                && (self.get_value_kind(*lhs) == 'f'
                    || self.get_value_kind(*rhs) == 'f'
                    || *result_ty == ValueType::Float) =>
            {
                let canonical = match canonical_float_arith_binop(binop_name)
                    .expect("guard checked float arithmetic op")
                {
                    "div" => "truediv", // RPython lltype: `float_truediv`
                    other => other,
                };
                self.stamp_value_kind(
                    op.result,
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(*lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(*rhs);
                ops.extend(rhs_ops);
                ops.push(SpaceOperation {
                    result: op.result,
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
                && is_float_rewrite_domain(self.get_value_kind(*lhs))
                && is_float_rewrite_domain(self.get_value_kind(*rhs))
                && (self.get_value_kind(*lhs) == 'f' || self.get_value_kind(*rhs) == 'f') =>
            {
                self.stamp_value_kind(
                    op.result,
                    crate::jit_codewriter::type_state::ConcreteType::Signed,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(*lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(*rhs);
                ops.extend(rhs_ops);
                ops.push(SpaceOperation {
                    result: op.result,
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
                && is_float_rewrite_domain(self.get_value_kind(*lhs))
                && is_float_rewrite_domain(self.get_value_kind(*rhs))
                && (self.get_value_kind(*lhs) == 'f'
                    || self.get_value_kind(*rhs) == 'f'
                    || *result_ty == ValueType::Float) =>
            {
                self.stamp_value_kind(
                    op.result,
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(*lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(*rhs);
                ops.extend(rhs_ops);
                let target = CallTarget::function_path(["ll_math_fmod"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(&target);
                ops.push(funcptr_op);
                ops.push(SpaceOperation {
                    result: op.result,
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
            } if unop_name == "neg" && self.get_value_kind(*operand) == 'f' => {
                self.stamp_value_kind(
                    op.result,
                    crate::jit_codewriter::type_state::ConcreteType::Float,
                );
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::UnaryOp {
                        op: "float_neg".into(),
                        operand: *operand,
                        result_ty: ValueType::Float,
                    },
                }])
            }
            // pyre front-end uses Rust's syn::BinOp / syn::UnOp identifiers
            // (`add_assign`, `deref`) which have no RPython counterpart —
            // RPython's flow-space never sees Python-level `+=` or `*x` as
            // distinct ops (Python bytecode expands `+=` to BINARY_ADD
            // followed by STORE_FAST, and `*x` disappears at rtyper). After
            // the float/int coercion arms above have had a chance to match,
            // the remaining in-place ops degenerate to their plain RPython
            // binary opnames.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if canonical_assign_binop(binop_name).is_some() => {
                let canonical = canonical_assign_binop(binop_name)
                    .expect("guard checked assign binop")
                    .to_string();
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::BinOp {
                        op: canonical,
                        lhs: *lhs,
                        rhs: *rhs,
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            _ => RewriteResult::Keep,
        }
    }

    // ── helpers ──────────────────────────────────────────────

    /// RPython: `Transformer.make_three_lists(vars)` (jtransform.py:437-445).
    /// Split args into three lists by kind (int, ref, float).
    ///
    /// RPython: `add_in_correct_list(v, lst_i, lst_r, lst_f)` checks
    /// `getkind(v.concretetype)` and appends to the matching list.
    /// Void args are skipped.
    fn make_three_lists(&self, args: &[ValueId]) -> (Vec<ValueId>, Vec<ValueId>, Vec<ValueId>) {
        let mut args_i = Vec::new();
        let mut args_r = Vec::new();
        let mut args_f = Vec::new();
        for &v in args {
            let kind = self.get_value_kind(v);
            match kind {
                'i' => args_i.push(v),
                'r' => args_r.push(v),
                'f' => args_f.push(v),
                'v' => {}            // void — skip (RPython jtransform.py:449)
                _ => args_r.push(v), // unknown → ref
            }
        }
        (args_i, args_r, args_f)
    }

    /// RPython: `getkind(v.concretetype)` — get the kind of a value.
    /// Uses type_state if available, falls back to 'r' for unknown.
    fn get_value_kind(&self, v: ValueId) -> char {
        if let Some(ct) = self.synth_kinds.get(&v) {
            return match ct {
                crate::jit_codewriter::type_state::ConcreteType::Signed => 'i',
                crate::jit_codewriter::type_state::ConcreteType::GcRef => 'r',
                crate::jit_codewriter::type_state::ConcreteType::Float => 'f',
                crate::jit_codewriter::type_state::ConcreteType::Void => 'v',
                crate::jit_codewriter::type_state::ConcreteType::Unknown => 'r',
            };
        }
        if let Some(ts) = self.type_state {
            if let Some(ct) = ts.concrete_types.get(&v) {
                return match ct {
                    crate::jit_codewriter::type_state::ConcreteType::Signed => 'i',
                    crate::jit_codewriter::type_state::ConcreteType::GcRef => 'r',
                    crate::jit_codewriter::type_state::ConcreteType::Float => 'f',
                    crate::jit_codewriter::type_state::ConcreteType::Void => 'v',
                    crate::jit_codewriter::type_state::ConcreteType::Unknown => 'r',
                };
            }
        }
        'r' // default: ref (most Python values are GC refs)
    }

    fn get_value_type(&self, v: ValueId) -> Option<ValueType> {
        let ts = self.type_state?;
        let ct = ts.concrete_types.get(&v)?;
        match ct {
            crate::jit_codewriter::type_state::ConcreteType::Signed => Some(ValueType::Int),
            crate::jit_codewriter::type_state::ConcreteType::GcRef => Some(ValueType::Ref),
            crate::jit_codewriter::type_state::ConcreteType::Float => Some(ValueType::Float),
            crate::jit_codewriter::type_state::ConcreteType::Void => Some(ValueType::Void),
            crate::jit_codewriter::type_state::ConcreteType::Unknown => None,
        }
    }

    /// RPython's float rtyper calls `hop.inputargs(Float, Float)`, which
    /// inserts `cast_int_to_float` for mixed int/float operands before
    /// emitting `float_*` or the `math.fmod` helper call.  Pyre's float
    /// rewrite arms only enter when both operands are
    /// `is_float_rewrite_domain` (i.e. `'i'` or `'f'`); Ref operands do
    /// not reach this helper.
    fn coerce_operand_to_float_domain(&mut self, value: ValueId) -> (ValueId, Vec<SpaceOperation>) {
        match self.get_value_kind(value) {
            'f' => (value, Vec::new()),
            'i' => self.coerce_operand_to_float(value),
            _ => (value, Vec::new()),
        }
    }

    /// RPython's float rtyper calls `hop.inputargs(Float, Float)`, which
    /// inserts `cast_int_to_float` for mixed int/float operands before
    /// emitting `float_*` or the `math.fmod` helper call.
    fn coerce_operand_to_float(&mut self, value: ValueId) -> (ValueId, Vec<SpaceOperation>) {
        if self.get_value_kind(value) != 'i' {
            return (value, Vec::new());
        }
        let coerced = self
            .fresh_synthetic_value_typed(crate::jit_codewriter::type_state::ConcreteType::Float);
        (
            coerced,
            vec![SpaceOperation {
                result: Some(coerced),
                kind: OpKind::UnaryOp {
                    op: "cast_int_to_float".into(),
                    operand: value,
                    result_ty: ValueType::Float,
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
        args: &[ValueId],
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
                if let Some(arg) = args.first().copied() {
                    RewriteResult::Identity(arg)
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
                if let Some(arg) = args.first().copied() {
                    if let Some(result) = op.result {
                        self.aliases
                            .insert(result, resolve_alias(arg, &self.aliases));
                    }
                }
                RewriteResult::Replace(vec![SpaceOperation {
                    result: None,
                    kind: OpKind::VableForce,
                }])
            }
        }
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
            .and_then(|result| self.get_value_type(result))
            .unwrap_or_else(|| ty.clone());
        // Track virtualizable array field reads
        if let Some(array_field) = self.config.vable_arrays.iter().find(|c| c.matches(field)) {
            if let Some(result) = op.result {
                // RPython: vable_array_vars[result] = (v_base, arrayfielddescr, arraydescr)
                // We store the vable base plus the arraydescr properties.
                let base = match &op.kind {
                    OpKind::FieldRead { base, .. } => *base,
                    _ => unreachable!("rewrite_op_getfield called on non-FieldRead op"),
                };
                let itemsize = array_field.array_itemsize.unwrap_or(8);
                let is_signed = array_field.array_is_signed.unwrap_or(false);
                self.vable_array_vars
                    .insert(result, (base, array_field.index, itemsize, is_signed));
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
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::VableFieldRead {
                    base: match &op.kind {
                        OpKind::FieldRead { base, .. } => *base,
                        _ => unreachable!("rewrite_op_getfield called on non-FieldRead op"),
                    },
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
                            base: *base,
                            field: field.clone(),
                            mutate_field,
                        },
                    },
                    SpaceOperation {
                        result: op.result,
                        kind: OpKind::FieldRead {
                            base: *base,
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
                    result: op.result,
                    kind: OpKind::FieldRead {
                        base: *base,
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
                result: op.result,
                kind: OpKind::FieldRead {
                    base: *base,
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
        value: ValueId,
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
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::VableFieldWrite {
                    base: match &op.kind {
                        OpKind::FieldWrite { base, .. } => *base,
                        _ => unreachable!("rewrite_op_setfield called on non-FieldWrite op"),
                    },
                    field_index: vable_field.index,
                    value,
                    ty: typed_ty,
                },
            }]);
        }
        if &typed_ty != ty {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::FieldWrite {
                    base: match &op.kind {
                        OpKind::FieldWrite { base, .. } => *base,
                        _ => unreachable!("rewrite_op_setfield called on non-FieldWrite op"),
                    },
                    field: field.clone(),
                    value,
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
        base: ValueId,
        index: ValueId,
        item_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let typed_item_ty = op
            .result
            .and_then(|result| self.get_value_type(result))
            .unwrap_or_else(|| item_ty.clone());
        if let Some(&(vable_base, arr_idx, itemsize, is_signed)) = self.vable_array_vars.get(&base)
        {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("rewrite: array[idx] → VableArrayRead[{arr_idx}]"),
            });
            self.vable_rewrites += 1;
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::VableArrayRead {
                    base: vable_base,
                    array_index: arr_idx,
                    elem_index: index,
                    item_ty: typed_item_ty,
                    array_itemsize: itemsize,
                    array_is_signed: is_signed,
                },
            }]);
        }
        if &typed_item_ty != item_ty {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::ArrayRead {
                    base,
                    index,
                    item_ty: typed_item_ty,
                    array_type_id: match &op.kind {
                        OpKind::ArrayRead { array_type_id, .. } => array_type_id.clone(),
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
        base: ValueId,
        index: ValueId,
        value: ValueId,
        item_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let typed_item_ty = self
            .get_value_type(value)
            .unwrap_or_else(|| item_ty.clone());
        if let Some(&(vable_base, arr_idx, itemsize, is_signed)) = self.vable_array_vars.get(&base)
        {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("rewrite: array[idx] = v → VableArrayWrite[{arr_idx}]"),
            });
            self.vable_rewrites += 1;
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::VableArrayWrite {
                    base: vable_base,
                    array_index: arr_idx,
                    elem_index: index,
                    value,
                    item_ty: typed_item_ty,
                    array_itemsize: itemsize,
                    array_is_signed: is_signed,
                },
            }]);
        }
        if &typed_item_ty != item_ty {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::ArrayWrite {
                    base,
                    index,
                    value,
                    item_ty: typed_item_ty,
                    array_type_id: match &op.kind {
                        OpKind::ArrayWrite { array_type_id, .. } => array_type_id.clone(),
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
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
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
            return RewriteResult::Identity(args[0]);
        }
        // RPython: guess_call_kind(op) → dispatch to handle_*_call
        if let Some(cc) = self.callcontrol.as_mut() {
            let kind = cc.guess_call_kind(op);
            return match kind {
                crate::call::CallKind::Regular => {
                    self.handle_regular_call(op, target, args, result_ty, graph_name)
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
                    let non_void_args = resolve_non_void_arg_types(args, self.type_state);
                    let result_ir_type = value_type_to_ir_type(result_ty);
                    let extraeffect = classify_call(target, &self.config.call_effects)
                        .map(|(d, _)| d.extra_info.extraeffect);
                    let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
                    let descriptor = cc_ref.getcalldescr(
                        op,
                        non_void_args,
                        result_ir_type,
                        OopSpecIndex::None,
                        extraeffect,
                        &mut self.analysis_cache,
                    );
                    self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
                }
                crate::call::CallKind::Builtin => {
                    self.handle_builtin_call(op, target, args, result_ty, graph_name)
                }
                crate::call::CallKind::Recursive => {
                    self.handle_recursive_call(op, target, args, result_ty, graph_name)
                }
            };
        }

        // Fallback when no CallControl: effect-only classification (legacy path).
        // RPython: always residual_call_*, effect only in calldescr.
        if let Some((descriptor, _effect)) = classify_call(target, &self.config.call_effects) {
            let non_void_args = resolve_non_void_arg_types(args, self.type_state);
            let descriptor =
                descriptor.with_signature(&non_void_args, value_type_to_ir_type(result_ty));
            self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
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
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // RPython jtransform.py:484-520: __handle_oopspec_call pattern.
        //   calldescr = self.callcontrol.getcalldescr(op, oopspecindex, extraeffect)
        //   self.callcontrol.callinfocollection.add(oopspecindex, calldescr, func)

        // Look up oopspec/effect info: config overrides first, then static table.
        // rlib/jit.py:250 — user-level `@oopspec(spec)` is registered via
        // `call_control.mark_oopspec(path, spec)`; look it up here before
        // falling back to the static builtin table.
        let user_oopspec: Option<String> = self
            .callcontrol
            .as_deref()
            .and_then(|cc| cc.get_oopspec(target).map(|s| s.to_string()));

        // jtransform.py:497-498 — oopspec dispatch by prefix.
        if let Some(spec) = user_oopspec.as_deref() {
            let base = spec.split('(').next().unwrap_or(spec).trim();
            // jtransform.py:497 — jit.* oopspecs → __handle_jit_call
            if base.starts_with("jit.") {
                return self._handle_jit_call(base, op, target, args, result_ty, graph_name);
            }
            // NOTE: conditional_call!/conditional_call_elidable!/record_known_result!
            // are handled by jitcode_lower (proc-macro level), NOT here.
            // The codewriter AST parser does not expand macro_rules!, so these
            // macros appear as Stmt::Macro → UnknownKind::MacroStmt.
            // The jitcode_lower proc-macro intercepts the macros directly and
            // emits BC_COND_CALL_* / BC_RECORD_KNOWN_RESULT_* bytecodes.
        }
        let (oopspecindex, extraeffect_override) =
            if let Some((descriptor, _)) = classify_call(target, &self.config.call_effects) {
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
        let non_void_args = resolve_non_void_arg_types(args, self.type_state);
        let result_ir_type = value_type_to_ir_type(result_ty);
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                oopspecindex,
                extraeffect_override,
                &mut self.analysis_cache,
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
        self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
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
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
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
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);
        self.stamp_value_kind_from_value_type(op.result, result_ty);

        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → inline_call[jitcode={:?}]", jitcode.name),
        });
        self.calls_classified += 1;
        // RPython jtransform.py:480-481: inline_call always followed by -live-
        RewriteResult::Replace(vec![
            SpaceOperation {
                result: op.result,
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
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
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
        let (greens_i, greens_r, greens_f) = self.make_three_lists(green_args);
        let (reds_i, reds_r, reds_f) = self.make_three_lists(red_args);
        let result_kind = value_type_to_kind(result_ty);
        self.stamp_value_kind_from_value_type(op.result, result_ty);

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
            result: op.result,
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
    fn promote_greens(&self, green_args: &[ValueId]) -> Vec<SpaceOperation> {
        let mut ops = Vec::new();
        for &v in green_args {
            let kind = self.value_kind(v);
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
                    value: v,
                    kind_char: kind,
                },
            });
        }
        ops
    }

    /// RPython: `getkind(v.concretetype)` — get the kind of a value.
    /// Delegates to `get_value_kind()` which consults type_state.
    fn value_kind(&self, v: ValueId) -> char {
        self.get_value_kind(v)
    }

    /// RPython: `Transformer.__handle_jit_call(op, oopspec_name, args)` (jtransform.py:1730-1757).
    /// Dispatches jit.* oopspec calls to dedicated opcodes or __handle_oopspec_call.
    fn _handle_jit_call(
        &mut self,
        oopspec_name: &str,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
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
                let value = args[0];
                let kind_char = self.get_value_kind(value);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.assert_green → {kind_char}_assert_green"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: None,
                    kind: OpKind::AssertGreen { value, kind_char },
                }])
            }
            // jtransform.py:1736-1737
            "jit.current_trace_length" => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: "jit.current_trace_length → current_trace_length".to_string(),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::CurrentTraceLength,
                }])
            }
            // jtransform.py:1738-1740
            "jit.isconstant" => {
                let value = args[0];
                let kind_char = self.get_value_kind(value);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.isconstant → {kind_char}_isconstant"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::IsConstant { value, kind_char },
                }])
            }
            // jtransform.py:1741-1743
            "jit.isvirtual" => {
                let value = args[0];
                let kind_char = self.get_value_kind(value);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.isvirtual → {kind_char}_isvirtual"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::IsVirtual { value, kind_char },
                }])
            }
            // jtransform.py:1744-1747
            "jit.force_virtual" => self._handle_oopspec_call(
                op,
                target,
                args,
                result_ty,
                graph_name,
                OopSpecIndex::JitForceVirtual,
                Some(majit_ir::descr::ExtraEffect::ForcesVirtualOrVirtualizable),
            ),
            // jtransform.py:1748-1755
            "jit.not_in_trace" => {
                // jtransform.py:1750-1753: not_in_trace must return void
                assert!(
                    *result_ty == ValueType::Void,
                    "jit.not_in_trace() function must return None"
                );
                self._handle_oopspec_call(
                    op,
                    target,
                    args,
                    result_ty,
                    graph_name,
                    OopSpecIndex::NotInTrace,
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
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
        oopspecindex: OopSpecIndex,
        extraeffect: Option<majit_ir::descr::ExtraEffect>,
    ) -> RewriteResult {
        // jtransform.py:1990-1993
        let non_void_args = resolve_non_void_arg_types(args, self.type_state);
        let result_ir_type = value_type_to_ir_type(result_ty);
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                oopspecindex,
                extraeffect,
                &mut self.analysis_cache,
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
        self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
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
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
        is_value: bool,
    ) -> RewriteResult {
        // jtransform.py:1666-1672: validate no floats, ≤4+2 args
        for &arg in args {
            if self.get_value_kind(arg) == 'f' {
                panic!("Conditional call does not support floats");
            }
        }
        if args.len() > 4 + 2 {
            panic!("Conditional call does not support more than 4 arguments");
        }
        // jtransform.py:1673-1676: calldescr from function call (args[1:] → result)
        let condition_or_value = args[0];
        let func_args = if args.len() > 1 { &args[1..] } else { &[] };
        let non_void_args = resolve_non_void_arg_types(func_args, self.type_state);
        let result_ir_type = value_type_to_ir_type(result_ty);
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                OopSpecIndex::None,
                None,
                &mut self.analysis_cache,
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
        let (args_i, args_r, args_f) = self.make_three_lists(func_args);
        assert!(
            args_f.is_empty(),
            "force_ir: no float args in conditional_call"
        );
        let result_kind = value_type_to_kind(result_ty);
        self.stamp_value_kind_from_value_type(op.result, result_ty);
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
                value: condition_or_value,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
                result_kind,
            }
        } else {
            OpKind::ConditionalCall {
                condition: condition_or_value,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
            }
        };
        // jtransform.py:1681-1682: -live- if calldescr_canraise
        let mut ops = vec![SpaceOperation {
            result: op.result,
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
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self._rewrite_op_cond_call(op, target, args, result_ty, graph_name, false)
    }

    /// RPython: `Transformer.rewrite_op_jit_conditional_call_value(op)`
    /// (jtransform.py:1687-1688). Dispatch wrapper kept for structural
    /// parity; pyre's `rewrite_operation` match does not reach it.
    #[allow(dead_code)]
    fn rewrite_op_jit_conditional_call_value(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self._rewrite_op_cond_call(op, target, args, result_ty, graph_name, true)
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
        args: &[ValueId],
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
                let (greens_i, greens_r, greens_f) =
                    split_args_by_kind(greens_raw, self.type_state);
                let (reds_i, reds_r, reds_f) =
                    split_args_by_kind(&user_args[num_greens..], self.type_state);
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
        greens_i: Vec<ValueId>,
        greens_r: Vec<ValueId>,
        greens_f: Vec<ValueId>,
        reds_i: Vec<ValueId>,
        reds_r: Vec<ValueId>,
        reds_f: Vec<ValueId>,
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
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        _result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // jtransform.py:293-295: validate no floats
        for &arg in args {
            if self.get_value_kind(arg) == 'f' {
                panic!("record_known_result does not support floats");
            }
        }
        // jtransform.py:298-300: calldescr from function (args[1:] → args[0])
        // args[0] = known result, args[1..] = function args
        let result_value = args[0];
        let func_args = if args.len() > 1 { &args[1..] } else { &[] };
        let result_kind = self.get_value_kind(result_value);
        let result_ir_type = match result_kind {
            'i' => majit_ir::value::Type::Int,
            'r' => majit_ir::value::Type::Ref,
            _ => {
                panic!("record_known_result: unsupported result kind '{result_kind}'");
            }
        };
        let non_void_args = resolve_non_void_arg_types(func_args, self.type_state);
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                OopSpecIndex::None,
                None,
                &mut self.analysis_cache,
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
        let (args_i, args_r, args_f) = self.make_three_lists(func_args);
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
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.handle_residual_call_with_targets(
            op, target, descriptor, args, result_ty, graph_name, None,
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
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[ValueId],
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
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);
        self.stamp_value_kind_from_value_type(op.result, result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(target);
        // RPython jtransform.py:469-470: residual_call followed by -live-
        // if the call can raise or may call jitcodes.
        // jtransform.py:547: `handle_regular_indirect_call` passes
        // `may_call_jitcodes=True`, which forces a trailing `-live-`.
        let can_raise = descriptor.extra_info.check_can_raise(false) || indirect_targets.is_some();
        let mut ops = vec![
            funcptr_op,
            SpaceOperation {
                result: op.result,
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
    /// `funcptr` is the runtime ValueId already produced by the
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
        funcptr: ValueId,
        args: &[ValueId],
        graphs: Option<&[crate::parse::CallPath]>,
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);
        self.stamp_value_kind_from_value_type(op.result, result_ty);
        let non_void_args = resolve_non_void_arg_types(args, self.type_state);
        let result_ir_type = value_type_to_ir_type(result_ty);
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
                        value: funcptr,
                        kind_char: 'i',
                    },
                });
                ops.push(SpaceOperation {
                    result: op.result,
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr),
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
                    result: op.result,
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr),
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
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → elidable"),
        });
        self.calls_classified += 1;
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);
        self.stamp_value_kind_from_value_type(op.result, result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(target);
        RewriteResult::Replace(vec![
            funcptr_op,
            SpaceOperation {
                result: op.result,
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
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → may_force"),
        });
        self.calls_classified += 1;
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);
        self.stamp_value_kind_from_value_type(op.result, result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(target);
        // RPython: call_may_force always followed by -live-
        RewriteResult::Replace(vec![
            funcptr_op,
            SpaceOperation {
                result: op.result,
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
        args: &[ValueId],
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
        // arg/result IR-kind parity. `resolve_non_void_arg_types`
        // returns `Type::Ref` when type_state is missing or the
        // value is unknown — that's the same default
        // `value_type_to_kind` applies for an `Unknown` result, so
        // the comparison stays sound under partial type info.
        let arg_types = resolve_non_void_arg_types(args, self.type_state);
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
fn resolve_non_void_arg_types(
    args: &[ValueId],
    type_state: Option<&crate::jit_codewriter::type_state::TypeResolutionState>,
) -> Vec<majit_ir::value::Type> {
    args.iter()
        .filter_map(|&v| {
            let kind = if let Some(ts) = type_state {
                if let Some(ct) = ts.concrete_types.get(&v) {
                    match ct {
                        crate::jit_codewriter::type_state::ConcreteType::Signed => 'i',
                        crate::jit_codewriter::type_state::ConcreteType::GcRef => 'r',
                        crate::jit_codewriter::type_state::ConcreteType::Float => 'f',
                        crate::jit_codewriter::type_state::ConcreteType::Void => 'v',
                        crate::jit_codewriter::type_state::ConcreteType::Unknown => 'r',
                    }
                } else {
                    'r'
                }
            } else {
                'r'
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
        ValueType::Int | ValueType::State => 'i',
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
        ValueType::Int | ValueType::State => majit_ir::value::Type::Int,
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

fn resolve_alias(
    mut value: ValueId,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> ValueId {
    while let Some(next) = aliases.get(&value).copied() {
        if next == value {
            break;
        }
        value = next;
    }
    value
}

fn remap_value(value: ValueId, aliases: &std::collections::HashMap<ValueId, ValueId>) -> ValueId {
    resolve_alias(value, aliases)
}

fn remap_list(
    values: &[ValueId],
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> Vec<ValueId> {
    values
        .iter()
        .copied()
        .map(|v| remap_value(v, aliases))
        .collect()
}

fn remap_call_funcptr(
    funcptr: &CallFuncPtr,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> CallFuncPtr {
    match funcptr {
        CallFuncPtr::Target(target) => CallFuncPtr::Target(target.clone()),
        CallFuncPtr::Value(value) => CallFuncPtr::Value(remap_value(*value, aliases)),
    }
}

fn remap_op(
    op: &SpaceOperation,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> SpaceOperation {
    let kind = match &op.kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstFloat(_)
        | OpKind::VableForce
        | OpKind::CurrentTraceLength
        | OpKind::Live
        | OpKind::LoopHeader { .. }
        | OpKind::GuardValue { .. }
        | OpKind::VtableMethodPtr { .. }
        | OpKind::Abort { .. } => op.kind.clone(),
        OpKind::JitMergePoint {
            jitdriver_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
        } => OpKind::JitMergePoint {
            jitdriver_index: *jitdriver_index,
            greens_i: remap_list(greens_i, aliases),
            greens_r: remap_list(greens_r, aliases),
            greens_f: remap_list(greens_f, aliases),
            reds_i: remap_list(reds_i, aliases),
            reds_r: remap_list(reds_r, aliases),
            reds_f: remap_list(reds_f, aliases),
        },
        OpKind::IndirectCall {
            funcptr,
            args,
            graphs,
            result_ty,
        } => OpKind::IndirectCall {
            funcptr: remap_value(*funcptr, aliases),
            args: remap_list(args, aliases),
            graphs: graphs.clone(),
            result_ty: result_ty.clone(),
        },
        OpKind::RecordQuasiImmutField {
            base,
            field,
            mutate_field,
        } => OpKind::RecordQuasiImmutField {
            base: remap_value(*base, aliases),
            field: field.clone(),
            mutate_field: mutate_field.clone(),
        },
        OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        } => OpKind::FieldRead {
            base: remap_value(*base, aliases),
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
            base: remap_value(*base, aliases),
            field: field.clone(),
            value: remap_value(*value, aliases),
            ty: ty.clone(),
        },
        OpKind::ArrayRead {
            base,
            index,
            item_ty,
            array_type_id,
        } => OpKind::ArrayRead {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
            array_type_id,
        } => OpKind::ArrayWrite {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::InteriorFieldRead {
            base,
            index,
            field,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldRead {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
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
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            field: field.clone(),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::Call {
            target,
            args,
            result_ty,
        } => OpKind::Call {
            target: target.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::GuardTrue { cond } => OpKind::GuardTrue {
            cond: remap_value(*cond, aliases),
        },
        OpKind::GuardFalse { cond } => OpKind::GuardFalse {
            cond: remap_value(*cond, aliases),
        },
        OpKind::VableFieldRead {
            base,
            field_index,
            ty,
        } => OpKind::VableFieldRead {
            base: remap_value(*base, aliases),
            field_index: *field_index,
            ty: ty.clone(),
        },
        OpKind::VableFieldWrite {
            base,
            field_index,
            value,
            ty,
        } => OpKind::VableFieldWrite {
            base: remap_value(*base, aliases),
            field_index: *field_index,
            value: remap_value(*value, aliases),
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
            base: remap_value(*base, aliases),
            array_index: *array_index,
            elem_index: remap_value(*elem_index, aliases),
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
            base: remap_value(*base, aliases),
            array_index: *array_index,
            elem_index: remap_value(*elem_index, aliases),
            value: remap_value(*value, aliases),
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
            lhs: remap_value(*lhs, aliases),
            rhs: remap_value(*rhs, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } => OpKind::UnaryOp {
            op: op.clone(),
            operand: remap_value(*operand, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::JitDebug { args } => OpKind::JitDebug {
            args: remap_list(args, aliases),
        },
        OpKind::RecordKnownResult {
            result_value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::RecordKnownResult {
            result_value: remap_value(*result_value, aliases),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            result_kind: *result_kind,
        },
        OpKind::AssertGreen { value, kind_char } => OpKind::AssertGreen {
            value: remap_value(*value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsConstant { value, kind_char } => OpKind::IsConstant {
            value: remap_value(*value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsVirtual { value, kind_char } => OpKind::IsVirtual {
            value: remap_value(*value, aliases),
            kind_char: *kind_char,
        },
        OpKind::CallElidable {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::CallElidable {
            funcptr: remap_call_funcptr(funcptr, aliases),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            result_kind: *result_kind,
        },
        OpKind::CallResidual {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            indirect_targets,
        } => OpKind::CallResidual {
            funcptr: remap_call_funcptr(funcptr, aliases),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            indirect_targets: indirect_targets.clone(),
            result_kind: *result_kind,
        },
        OpKind::CallMayForce {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::CallMayForce {
            funcptr: remap_call_funcptr(funcptr, aliases),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            result_kind: *result_kind,
        },
        OpKind::InlineCall {
            jitcode,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::InlineCall {
            jitcode: jitcode.clone(),
            args_i: args_i
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            args_r: args_r
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            args_f: args_f
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_kind: *result_kind,
        },
        OpKind::RecursiveCall {
            jd_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            result_kind,
        } => OpKind::RecursiveCall {
            jd_index: *jd_index,
            greens_i: greens_i
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            greens_r: greens_r
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            greens_f: greens_f
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            reds_i: reds_i
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            reds_r: reds_r
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            reds_f: reds_f
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_kind: *result_kind,
        },
        OpKind::ConditionalCall {
            condition,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
        } => OpKind::ConditionalCall {
            condition: remap_value(*condition, aliases),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
        },
        OpKind::ConditionalCallValue {
            value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::ConditionalCallValue {
            value: remap_value(*value, aliases),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            result_kind: *result_kind,
        },
    };
    SpaceOperation {
        result: op.result,
        kind,
    }
}

fn classify_vable_hint(target: &CallTarget) -> Option<crate::hints::VirtualizableHintKind> {
    target
        .path_segments()
        .and_then(|segments| crate::hints::classify_virtualizable_hint_segments(segments))
}

/// Match CallTarget loosely: generic receivers (lowercase like "handler",
/// "self") match any pattern receiver type.
fn call_target_matches_loose(pattern: &CallTarget, target: &CallTarget) -> bool {
    match (pattern, target) {
        (
            CallTarget::Method {
                name: pn,
                receiver_root: pr,
            },
            CallTarget::Method {
                name: tn,
                receiver_root: tr,
            },
        ) => {
            if pn != tn {
                return false;
            }
            match (pr.as_deref(), tr.as_deref()) {
                (Some(p), Some(t)) => p == t || crate::call::is_generic_receiver(t),
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
        .find(|override_| call_target_matches_loose(&override_.target, target))
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
    use crate::jit_codewriter::type_state::{ConcreteType, TypeResolutionState};
    use crate::model::{CallFuncPtr, CallTarget, FunctionGraph, OpKind, ValueType};

    #[test]
    fn rewrite_graph_canonicalizes_frontend_bitops() {
        let mut graph = FunctionGraph::new("bitops");
        let lhs = graph.alloc_value();
        let rhs = graph.alloc_value();
        let result = graph
            .push_op(
                graph.startblock,
                OpKind::BinOp {
                    op: "bitxor".to_string(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));

        let transformed = rewrite_graph(&graph, &GraphTransformConfig::default());
        match &transformed.graph.block(graph.startblock).operations[0].kind {
            OpKind::BinOp { op, .. } => assert_eq!(op, "xor"),
            other => panic!("expected canonical BinOp, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_coerces_mixed_float_add() {
        let mut graph = FunctionGraph::new("mixed_float_add");
        let lhs = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));

        let mut type_state = TypeResolutionState::new();
        type_state.concrete_types.insert(lhs, ConcreteType::Signed);
        type_state.concrete_types.insert(rhs, ConcreteType::Float);
        type_state
            .concrete_types
            .insert(result, ConcreteType::Float);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, lhs);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        match &ops[3].kind {
            OpKind::BinOp {
                op,
                lhs: rewritten_lhs,
                rhs: rewritten_rhs,
                result_ty,
            } => {
                assert_eq!(op, "float_add");
                assert_eq!(*rewritten_lhs, cast_result);
                assert_eq!(*rewritten_rhs, rhs);
                assert_eq!(*result_ty, ValueType::Float);
            }
            other => panic!("expected float_add, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_coerces_mixed_float_comparison() {
        let mut graph = FunctionGraph::new("mixed_float_eq");
        let lhs = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op(
                graph.startblock,
                OpKind::BinOp {
                    op: "eq".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));

        let mut type_state = TypeResolutionState::new();
        type_state.concrete_types.insert(lhs, ConcreteType::Float);
        type_state.concrete_types.insert(rhs, ConcreteType::Signed);
        type_state
            .concrete_types
            .insert(result, ConcreteType::Signed);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, rhs);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        match &ops[3].kind {
            OpKind::BinOp {
                op,
                lhs: rewritten_lhs,
                rhs: rewritten_rhs,
                result_ty,
            } => {
                assert_eq!(op, "float_eq");
                assert_eq!(*rewritten_lhs, lhs);
                assert_eq!(*rewritten_rhs, cast_result);
                assert_eq!(*result_ty, ValueType::Int);
            }
            other => panic!("expected float_eq, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_lowers_float_mod_to_ll_math_fmod_residual_call() {
        let mut graph = FunctionGraph::new("float_mod");
        let lhs = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op(
                graph.startblock,
                OpKind::BinOp {
                    op: "mod".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));

        let mut type_state = TypeResolutionState::new();
        type_state.concrete_types.insert(lhs, ConcreteType::Signed);
        type_state.concrete_types.insert(rhs, ConcreteType::Float);
        type_state
            .concrete_types
            .insert(result, ConcreteType::Float);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 6, "Input + Input + cast + fnptr + call + Live");
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, lhs);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        assert!(matches!(ops[3].kind, OpKind::ConstInt(_)));
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
                assert_eq!(args_f, &vec![cast_result, rhs]);
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
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
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
        assert!(
            matches!(
                &rewritten_op.kind,
                OpKind::VableFieldRead {
                    base: rewritten_base,
                    field_index: 0,
                    ..
                } if *rewritten_base == base
            ),
            "expected VableFieldRead, got {:?}",
            rewritten_op.kind
        );
    }

    #[test]
    fn rewrite_graph_tags_vable_arrays_with_explicit_base() {
        let mut graph = FunctionGraph::new("test");
        let base = graph.alloc_value();
        let index = graph.alloc_value();
        let array = graph
            .push_op(
                graph.startblock,
                OpKind::FieldRead {
                    base,
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
        graph.push_op(
            graph.startblock,
            OpKind::ArrayRead {
                base: array,
                index,
                item_ty: ValueType::Int,
                array_type_id: None,
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
        assert!(
            matches!(
                &rewritten_op.kind,
                OpKind::VableArrayRead {
                    base: rewritten_base,
                    array_index: 0,
                    ..
                } if *rewritten_base == base
            ),
            "expected VableArrayRead with explicit base, got {:?}",
            rewritten_op.kind
        );
    }

    #[test]
    fn rewrite_graph_requires_matching_field_owner_root() {
        let mut graph = FunctionGraph::new("test");
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
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
        let base = graph.alloc_value();
        let value = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                value,
                ty: ValueType::Unknown,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let mut type_state = crate::jit_codewriter::type_state::TypeResolutionState::new();
        type_state.concrete_types.insert(
            value,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::FieldWrite { ty, .. } => assert_eq!(*ty, ValueType::Int),
            other => panic!("expected FieldWrite, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_types_arraywrite_from_value_kind() {
        let mut graph = FunctionGraph::new("test");
        let base = graph.alloc_value();
        let index = graph.alloc_value();
        let value = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty: ValueType::Unknown,
                array_type_id: None,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let mut type_state = crate::jit_codewriter::type_state::TypeResolutionState::new();
        type_state.concrete_types.insert(
            value,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::ArrayWrite { item_ty, .. } => assert_eq!(*item_ty, ValueType::Int),
            other => panic!("expected ArrayWrite, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_types_fieldread_from_result_kind() {
        let mut graph = FunctionGraph::new("test");
        let base = graph.alloc_value();
        let result = graph
            .push_op(
                graph.startblock,
                OpKind::FieldRead {
                    base,
                    field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Unknown,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));

        let mut type_state = crate::jit_codewriter::type_state::TypeResolutionState::new();
        type_state.concrete_types.insert(
            result,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::FieldRead { ty, .. } => assert_eq!(*ty, ValueType::Int),
            other => panic!("expected FieldRead, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_types_arrayread_from_result_kind() {
        let mut graph = FunctionGraph::new("test");
        let base = graph.alloc_value();
        let index = graph.alloc_value();
        let result = graph
            .push_op(
                graph.startblock,
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty: ValueType::Unknown,
                    array_type_id: None,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));

        let mut type_state = crate::jit_codewriter::type_state::TypeResolutionState::new();
        type_state.concrete_types.insert(
            result,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::ArrayRead { item_ty, .. } => assert_eq!(*item_ty, ValueType::Int),
            other => panic!("expected ArrayRead, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_classifies_calls() {
        let mut graph = FunctionGraph::new("test");
        graph.push_op(
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
    fn residual_direct_call_materializes_funcptr_const() {
        let mut cc = crate::call::CallControl::new();
        let target = CallTarget::function_path(["custom_reader"]);
        let mut graph = FunctionGraph::new("test");
        let arg = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "arg".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: target.clone(),
                args: vec![arg],
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
        graph.push_op(
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
        graph.push_op(
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
        let frame = graph.alloc_value();
        let hinted = graph.alloc_value();
        graph.block_mut(graph.startblock).inputargs.push(frame);
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_access_directly"]),
                args: vec![frame],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(hinted);
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base: hinted,
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
        let frame = graph.alloc_value();
        let forced = graph.alloc_value();
        graph.block_mut(graph.startblock).inputargs.push(frame);
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_force_virtualizable"]),
                args: vec![frame],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(forced);
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base: forced,
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
        assert!(matches!(ops[0].kind, OpKind::VableForce));
        assert!(matches!(
            ops[1].kind,
            OpKind::VableFieldRead { field_index: 0, .. }
        ));
    }

    // ── RPython indirect_call plumbing tests — plan §Tests ──────────

    /// Build a graph that calls `receiver.run()` on a `dyn Handler` receiver
    /// with two registered impls, run the rtyper-equivalent
    /// `lower_indirect_calls` pass and then `Transformer::transform`, and
    /// assert the post-jtransform sequence is exactly
    /// `[VtableMethodPtr, Live, GuardValue{kind='i'},
    ///   CallResidual{funcptr=Value(_), indirect_targets=Some}, Live]`.
    /// RPython `jtransform.py:410-412 + 538-553` orthodox port parity.
    /// Build an impl-method graph with a single `Input { ty: Ref }`
    /// representing the receiver `self` — mirrors the one-arg signature
    /// `fn run(&self)` that `FunctionReprBase.call` feeds into
    /// `indirect_call(funcptr, self, c_graphs)` (rpbc.py:207-217).
    fn build_handler_run_impl_graph(name: &str) -> FunctionGraph {
        let mut graph = FunctionGraph::new(name);
        graph
            .push_op(
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

    #[test]
    fn lower_indirect_call_op_emit_order() {
        use crate::call::CallControl;
        use crate::translate_legacy::annotator::annrpython::annotate;
        use crate::translate_legacy::rtyper::rtyper::resolve_types;
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

        // Build a graph: `receiver = input(); receiver.run()`.
        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        // Run the rtyper-equivalent lowering before jtransform — this is
        // what `codewriter.rs::transform_graph_to_jitcode` does for the
        // canonical pipeline.
        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_type_state(&type_state);
        let result = transformer.transform(&graph);

        let ops = &result.graph.block(graph.startblock).operations;
        // Strip the initial Input op.
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
        use crate::translate_legacy::annotator::annrpython::annotate;
        use crate::translate_legacy::rtyper::rtyper::resolve_types;
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
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_type_state(&type_state);
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
    fn check_indirect_regular_call_kind(
        extras: &[ValueType],
        result_ty: ValueType,
        expect: (usize, usize, usize),
        expect_res_kind: char,
    ) {
        use crate::call::CallControl;
        use crate::translate_legacy::annotator::annrpython::annotate;
        use crate::translate_legacy::rtyper::rtyper::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let build_impl = |name: &str| {
            let mut g = FunctionGraph::new(name);
            g.push_op(
                g.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
            for (i, ty) in extras.iter().enumerate() {
                g.push_op(
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
            ValueType::Ref | ValueType::Unknown => "String",
            ValueType::Float => "f64",
            ValueType::Void => "",
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
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let mut args = vec![receiver];
        for (i, ty) in extras.iter().enumerate() {
            args.push(
                graph
                    .push_op(
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
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("T", "m"),
                args,
                result_ty,
            },
            has_result,
        );
        graph.set_return(graph.startblock, None);

        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_type_state(&type_state);
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
    // unconditionally and provides the runtime funcptr ValueId consumed
    // by the `CallResidual.funcptr` operand.

    /// Same skeleton as `check_indirect_regular_call_kind` but drops
    /// `find_all_graphs_for_tests`, so `candidate_graphs` stays empty
    /// and `guess_call_kind(op)` falls through to `CallKind::Residual`
    /// via the `graphs_from(op)` call.py:137-139 fall-through.
    /// Emit is `[VtableMethodPtr, CallResidual{indirect_targets: None},
    /// Live?]` — the trailing `-live-` appears only when the descriptor's
    /// `can_raise` is true, which is the default for non-elidable
    /// family calls.
    fn check_indirect_residual_call_kind(
        extras: &[ValueType],
        result_ty: ValueType,
        expect: (usize, usize, usize),
        expect_res_kind: char,
    ) {
        use crate::call::CallControl;
        use crate::translate_legacy::annotator::annrpython::annotate;
        use crate::translate_legacy::rtyper::rtyper::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let build_impl = |name: &str| {
            let mut g = FunctionGraph::new(name);
            g.push_op(
                g.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
            for (i, ty) in extras.iter().enumerate() {
                g.push_op(
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
            ValueType::Ref | ValueType::Unknown => "String",
            ValueType::Float => "f64",
            ValueType::Void => "",
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
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let mut args = vec![receiver];
        for (i, ty) in extras.iter().enumerate() {
            args.push(
                graph
                    .push_op(
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
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("T", "m"),
                args,
                result_ty,
            },
            has_result,
        );
        graph.set_return(graph.startblock, None);

        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_type_state(&type_state);
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
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "cell".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
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
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "p".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
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
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_portal_jd(Some(3));
        let ops = transformer.handle_jit_marker__jit_merge_point(
            vec![ValueId(0)],
            vec![],
            vec![],
            vec![ValueId(1), ValueId(2)],
            vec![ValueId(3)],
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
                assert_eq!(greens_i, &vec![ValueId(0)]);
                assert_eq!(reds_i, &vec![ValueId(1), ValueId(2)]);
                assert_eq!(reds_r, &vec![ValueId(3)]);
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
        let greens = vec![ValueId(0), ValueId(1), ValueId(2)];
        let ops = transformer.promote_greens(&greens);
        assert_eq!(ops.len(), 6, "expect 2 ops per green");
        for i in 0..greens.len() {
            assert!(
                matches!(ops[i * 2].kind, OpKind::Live),
                "slot {i} should start with Live"
            );
            match &ops[i * 2 + 1].kind {
                OpKind::GuardValue { value, kind_char } => {
                    assert_eq!(*value, greens[i]);
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
    /// r1])` with two greens and one red, all int-typed via
    /// TypeResolutionState. Expected output (`promote_greens` +
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
        use crate::jit_codewriter::type_state::{ConcreteType, TypeResolutionState};
        use crate::parse::CallPath;

        let g1 = ValueId(10);
        let g2 = ValueId(11);
        let r1 = ValueId(12);
        let receiver = ValueId(99);

        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["test", "portal"]),
            vec!["green1".into(), "green2".into()],
            vec!["red1".into()],
            Vec::new(),
            Vec::new(),
        );

        let mut ts = TypeResolutionState::new();
        ts.concrete_types.insert(g1, ConcreteType::Signed);
        ts.concrete_types.insert(g2, ConcreteType::Signed);
        ts.concrete_types.insert(r1, ConcreteType::Signed);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_portal_jd(Some(0))
            .with_type_state(&ts);

        let args = [receiver, g1, g2, r1];
        let ops = transformer
            .try_handle_jit_marker(JitMarkerKey::JitMergePoint, &args)
            .expect("portal_jd + cc + 2-greens + 1-red satisfies dispatch preconditions");

        assert_eq!(ops.len(), 7, "promote_greens(2 greens)*2 + merge*3 = 7");

        // promote_greens prefix: -live-, int_guard_value(g1), -live-, int_guard_value(g2)
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue { value, kind_char } => {
                assert_eq!(*value, g1);
                assert_eq!(*kind_char, 'i');
            }
            other => panic!("ops[1] expected GuardValue(g1, 'i'), got {other:?}"),
        }
        assert!(matches!(ops[2].kind, OpKind::Live));
        match &ops[3].kind {
            OpKind::GuardValue { value, kind_char } => {
                assert_eq!(*value, g2);
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
                assert_eq!(greens_i, &vec![g1, g2]);
                assert!(greens_r.is_empty());
                assert!(greens_f.is_empty());
                assert_eq!(reds_i, &vec![r1]);
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
        let mut ts = TypeResolutionState::new();
        let arg = ValueId(1);
        ts.concrete_types.insert(arg, ConcreteType::GcRef);
        let transformer = Transformer::new(&config).with_type_state(&ts);
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
        let mut ts = TypeResolutionState::new();
        let arg = ValueId(1);
        ts.concrete_types.insert(arg, ConcreteType::GcRef);
        let transformer = Transformer::new(&config).with_type_state(&ts);
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
        let mut ts = TypeResolutionState::new();
        let arg = ValueId(1);
        ts.concrete_types.insert(arg, ConcreteType::GcRef);
        let transformer = Transformer::new(&config).with_type_state(&ts);
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
        let mut ts = TypeResolutionState::new();
        let arg = ValueId(1);
        ts.concrete_types.insert(arg, ConcreteType::Signed);
        let transformer = Transformer::new(&config).with_type_state(&ts);
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
        let mut ts = TypeResolutionState::new();
        let arg = ValueId(1);
        ts.concrete_types.insert(arg, ConcreteType::GcRef);
        let transformer = Transformer::new(&config).with_type_state(&ts);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::function_path(["Result", "Ok"]),
            &[arg],
            &ValueType::Ref,
        ));
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Foo"),
            &[arg],
            &ValueType::Ref,
        ));
        assert!(transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Err"),
            &[arg],
            &ValueType::Ref,
        ));
        assert!(transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Some"),
            &[arg],
            &ValueType::Ref,
        ));
    }
}
