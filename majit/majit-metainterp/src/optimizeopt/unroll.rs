/// Loop unrolling pass (peel one iteration).
///
/// Detects loops ending with a `Jump` back-edge and peels one iteration:
/// the loop body is duplicated to create a "preamble" that executes once
/// before the main loop. Guards in the peeled preamble serve as initial
/// type checks, enabling downstream passes to remove redundant guards
/// from the main loop body.
///
/// The peeled structure looks like:
///
/// ```text
/// [peeled body]          ← first iteration (preamble), guards act as type checks
///   Label(...)           ← loop header
/// [original body]        ← main loop body
///   Jump(...)            ← back-edge to Label
/// ```
///
/// OpRefs in the peeled iteration are remapped to new positions so they
/// don't collide with the original ops.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use majit_ir::{DescrRef, Op, OpCode, OpRef, Type, Value};

use crate::optimizeopt::{
    OptContext, Optimization, OptimizationResult, SnapshotBoxes, SnapshotFramePcs,
    SnapshotFrameSizes, next_snapshot_pos, snapshot_get, snapshot_insert,
};
use crate::resume::SnapshotBox;

fn is_trace_constant_ref(opref: OpRef, constants: &HashMap<u32, i64>) -> bool {
    !opref.is_none() && constants.contains_key(&opref.raw())
}

fn is_trace_runtime_ref(opref: OpRef, constants: &HashMap<u32, i64>) -> bool {
    !opref.is_none() && !is_trace_constant_ref(opref, constants)
}

fn p1_full_prefix_from_box_pool_snapshot(
    snapshot: Option<&[crate::r#box::BoxRef]>,
) -> Option<Vec<crate::r#box::BoxRef>> {
    snapshot.map(|pool| pool.to_vec())
}

/// unroll.py: UnrollOptimizer — high-level loop optimization controller.
///
/// Wraps the streaming OptUnroll pass with RPython's UnrollOptimizer API:
/// - optimize_preamble: process and optimize the first iteration
/// - optimize_peeled_loop: optimize the main loop body
pub struct UnrollOptimizer {
    /// The short preamble from the preamble optimization pass.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// history.py: JitCellToken.target_tokens — compiled versions of this loop.
    /// Each TargetToken has its own virtual state and short preamble.
    pub target_tokens: Vec<TargetToken>,
    /// history.py: JitCellToken.retraced_count — number of times this loop
    /// has been retraced. Compared against retrace_limit to prevent infinite
    /// retracing.
    pub retraced_count: u32,
    /// warmstate.py: retrace_limit parameter. When retraced_count reaches
    /// this limit, jump_to_preamble is forced instead of creating a new
    /// target token.
    pub retrace_limit: u32,
    /// warmstate.py: max_retrace_guards parameter. If a compiled trace has
    /// more guards than this, retracing is permanently disabled.
    pub max_retrace_guards: u32,
    /// Constant type hints from ConstantPool, propagated to inner Optimizer.
    pub constant_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// compile.py:362: pre-imported ExportedState for compile_retrace.
    /// When set, Phase 1 (preamble) is skipped and Phase 2 uses this state
    /// directly, matching UnrolledLoopData.optimize → optimize_peeled_loop.
    pub imported_state: Option<ExportedState>,
    /// Phase 1's finalized ExportedState, retained across the Phase 2 run
    /// so the caller of `optimize_trace_*` can consult the renamed inputarg
    /// types that the optimizer decided on. RPython does not need this
    /// (Box.type is intrinsic), but majit's `InputArg.tp` side table is
    /// otherwise disconnected from the optimizer's reduced LABEL.
    pub final_exported_state: Option<ExportedState>,
    // RPython compile.py:278-284: Phase 1 results for retrace_needed.
    // In RPython, Phase 1 and Phase 2 are separate calls, so Phase 1
    // results are naturally accessible. In pyre, Phase 1 results are
    // returned via the phase1_out output parameter to the caller's
    // stack frame (survives Phase 2 panic).
    /// resume.py parity: per-guard snapshot boxes from tracing time.
    /// Passed through to Phase 1 and Phase 2 optimizers for
    /// store_final_boxes_in_guard snapshot-based fail_args rebuild.
    pub snapshot_boxes: SnapshotBoxes,
    /// Per-frame box counts for multi-frame snapshots.
    pub snapshot_frame_sizes: SnapshotFrameSizes,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: SnapshotBoxes,
    /// Per-guard virtualref boxes from tracing-time snapshots
    /// (resume.py:243-247 vref_array — _number_boxes consumes them
    /// after the virtualizable array).
    pub snapshot_vref_boxes: SnapshotBoxes,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: SnapshotFramePcs,
    /// pyjitpl.py:2289 all_descrs: dense list indexed by descr_index.
    /// Threaded through inner Optimizer instances for inline registration.
    pub all_descrs: Vec<majit_ir::descr::DescrRef>,
    /// RPython parity: GcRef constants that need Ref type for resume data.
    pub numbering_type_overrides: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython Box type parity: trace inputarg types from recorder.
    /// Each RPython Box carries its type; in majit OpRef is untyped u32.
    /// Propagated to Phase 1 and Phase 2 Optimizer.trace_inputarg_types
    /// so value_types covers inputarg OpRefs.
    pub trace_inputarg_types: Vec<majit_ir::Type>,
    /// Phase 1 emit ops (filtered to non-NONE pos, non-Void type) carried
    /// into Phase 2 so that `OptContext::op_at` resolves Phase 1 OpRefs
    /// directly via `op.type_` (history.py:220 box.type parity).
    phase1_emit_ops: Vec<majit_ir::Op>,
    /// RPython: same Optimizer instance across phases keeps patchguardop.
    /// In majit, separate instances — forward explicitly.
    phase1_patchguardop: Option<majit_ir::Op>,
    /// Cross-phase fresh OpRef high water (majit-specific companion to
    /// RPython's `TraceIterator._index`).
    ///
    /// In RPython each `TraceIterator.next()` allocates a fresh `cls()`
    /// ResOperation whose Python identity distinguishes Phase 1 from
    /// Phase 2 boxes; majit's `OpRef::from_raw(u32)` IS the identity, so Phase 2 must
    /// continue allocating *above* Phase 1's high water mark to keep the
    /// two phases' OpRef sets disjoint. After Phase 1 finishes,
    /// `next_global_opref` holds the smallest OpRef Phase 2 may emit; it
    /// is the `start_fresh` argument the next `TraceIterator::new` call
    /// (or bridge entry) should use. Initialized to 0.
    #[allow(dead_code)]
    pub(crate) next_global_opref: u32,
    /// RPython metainterp_sd.callinfocollection parity.
    /// Maps oopspec indices to (calldescr, func_ptr) for generate_modified_call.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// compile.py:221 + optimizer.py:530: call_pure_results from tracing.
    /// Passed through to the inner Optimizer for cross-iteration CALL_PURE folding.
    pub call_pure_results: std::collections::HashMap<Vec<majit_ir::Value>, majit_ir::Value>,
}

impl UnrollOptimizer {
    pub fn new() -> Self {
        UnrollOptimizer {
            short_preamble: None,
            target_tokens: Vec::new(),
            retraced_count: 0,
            constant_types: std::collections::HashMap::new(),
            numbering_type_overrides: std::collections::HashMap::new(),
            retrace_limit: 5,
            max_retrace_guards: 15,
            imported_state: None,
            final_exported_state: None,
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: Vec::new(),
            snapshot_vable_boxes: Vec::new(),
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: Vec::new(),
            all_descrs: Vec::new(),
            trace_inputarg_types: Vec::new(),
            phase1_emit_ops: Vec::new(),
            phase1_patchguardop: None,
            next_global_opref: 0,
            callinfocollection: None,
            call_pure_results: std::collections::HashMap::new(),
        }
    }

    /// unroll.py: optimize_preamble(trace, runtime_boxes)
    /// Optimize the preamble (first iteration) of a loop trace.
    /// Returns the optimized preamble ops + the peeled loop ops.
    pub fn optimize_preamble(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        optimizer.propagate_all_forward(ops)
    }

    /// unroll.py: optimize_peeled_loop(trace)
    /// Optimize the loop body AFTER preamble peeling.
    /// The peeled preamble has already established the type/class/bounds
    /// information; this method optimizes the repeating body.
    pub fn optimize_peeled_loop(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        optimizer.propagate_all_forward(ops)
    }

    /// unroll.py:238-242: jump_to_preamble(cell_token, jump_op).
    ///
    /// Redirect the closing JUMP to the preamble entry token
    /// (target_tokens[0], virtual_state=None). Only changes the
    /// descriptor, keeping arglist intact — RPython parity.
    pub fn jump_to_preamble(body_ops: &[Op], preamble_target: &TargetToken) -> Vec<Op> {
        assert!(
            preamble_target.virtual_state.is_none(),
            "jump_to_preamble expects the start/preamble target token"
        );
        let mut result = body_ops.to_vec();
        if let Some(jump) = result.iter_mut().rfind(|op| op.opcode == OpCode::Jump) {
            jump.descr = Some(preamble_target.as_jump_target_descr());
        }
        result
    }

    fn ensure_preamble_target_token(&mut self) {
        if self
            .target_tokens
            .first()
            .is_some_and(|token| token.virtual_state.is_none())
        {
            return;
        }
        self.target_tokens.insert(0, TargetToken::new_preamble(0));
    }

    /// unroll.py: optimize_trace(trace, call_pure_results)
    /// Full trace optimization: peel → optimize preamble → optimize body.
    /// Returns the optimized peeled+body trace.
    pub fn optimize_trace(&mut self, ops: &[Op]) -> Vec<Op> {
        let result = self.optimize_preamble(ops);
        // After peeling, extract short preamble from the result.
        let sp = crate::optimizeopt::shortpreamble::extract_short_preamble(&result);
        if !sp.is_empty() {
            self.short_preamble = Some(sp);
        }
        result
    }

    /// unroll.py: optimize_trace_with_constants
    /// Same as optimize_trace but with known constants.
    pub fn optimize_trace_with_constants(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
    ) -> Vec<Op> {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        let result = optimizer.optimize_with_constants(ops, constants);
        let sp = crate::optimizeopt::shortpreamble::extract_short_preamble(&result);
        if !sp.is_empty() {
            self.short_preamble = Some(sp);
        }
        result
    }

    /// optimize_trace with constants AND explicit num_inputs.
    /// compile.py: compile_loop → optimize with preamble peeling.
    pub fn optimize_trace_with_constants_and_inputs(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
    ) -> (Vec<Op>, usize) {
        self.optimize_trace_with_constants_and_inputs_vable(ops, constants, num_inputs, None)
    }

    /// compile.py:275-308: compile_loop — 2-phase preamble peeling.
    /// compile.py:275-338: 2-phase preamble peeling (RPython parity).
    ///
    /// Phase 1 (optimize_preamble): full pipeline on trace → preamble_ops.
    /// export_state: capture the preamble's exported optimizer state.
    /// Phase 2 (optimize_peeled_loop): import_state + full pipeline → body_ops.
    /// Assembly: [preamble_no_jump] + Label(label_args) + [body_with_jump].
    pub fn optimize_trace_with_constants_and_inputs_vable(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        vable_config: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
    ) -> (Vec<Op>, usize) {
        self.optimize_trace_with_constants_and_inputs_vable_out(
            ops,
            constants,
            num_inputs,
            vable_config,
            None,
        )
    }

    /// Same as optimize_trace_with_constants_and_inputs_vable but with an
    /// output parameter for Phase 1 results. RPython compile.py:278-294
    /// parity: Phase 1 results (preamble_ops + exported_state) are written
    /// to `phase1_out` before Phase 2 starts. If Phase 2 panics, the caller
    /// still has the Phase 1 results for retrace_needed.
    pub fn optimize_trace_with_constants_and_inputs_vable_out(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        vable_config: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
        phase1_out: Option<&mut Option<(Vec<Op>, ExportedState)>>,
    ) -> (Vec<Op>, usize) {
        // compile.py:362: if imported_state is pre-set (compile_retrace path),
        // skip Phase 1 and go directly to Phase 2 with the imported state.
        let (mut exported_state, consts_p1, p1_ops, p1_full_prefix) = if let Some(pre_imported) =
            self.imported_state.take()
        {
            // RPython uses object identity for Boxes, so a TraceIterator over a
            // retrace can never numerically collide with the already optimized
            // partial preamble. Majit's OpRef is the identity; when Phase 1 is
            // skipped, recover the partial preamble high-water from the imported
            // state before allocating Phase 2 input/result OpRefs.
            self.next_global_opref = self
                .next_global_opref
                .max(num_inputs as u32)
                .max(pre_imported.opref_high_water());
            // Retrace path: Phase 1 already done; preamble ops are in
            // the caller's partial_trace, not produced here.
            // RPython: same Optimizer persists patchguardop. Recover here.
            if self.phase1_patchguardop.is_none() {
                self.phase1_patchguardop = pre_imported.patchguardop.clone();
            }
            // compile_retrace has no `opt_p1.final_ctx` (Phase 1 was
            // skipped), but the failed attempt's Phase 1 captured its full
            // `box_pool` into `ExportedState.box_pool_snapshot` at export
            // time (`export_state_with_bounds`). TraceIterator's
            // `p1_full_prefix` contract: deliver the full Phase 1 box_pool
            // including inputarg BoxRefs at `[0..num_inputs)`, so
            // `import_state` setting `Forwarded::Box(target)` for
            // `target.raw() < num_inputs` lands the chain on the same
            // Phase 1 inputarg cell — PyPy Phase 2 reads Phase 1
            // `_forwarded` through shared Box identity; preserving the full
            // raw position alignment (inputargs + emit ops) reproduces
            // that identity here.
            let prefix =
                p1_full_prefix_from_box_pool_snapshot(pre_imported.box_pool_snapshot.as_deref());
            (pre_imported, constants.clone(), Vec::new(), prefix)
        } else {
            // ── Phase 1: PreambleCompileData.optimize() ──
            // ── Phase 1: optimize_preamble (compile.py:275-276) ──
            let mut consts_p1 = constants.clone();
            let mut opt_p1 = match vable_config.as_ref() {
                Some(c) => {
                    crate::optimizeopt::optimizer::Optimizer::default_pipeline_with_virtualizable(
                        c.clone(),
                    )
                }
                None => crate::optimizeopt::optimizer::Optimizer::default_pipeline(),
            };
            opt_p1.all_descrs = std::mem::take(&mut self.all_descrs);
            opt_p1.constant_types = self.constant_types.clone();
            opt_p1.callinfocollection = self.callinfocollection.clone();
            opt_p1.numbering_type_overrides = self.numbering_type_overrides.clone();
            opt_p1.trace_inputarg_types = self.trace_inputarg_types.clone();
            opt_p1.snapshot_boxes = self.snapshot_boxes.clone();
            opt_p1.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
            opt_p1.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
            opt_p1.snapshot_vref_boxes = self.snapshot_vref_boxes.clone();
            opt_p1.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
            opt_p1.call_pure_results = self.call_pure_results.clone();
            // RPython optimize_preamble (unroll.py:101-103): flush=False.
            // JUMP/FINISH is NOT sent through the pass pipeline; it's
            // returned in info.jump_op for Phase 2 to consume.
            opt_p1.skip_flush = true;
            // RPython unroll.py:101-103 `optimize_preamble` calls
            // `propagate_all_forward(trace.get_iter())`. `trace.get_iter()`
            // is a fresh `TraceIterator` whose `next()` produces a freshly
            // allocated `cls()` ResOperation for every visited op.
            //
            // Phase 1 routes the input ops through `TraceIterator::new`
            // with `start_fresh = 0`. The recorder emits ops at
            // monotonically increasing positions starting from
            // `num_inputs` (recorder.rs `record_op` uses `op_count` for
            // BOTH inputargs and ops, and ops follow inputargs), and
            // `TraceIterator::next` allocates fresh OpRefs from `_fresh`
            // which is also seeded at `num_inputs` after the inputarg
            // pre-seed loop. Both void and non-void ops advance `_fresh`
            // (see opencoder.rs::next), so the freshly produced OpRef
            // sequence is bit-identical to the input — this wrap is a
            // structural alignment with RPython's `trace.get_iter()`
            // call site, not a functional change.
            // opencoder.py:264 `inputarg_from_tp(arg.type)` — fresh inputargs
            // are typed via `self.trace_inputarg_types`, the recorder-supplied
            // per-arg type list (length must equal `num_inputs`).
            let p1_inputarg_types: &[majit_ir::Type] = &self.trace_inputarg_types;
            debug_assert_eq!(p1_inputarg_types.len(), num_inputs);
            let mut p1_iter = crate::opencoder::TraceIterator::new(
                ops,
                0,
                ops.len(),
                None,
                p1_inputarg_types,
                0,    // start_fresh = 0 — inputargs at [0..num_inputs)
                None, // p1_full_prefix — Phase 1 has no prior phase
            );
            let mut p1_ops_in: Vec<Op> = Vec::with_capacity(ops.len());
            while let Some(op) = p1_iter.next() {
                p1_ops_in.push(op);
            }
            let p1_iter_fresh_hw = p1_iter._fresh;
            opt_p1.runtime_boxes = ops
                .iter()
                .rfind(|op| op.opcode == OpCode::Jump)
                .map(|op| op.args.to_vec())
                .unwrap_or_default();
            // Slice 77b.B: hand opt_p1 the per-iter BoxRef pool that p1_iter
            // allocated (slice 77b.A). trace.get_iter() per-call
            // inputarg_from_tp(...) / cls() — each phase optimizes against a
            // fresh Box identity set so _forwarded mutations cannot alias
            // across phases.
            //
            // Const BoxRefs are NOT cached: `mirror_forwarded_to_box` and
            // `get_box_replacement_box` allocate `BoxRef::new_const(value)`
            // per call from `const_pool` (`history.py:220` ConstInt(value)
            // per-call-site parity). opt_p1's entry path seeds `const_pool`
            // from the shared `constants` map (`optimizer.rs:1944`).
            opt_p1.set_pending_box_pool(p1_iter.box_pool.clone());
            let p1_ops =
                opt_p1.optimize_with_constants_and_inputs(&p1_ops_in, &mut consts_p1, num_inputs);
            // RPython parity: Phase 1 optimizer may discover new constants
            // via make_constant (e.g., constant-folded heap reads, guard
            // class pointers). These live in ctx.constants but not in
            // consts_p1 (which was only seeded from the input constants).
            // Merge them back so build_short_preamble_from_exported_boxes
            // can capture all constants referenced by short preamble ops.
            if let Some(ref final_ctx) = opt_p1.final_ctx {
                crate::optimizeopt::optimizer::merge_backend_constants_from_ctx(
                    final_ctx,
                    &mut consts_p1,
                );
            }
            let p1_ni = opt_p1.final_num_inputs();

            match opt_p1.exported_loop_state.take() {
                Some(mut state) => {
                    // end_arg_types is already populated by
                    // `Optimizer::optimize_with_constants_and_inputs_at`
                    // using the optimizer-visible `ctx.opref_type()` (see
                    // optimizer.rs:2405-2416). Overwriting it here with
                    // `opcode.result_type()` + `Type::Ref` default would
                    // retype Phase 1 inputarg OpRefs (absent from `p1_ops`)
                    // as `Ref`, which later feeds the cross-type forward
                    // assertion in `OptContext::replace_op`.
                    // RPython Phase 1 → Phase 2 heap cache transfer:
                    // RPython does NOT serialize heap cache in ExportedState.
                    // HeapOps in the short preamble are replayed during Phase 2's
                    // inline_short_preamble, populating the heap cache naturally
                    // through OptHeap. serialize_optheap is only for bridgeopt.
                    // opencoder.py:271 _index parity: Phase 2's TraceIterator
                    // must allocate fresh boxes ABOVE Phase 1's high water
                    // mark so the two phases' OpRef namespaces are disjoint
                    // (RPython relies on Python identity to distinguish them;
                    // majit relies on disjoint integer ranges). The high
                    // water is `final_ctx.next_pos` after Phase 1 emit, with
                    // a floor of `num_inputs` for empty traces.
                    // Phase 1's emit-side high water (`final_ctx.next_pos`)
                    // reflects only the positions `ctx.reserve_pos` handed
                    // out; the per-iteration TraceIterator additionally
                    // allocates fresh OpRefs via its `_fresh` counter and
                    // those values can exceed `next_pos` for traces where
                    // Phase 1 dropped or folded many ops. Taking the max
                    // with `p1_iter_fresh_hw` keeps Phase 2's inputarg base
                    // strictly above every OpRef Phase 1 ever allocated,
                    // so Phase 2's own fresh OpRefs cannot collide with
                    // positions in `phase1_emit_ops` / typed snapshots.
                    self.next_global_opref = opt_p1
                        .final_ctx
                        .as_ref()
                        .map(|c| c.next_pos)
                        .unwrap_or(num_inputs as u32)
                        .max(num_inputs as u32)
                        .max(p1_iter_fresh_hw);
                    // RPython Box type parity: Phase 1's emit ops carry
                    // `op.type_` intrinsically; Phase 2's `op_at` reads it
                    // directly to resolve cross-phase OpRefs that appear as
                    // imported_label_args / fail_args / record_same_as
                    // sources (history.py:220 parity).
                    self.phase1_emit_ops = std::mem::take(&mut opt_p1.phase1_emit_ops);
                    // RPython: same Optimizer instance keeps patchguardop.
                    if self.phase1_patchguardop.is_none() {
                        self.phase1_patchguardop = opt_p1.patchguardop.clone();
                    }
                    state.patchguardop = opt_p1.patchguardop.clone();
                    self.all_descrs = std::mem::take(&mut opt_p1.all_descrs);
                    // Box.type parity: retain Phase 1's renamed_inputarg_types
                    // so the backend can see the reduced LABEL's declared
                    // types. Clone only the field we need to avoid keeping
                    // Phase 1 short preamble data alive longer than before.
                    self.final_exported_state = Some(ExportedState {
                        end_args: Vec::new(),
                        next_iteration_args: Vec::new(),
                        end_arg_types: Vec::new(),
                        virtual_state: state.virtual_state.clone(),
                        exported_infos: std::collections::HashMap::new(),
                        exported_short_boxes: Vec::new(),
                        short_boxes: Vec::new(),
                        short_box_const_values: std::collections::HashMap::new(),
                        short_preamble: None,
                        renamed_inputargs: state.renamed_inputargs.clone(),
                        renamed_inputarg_types: state.renamed_inputarg_types.clone(),
                        short_inputargs: Vec::new(),
                        runtime_boxes: Vec::new(),
                        patchguardop: None,
                        phase1_emit_high_water: state.phase1_emit_high_water,
                        box_pool_snapshot: None,
                        rooted_refs: Vec::new(),
                        shadow_stack_base: 0,
                    });
                    // Codex plan step 3: snapshot Phase 1 emit-op slice as
                    // Rc::clone vector. Phase 2's TraceIterator embeds these
                    // BoxRefs at its placeholder prefix `[num_inputs..end)`
                    // so chain walks via `Forwarded::Box(rc)` observe Phase 1's
                    // accumulated `_forwarded` info naturally — `unroll.py`
                    // Phase 2 reading Phase 1 box._forwarded through shared
                    // Python identity parity. After Phase 1 finishes Phase 2
                    // is the sole writer of these Rcs (audit memo Risk 1).
                    //
                    // Slice end is `box_pool.len()` to cover the full Phase 1
                    // allocation high water — Phase 2's `next_global_opref` is
                    // `max(next_pos, p1_iter_fresh_hw)` which can exceed
                    // `next_pos` when Phase 1 dropped/folded ops (the iterator
                    // advanced `_fresh` past those positions while
                    // `OptContext::next_pos` only counts `new_operations`).
                    // `p1_iter.box_pool` was seeded into opt_p1 via
                    // `set_pending_box_pool` so `final_ctx.box_pool` carries
                    // the iterator's full allocation range; transferring up to
                    // `box_pool.len()` ensures Phase 2 chain walks observe the
                    // real Phase 1 BoxRefs at every reachable position rather
                    // than Type::Void placeholders.
                    let p1_full_prefix: Option<Vec<crate::r#box::BoxRef>> =
                        opt_p1.final_ctx.as_ref().map(|c| c.box_pool.to_vec());
                    (state, consts_p1, p1_ops, p1_full_prefix)
                }
                None => {
                    *constants = consts_p1;
                    // RPython: compile_loop uses flush=True — terminal op
                    // (Finish/Jump) goes through the pass pipeline normally.
                    // majit: flush=False stores it in terminal_op; restore it
                    // here for non-peeled traces that return directly.
                    let mut ops = p1_ops;
                    if let Some(terminal) = opt_p1.terminal_op.take() {
                        ops.push(terminal);
                    }
                    let loop_arity = closing_loop_contract_arity(&ops, p1_ni);
                    return (ops, loop_arity);
                }
            }
        };
        // unroll.py:454 end_args carry type via Box; export_state already
        // populated `exported_state.end_arg_types` from ctx.
        // RPython parity: Phase 2 needs patchguardop from Phase 1's
        // GuardFutureCondition (unroll.py:333). Extract before dropping opt_p1.
        let p1_patchguardop = exported_state.patchguardop.clone();

        self.ensure_preamble_target_token();
        // ── Phase 2: optimize_peeled_loop (compile.py:291-292) ──
        let body_num_inputs = num_inputs;

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] preamble peeling: {} virtual(s), phase1 end_args={} p1_patchguardop={}",
                exported_state
                    .virtual_state
                    .state
                    .iter()
                    .filter(|s| s.is_virtual())
                    .count(),
                exported_state.end_args.len(),
                p1_patchguardop
                    .as_ref()
                    .map(|p| p.rd_resume_position)
                    .unwrap_or(-99),
            );
        }

        // opencoder.py:259-404 parity: Phase 2 uses the same ops as Phase 1.
        // RPython TraceIterator creates fresh Box objects per phase — each
        // iterator has its own _cache, so Phase 2 results never collide with
        // Phase 1. In majit, Phase 2 gets a separate OptContext, achieving
        // the same isolation.
        let mut consts_p2 = consts_p1.clone();

        let mut opt_p2 = match vable_config.as_ref() {
            Some(c) => {
                crate::optimizeopt::optimizer::Optimizer::default_pipeline_with_virtualizable(
                    c.clone(),
                )
            }
            None => crate::optimizeopt::optimizer::Optimizer::default_pipeline(),
        };
        opt_p2.all_descrs = std::mem::take(&mut self.all_descrs);
        opt_p2.constant_types = self.constant_types.clone();
        opt_p2.callinfocollection = self.callinfocollection.clone();
        opt_p2.numbering_type_overrides = self.numbering_type_overrides.clone();
        opt_p2.trace_inputarg_types = self.trace_inputarg_types.clone();
        opt_p2.phase1_emit_ops = self.phase1_emit_ops.clone();
        opt_p2.snapshot_boxes = self.snapshot_boxes.clone();
        opt_p2.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
        opt_p2.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
        opt_p2.snapshot_vref_boxes = self.snapshot_vref_boxes.clone();
        opt_p2.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
        opt_p2.call_pure_results = self.call_pure_results.clone();
        // RPython: same Optimizer instance keeps patchguardop across phases.
        // Phase 1 processes GUARD_FUTURE_CONDITION (from close_loop_args_at)
        // which sets patchguardop. optimizer.py:294 parity — no synthetic
        // fallback; the actual GFC provides rd_resume_position.
        opt_p2.patchguardop = self.phase1_patchguardop.clone();
        // gcreftracer.py parity: root GcRef values on the shadow stack.
        // RPython: single Python object — GC traces automatically.
        // Rust: LIFO shadow stack requires longer-lived roots at lower depth.
        //
        // Order: (1) phase1_out clone rooted first (survives beyond this
        // function for retrace — pyre-specific panic safety backup).
        // (2) original rooted second (lives until opt_p2 drops at Phase 2
        // end — shorter-lived, higher depth, dropped first).
        if let Some(out) = phase1_out {
            let mut backup = exported_state.clone();
            backup.root_all_gcrefs();
            *out = Some((p1_ops.clone(), backup));
        }
        exported_state.root_all_gcrefs();
        opt_p2.imported_loop_state = Some(exported_state);
        // Set imported_virtuals so Phase 2 intercepts GetfieldGcR(pool)
        // and sets up VirtualStruct PtrInfo for the imported head.
        // Virtual structure is derived from VirtualState (ExportedState).
        opt_p2.imported_virtuals =
            build_imported_virtuals_from_state(opt_p2.imported_loop_state.as_ref().unwrap());
        // RPython: propagate_all_forward(trace, flush=False) for Phase 2.
        // Don't flush lazy sets — virtuals remain virtual until JUMP handling.
        opt_p2.skip_flush = true;
        // RPython parity: Phase 2 DOES virtualize New(). Guard recovery uses
        // rd_virtuals (generated by finalize_guard_resume_data)
        // for virtual materialization on guard failure.
        // Previously disabled (set_phase2(true)) due to missing rd_virtuals;
        // now enabled after compile.rs rd_virtuals→rd_virtuals generation.
        // RPython parity: Phase 2 imports heap cache via short preamble
        // RPython: Phase 2 heap cache is populated by inline_short_preamble
        // replaying HeapOps through send_extra_operation.
        if std::env::var_os("MAJIT_LOG").is_some() {
            let gc_before = ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!(
                "[jit] phase 2 input: {} ops, {} guards, body_ni={}",
                ops.len(),
                gc_before,
                body_num_inputs
            );
        }

        // opencoder.py:249-406 TraceIterator parity for Phase 2.
        //
        // RPython's `optimize_peeled_loop` calls `trace.get_iter()` which
        // constructs a FRESH TraceIterator whose `__init__` allocates new
        // Box objects for every inputarg (inputarg_from_tp) and whose
        // `next()` allocates new cls() ResOperation instances for every
        // emitted op. Each iteration over the same trace produces a
        // completely disjoint set of Python identities; the cache
        // `_cache[raw_position]` records the per-iteration fresh box so
        // later references resolve to the iteration-local identity.
        //
        // majit's `OpRef::from_raw(u32)` IS the identity, so "fresh per iteration"
        // means disjoint integer ranges. Phase 2 must not emit op results
        // at OpRefs that collide with Phase 1's emitted positions — doing
        // so reintroduces the box-identity collision that the reactive
        // check at `mod.rs::emit` (collision detection + forwarding
        // redirect) currently compensates for.
        //
        // This step (Commit D2 of the Box identity plan): run Phase 2
        // ops through TraceIterator with `start_index = next_global_opref`,
        // which gives BOTH inputargs AND op results fresh OpRefs in a
        // disjoint range. Phase 2 inputargs live at
        // `[next_global_opref..next_global_opref+body_num_inputs)` and
        // op results at `[next_global_opref+body_num_inputs..)`. This is
        // the RPython-literal model where each TraceIterator produces
        // freshly allocated InputArg and ResOp Python instances for
        // every iteration — `import_state`'s `assert source is not
        // target` (unroll.py:483) now holds by construction because the
        // Phase 2 source slot OpRefs are always distinct from any
        // Phase 1 end_arg OpRef.
        //
        // After optimization, Phase 2 output is post-translated back to
        // the shared `[0..body_num_inputs)` inputarg layout for the
        // final assembly (see the `shift_back` pass below). The
        // assembly and downstream consumers continue to assume the
        // shared-inputarg layout; only the Phase 2 optimizer internals
        // see the disjoint range.
        let phase2_inputarg_base = self.next_global_opref.max(body_num_inputs as u32);
        // opencoder.py:264 `inputarg_from_tp(arg.type)` — same per-arg types
        // as Phase 1 (Phase 2 walks the body half of the same trace).
        let p2_inputarg_types: &[majit_ir::Type] = &self.trace_inputarg_types;
        debug_assert_eq!(p2_inputarg_types.len(), body_num_inputs);
        let mut iter = crate::opencoder::TraceIterator::new(
            &ops,
            0,
            ops.len(),
            None,
            p2_inputarg_types,
            phase2_inputarg_base, // fresh inputargs at [phase2_inputarg_base..)
            p1_full_prefix, // Codex plan step 3 slice B: full Phase 1 box_pool (inputargs + emit ops)
        );
        let mut p2_ops_in: Vec<Op> = Vec::with_capacity(ops.len());
        while let Some(op) = iter.next() {
            p2_ops_in.push(op);
        }
        let p2_high_water = iter._fresh;
        let p2_cache = iter._cache;
        // opencoder.py:286-289 `_get(self, i)` parity. `p2_cache[raw_pos]`
        // holds the fresh per-iteration box for every Phase 1 input/op
        // position that Phase 2's TraceIterator walks. snapshot_boxes /
        // snapshot_vable_boxes are populated during Phase 1 against
        // those same Phase 1 positions, so each entry must hit the
        // cache. A miss means the snapshot references a Phase 1 OpRef
        // that does not appear in Phase 2's input ops (e.g. a stale
        // reference to a Phase 1-DCE'd op). The previous
        // `unwrap_or(opref)` silently leaked such stale OpRefs into
        // Phase 2 namespace; production probes (nbody / fannkuch /
        // fib_loop / fib_recursive / spectral_norm) show zero misses,
        // so promote to a strict panic matching RPython's _get assert.
        let translate_opref = |opref: OpRef| -> OpRef {
            if opref.is_none() || opref.is_constant() {
                return opref;
            }
            p2_cache
                .get(opref.raw() as usize)
                .copied()
                .flatten()
                .unwrap_or_else(|| {
                    panic!(
                        "phase2 snapshot remap cache miss for {opref:?} \
                         (cache_len={} body_ni={} phase2_inputarg_base={})",
                        p2_cache.len(),
                        body_num_inputs,
                        phase2_inputarg_base,
                    )
                })
        };
        for boxes in opt_p2.snapshot_boxes.iter_mut().flatten() {
            for r in boxes.iter_mut() {
                *r = r.map_opref(translate_opref);
            }
        }
        for boxes in opt_p2.snapshot_vable_boxes.iter_mut().flatten() {
            for r in boxes.iter_mut() {
                *r = r.map_opref(translate_opref);
            }
        }
        for boxes in opt_p2.snapshot_vref_boxes.iter_mut().flatten() {
            for r in boxes.iter_mut() {
                *r = r.map_opref(translate_opref);
            }
        }
        // Phase 1's emitted ops are already in Phase 1's emitted
        // namespace `[num_inputs..next_global_opref)`. Phase 2 body may
        // reference these via `imported_label_args`. They are NOT in the
        // `p2_cache` (only raw trace positions are), so leave
        // `phase1_emit_ops` untranslated.
        // Slice 77b.B: hand opt_p2 the per-iter BoxRef pool that the Phase 2
        // iter allocated. Disjoint from opt_p1's pool — _forwarded mutations
        // recorded against Phase 1 boxes do not alias Phase 2 boxes for the
        // same OpRef raw index, fixing the Rc<Box> split-brain that broke
        // the first 77b attempt.
        //
        // Const BoxRefs: see opt_p1 plumb above — fresh per-call from
        // `const_pool` via `BoxRef::new_const(value)`, no dedup.
        opt_p2.set_pending_box_pool(iter.box_pool.clone());
        let mut p2_ops = opt_p2.optimize_with_constants_and_inputs_at(
            &p2_ops_in,
            &mut consts_p2,
            body_num_inputs,
            phase2_inputarg_base, // inputarg_base — Phase 2 inputargs at [phase2_inputarg_base..)
            p2_high_water,
        );
        // RPython parity: by the time Phase 2 reaches the assembly path,
        // body op args have already been canonicalized through
        // `get_box_replacement` (optimizer.rs:2932-2940).  The flat OpRef
        // port keeps Phase 2 source slots in a disjoint namespace
        // `[phase2_inputarg_base..)` during optimization; re-resolve the
        // finalized body ops here so no raw Phase 2 inputarg slot leaks into
        // `assemble_peeled_trace_with_jump_args`.
        if let Some(final_ctx) = opt_p2.final_ctx.as_ref() {
            for op in &mut p2_ops {
                apply_box_replacements(op, final_ctx);
            }
        }
        // Post-translate Phase 2 output back to the shared-inputarg
        // layout expected by `assemble_peeled_trace_with_jump_args`.
        //
        // Phase 2's OptContext internally used disjoint inputarg OpRefs
        // at `[phase2_inputarg_base..phase2_inputarg_base+body_num_inputs)`
        // so the RPython `import_state` "source is not target"
        // (unroll.py:483) invariant held by construction. The final
        // assembled trace, however, uses shared body inputargs at
        // `[0..body_num_inputs)` so that the preamble and body share
        // the same inputarg slots — the `Label(label_args)` op at the
        // body boundary binds them to the preamble's end values, and
        // downstream consumers (Cranelift regalloc, resume data) see
        // a single consistent inputarg numbering.
        //
        // This shift only touches op args, fail_args, and the label /
        // source_slot vectors. Op positions (the emitted results) are
        // already in `[p2_high_water..)`, which is above both the
        // shared range `[0..body_num_inputs)` and the Phase 1 emitted
        // range `[num_inputs..next_global_opref)`, so they stay put.
        // D2 shift_back shim removed: the assembly now natively handles
        // disjoint Phase 2 inputarg OpRefs via the inputarg_base parameter.
        // Phase 2 inputarg OpRefs at [phase2_inputarg_base..+body_num_inputs)
        // flow directly into the assembly without translation.
        // Phase 2 may discover new constants via make_constant (e.g., guard
        // class pointers from collect_use_box_guards).
        // Merge back into consts_p2 so the backend can resolve them.
        if let Some(ref final_ctx) = opt_p2.final_ctx {
            crate::optimizeopt::optimizer::merge_backend_constants_from_ctx(
                final_ctx,
                &mut consts_p2,
            );
        }
        let body_terminal_op = opt_p2.terminal_op.clone();
        let p2_ni = opt_p2.final_num_inputs();
        self.all_descrs = std::mem::take(&mut opt_p2.all_descrs);
        let label_args = opt_p2
            .imported_label_args
            .clone()
            .expect("phase 2 missing import_state label_args");

        if std::env::var_os("MAJIT_LOG").is_some() {
            for op in &p2_ops {
                if op.opcode.is_guard() {
                    let rd_numb_len = op.resolved_rd_numb().map(|s| s.len()).unwrap_or(0);
                    if let Some(ref fa) = op.fail_args {
                        let fa_raw: Vec<String> = fa
                            .iter()
                            .map(|a| format!("OpRef::from_raw({})", a.raw()))
                            .collect();
                        eprintln!(
                            "[jit] p2 guard {:?} pos={:?} resume_pos={} rd_numb={} fail_args_raw=[{}]",
                            op.opcode,
                            op.pos,
                            op.rd_resume_position,
                            rd_numb_len,
                            fa_raw.join(", ")
                        );
                    } else {
                        eprintln!(
                            "[jit] p2 guard {:?} pos={:?} resume_pos={} rd_numb={} fail_args_raw=<none>",
                            op.opcode, op.pos, op.rd_resume_position, rd_numb_len,
                        );
                    }
                }
            }
            let nc = p2_ops
                .iter()
                .filter(|o| o.opcode == OpCode::New || o.opcode == OpCode::NewWithVtable)
                .count();
            let gc = p2_ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!(
                "[jit] phase 2: {} ops, {} New, {} guards, p2_ni={}",
                p2_ops.len(),
                nc,
                gc,
                p2_ni
            );
            for (i, op) in p2_ops.iter().enumerate() {
                eprintln!(
                    "[jit] p2[{i}]: {:?} pos={:?} args={:?}",
                    op.opcode, op.pos, op.args
                );
            }
        }

        // ── unroll.py:140-175: finalize + jump_to_existing_trace ──
        let imported_short_aliases = opt_p2.imported_short_aliases.clone();
        // finalize_short_preamble: create TargetToken for this loop version
        // RPython parity: short preamble ops reference constant OpRefs from
        // the loop's constant pool. Pass consts_p1 so the ShortPreamble
        // captures (value, type) for each constant, enabling bridges to
        // re-register them in their own pool (RPython embeds Const objects
        // directly in op args; majit uses separate constant pool indices).
        // RPython parity: read back from the same ExportedState that
        // import_state used (Python reference semantics — one object).
        // exported_state was moved into opt_p2.imported_loop_state.
        // Extract needed fields before opt_p2 is borrowed mutably below.
        let (
            exported_vs,
            exported_end_args,
            exported_short_inputargs,
            exported_short_boxes_produced,
            exported_renamed_inputargs,
        ) = {
            let es = opt_p2
                .imported_loop_state
                .as_ref()
                .expect("imported_loop_state must survive Phase 2");
            (
                es.virtual_state.clone(),
                es.end_args.clone(),
                es.short_inputargs.clone(),
                es.short_boxes.clone(),
                es.renamed_inputargs.clone(),
            )
        };
        // RPython unroll.py:124-141 performs an extra end-of-preamble forcing
        // pass on the closing JUMP args before finalize_short_preamble():
        //   for a in end_jump.getarglist():
        //       self.force_box_for_end_of_preamble(get_box_replacement(a))
        //   current_vs = self.optunroll.get_virtual_state(end_jump.getarglist())
        //   target_virtual_state = self.optunroll.pick_virtual_state(...)
        //   args = target_virtual_state.make_inputargs(..., force_boxes=True)
        //   for arg in args:
        //       self.force_box(arg)
        //
        // The optimizer-level imported_short_preamble snapshot is taken before
        // this step, so rebuild from the live Phase 2 context here instead of
        // trusting the stale copy.
        let body_jump_args: Vec<OpRef> = body_terminal_op
            .as_ref()
            .map(|jump| jump.args.to_vec())
            .or_else(|| {
                p2_ops
                    .iter()
                    .rfind(|op| op.opcode == OpCode::Jump)
                    .map(|jump| jump.args.to_vec())
            })
            .unwrap_or_default();
        let (imported_short_preamble_builder, rebuilt_imported_short_preamble) =
            if let Some(mut final_ctx) = opt_p2.final_ctx.take() {
                if !body_jump_args.is_empty() {
                    // unroll.py:126-127 parity:
                    //   for a in end_jump.getarglist():
                    //       self.force_box_for_end_of_preamble(get_box_replacement(a))
                    //
                    // Route through `force_box_for_end_of_preamble` (the
                    // per-box type-gating wrapper) rather than the inner
                    // dispatcher, matching optimizer.py:306-319.
                    let resolved_jump_args: Vec<OpRef> = body_jump_args
                        .iter()
                        .map(|&arg| final_ctx.get_box_replacement(arg))
                        .collect();
                    for &arg in &resolved_jump_args {
                        let _ = opt_p2.force_box_for_end_of_preamble(arg, &mut final_ctx);
                    }
                    let forced_jump_args: Vec<OpRef> = body_jump_args
                        .iter()
                        .map(|&arg| final_ctx.get_box_replacement(arg))
                        .collect();
                    let current_vs = crate::optimizeopt::virtualstate::export_state(
                        &forced_jump_args,
                        &final_ctx,
                    );
                    let mut target_states: Vec<crate::optimizeopt::virtualstate::VirtualState> =
                        self.target_tokens
                            .iter()
                            .filter_map(|token| token.virtual_state.clone())
                            .collect();
                    let target_virtual_state =
                        if let Some(idx) = pick_virtual_state(&current_vs, &target_states) {
                            target_states.swap_remove(idx)
                        } else {
                            exported_vs.clone()
                        };
                    if let Ok(args) = target_virtual_state.make_inputargs(
                        &forced_jump_args,
                        &mut opt_p2,
                        &mut final_ctx,
                        true,
                    ) {
                        for arg in args {
                            if !arg.is_none() {
                                let _ = opt_p2.force_box(arg, &mut final_ctx);
                            }
                        }
                    }
                }
                // PRE-EXISTING-ADAPTATION (B.6.5): RPython's
                // `force_op_from_preamble` (unroll.py:26-39) only seeds
                // `potential_extra_ops`. The orthodox path that fills
                // `used_boxes` / `short_preamble_jump` / `extra_same_as`
                // is `force_box -> potential_extra_ops.pop ->
                // add_preamble_op` (shortpreamble.py:432-440), which
                // RPython runs in `optimize_op` whenever a body op
                // forces an imported Box. RPython's Box identity then
                // keeps that Box live across the loop boundary
                // implicitly.
                //
                // majit's disjoint Phase 1 / Phase 2 OpRef namespaces
                // strip the Box-identity transparency. Body ops that
                // CSE-replace through `force_op_from_preamble_op` and
                // never trigger `force_box` leave their imported entry
                // in `potential_extra_ops`, so the LABEL is not
                // extended with the corresponding `used_box`. To
                // recover orthodox shape, walk body args + fail_args
                // in body-occurrence order before
                // `build_imported_short_preamble` and call the
                // orthodox `force_box` for every body OpRef that has a
                // pending `potential_extra_ops` entry. `force_box`
                // routes through `add_preamble_op_from_pop`
                // (optimizer.rs `force_box`), so used_boxes /
                // short_preamble_jump / extra_same_as fill via the
                // canonical RPython path.
                {
                    let mut visited_force: std::collections::HashSet<OpRef> =
                        std::collections::HashSet::new();
                    for op in p2_ops.iter() {
                        if op.opcode == OpCode::Jump {
                            // The terminal jump's args are already
                            // run through `force_box_for_end_of_preamble`
                            // (unroll.py:126-127) above.
                            continue;
                        }
                        let arg_iter = op
                            .args
                            .iter()
                            .copied()
                            .chain(op.fail_args.iter().flatten().copied());
                        for arg in arg_iter {
                            if !is_trace_runtime_ref(arg, &consts_p2) {
                                continue;
                            }
                            if !visited_force.insert(arg) {
                                continue;
                            }
                            let resolved = final_ctx.get_box_replacement(arg);
                            let needs_force = final_ctx.potential_extra_ops.contains_key(&arg)
                                || final_ctx.potential_extra_ops.contains_key(&resolved);
                            if needs_force {
                                let _ = opt_p2.force_box(arg, &mut final_ctx);
                            }
                        }
                    }
                }
                let rebuilt = final_ctx.build_imported_short_preamble();
                let builder = final_ctx.imported_short_preamble_builder.clone();
                opt_p2.final_ctx = Some(final_ctx);
                (builder, rebuilt)
            } else {
                (
                    opt_p2.imported_short_preamble_builder.clone(),
                    opt_p2.imported_short_preamble.clone(),
                )
            };
        let mut initial_sp = rebuilt_imported_short_preamble.unwrap_or_else(|| {
            // RPython unroll.py:497-504 `for produced_op in
            // exported_state.short_boxes` parity: consume the eagerly-derived
            // ProducedShortOp list stored on ExportedState at export time.
            crate::optimizeopt::shortpreamble::build_short_preamble_from_produced_boxes(
                &exported_end_args,
                &exported_short_inputargs,
                &exported_short_boxes_produced,
                &consts_p1,
                &self.constant_types,
            )
        });
        // shortpreamble.py:414-425 parity: store PtrInfo for each inputarg.
        // In RPython, preamble_op.set_forwarded(info) attaches PtrInfo to
        // the short_inputarg Box. In majit, we extract from Phase 2 final_ctx
        // so inline_short_preamble can propagate to jump_args.
        if let Some(ref final_ctx) = opt_p2.final_ctx {
            let mut infos = Vec::with_capacity(initial_sp.inputargs.len());
            for &inputarg in &initial_sp.inputargs {
                infos.push(final_ctx.peek_ptr_info_via_box(inputarg));
            }
            initial_sp.inputarg_infos = infos;
        }
        let opt_unroll = OptUnroll::new();
        let target_token = opt_unroll.finalize_short_preamble(
            self.target_tokens.len() as u64,
            exported_vs,
            initial_sp.clone(),
            imported_short_preamble_builder.as_ref(),
        );
        self.target_tokens.push(target_token);

        // unroll.py:176-177: disable_retracing_if_max_retrace_guards
        if Self::disable_retracing_if_max_retrace_guards(&p2_ops, self.max_retrace_guards) {
            self.retraced_count = u32::MAX;
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] too many guards (>{}), disabling retracing",
                    self.max_retrace_guards
                );
            }
        }

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] finalize_short_preamble: target_tokens={}",
                self.target_tokens.len()
            );
        }

        // ── unroll.py:207-230: jump_to_existing_trace / retrace_limit ──
        // Try to match the body's JUMP virtual state to an existing target.
        // RPython: new_virtual_state = jump_to_existing_trace(end_jump, ...)
        //
        // RPython parity: never skip jump_to_existing_trace based on
        // guard count. RPython's unroll.py always attempts
        // jump_to_existing_trace regardless of body size.
        let p2_guard_count = p2_ops.iter().filter(|o| o.opcode.is_guard()).count();
        let skip_jump_to_existing = false;
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] post-finalize: entering jump_to_existing_trace section (p2_guards={}, skip={})",
                p2_guard_count, skip_jump_to_existing
            );
        }
        let mut body_ops = p2_ops;
        let mut redirected_tail_ops = Vec::new();
        let jump_to_self = {
            let body_jump_args: Vec<OpRef> = body_terminal_op
                .as_ref()
                .map(|jump| jump.args.to_vec())
                .or_else(|| {
                    body_ops
                        .iter()
                        .rfind(|o| o.opcode == OpCode::Jump)
                        .map(|j| j.args.to_vec())
                })
                .unwrap_or_default();
            let mut current_label_args = label_args.clone();
            // RPython Box parity: each used_box is a distinct Box even
            // when two virtuals share the same OpRef. Allocate fresh
            // OpRefs for duplicates so the LABEL carries independent slots.
            {
                let mut seen_used = std::collections::HashSet::new();
                let mut next_fresh = current_label_args
                    .iter()
                    .chain(initial_sp.used_boxes.iter())
                    .map(|a| a.raw())
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1)
                    .max(body_num_inputs as u32 + 100);
                for &ub in &initial_sp.used_boxes {
                    if seen_used.insert(ub) {
                        current_label_args.push(ub);
                    } else {
                        current_label_args.push(OpRef::from_raw(next_fresh));
                        next_fresh += 1;
                    }
                }
            }
            let opt_unroll = OptUnroll::new();
            // Use Phase 2's final context for virtual state matching.
            let mut jump_ctx = opt_p2.final_ctx.take().unwrap_or_else(|| {
                // opencoder.py:259 inputarg_from_tp parity — Phase 2 inputargs
                // carry the same producer-side types as Phase 1's; fall back
                // to Ref when types are unavailable.
                let types: Vec<majit_ir::Type> = opt_p2
                    .trace_inputarg_types
                    .get(..body_num_inputs)
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| vec![majit_ir::Type::Ref; body_num_inputs]);
                crate::optimizeopt::OptContext::with_inputarg_types(32, &types)
            });

            // unroll.py:151-158: jump_to_existing_trace(force_boxes=False)
            // RPython: except InvalidLoop → jump_to_preamble immediately,
            // NO retry. The big comment at unroll.py:305-316 explains why
            // continuing after partial inlining is unsafe.
            let runtime_boxes = body_jump_args.clone();
            let mut invalid_loop = false;
            let mut jumped = if skip_jump_to_existing {
                false
            } else {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    opt_unroll
                        .jump_to_existing_trace(
                            &body_jump_args,
                            Some(&current_label_args),
                            &mut self.target_tokens,
                            &mut opt_p2,
                            &mut jump_ctx,
                            false,
                            Some(&runtime_boxes),
                        )
                        .is_none()
                })) {
                    Ok(result) => result,
                    Err(payload) => {
                        if payload
                            .downcast_ref::<crate::optimize::InvalidLoop>()
                            .is_some()
                        {
                            if let Some(reason) =
                                payload.downcast_ref::<crate::optimize::InvalidLoop>()
                            {
                                if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                                    eprintln!(
                                        "[jit][jte] InvalidLoop during force_boxes=false: {}",
                                        reason.0
                                    );
                                }
                            }
                            // unroll.py:154-158: except InvalidLoop →
                            // jump_to_preamble, skip retry
                            invalid_loop = true;
                            false
                        } else {
                            std::panic::resume_unwind(payload);
                        }
                    }
                }
            };
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] jump_to_existing_trace(force_boxes=false) result: jumped={}, invalid_loop={}",
                    jumped, invalid_loop
                );
            }

            // unroll.py:154-158: on InvalidLoop, skip retry entirely
            if !jumped && !skip_jump_to_existing && !invalid_loop {
                // unroll.py:161-174: virtual state not matched, retry
                if self.retraced_count < self.retrace_limit {
                    self.retraced_count += 1;
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[jit] Retracing ({}/{})",
                            self.retraced_count, self.retrace_limit
                        );
                    }
                    // unroll.py:164-168: force_boxes=True, except InvalidLoop: pass
                    jumped = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        opt_unroll
                            .jump_to_existing_trace(
                                &body_jump_args,
                                Some(&current_label_args),
                                &mut self.target_tokens,
                                &mut opt_p2,
                                &mut jump_ctx,
                                true,
                                Some(&runtime_boxes),
                            )
                            .is_none()
                    })) {
                        Ok(result) => result,
                        Err(payload) => {
                            if payload
                                .downcast_ref::<crate::optimize::InvalidLoop>()
                                .is_some()
                            {
                                if let Some(reason) =
                                    payload.downcast_ref::<crate::optimize::InvalidLoop>()
                                {
                                    if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                                        eprintln!(
                                            "[jit][jte] InvalidLoop during force_boxes=true retrace: {}",
                                            reason.0
                                        );
                                    }
                                }
                                false // unroll.py:167-168: except InvalidLoop: pass
                            } else {
                                std::panic::resume_unwind(payload);
                            }
                        }
                    };
                } else {
                    // unroll.py:220-226: limit reached, try force_boxes=true
                    jumped = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        opt_unroll
                            .jump_to_existing_trace(
                                &body_jump_args,
                                Some(&current_label_args),
                                &mut self.target_tokens,
                                &mut opt_p2,
                                &mut jump_ctx,
                                true,
                                Some(&runtime_boxes),
                            )
                            .is_none()
                    })) {
                        Ok(result) => result,
                        Err(payload) => {
                            if payload
                                .downcast_ref::<crate::optimize::InvalidLoop>()
                                .is_some()
                            {
                                if let Some(reason) =
                                    payload.downcast_ref::<crate::optimize::InvalidLoop>()
                                {
                                    if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                                        eprintln!(
                                            "[jit][jte] InvalidLoop during force_boxes=true limit: {}",
                                            reason.0
                                        );
                                    }
                                }
                                false // unroll.py:224-225: except InvalidLoop: pass
                            } else {
                                std::panic::resume_unwind(payload);
                            }
                        }
                    };
                    if !jumped {
                        // unroll.py:228: "Retrace count reached, jumping to preamble"
                        if std::env::var_os("MAJIT_LOG").is_some() {
                            eprintln!("[jit] Retrace count reached, jumping to preamble");
                        }
                        // jumped stays false → jump_to_preamble below
                    }
                }
            }
            if jumped && redirected_tail_ops.is_empty() {
                // Only take jump_ctx ops if we don't already have
                // a self-loop Jump from the retrace path.
                redirected_tail_ops = std::mem::take(&mut jump_ctx.new_operations);
                // Check if the redirected Jump targets the current body token
                // (last in target_tokens) or an external token from a previous
                // compilation.  The Cranelift backend compiles each trace as a
                // single function — cross-function jumps to external target
                // tokens are not supported.  Discard the redirected tail and
                // restore the body's original self-loop Jump instead.
                let current_body_descr_idx = self
                    .target_tokens
                    .last()
                    .map(|t| t.as_jump_target_descr().index());
                let redirected_jump_descr_idx = redirected_tail_ops
                    .iter()
                    .rfind(|o| o.opcode == OpCode::Jump)
                    .and_then(|o| o.descr.as_ref())
                    .map(|d| d.index());
                if redirected_jump_descr_idx != current_body_descr_idx {
                    // RPython parity: the Cranelift backend can't jump to
                    // code from a previous compilation (separate function).
                    // Fall back to jump_to_preamble, matching RPython's
                    // behavior when the target isn't reachable (unroll.py:228).
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[jit] jump_to_existing_trace: external target {:?} != body {:?}, falling back to preamble",
                            redirected_jump_descr_idx, current_body_descr_idx
                        );
                    }
                    redirected_tail_ops.clear();
                    jumped = false;
                }
            }
            opt_p2.final_ctx = Some(jump_ctx);
            jumped
        };

        let sp = self
            .target_tokens
            .last()
            .and_then(|target| target.short_preamble.clone())
            .unwrap_or(initial_sp);
        if !sp.is_empty() {
            self.short_preamble = Some(sp.clone());
        }
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] assembly_contract: label_args={:?} used_boxes={:?} jump_args={:?}",
                label_args, sp.used_boxes, sp.jump_args
            );
        }

        if !jump_to_self {
            // unroll.py:170-171: jump_to_preamble — body JUMP → preamble Label
            //
            // RPython parity: force_box_for_end_of_preamble (unroll.py:126-127)
            // re-boxes unboxed values before the JUMP so types match the
            // preamble inputargs. Without force_box, the body JUMP may pass
            // Float/Int values at Ref-typed positions, causing the preamble's
            // guard checks to dereference non-pointer values → segfault.
            //
            // Until force_box_for_end_of_preamble is implemented, reject
            // traces where the body JUMP types don't match preamble inputarg
            // types. The metainterp falls back to interpretation.
            let preamble_target = self
                .target_tokens
                .first()
                .expect("preamble target token must exist before jump_to_preamble")
                .clone();
            let preamble_arity = exported_renamed_inputargs.len();
            if std::env::var_os("MAJIT_LOG").is_some() {
                let body_jump_arity = body_terminal_op.as_ref().map(|j| j.args.len()).unwrap_or(0);
                eprintln!(
                    "[jit] jump_to_preamble: body_jump_args={} preamble_arity={} start_label_args={:?}",
                    body_jump_arity, preamble_arity, exported_renamed_inputargs,
                );
            }
            if let Some(mut end_jump) = body_terminal_op {
                end_jump.descr = Some(preamble_target.as_jump_target_descr());
                if let Some(mut final_ctx) = opt_p2.final_ctx.take() {
                    // unroll.py:238-242 parity: jump_to_preamble retargets
                    // the live end_jump and routes it through
                    // send_extra_operation, preserving any force_box /
                    // partial-inline operations already appended to
                    // _newoperations.
                    opt_p2.send_extra_operation(&end_jump, &mut final_ctx);
                    let redirected_tail_ops = std::mem::take(&mut final_ctx.new_operations);
                    opt_p2.final_ctx = Some(final_ctx);
                    body_ops = splice_redirected_tail(&body_ops, &redirected_tail_ops);
                } else {
                    body_ops = replace_terminal_jump(&body_ops, end_jump);
                }
            } else {
                body_ops = Self::jump_to_preamble(&body_ops, &preamble_target);
            }
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!("[jit] jump_to_preamble: body JUMP retargeted to start descr");
            }
        } else if !redirected_tail_ops.is_empty() {
            body_ops = splice_redirected_tail(&body_ops, &redirected_tail_ops);
        } else if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[jit] jump_to_existing_trace: body JUMP → self-loop");
        }

        // ── Assembly (compile.py:310-338) ──
        let mut combined = assemble_peeled_trace_with_jump_args(
            &p1_ops,
            &body_ops,
            &label_args,
            &exported_renamed_inputargs,
            &sp.used_boxes,
            &sp.jump_args,
            p2_ni,
            phase2_inputarg_base,
            jump_to_self,
            &imported_short_aliases,
            &consts_p2,
            self.target_tokens
                .first()
                .map(|target| target.as_jump_target_descr()),
            self.target_tokens
                .last()
                .map(|target| target.as_jump_target_descr()),
            &exported_end_args,
            opt_p2
                .final_ctx
                .as_mut()
                .expect("assemble peeled trace requires Phase 2 OptContext for Box.type parity"),
        );
        // RPython Box parity: drop duplicate-position ops. In RPython
        // each Box is unique so collisions can't happen. Keep first.
        {
            let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
            combined.retain(|op| {
                if op.pos.is_none() || op.result_type() == Type::Void {
                    return true;
                }
                seen.insert(op.pos.raw())
            });
        }
        crate::optimizeopt::optimizer::sanitize_backend_constants_for_ops(
            &combined,
            &mut consts_p2,
        );
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("--- peeled trace (assembled) ---");
            eprint!("{}", majit_ir::format_trace(&combined, &consts_p2));
            let mut sorted_consts: Vec<_> =
                consts_p2.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
            sorted_consts.sort_by_key(|(k, _)| *k);
            eprintln!("[jit] consts_p2: {:?}", sorted_consts);
        }
        *constants = consts_p2;
        // Merge Phase 2 constant types back so build_guard_metadata
        // can resolve Phase 2 allocated constants for rd_virtuals.
        for (k, v) in &opt_p2.constant_types {
            self.constant_types.entry(*k).or_insert(*v);
        }
        (combined, p2_ni)
    }

    /// RPython compile.py uses the optimized loop state's inputargs contract,
    /// not a stale input counter. When majit falls back to the phase-1 trace,
    /// derive the live loop arity from the actual closing Label/Jump.
    pub fn closing_loop_contract_arity(ops: &[Op], fallback: usize) -> usize {
        closing_loop_contract_arity(ops, fallback)
    }

    /// Count the guards in an optimized trace (for retrace_limit checks).
    pub fn count_guards(ops: &[Op]) -> u32 {
        ops.iter().filter(|op| op.opcode.is_guard()).count() as u32
    }

    /// unroll.py: _map_args(mapping, arglist)
    /// Remap a list of OpRefs through a forwarding mapping.
    /// Constant OpRefs are left unchanged because they are not remapped.
    pub fn map_args(
        mapping: &std::collections::HashMap<OpRef, OpRef>,
        args: &[OpRef],
    ) -> Vec<OpRef> {
        args.iter()
            .map(|&arg| mapping.get(&arg).copied().unwrap_or(arg))
            .collect()
    }

    /// unroll.py: _check_no_forwarding(lsts)
    /// Debug assertion: verify no OpRef in the lists has been forwarded.
    pub fn check_no_forwarding(ctx: &crate::optimizeopt::OptContext, oprefs: &[OpRef]) -> bool {
        oprefs.iter().all(|&r| ctx.get_box_replacement(r) == r)
    }

    /// unroll.py: disable_retracing_if_max_retrace_guards(ops, target_token)
    /// If the trace has too many guards, disable retracing for this location.
    /// Returns true if retracing was disabled.
    pub fn disable_retracing_if_max_retrace_guards(ops: &[Op], max_retrace_guards: u32) -> bool {
        let guard_count = Self::count_guards(ops);
        guard_count > max_retrace_guards
    }

    /// unroll.py: get_virtual_state(args)
    /// Build a VirtualState from the optimizer's current knowledge about args.
    pub fn get_virtual_state(
        args: &[OpRef],
        ctx: &crate::optimizeopt::OptContext,
    ) -> crate::optimizeopt::virtualstate::VirtualState {
        crate::optimizeopt::virtualstate::export_state(args, ctx)
    }
}

fn closing_loop_contract_arity(ops: &[Op], fallback: usize) -> usize {
    ops.iter()
        .rev()
        .find_map(|op| match op.opcode {
            OpCode::Label | OpCode::Jump => Some(op.args.len()),
            _ => None,
        })
        .unwrap_or(fallback)
}

impl Default for UnrollOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// unroll.py: ExportedState — snapshot of optimizer state at the end of
/// the preamble, used to initialize the peeled loop body.
///
/// Contains the virtual state, short preamble boxes, arg mappings,
/// and exported infos needed to resume optimization after peeling.
///
/// gcreftracer.py parity: GcRef values in exported_infos are rooted on
/// the shadow stack. ExportedState persists between Phase 1 (preamble)
/// and Phase 2 (body), during which GC can run and move objects.
#[derive(Debug)]
pub struct ExportedState {
    /// Label args at the end of the preamble (after forcing).
    pub end_args: Vec<OpRef>,
    /// Args for the next iteration (before forcing).
    pub next_iteration_args: Vec<OpRef>,
    /// Types of end_args as determined by Phase 1 optimization.
    /// Used by Phase 2 import_state to propagate unboxed types.
    pub end_arg_types: Vec<Type>,
    /// Virtual state at the loop boundary.
    pub virtual_state: crate::optimizeopt::virtualstate::VirtualState,
    /// unroll.py:548 ExportedState.exported_infos — optimizer knowledge from preamble.
    /// Maps OpRef → info for all args including virtual field contents.
    ///
    /// RPython stores one of `PtrInfo` / `IntBound` / `FloatConstInfo` per box,
    /// dispatched via `isinstance` in `setinfo_from_preamble` (unroll.py:53-98).
    /// Majit uses the existing `OpInfo` enum (info.rs:137) as the discriminated
    /// union of these three cases.
    pub exported_infos: HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
    /// RPython shortpreamble.py: produced short boxes in preamble order.
    /// This preserves the original preamble ops so the active path can build
    /// short preambles without re-extracting them from the peeled trace.
    pub exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// RPython unroll.py:466-477 `short_boxes = sb.create_short_boxes(...)`
    /// stored directly on ExportedState. Consumer sites
    /// (`build_imported_short_preamble`, `import_short_preamble_state`)
    /// iterate this list verbatim. Pyre derives this eagerly from
    /// `exported_short_boxes + label_args + short_inputargs` at export time
    /// (`shortpreamble.py:269-270 ShortBoxes.create_short_boxes` parity).
    pub short_boxes: Vec<(OpRef, crate::optimizeopt::shortpreamble::ProducedShortOp)>,
    /// PRE-EXISTING-ADAPTATION: producer-side const value snapshot for any
    /// const-namespace OpRef referenced by `short_boxes` op args. RPython
    /// gets this for free because `Const` Box objects carry their value as
    /// an attribute and persist across optimization phases. In majit, OpRef
    /// is a flat trace-local index — `OpRef::from_const(N)` only resolves
    /// to a value through `OptContext::const_pool[N]`, and consumer-side ctx
    /// (Phase 2 / bridge) may not have the producer's slot N populated.
    ///
    /// The legacy `ExportedShortOp::Pure { args: Vec<ExportedShortArg> }`
    /// path captured constant args inline as `ExportedShortArg::Const
    /// { source, value }`. Phase B.1's polymorphic `ProducedShortOp::
    /// produce_op` reads raw OpRef args, so we lift the value snapshot to
    /// this sidecar map keyed by source OpRef. `produce_op` /
    /// `classify_short_arg` (shortpreamble.rs) read this map first when
    /// classifying a Const arg.
    ///
    /// Convergence path: this map disappears once consumer ctxs are
    /// guaranteed to have producer constants pre-seeded (production
    /// already does this at `optimizer.rs:1927`; bridges and unit tests
    /// remain the open cases). At that point `classify_short_arg` can
    /// read from `ctx.get_constant(arg)` exclusively.
    pub short_box_const_values: HashMap<OpRef, majit_ir::Value>,
    /// Short preamble builder for bridge entry.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// Renamed inputargs from the preamble.
    pub renamed_inputargs: Vec<OpRef>,
    /// Types of renamed_inputargs. RPython Box objects carry type
    /// intrinsically; majit stores it separately.
    pub renamed_inputarg_types: Vec<Type>,
    /// Short inputargs for the short preamble.
    pub short_inputargs: Vec<OpRef>,
    /// unroll.py: runtime_boxes — live values at the original jump point.
    /// Threaded into Phase 2 import as `runtime_boxes` for guard generation.
    /// Default `Vec::new()` until the export site populates it; callers
    /// that need it write the field directly after `ExportedState::new`.
    pub runtime_boxes: Vec<OpRef>,
    /// RPython parity: patchguardop from Phase 1's GuardFutureCondition.
    /// Phase 2's extra_guards (from virtualstate) need rd_resume_position
    /// from this patchguardop (unroll.py:333-336).
    pub patchguardop: Option<majit_ir::Op>,
    /// Smallest fresh OpRef strictly above every Phase 1 emit position
    /// (i.e. `max(op.pos.raw()) + 1` over `phase1_emit_ops`). Used by
    /// `opref_high_water` so retrace's Phase 2 namespace stays disjoint
    /// from intermediate Phase 1 emit OpRefs that were forwarded /
    /// folded and never survive as an end_arg / short_op source.
    /// `0` when there are no Phase 1 emit positions to track.
    pub phase1_emit_high_water: u32,
    /// Snapshot of Phase 1's `OptContext::box_pool` taken at export time.
    /// Used by `compile_retrace`'s Phase 2 TraceIterator as the
    /// `p1_full_prefix` (inputarg + emit BoxRefs) so chain walks via
    /// `Forwarded::Box(rc)` observe the failed attempt's accumulated
    /// `_forwarded` info on every Phase 1 position — `unroll.py` Phase 2
    /// reading Phase 1 box.\_forwarded through Python identity parity.
    ///
    /// `None` for ExportedStates produced by paths that do not feed into a
    /// retrace attempt (test fixtures, in-flight construction). Populated
    /// only by `export_state_with_bounds` at the canonical Phase 1 export
    /// site.
    pub box_pool_snapshot: Option<Vec<crate::r#box::BoxRef>>,
    /// Shadow stack rooting for GcRef values in exported_infos.
    /// (OpRef key, field kind, shadow stack index).
    rooted_refs: Vec<(OpRef, ExportedGcRefField, usize)>,
    /// Shadow stack depth at creation. release_roots pops to here.
    shadow_stack_base: usize,
}

// unroll.py:529 `exported_infos - a mapping from ops to infos, including inputargs`
// The per-entry info type is now `OpInfo` (info.rs:137), the discriminated
// union matching RPython's `PtrInfo | IntBound | FloatConstInfo` dispatched
// via `isinstance` in `setinfo_from_preamble` (unroll.py:53-98). The earlier
// majit-only bundle `ExportedValueInfo { constant, ptr_info, int_bound }`
// is removed as part of the Box identity plan Phase D.

/// Identifies which GcRef-bearing field inside an ExportedState
/// is rooted at a particular shadow stack slot.
#[derive(Clone, Copy, Debug)]
enum ExportedGcRefField {
    /// exported_infos[OpRef].constant (Value::Ref)
    InfoConstantRef,
    /// exported_infos[OpRef].ptr_info = PtrInfo::Constant(GcRef)
    InfoPtrInfoConstant,
    /// exported_infos[OpRef].ptr_info = PtrInfo::Instance with known_class set
    /// (PyPy `InstancePtrInfo._known_class`; majit folds the former
    /// `KnownClass` variant into `Instance(descr=None, known_class=Some(_))`).
    InfoPtrInfoKnownClass,
    /// virtual_state.state[index] = KnownClass { class_ptr }
    VirtualStateKnownClass(usize),
    /// virtual_state.state[index] = Virtual { known_class }
    VirtualStateVirtualClass(usize),
    /// virtual_state.state[index] = Constant(Value::Ref)
    VirtualStateConstantRef(usize),
    /// short_box_const_values[OpRef] = Value::Ref(...)
    ShortBoxConstValue(OpRef),
    /// box_pool_snapshot[index].forwarded = Info(OpInfo::Constant(Value::Ref(_)))
    BoxPoolInfoConstantRef(usize),
    /// box_pool_snapshot[index].forwarded = Info(OpInfo::Ptr(PtrInfo::Constant(_)))
    BoxPoolInfoPtrInfoConstant(usize),
    /// box_pool_snapshot[index].forwarded = Info(OpInfo::Ptr(PtrInfo::Instance{known_class: Some(_)}))
    BoxPoolInfoPtrInfoKnownClass(usize),
    /// box_pool_snapshot[index].forwarded = Box(target) where target is
    /// `BoxKind::Const(Value::Ref(_))` — `make_constant` /
    /// `replace_op(... const ...)` writer mirror shape.
    BoxPoolBoxConstRef(usize),
}

impl ExportedState {
    /// unroll.py: ExportedState.__init__
    pub fn new(
        end_args: Vec<OpRef>,
        next_iteration_args: Vec<OpRef>,
        virtual_state: crate::optimizeopt::virtualstate::VirtualState,
        exported_infos: HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
        exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
        renamed_inputargs: Vec<OpRef>,
        renamed_inputarg_types: Vec<Type>,
        short_inputargs: Vec<OpRef>,
    ) -> Self {
        // unroll.py:466-477 `sb.create_short_boxes(...)` parity: pyre
        // pre-derives the per-OpRef ProducedShortOp view (with label-arg
        // refs renamed to short-inputarg slots) and stores it directly,
        // matching RPython's `ExportedState.short_boxes`.
        let short_boxes =
            crate::optimizeopt::shortpreamble::produced_short_boxes_from_exported_boxes(
                &end_args,
                &short_inputargs,
                &exported_short_boxes,
            );
        ExportedState {
            end_args,
            next_iteration_args,
            end_arg_types: Vec::new(),
            virtual_state,
            exported_infos,
            exported_short_boxes,
            short_boxes,
            short_box_const_values: HashMap::new(),
            short_preamble: None,
            renamed_inputargs,
            renamed_inputarg_types,
            short_inputargs,
            runtime_boxes: Vec::new(),
            patchguardop: None,
            phase1_emit_high_water: 0,
            box_pool_snapshot: None,
            rooted_refs: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
        // gcreftracer.py parity: RPython ExportedState is a Python object
        // whose GcRef fields are automatically traced by the GC. In Rust,
        // root_all_gcrefs() must be called at each storage site in LIFO
        // order (longer-lived copy rooted first → lower shadow stack depth).
        // new() does NOT auto-root because the LIFO ordering depends on
        // the caller's storage pattern.
    }

    /// Smallest fresh OpRef that is guaranteed not to collide with any Box
    /// identity stored in this exported preamble state.
    ///
    /// RPython keeps these as object identities in `partial_trace` /
    /// `ExportedState`. Majit uses integer OpRefs, so compile_retrace must
    /// reconstruct this high-water mark before creating the fresh Phase 2
    /// TraceIterator namespace.
    fn opref_high_water(&self) -> u32 {
        let mut high = 0_u32;
        let mut visit = |opref: OpRef| {
            if !opref.is_none() && !opref.is_constant() {
                high = high.max(opref.raw().saturating_add(1));
            }
        };
        let visit_op = |op: &Op, visit: &mut dyn FnMut(OpRef)| {
            visit(op.pos);
            for &arg in &op.args {
                visit(arg);
            }
            if let Some(ref fail_args) = op.fail_args {
                for &arg in fail_args {
                    visit(arg);
                }
            }
        };
        for &opref in &self.end_args {
            visit(opref);
        }
        for &opref in &self.next_iteration_args {
            visit(opref);
        }
        for &opref in &self.renamed_inputargs {
            visit(opref);
        }
        for &opref in &self.short_inputargs {
            visit(opref);
        }
        for &opref in &self.runtime_boxes {
            visit(opref);
        }

        for (&opref, info) in &self.exported_infos {
            visit(opref);
            if let crate::optimizeopt::info::OpInfo::Ptr(ptr_info) = info {
                for child in ptr_info.visitor_walk_recursive() {
                    visit(child);
                }
            }
        }

        for preamble_op in &self.exported_short_boxes {
            visit_op(&preamble_op.op, &mut visit);
            if let Some(source) = preamble_op.same_as_source {
                visit(source);
            }
        }

        if let Some(short_preamble) = &self.short_preamble {
            for opref in short_preamble
                .inputargs
                .iter()
                .chain(short_preamble.used_boxes.iter())
                .chain(short_preamble.jump_args.iter())
            {
                visit(*opref);
            }
            if let Some(phase1_inputargs) = &short_preamble.phase1_inputargs {
                for &opref in phase1_inputargs {
                    visit(opref);
                }
            }
            for short_op in &short_preamble.ops {
                visit_op(&short_op.op, &mut visit);
            }
            for ptr_info in short_preamble.inputarg_infos.iter().flatten() {
                for child in ptr_info.visitor_walk_recursive() {
                    visit(child);
                }
            }
        }

        // `phase1_emit_high_water` covers Phase 1 emit positions that are
        // not reachable via any of the structure-stored OpRef fields above
        // (e.g. intermediate ops that were forwarded and never survive as
        // an end_arg / short_op source). Retrace must keep its Phase 2
        // namespace disjoint from every one of those.
        high = high.max(self.phase1_emit_high_water);

        high
    }

    /// Push all GcRef values from exported_infos and virtual_state to
    /// shadow stack. gcreftracer.py parity: GC can run between Phase 1
    /// and Phase 2, and between Phase 1 and retrace.
    ///
    /// Must be called explicitly after construction — not auto-called in
    /// new(). This enables LIFO-correct rooting: root the longer-lived
    /// copy first (lower shadow stack depth), then the shorter-lived copy.
    pub fn root_all_gcrefs(&mut self) {
        use crate::optimizeopt::info::{OpInfo, PtrInfo};
        use crate::optimizeopt::virtualstate::VirtualStateInfo;
        self.shadow_stack_base = majit_gc::shadow_stack::depth();
        // ── exported_infos GcRef fields ──
        let mut keys: Vec<OpRef> = self.exported_infos.keys().copied().collect();
        keys.sort_by_key(|k| k.raw());
        for key in keys {
            match &self.exported_infos[&key] {
                // RPython ConstPtrInfo: GcRef constant stored directly in
                // PtrInfo. Root the GcRef so GC keeps it alive.
                OpInfo::Ptr(PtrInfo::Constant(gcref)) if !gcref.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*gcref);
                    self.rooted_refs
                        .push((key, ExportedGcRefField::InfoPtrInfoConstant, ss_idx));
                }
                // InstancePtrInfo.known_class: GcRef for the class pointer.
                OpInfo::Ptr(PtrInfo::Instance(iinfo)) => {
                    if let Some(gcref) = iinfo.known_class {
                        if !gcref.is_null() {
                            let ss_idx = majit_gc::shadow_stack::push(gcref);
                            self.rooted_refs.push((
                                key,
                                ExportedGcRefField::InfoPtrInfoKnownClass,
                                ss_idx,
                            ));
                        }
                    }
                }
                // Direct `OpInfo::Constant(Value::Ref(...))` — rare Int-typed
                // constant forward that carries a Ref payload.
                OpInfo::Constant(Value::Ref(gcref)) if !gcref.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*gcref);
                    self.rooted_refs
                        .push((key, ExportedGcRefField::InfoConstantRef, ss_idx));
                }
                _ => {}
            }
        }
        // ── virtual_state GcRef fields ──
        // VirtualStateInfo::KnownClass, Virtual{known_class}, Constant(Ref)
        // The rooted_refs `key` slot here just tags the field origin;
        // these entries are not keyed back into a typed-OpRef map, so
        // use the canonical `OpRef::NONE` sentinel rather than a raw
        // u32::MAX placeholder that would diverge from `OpRef::None`
        // under variant-aware Eq/Hash.
        let dummy_key = OpRef::NONE;
        for (i, entry) in self.virtual_state.state.iter().enumerate() {
            match &entry.info {
                VirtualStateInfo::KnownClass { class_ptr } if !class_ptr.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*class_ptr);
                    self.rooted_refs.push((
                        dummy_key,
                        ExportedGcRefField::VirtualStateKnownClass(i),
                        ss_idx,
                    ));
                }
                VirtualStateInfo::Virtual {
                    known_class: Some(gcref),
                    ..
                } if !gcref.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*gcref);
                    self.rooted_refs.push((
                        dummy_key,
                        ExportedGcRefField::VirtualStateVirtualClass(i),
                        ss_idx,
                    ));
                }
                VirtualStateInfo::Constant(Value::Ref(gcref)) if !gcref.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*gcref);
                    self.rooted_refs.push((
                        dummy_key,
                        ExportedGcRefField::VirtualStateConstantRef(i),
                        ss_idx,
                    ));
                }
                _ => {}
            }
        }
        // RPython Const boxes are GC-traced objects. The Rust producer-side
        // const snapshot must therefore root any Ref payload it carries.
        let mut const_keys: Vec<OpRef> = self.short_box_const_values.keys().copied().collect();
        const_keys.sort_by_key(|k| k.raw());
        for key in const_keys {
            if let Some(Value::Ref(gcref)) = self.short_box_const_values.get(&key)
                && !gcref.is_null()
            {
                let ss_idx = majit_gc::shadow_stack::push(*gcref);
                self.rooted_refs
                    .push((key, ExportedGcRefField::ShortBoxConstValue(key), ss_idx));
            }
        }
        // ── box_pool_snapshot Forwarded::Info GcRef fields ──
        // Phase 2 chain walks land on these BoxRef cells and read their
        // _forwarded info; if GC moves a Ref between Phase 1 export and
        // retrace those reads see a stale handle.
        if let Some(snapshot) = &self.box_pool_snapshot {
            for (i, b) in snapshot.iter().enumerate() {
                let forwarded = b.get_forwarded();
                if let crate::r#box::Forwarded::Info(opinfo) = &*forwarded {
                    match opinfo {
                        OpInfo::Constant(Value::Ref(gcref)) if !gcref.is_null() => {
                            let ss_idx = majit_gc::shadow_stack::push(*gcref);
                            self.rooted_refs.push((
                                dummy_key,
                                ExportedGcRefField::BoxPoolInfoConstantRef(i),
                                ss_idx,
                            ));
                        }
                        OpInfo::Ptr(PtrInfo::Constant(gcref)) if !gcref.is_null() => {
                            let ss_idx = majit_gc::shadow_stack::push(*gcref);
                            self.rooted_refs.push((
                                dummy_key,
                                ExportedGcRefField::BoxPoolInfoPtrInfoConstant(i),
                                ss_idx,
                            ));
                        }
                        OpInfo::Ptr(PtrInfo::Instance(iinfo)) => {
                            if let Some(gcref) = iinfo.known_class {
                                if !gcref.is_null() {
                                    let ss_idx = majit_gc::shadow_stack::push(gcref);
                                    self.rooted_refs.push((
                                        dummy_key,
                                        ExportedGcRefField::BoxPoolInfoPtrInfoKnownClass(i),
                                        ss_idx,
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                } else if let crate::r#box::Forwarded::Box(target) = &*forwarded {
                    if target.is_constant() {
                        if let Some(Value::Ref(gcref)) = target.const_value() {
                            if !gcref.is_null() {
                                let ss_idx = majit_gc::shadow_stack::push(gcref);
                                self.rooted_refs.push((
                                    dummy_key,
                                    ExportedGcRefField::BoxPoolBoxConstRef(i),
                                    ss_idx,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Update GcRef values from shadow stack — GC may have moved objects.
    ///
    /// VirtualStateInfo top-level entries are stored as
    /// `Rc<VirtualStateInfoNode>` (so two aliased jump args can share a single
    /// `Rc`); GcRef updates have to replace the entire `Rc` with a fresh
    /// one because the inner enum is immutable.
    ///
    /// **Aliasing preservation**: RPython virtualstate.py:712-728
    /// `VirtualStateConstructor.create_state` caches by Python object
    /// identity, so a GC pause never breaks the "two aliased jump args
    /// share one VirtualStateInfo" invariant. Rust's `Rc<...>` is immutable
    /// after construction, so each per-slot GcRef refresh would otherwise
    /// allocate an independent new `Rc` for every shared slot, breaking
    /// the `Rc::as_ptr` dedup the walker relies on. Snapshot the original
    /// `Rc::as_ptr` per slot, group slots that
    /// originally shared an `Rc`, and after the GcRef updates re-clone a
    /// single canonical `Rc` into every slot of each group so the
    /// post-refresh tree preserves the pre-GC aliasing.
    pub fn refresh_from_gc(&mut self) {
        use crate::optimizeopt::info::{OpInfo, PtrInfo};
        use crate::optimizeopt::virtualstate::{VirtualStateInfo, VirtualStateInfoNode};
        use std::rc::Rc;
        // virtualstate.py:712 cache parity: snapshot original Rc identities
        // so we can re-share post-update slots that aliased pre-GC.
        let original_ptrs: Vec<usize> = self
            .virtual_state
            .state
            .iter()
            .map(|rc| Rc::as_ptr(rc) as usize)
            .collect();
        let mut virtual_state_dirty = false;
        for &(key, ref field, ss_idx) in &self.rooted_refs {
            let updated = majit_gc::shadow_stack::get(ss_idx);
            match field {
                ExportedGcRefField::InfoConstantRef => {
                    if let Some(info) = self.exported_infos.get_mut(&key) {
                        *info = crate::optimizeopt::info::OpInfo::Constant(Value::Ref(updated));
                    }
                }
                ExportedGcRefField::InfoPtrInfoConstant => {
                    if let Some(info) = self.exported_infos.get_mut(&key) {
                        *info = crate::optimizeopt::info::OpInfo::Ptr(PtrInfo::Constant(updated));
                    }
                }
                ExportedGcRefField::InfoPtrInfoKnownClass => {
                    if let Some(info) = self.exported_infos.get_mut(&key) {
                        if let crate::optimizeopt::info::OpInfo::Ptr(PtrInfo::Instance(iinfo)) =
                            info
                        {
                            iinfo.known_class = Some(updated);
                        }
                    }
                }
                ExportedGcRefField::VirtualStateKnownClass(i) => {
                    if let Some(slot) = self.virtual_state.state.get_mut(*i)
                        && matches!(&slot.info, VirtualStateInfo::KnownClass { .. })
                    {
                        *slot = VirtualStateInfoNode::new_rc(VirtualStateInfo::KnownClass {
                            class_ptr: updated,
                        });
                        virtual_state_dirty = true;
                    }
                }
                ExportedGcRefField::VirtualStateVirtualClass(i) => {
                    if let Some(slot) = self.virtual_state.state.get_mut(*i) {
                        if let VirtualStateInfo::Virtual {
                            descr,
                            ob_type_descr,
                            fields,
                            field_descrs,
                            ..
                        } = &slot.info
                        {
                            *slot = VirtualStateInfoNode::new_rc(VirtualStateInfo::Virtual {
                                descr: descr.clone(),
                                known_class: Some(updated),
                                ob_type_descr: ob_type_descr.clone(),
                                fields: fields.clone(),
                                field_descrs: field_descrs.clone(),
                            });
                            virtual_state_dirty = true;
                        }
                    }
                }
                ExportedGcRefField::VirtualStateConstantRef(i) => {
                    if let Some(entry) = self.virtual_state.state.get_mut(*i) {
                        *entry = VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(
                            Value::Ref(updated),
                        ));
                        virtual_state_dirty = true;
                    }
                }
                ExportedGcRefField::ShortBoxConstValue(source) => {
                    if let Some(value) = self.short_box_const_values.get_mut(source) {
                        *value = Value::Ref(updated);
                    }
                }
                ExportedGcRefField::BoxPoolInfoConstantRef(i) => {
                    if let Some(snapshot) = self.box_pool_snapshot.as_ref()
                        && let Some(b) = snapshot.get(*i)
                    {
                        let new_info = {
                            let f = b.get_forwarded();
                            if matches!(
                                &*f,
                                crate::r#box::Forwarded::Info(OpInfo::Constant(Value::Ref(_)))
                            ) {
                                Some(OpInfo::Constant(Value::Ref(updated)))
                            } else {
                                None
                            }
                        };
                        if let Some(info) = new_info {
                            b.set_forwarded_info(info);
                        }
                    }
                }
                ExportedGcRefField::BoxPoolInfoPtrInfoConstant(i) => {
                    if let Some(snapshot) = self.box_pool_snapshot.as_ref()
                        && let Some(b) = snapshot.get(*i)
                    {
                        let new_info = {
                            let f = b.get_forwarded();
                            if matches!(
                                &*f,
                                crate::r#box::Forwarded::Info(OpInfo::Ptr(PtrInfo::Constant(_)))
                            ) {
                                Some(OpInfo::Ptr(PtrInfo::Constant(updated)))
                            } else {
                                None
                            }
                        };
                        if let Some(info) = new_info {
                            b.set_forwarded_info(info);
                        }
                    }
                }
                ExportedGcRefField::BoxPoolInfoPtrInfoKnownClass(i) => {
                    if let Some(snapshot) = self.box_pool_snapshot.as_ref()
                        && let Some(b) = snapshot.get(*i)
                    {
                        let new_info = {
                            let f = b.get_forwarded();
                            if let crate::r#box::Forwarded::Info(OpInfo::Ptr(PtrInfo::Instance(
                                iinfo,
                            ))) = &*f
                            {
                                let mut new_iinfo = iinfo.clone();
                                new_iinfo.known_class = Some(updated);
                                Some(OpInfo::Ptr(PtrInfo::Instance(new_iinfo)))
                            } else {
                                None
                            }
                        };
                        if let Some(info) = new_info {
                            b.set_forwarded_info(info);
                        }
                    }
                }
                ExportedGcRefField::BoxPoolBoxConstRef(i) => {
                    if let Some(snapshot) = self.box_pool_snapshot.as_ref()
                        && let Some(b) = snapshot.get(*i)
                    {
                        // BoxKind::Const is immutable, so swap in a fresh
                        // ConstRef BoxRef carrying the updated handle.
                        // `set_forwarded_box` enforces the AbstractValue
                        // invariant (Const has no _forwarded slot of its
                        // own) for the source, which is a non-Const ResOp
                        // / InputArg here.
                        let needs_swap = {
                            let f = b.get_forwarded();
                            if let crate::r#box::Forwarded::Box(target) = &*f {
                                target.is_constant()
                                    && matches!(target.const_value(), Some(Value::Ref(_)))
                            } else {
                                false
                            }
                        };
                        if needs_swap {
                            b.set_forwarded_box(crate::r#box::BoxRef::new_const(Value::Ref(
                                updated,
                            )));
                        }
                    }
                }
            }
        }
        if virtual_state_dirty {
            // Re-share slots that originally aliased: walk the snapshot
            // map and copy each group's first canonical Rc into every
            // peer slot, restoring the pre-GC `Rc::as_ptr` equivalences.
            let mut canonical_by_old: std::collections::HashMap<usize, Rc<VirtualStateInfoNode>> =
                std::collections::HashMap::new();
            for (slot_idx, &old_ptr) in original_ptrs.iter().enumerate() {
                let entry = canonical_by_old
                    .entry(old_ptr)
                    .or_insert_with(|| Rc::clone(&self.virtual_state.state[slot_idx]));
                if !Rc::ptr_eq(entry, &self.virtual_state.state[slot_idx]) {
                    self.virtual_state.state[slot_idx] = Rc::clone(entry);
                }
            }
            // Rc identities have shifted, so position cells must be
            // re-derived against the canonical state graph.
            self.virtual_state.enum_top_level();
        }
    }

    /// Release shadow stack roots.
    fn release_roots(&mut self) {
        if !self.rooted_refs.is_empty() {
            majit_gc::shadow_stack::pop_to(self.shadow_stack_base);
            self.rooted_refs.clear();
        }
    }

    /// unroll.py: final() — ExportedState is never final (loop continues).
    pub fn is_final(&self) -> bool {
        false
    }
}

impl Clone for ExportedState {
    /// Pure data clone — no shadow stack side effects.
    ///
    /// RPython has no clone (single Python object shared by reference).
    /// When a Rust clone is stored long-term (across potential GC points),
    /// the caller must call root_all_gcrefs() explicitly.
    fn clone(&self) -> Self {
        ExportedState {
            end_args: self.end_args.clone(),
            next_iteration_args: self.next_iteration_args.clone(),
            end_arg_types: self.end_arg_types.clone(),
            virtual_state: self.virtual_state.clone(),
            exported_infos: self.exported_infos.clone(),
            exported_short_boxes: self.exported_short_boxes.clone(),
            short_boxes: self.short_boxes.clone(),
            short_box_const_values: self.short_box_const_values.clone(),
            short_preamble: self.short_preamble.clone(),
            renamed_inputargs: self.renamed_inputargs.clone(),
            renamed_inputarg_types: self.renamed_inputarg_types.clone(),
            short_inputargs: self.short_inputargs.clone(),
            runtime_boxes: self.runtime_boxes.clone(),
            patchguardop: self.patchguardop.clone(),
            phase1_emit_high_water: self.phase1_emit_high_water,
            box_pool_snapshot: self.box_pool_snapshot.clone(),
            rooted_refs: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
    }
}

impl Drop for ExportedState {
    fn drop(&mut self) {
        self.release_roots();
    }
}

/// history.py: TargetToken — describes one compiled version of a loop.
///
/// Each peeled loop body creates a TargetToken that records the virtual state
/// and short preamble needed for bridge entry. Multiple TargetTokens can exist
/// per loop (from retracing with different virtual states).
#[derive(Clone, Debug)]
pub struct TargetToken {
    /// RPython history.py: identity of this target token within the current
    /// JitCellToken.target_tokens list. Debug-display only — backend identity
    /// is the `jump_target_descr` Arc address.
    pub token_id: u64,
    /// compile.py: start_descr — the preamble target token has no virtual
    /// state and lives at target_tokens[0].
    pub is_preamble_target: bool,
    /// Virtual state at this loop entry point.
    /// Used by _jump_to_existing_trace to check compatibility.
    pub virtual_state: Option<crate::optimizeopt::virtualstate::VirtualState>,
    /// Short preamble: ops to replay when entering from a bridge.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// RPython unroll.py: active ExtendedShortPreambleBuilder for the target
    /// token currently being finalized.
    pub short_preamble_producer:
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    jump_target_descr: Arc<LoopTargetDescr>,
}

impl TargetToken {
    pub fn new() -> Self {
        TargetToken {
            token_id: 0,
            is_preamble_target: false,
            virtual_state: None,
            short_preamble: None,
            short_preamble_producer: None,
            jump_target_descr: Arc::new(LoopTargetDescr::new(0, false)),
        }
    }

    pub fn new_loop(token_id: u64) -> Self {
        let mut token = Self::new();
        token.token_id = token_id;
        token.jump_target_descr = Arc::new(LoopTargetDescr::new(token_id, false));
        token
    }

    pub fn new_preamble(token_id: u64) -> Self {
        let mut token = Self::new();
        token.token_id = token_id;
        token.is_preamble_target = true;
        token.jump_target_descr = Arc::new(LoopTargetDescr::new(token_id, true));
        token
    }

    pub fn as_jump_target_descr(&self) -> majit_ir::DescrRef {
        self.jump_target_descr.clone()
    }

    /// `compile.py:237` / `compile.py:289` — bind the freshly-made
    /// JitCellToken's `number` to this TargetToken's `original_jitcell_token`.
    /// Walker (`record_loop_or_bridge`, `compile.py:197-199`) reads this to
    /// determine whether a JUMP crosses to a different loop.
    pub fn set_original_jitcell_token_number(&self, num: u64) {
        majit_ir::LoopTargetDescr::set_original_jitcell_token_number(
            self.jump_target_descr.as_ref(),
            num,
        );
    }
}

#[derive(Debug, Default)]
struct LoopTargetDescrState {
    ll_loop_code: usize,
    target_arglocs: Vec<majit_ir::TargetArgLoc>,
    /// `history.py:493 self.original_jitcell_token`. Backfilled once the
    /// owning JitCellToken is created (`pyjitpl/mod.rs:3853` etc., the
    /// counterpart to `compile.py:237` / `compile.py:289`).
    original_jitcell_token_number: Option<u64>,
}

#[derive(Debug)]
struct LoopTargetDescr {
    token_id: u64,
    is_preamble_target: bool,
    state: Mutex<LoopTargetDescrState>,
}

impl LoopTargetDescr {
    fn new(token_id: u64, is_preamble_target: bool) -> Self {
        Self {
            token_id,
            is_preamble_target,
            state: Mutex::new(LoopTargetDescrState::default()),
        }
    }
}

impl majit_ir::Descr for LoopTargetDescr {
    fn index(&self) -> u32 {
        self.token_id as u32
    }

    fn repr(&self) -> String {
        if self.is_preamble_target {
            format!("LoopTargetDescr(start:{})", self.token_id)
        } else {
            format!("LoopTargetDescr({})", self.token_id)
        }
    }

    fn as_loop_target_descr(&self) -> Option<&dyn majit_ir::LoopTargetDescr> {
        Some(self)
    }
}

impl majit_ir::LoopTargetDescr for LoopTargetDescr {
    fn token_id(&self) -> u64 {
        self.token_id
    }

    fn is_preamble_target(&self) -> bool {
        self.is_preamble_target
    }

    fn ll_loop_code(&self) -> usize {
        self.state.lock().unwrap().ll_loop_code
    }

    fn set_ll_loop_code(&self, loop_code: usize) {
        self.state.lock().unwrap().ll_loop_code = loop_code;
    }

    fn target_arglocs(&self) -> Vec<majit_ir::TargetArgLoc> {
        self.state.lock().unwrap().target_arglocs.clone()
    }

    fn set_target_arglocs(&self, arglocs: Vec<majit_ir::TargetArgLoc>) {
        self.state.lock().unwrap().target_arglocs = arglocs;
    }

    fn original_jitcell_token_number(&self) -> Option<u64> {
        self.state.lock().unwrap().original_jitcell_token_number
    }

    fn set_original_jitcell_token_number(&self, num: u64) {
        self.state.lock().unwrap().original_jitcell_token_number = Some(num);
    }
}

/// unroll.py: UnrollInfo(BasicLoopInfo) — return type from optimize_peeled_loop.
///
/// Carries the target_token, label_op, and extra_same_as needed to
/// finalize compilation after the peeled loop body is optimized.
#[derive(Clone, Debug)]
pub struct UnrollInfo {
    /// The target token for this loop's entry point.
    pub target_token: u64,
    /// Extra same_as ops added during finalization.
    pub extra_same_as: Vec<Op>,
    /// Quasi-immutable dependencies discovered during optimization.
    pub quasi_immutable_deps: std::collections::HashSet<(u64, u32)>,
    /// Extra ops to insert before the label (from bridge inlining).
    pub extra_before_label: Vec<Op>,
}

impl UnrollInfo {
    /// unroll.py: final() — UnrollInfo is always final.
    pub fn is_final(&self) -> bool {
        true
    }
}

impl OptUnroll {
    /// unroll.py:264-272 disable_retracing_if_max_retrace_guards.
    ///
    /// When the peeled body contains more guards than `max_retrace_guards`,
    /// set `retraced_count = u32::MAX` on the targeting JitCellToken so that
    /// future bridges jump to the preamble instead of requesting a retrace.
    pub fn disable_retracing_if_max_retrace_guards(
        ops: &[majit_ir::Op],
        retraced_count: &mut u32,
        max_retrace_guards: u32,
    ) {
        let guard_count = ops.iter().filter(|op| op.opcode.is_guard()).count();
        if guard_count > max_retrace_guards as usize {
            *retraced_count = u32::MAX;
        }
    }

    /// unroll.py: export_state — capture optimizer state at end of preamble.
    ///
    /// After the preamble is optimized, snapshot:
    /// - end_args: forced versions of label args
    /// - virtual_state: abstract info for loop-carried values
    /// - short boxes: mapping of preamble ops to label args
    pub fn export_state(
        &self,
        original_label_args: &[OpRef],
        renamed_inputargs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> ExportedState {
        self.export_state_with_bounds(original_label_args, renamed_inputargs, optimizer, ctx, None)
    }

    /// unroll.py:452-477: export_state implementation.
    fn export_state_with_bounds(
        &self,
        original_label_args: &[OpRef],
        renamed_inputargs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
    ) -> ExportedState {
        // unroll.py:454: end_args = [force_at_the_end_of_preamble(a) ...]
        let end_args: Vec<OpRef> = ctx.preamble_end_args.clone().unwrap_or_else(|| {
            original_label_args
                .iter()
                .map(|&a| ctx.get_box_replacement(a))
                .collect()
        });
        // unroll.py:457 `virtual_state = self.get_virtual_state(end_args)`
        // — VS captured AFTER `force_box_for_end_of_preamble` and AFTER
        // `flush()`. The caller (`Optimizer::optimize_with_constants_and_inputs_at`)
        // already ran both passes before invoking us, so `end_args` is in
        // the same post-force, post-flush state RPython feeds in.
        let virtual_state = crate::optimizeopt::virtualstate::export_state(&end_args, ctx);
        // unroll.py:459-461: infos = {}; for arg in end_args: _expand_info(arg, infos)
        let mut infos: HashMap<OpRef, crate::optimizeopt::info::OpInfo> = HashMap::new();
        for &arg in &end_args {
            self.expand_info(arg, ctx, exported_int_bounds, &mut infos);
        }
        // unroll.py:462-463 `label_args, virtuals =
        //   virtual_state.make_inputargs_and_virtuals(end_args, self.optimizer)`.
        let (label_args, virtuals) = virtual_state
            .make_inputargs_and_virtuals(&end_args, optimizer, ctx, false)
            .expect("export_state make_inputargs_and_virtuals failed");
        // unroll.py:464-465: for arg in label_args: _expand_info(arg, infos)
        for &arg in &label_args {
            self.expand_info(arg, ctx, exported_int_bounds, &mut infos);
        }
        let mut short_args = label_args.to_vec();
        short_args.extend(virtuals);
        let exported_short_boxes = ctx.exported_short_boxes.clone();
        // unroll.py:466-473 line-by-line:
        //
        //   sb = ShortBoxes()
        //   short_boxes = sb.create_short_boxes(self.optimizer,
        //       renamed_inputargs, label_args + virtuals)
        //   short_inputargs = sb.create_short_inputargs(label_args + virtuals)
        //   for produced_op in short_boxes:
        //       op = produced_op.short_op.res
        //       if not isinstance(op, Const):
        //           self._expand_info(op, infos)
        //
        // RPython expands info from the post-`create_short_boxes` list (the
        // same `short_boxes` that survives into `ExportedState.short_boxes`).
        // pyre's analog is `produced_short_boxes_from_exported_boxes` (which
        // applies the GuardOverflow filter and label-arg → short-inputarg
        // rename); iterating the raw `exported_short_boxes` here would
        // expand info for entries (e.g. standalone `GuardOverflow`) that
        // PyPy never carries into `short_boxes`, polluting the dict.
        let short_boxes_for_info =
            crate::optimizeopt::shortpreamble::produced_short_boxes_from_exported_boxes(
                &label_args,
                &short_args,
                &exported_short_boxes,
            );
        for (_, produced_op) in &short_boxes_for_info {
            let op = produced_op.preamble_op.pos;
            if !op.is_constant() {
                self.expand_info(op, ctx, exported_int_bounds, &mut infos);
            }
        }

        // RPython unroll.py:467: next_iteration_args = end_args (post-force).
        // Aliased boxes (same resolved OpRef) are handled by export_state's
        // create_state cache + make_inputargs' position_in_notvirtuals dedup.
        // unroll.py:454 `end_args` are typed Boxes — port reads each arg's
        // authoritative type from `ctx` at export time so downstream consumers
        // never have to reconstruct types from op result shape.
        // RPython Box type parity: `renamed_inputargs` are the fresh per-iteration
        // InputArg Boxes (unroll.py `trace.get_iter()` inputargs). Their types
        // are intrinsic on RPython Boxes; pyre reconstructs them via
        // `ctx.opref_type` for each renamed OpRef directly. Using
        // `end_arg_types` (computed over `end_args`) here would be wrong:
        // `end_args` are the post-force JUMP operands, which may be distinct
        // OpRefs from the renamed inputargs (and can have different types
        // when forcing virtuals replaces an InputArg slot with a fresh
        // NewWithVtable / SameAs alias).
        // RPython unroll.py:452 `export_state()` reads each `info.inputargs`
        // Box's `.type` (history.py:220 InputArg{Int,Ref,Float}) directly —
        // the inputarg Box's intrinsic type, not the post-forwarding
        // target's. Pyre's `renamed_inputargs` are the canonical inputarg
        // slot OpRefs; reading `inputarg_type` *before* `opref_type`
        // forwarding mirrors RPython's Box-identity lookup. If the slot
        // was forwarded onto something that itself lacks a recorded type
        // (e.g. a virtual-head SameAs alias whose `value_types` entry
        // was Void-filtered upstream), the fall-through `opref_type` path
        // still resolves it, and finally the strict panic surfaces the
        // genuine missing-Box.type case RPython would never reach.
        let renamed_inputarg_types: Vec<Type> = renamed_inputargs
            .iter()
            .map(|&opref| {
                ctx.inputarg_type(opref)
                    .or_else(|| ctx.opref_type(opref))
                    .unwrap_or_else(|| {
                        panic!(
                            "missing type for renamed inputarg {:?} in export_state \
                             (history.py:220 Box.type invariant)",
                            opref
                        )
                    })
            })
            .collect();
        // RPython parity: store next_iteration_args in their post-forwarding
        // form so the cache key invariant established by `export_single_value`
        // (which uses `get_box_replacement` as the cache key, virtualstate.rs:2213)
        // survives the phase boundary. `import_state` later runs in a fresh
        // Phase 2 ctx that does NOT inherit Phase 1's forwarding map; if we
        // stored raw `end_args[i]` here, two top-level entries that shared
        // the same `VirtualStateInfoNode` Rc (because their post-resolution
        // targets coincided in Phase 1) would `replace_op(source(i), raw_i)`
        // to *different* raw OpRefs, and `make_inputargs` in Phase 2 would
        // see divergent forwarding for the supposedly-shared state slot.
        // RPython sidesteps this naturally because `box._forwarded` persists
        // across phases. virtualstate.py make_inputargs assumes shared state
        // ⇒ same forwarded target.
        // unroll.py:458 `end_args = [get_box_replacement(arg) for arg in end_args]`.
        let resolved_next_iteration_args: Vec<OpRef> = end_args
            .iter()
            .map(|&a| ctx.get_box_replacement(a))
            .collect();
        // Phase B B1: `produced_short_boxes` is derived from
        // `exported_short_boxes` lazily at the consumer site
        // (`build_short_preamble_from_produced_boxes` in `import_state`)
        // via `produced_short_boxes_from_exported_boxes`
        // (`shortpreamble.rs:2100`). Storing both shapes on
        // `ExportedState` would risk drift across mutations.
        let mut state = ExportedState::new(
            label_args.clone(),
            resolved_next_iteration_args,
            virtual_state,
            infos,
            exported_short_boxes,
            renamed_inputargs.to_vec(),
            renamed_inputarg_types,
            short_args,
        );
        // Smallest fresh OpRef strictly above every Phase 1 emit position.
        // Retrace's `opref_high_water` consults this so Phase 2's namespace
        // stays disjoint from intermediate Phase 1 ops that were forwarded
        // and never survive as an end_arg / short_op source. `op.pos.0`
        // is already non-NONE / non-Void by the rebuild filter.
        state.phase1_emit_high_water = optimizer
            .phase1_emit_ops
            .iter()
            .map(|op| op.pos.raw().saturating_add(1))
            .max()
            .unwrap_or(0);
        // Capture the full Phase 1 BoxRef pool so a subsequent
        // `compile_retrace` attempt can hand it back to Phase 2 as
        // `p1_full_prefix` (`unroll.rs:282-303`). PyPy `unroll.py` Phase 2
        // observes Phase 1's `_forwarded` mutations through Python identity
        // on the same Box objects; pyre needs the Rc::cloned BoxRef vector
        // to recreate that observation across the in-memory ExportedState
        // hand-off. Each entry is `Rc::clone` cheap; vec retention is
        // bounded by `ExportedState` lifetime.
        state.box_pool_snapshot = Some(ctx.box_pool.to_vec());
        // PRE-EXISTING-ADAPTATION: snapshot producer-side const values for
        // any const-namespace OpRef referenced by `short_boxes` op args.
        // Phase B.2 `ProducedShortOp::produce_op` reads raw OpRefs (not the
        // legacy `ExportedShortArg::Const { source, value }` enum), so we
        // capture the const value here for the consumer's
        // `classify_short_arg` to find. Bridges and unit tests run with
        // a consumer ctx that may not have the producer's const slot
        // populated; without this snapshot, the import path silently
        // skips the short op.
        for (_, produced) in &state.short_boxes {
            for &arg in &produced.preamble_op.args {
                if !arg.is_constant() {
                    continue;
                }
                if state.short_box_const_values.contains_key(&arg) {
                    continue;
                }
                if let Some(value) = ctx.get_constant(arg) {
                    state.short_box_const_values.insert(arg, value);
                }
            }
        }
        state
    }

    /// unroll.py:432-443: _expand_info
    fn expand_info(
        &self,
        arg: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
        infos: &mut HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
    ) {
        let resolved = ctx.get_box_replacement(arg);
        if infos.contains_key(&resolved) {
            // Also store under the original key so import_state can
            // find the info using the unresolved next_iteration_args key.
            if arg != resolved {
                if let Some(info) = infos.get(&resolved).cloned() {
                    infos.insert(arg, info);
                }
            }
            return;
        }
        // unroll.py:438-443 `_expand_info`:
        //     if arg in infos:
        //         return
        //     if info:
        //         infos[arg] = info
        //         if info.is_virtual():
        //             self._expand_infos_from_virtual(info, infos)
        //
        // RPython stores the entry only when `info` is truthy — a falsy
        // `info` (None) simply skips the insert, so downstream
        // `setinfo_from_preamble_list` at unroll.py:45-51 sees "no entry"
        // and drops any inherited forwarded via `item.set_forwarded(None)`.
        let Some(info) = self.collect_exported_info(resolved, ctx, exported_int_bounds) else {
            return;
        };
        let has_fields = matches!(ctx.peek_ptr_info_via_box(resolved), Some(pi) if pi.is_virtual() || !pi.all_items().is_empty());
        infos.insert(resolved, info.clone());
        // Also store under the original (unresolved) key.
        if arg != resolved {
            infos.insert(arg, info);
        }
        if has_fields {
            self.expand_infos_from_virtual(resolved, ctx, exported_int_bounds, infos);
        }
    }

    /// unroll.py:445-450: _expand_infos_from_virtual
    fn expand_infos_from_virtual(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
        infos: &mut HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
    ) {
        let fields: Vec<OpRef> = match ctx.peek_ptr_info_via_box(opref) {
            Some(crate::optimizeopt::info::PtrInfo::Virtual(v)) => {
                v.fields.iter().map(|(_, r)| *r).collect()
            }
            Some(crate::optimizeopt::info::PtrInfo::VirtualStruct(v)) => {
                v.fields.iter().map(|(_, r)| *r).collect()
            }
            Some(crate::optimizeopt::info::PtrInfo::VirtualArray(v)) => v.items.clone(),
            Some(crate::optimizeopt::info::PtrInfo::Instance(v)) if !v.fields.is_empty() => {
                v.fields.iter().map(|(_, e)| e.as_seen_opref()).collect()
            }
            Some(crate::optimizeopt::info::PtrInfo::Struct(v)) if !v.fields.is_empty() => {
                v.fields.iter().map(|(_, e)| e.as_seen_opref()).collect()
            }
            _ => return,
        };
        for field in fields {
            if field.is_none() {
                continue;
            }
            self.expand_info(field, ctx, exported_int_bounds, infos);
        }
    }

    /// unroll.py:284-301: finalize_short_preamble — create a TargetToken
    /// and attach the short preamble to it. Called at the end of
    /// optimize_peeled_loop after the loop body is optimized.
    ///
    /// Returns the new TargetToken with virtual_state and short_preamble set.
    pub fn finalize_short_preamble(
        &self,
        token_id: u64,
        virtual_state: crate::optimizeopt::virtualstate::VirtualState,
        short_preamble: crate::optimizeopt::shortpreamble::ShortPreamble,
        short_preamble_builder: Option<&crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    ) -> TargetToken {
        let mut target_token = TargetToken::new();
        target_token.token_id = token_id;
        target_token.jump_target_descr = Arc::new(LoopTargetDescr::new(token_id, false));
        target_token.virtual_state = Some(virtual_state);
        target_token.short_preamble = Some(short_preamble);
        target_token.short_preamble_producer = short_preamble_builder.map(|builder| {
            crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder::new(token_id, builder)
        });
        target_token
    }

    /// unroll.py:320-362: _jump_to_existing_trace — check if any existing
    /// compiled trace (target_token) has a compatible virtual state.
    /// If so, generate extra guards, inline short preamble, and redirect jump.
    ///
    /// Returns None if jumped successfully, Some(virtual_state) otherwise.
    /// unroll.py:304-362: jump_to_existing_trace
    ///
    /// `runtime_boxes`: live values at the jump point, used by
    /// generate_guards to emit GUARD_VALUE when the runtime value
    /// matches a known constant. Pass None when unavailable (bridges).
    pub fn jump_to_existing_trace(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &mut [TargetToken],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: Option<&[OpRef]>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        self.jump_to_existing_trace_with_vs(
            jump_args,
            current_label_args,
            target_tokens,
            optimizer,
            ctx,
            force_boxes,
            runtime_boxes,
            None,
        )
    }

    /// Like jump_to_existing_trace, but with an optional pre-computed
    /// virtual_state. Used by optimize_bridge where force_at_the_end_of_preamble
    /// may change forwarding chains after the virtual state was exported.
    pub fn jump_to_existing_trace_with_vs(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &mut [TargetToken],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: Option<&[OpRef]>,
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        optimizer.disable_guard_replacement();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.jump_to_existing_trace_impl(
                jump_args,
                current_label_args,
                target_tokens,
                optimizer,
                ctx,
                force_boxes,
                runtime_boxes,
                pre_vs,
            )
        }));
        optimizer.enable_guard_replacement();
        match result {
            Ok(vs) => vs,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn jump_to_existing_trace_impl(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &mut [TargetToken],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: Option<&[OpRef]>,
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        let mut virtual_state = pre_vs
            .unwrap_or_else(|| crate::optimizeopt::virtualstate::export_state(jump_args, ctx));
        let mut args: Vec<OpRef> = jump_args
            .iter()
            .map(|&a| ctx.get_box_replacement(a))
            .collect();

        for (tt_idx, target_token) in target_tokens.iter_mut().enumerate() {
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!("[jit][jump_to_existing] trying target_token #{tt_idx}");
            }
            let target_vs = match &target_token.virtual_state {
                Some(vs) => vs,
                None => continue,
            };

            // RPython unroll.py:333: patchguardop = self.optimizer.patchguardop
            // Ensure ctx.patchguardop is set before generate_guards so that
            // extra guards get a valid rd_resume_position (resume.py:397).
            if ctx.patchguardop.is_none() {
                ctx.patchguardop = optimizer.patchguardop.clone();
            }

            // RPython unroll.py:315 parity: try generate_guards directly
            // instead of gating on generalization_of. If guards can't be
            // generated (VirtualStatesCantMatch), skip this target.
            let extra_guards = match target_vs.generate_guards(
                &virtual_state,
                &args,
                runtime_boxes,
                ctx,
                force_boxes,
            ) {
                Ok(guards) => guards,
                Err(()) => {
                    if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} generate_guards failed (force_boxes={force_boxes})",
                        );
                    }
                    continue;
                }
            };
            // unroll.py:333-338 parity: read patchguardop.rd_resume_position
            // and stamp it onto every extra guard. RPython has no fallback —
            // `optimize_GUARD_FUTURE_CONDITION` (rewrite.py / simplify.py) runs
            // unconditionally on the GUARD_FUTURE_CONDITION emitted at
            // `reached_loop_header` (pyjitpl.py:2969), so by the time
            // `_jump_to_existing_trace` runs `self.optimizer.patchguardop` is
            // always populated. pyre mirrors this: `close_loop_args_at`
            // (trace_opcode.rs:1863) emits the same GFC, and Phase 1 captures
            // it into `patchguardop` (rewrite.rs:2737), which Phase 2 inherits
            // (unroll.rs:513). Skip the read when `extra_guards` is empty —
            // RPython's `for guard in extra_guards.extra_guards` loop body is
            // also skipped in that case.
            for guard_req in &extra_guards {
                if let Some(mut guard_op) = guard_req.to_op(&args, ctx) {
                    let patch = ctx.patchguardop.as_ref().unwrap_or_else(|| {
                        panic!(
                            "unroll.py:333 invariant: patchguardop must be set \
                             when extra_guards is non-empty (target_token #{}, \
                             {} extra guard(s), force_boxes={})",
                            tt_idx,
                            extra_guards.len(),
                            force_boxes,
                        )
                    });
                    if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                        let arg_values: Vec<_> = guard_op
                            .args
                            .iter()
                            .map(|&arg| (arg, ctx.get_constant(arg)))
                            .collect();
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} emit guard {:?} from {:?} args={:?}",
                            guard_op.opcode, guard_req, arg_values
                        );
                    }
                    // unroll.py:336: guard.rd_resume_position = patchguardop.rd_resume_position
                    guard_op.rd_resume_position = patch.rd_resume_position;
                    guard_op.descr = Some(crate::optimizeopt::make_resume_at_position_descr());
                    optimizer.send_extra_operation(&guard_op, ctx);
                }
            }

            // unroll.py:346-347: make_inputargs_and_virtuals
            // RPython: force_box emits New/SetfieldGc via emit_extra which
            // routes through passes AFTER Virtualize. The non-virtual PtrInfo
            // (Struct/Instance) on alloc_ref prevents re-absorption.
            let (target_args, virtuals) = match target_vs.make_inputargs_and_virtuals(
                &args,
                optimizer,
                ctx,
                force_boxes,
            ) {
                Ok(result) => result,
                Err(()) => {
                    if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} make_inputargs failed (force_boxes={force_boxes})",
                        );
                    }
                    if force_boxes {
                        args = jump_args
                            .iter()
                            .map(|&a| ctx.get_box_replacement(a))
                            .collect();
                        virtual_state = crate::optimizeopt::virtualstate::export_state(&args, ctx);
                    }
                    continue;
                }
            };
            let mut short_jump_args = target_args.clone();
            short_jump_args.extend(virtuals);

            // Ensure jump_args carry PtrInfo from Phase 2 body.
            // RPython Box identity preserves info across forwarding.
            // In majit, forwarding target may lack PtrInfo — propagate
            // from the original label arg (before forwarding).
            if let Some(label) = current_label_args {
                for (i, &jump_arg) in short_jump_args.iter().enumerate() {
                    let resolved = ctx.get_box_replacement(jump_arg);
                    if !ctx.has_ptr_info_via_box(resolved) {
                        // Try label arg at same index
                        if let Some(&label_arg) = label.get(i) {
                            if let Some(info) = ctx.peek_ptr_info_via_box(label_arg) {
                                ctx.ensure_ptr_info_preserve_forwarding(resolved, info);
                            }
                        }
                    }
                }
            }

            // unroll.py:353-356: inline short preamble
            let mut extra = Vec::new();
            if let Some(sp) = target_token.short_preamble.clone() {
                if let Some(mut builder) = target_token.short_preamble_producer.take() {
                    if let Some(label_args) = current_label_args {
                        // shortpreamble.py:283-296 / 311-341 parity:
                        // setup() returns false when an op references an
                        // unresolvable Phase 1 OpRef. Treat this exactly
                        // like RPython's "produce_arg returned None →
                        // add_op_to_short returned None" path: drop the
                        // peeled trace and let the unroll caller raise
                        // InvalidLoop, falling back to jump_to_preamble.
                        if !builder.setup(&sp, label_args) {
                            target_token.short_preamble_producer = Some(builder);
                            std::panic::panic_any(crate::optimize::InvalidLoop(
                                "short preamble has unresolvable Phase 1 args",
                            ));
                        }
                        ctx.activate_short_preamble_producer(builder);
                        extra = Self::inline_short_preamble(
                            &short_jump_args,
                            &target_args,
                            &sp,
                            optimizer,
                            ctx,
                        );
                        if let Some(builder) = ctx.take_active_short_preamble_producer() {
                            // RPython parity: extract constant pool from
                            // OptContext. In RPython, Const objects in short
                            // preamble ops are GC-tracked and survive across
                            // compilations. build_short_preamble_struct scans
                            // all op args to capture referenced constants.
                            let mut loop_constants: HashMap<u32, i64> = HashMap::new();
                            let mut loop_constant_types: HashMap<u32, majit_ir::Type> =
                                optimizer.constant_types.clone();
                            for (&const_idx, val) in &ctx.const_pool {
                                let (raw, tp) = match val {
                                    majit_ir::Value::Int(v) => (*v, majit_ir::Type::Int),
                                    majit_ir::Value::Float(f) => {
                                        (f.to_bits() as i64, majit_ir::Type::Float)
                                    }
                                    majit_ir::Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                                    majit_ir::Value::Void => (0, majit_ir::Type::Void),
                                };
                                let opref = OpRef::const_typed(const_idx, tp);
                                loop_constants.insert(opref.raw(), raw);
                                loop_constant_types.entry(opref.raw()).or_insert(tp);
                            }
                            // Merge previous short preamble's constants.
                            // RPython's Const objects survive across compilations
                            // via GC tracing. In majit, we must carry forward
                            // constants from the previous build that may not
                            // exist in the current Phase 2 context.
                            for (&k, &(v, tp)) in &sp.constants {
                                loop_constants.entry(k).or_insert(v);
                                loop_constant_types.entry(k).or_insert(tp);
                            }
                            target_token.short_preamble =
                                Some(builder.build_short_preamble_struct(
                                    &loop_constants,
                                    &loop_constant_types,
                                ));
                            target_token.short_preamble_producer = Some(builder);
                        }
                    } else {
                        extra = Self::inline_short_preamble(
                            &short_jump_args,
                            &target_args,
                            &sp,
                            optimizer,
                            ctx,
                        );
                        target_token.short_preamble_producer = Some(builder);
                    }
                } else {
                    extra = Self::inline_short_preamble(
                        &short_jump_args,
                        &target_args,
                        &sp,
                        optimizer,
                        ctx,
                    );
                    if crate::optimizeopt::majit_log_enabled() {
                        eprintln!("[jit][jte-isp] done, extra_len={}", extra.len());
                    }
                }
            }

            // unroll.py:357-359: emit JUMP to target
            let mut jump_args = target_args;
            jump_args.extend(extra);
            let mut jump = Op::new(OpCode::Jump, &jump_args);
            jump.descr = Some(target_token.as_jump_target_descr());
            optimizer.send_extra_operation(&jump, ctx);
            return None; // successfully jumped
        }

        Some(virtual_state)
    }

    /// unroll.py: inline_short_preamble — replay short preamble ops
    /// to re-populate the optimizer's cache when entering from a bridge.
    ///
    /// Maps short preamble input args to the jump args, then emits
    /// each short preamble op with remapped arguments.
    pub fn inline_short_preamble(
        jump_args: &[OpRef],
        args_no_virtuals: &[OpRef],
        short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        // RPython parity: in RPython, short preamble ops embed Const objects
        // (ConstPtr/ConstInt) that carry their own value and are GC-tracked.
        // _map_args skips Const args — they're always valid because the GC
        // keeps referenced objects alive via JitCellToken → TargetToken →
        // short_preamble → ResOperation → ConstPtr chain.
        //
        // In pyre, short preamble ops use OpRef indices from the loop's
        // constant pool. These OpRefs aren't in the bridge's context.
        // Register them as constants in the bridge's OptContext so the ops
        // can reference them correctly.
        for (&idx, &(val, tp)) in &short_preamble.constants {
            let value = match tp {
                majit_ir::Type::Int => majit_ir::Value::Int(val),
                majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(val as usize)),
                majit_ir::Type::Float => majit_ir::Value::Float(f64::from_bits(val as u64)),
                majit_ir::Type::Void => majit_ir::Value::Int(val),
            };
            // history.py:220/261/307: ConstInt/ConstFloat/ConstPtr are
            // disjoint Box subclasses pinned to `box.type` at construction.
            // Mint the typed `OpRef::const_*` variant so the imported
            // preamble const slot collides with the bridge's intra-pool
            // dedup (`get_or_insert_typed` returns the same typed OpRef
            // when value+type match) under variant-aware OpRef Eq
            // (resoperation.rs:290) — `Untyped(idx)` from this loop and
            // `ConstInt(idx)` from the bridge's `get_or_insert` would
            // otherwise hash as separate keys for the same slot.
            let typed_opref = match tp {
                majit_ir::Type::Int | majit_ir::Type::Void => majit_ir::OpRef::const_int(idx),
                majit_ir::Type::Ref => majit_ir::OpRef::const_ptr(idx),
                majit_ir::Type::Float => majit_ir::OpRef::const_float(idx),
            };
            ctx.make_constant(typed_opref, value);
            optimizer.constant_types.entry(idx).or_insert(tp);
        }

        let mut mapping: HashMap<OpRef, OpRef> = HashMap::new();

        for (i, &short_inputarg) in short_preamble.inputargs.iter().enumerate() {
            if let Some(&jump_arg) = jump_args.get(i) {
                mapping.insert(short_inputarg, jump_arg);
                // RPython: jump_arg Box inherits info via identity.
                // In majit, propagate PtrInfo from short_inputarg (which
                // has info from Phase 1 export) to the resolved jump_arg.
                // shortpreamble.py:414-425 parity: propagate PtrInfo from
                // Phase 1 export to jump_args so guards are redundant.
                let resolved = ctx.get_box_replacement(jump_arg);
                if !ctx.has_ptr_info_via_box(resolved) {
                    let info = ctx
                        .peek_ptr_info_via_box(jump_arg)
                        .or_else(|| ctx.peek_ptr_info_via_box(short_inputarg))
                        .or_else(|| {
                            short_preamble
                                .inputarg_infos
                                .get(i)
                                .and_then(|opt| opt.clone())
                        });
                    if let Some(info) = info {
                        ctx.ensure_ptr_info_preserve_forwarding(resolved, info);
                    }
                }
            }
        }

        // RPython parity: also map Phase 1 inputargs → jump_args.
        // Short ops may reference Phase 1 OpRefs (from produce_arg's
        // label_arg_positions check) that aren't in the current inputargs.
        // In RPython, renamed inputargs are stable across compilations,
        // so this situation doesn't arise.
        if let Some(ref phase1) = short_preamble.phase1_inputargs {
            for (i, &phase1_inputarg) in phase1.iter().enumerate() {
                if let Some(&jump_arg) = jump_args.get(i) {
                    mapping.entry(phase1_inputarg).or_insert(jump_arg);
                }
            }
        }

        let mut replay_index = 0;

        fn current_short_len(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
        ) -> usize {
            ctx.active_short_preamble_producer
                .as_ref()
                .map(|builder| builder.short_ops_len())
                .unwrap_or_else(|| short_preamble.ops.len())
        }

        fn current_short_op(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
            index: usize,
        ) -> Option<Op> {
            if let Some(builder) = ctx.active_short_preamble_producer.as_ref() {
                builder.short_op(index).cloned()
            } else {
                short_preamble.ops.get(index).map(|entry| entry.op.clone())
            }
        }

        fn current_short_jump_args(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
        ) -> Vec<OpRef> {
            ctx.active_short_preamble_producer
                .as_ref()
                .map(|builder| builder.jump_args().to_vec())
                .unwrap_or_else(|| short_preamble.jump_args.clone())
        }

        // unroll.py:398-427: fix-point loop, runs only once in almost all cases.
        // RPython uses `while 1:` until convergence, but in practice it should
        // converge in very few iterations. Add a safety cap to prevent hangs.
        let mut fixpoint_iter = 0u32;
        loop {
            fixpoint_iter += 1;
            if fixpoint_iter > 20 {
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[jit][inline_short_preamble] fixpoint loop exceeded 20 iterations, breaking"
                    );
                }
                break;
            }
            // unroll.py:402: while i < len(short) - 1
            // Use LIVE length — newly added ops (from use_box during
            // send_extra_operation) are replayed in the same iteration.
            while replay_index < current_short_len(short_preamble, ctx) {
                let Some(sp_op) = current_short_op(short_preamble, ctx, replay_index) else {
                    break;
                };
                let mut new_op = sp_op.clone();
                // unroll.py:404: _map_args(mapping, sop.getarglist())
                // Const passes through unchanged, non-Const must be in mapping.
                for arg in &mut new_op.args {
                    // unroll.py:367: isinstance(box, Const) — true Const objects
                    // only (ConstInt/ConstPtr/ConstFloat). make_constant'd values
                    // are NOT Const objects — they are regular boxes with forwarded
                    // set to a Const, so they must go through the mapping.
                    if arg.is_constant() || short_preamble.constants.contains_key(&arg.raw()) {
                        continue;
                    }
                    // unroll.py:404: _map_args — non-Const must be in mapping.
                    // RPython: mapping is complete (seeded from short_inputargs →
                    // jump_args, extended by mapping[sop] = op). Missing keys
                    // indicate a structural mismatch (e.g., cross-loop bridge
                    // with incompatible short preamble). Raise InvalidLoop.
                    match mapping.get(arg) {
                        Some(&mapped) => *arg = mapped,
                        None => {
                            // RPython: _map_args raises KeyError for unmapped
                            // args. This is equivalent to InvalidLoop — the
                            // short preamble is structurally incompatible.
                            // Panic propagates through jump_to_existing_trace
                            // to the caller, which catches it and falls back
                            // to jump_to_preamble (unroll.py:154-158, 209-211).
                            if crate::optimizeopt::majit_log_enabled() {
                                eprintln!(
                                    "[jit] inline_short_preamble: unmapped arg {:?} in {:?} — InvalidLoop",
                                    arg, new_op.opcode
                                );
                            }
                            std::panic::panic_any(crate::optimize::InvalidLoop(
                                "inline_short_preamble: unmapped arg in short preamble",
                            ));
                        }
                    }
                }
                // unroll.py:405-414: unified guard/non-guard handling.
                // RPython: both guards and non-guards follow the same path:
                //   copy_and_change → mapping[sop] = op → send_extra_operation(op)
                if new_op.opcode.is_guard() {
                    // unroll.py:406-409: copy_and_change with ResumeAtPositionDescr.
                    // The fresh ResumeAtPositionDescr has an empty RdPayload, so
                    // the new guard reads None for every rd_* until
                    // store_final_boxes_in_guard repopulates them from the live
                    // snapshot.
                    new_op.descr = Some(crate::optimizeopt::make_resume_at_position_descr());
                    new_op.fail_args = None;
                    new_op.fail_arg_types = None;
                    // unroll.py:409: op.rd_resume_position = patchguardop.rd_resume_position
                    // RPython: patchguardop is always set (from GUARD_FUTURE_CONDITION).
                    if let Some(ref patch) = ctx.patchguardop {
                        new_op.rd_resume_position = patch.rd_resume_position;
                    }
                    // Re-register guard constant args from preamble's constant pool.
                    for &arg in &new_op.args {
                        if let Some(&(val, tp)) = short_preamble.constants.get(&arg.raw()) {
                            let value = match tp {
                                majit_ir::Type::Int => majit_ir::Value::Int(val),
                                majit_ir::Type::Ref => {
                                    majit_ir::Value::Ref(majit_ir::GcRef(val as usize))
                                }
                                majit_ir::Type::Float => {
                                    majit_ir::Value::Float(f64::from_bits(val as u64))
                                }
                                majit_ir::Type::Void => majit_ir::Value::Int(val),
                            };
                            ctx.make_constant(arg, value);
                        }
                    }
                } else if let Some(ref mut fail_args) = new_op.fail_args {
                    for arg in fail_args.iter_mut() {
                        if let Some(&mapped) = mapping.get(arg) {
                            *arg = mapped;
                        }
                    }
                }
                let new_ref = ctx.alloc_op_position_typed(new_op.result_type());
                new_op.pos = new_ref;
                // unroll.py:412-414: mapping[sop] = op; i += 1; send_extra_operation(op)
                // RPython sets mapping BEFORE send_extra_operation.
                mapping.insert(sp_op.pos, new_ref);
                replay_index += 1;
                optimizer.send_extra_operation(&new_op, ctx);
            }

            // unroll.py:417-423: force all except virtuals.
            loop {
                let short_jump_args = current_short_jump_args(short_preamble, ctx);
                let num_short_jump_args = short_jump_args.len();
                // unroll.py:420: _map_args(mapping, short_jump_args)
                // Const passes through, non-Const requires mapping.
                let mapped_jump_args: Vec<OpRef> = short_jump_args
                    .iter()
                    .map(|jump_arg| {
                        let mapped = if short_preamble.constants.contains_key(&jump_arg.raw())
                            || jump_arg.is_constant()
                        {
                            *jump_arg
                        } else {
                            mapping[jump_arg]
                        };
                        ctx.get_box_replacement(mapped)
                    })
                    .collect();
                // unroll.py:419-421
                for &arg in args_no_virtuals.iter().chain(mapped_jump_args.iter()) {
                    let _ = optimizer.force_box(arg, ctx);
                }
                if current_short_jump_args(short_preamble, ctx).len() == num_short_jump_args {
                    break;
                }
            }
            // unroll.py:424
            optimizer.flush(ctx);
            // unroll.py:426: done unless "short" has grown again
            if replay_index == current_short_len(short_preamble, ctx) {
                break;
            }
        }

        // RPython: get_box_replacement follows forwarding after mapping
        current_short_jump_args(short_preamble, ctx)
            .iter()
            .map(|&jump_arg| {
                let mapped = *mapping.get(&jump_arg).unwrap_or(&jump_arg);
                ctx.get_box_replacement(mapped)
            })
            .collect()
    }

    /// unroll.py:479-504 import_state — line-by-line port.
    ///
    /// ```python
    /// def import_state(self, targetargs, exported_state):
    ///     assert len(exported_state.next_iteration_args) == len(targetargs)
    ///     for i, target in enumerate(exported_state.next_iteration_args):
    ///         source = targetargs[i]
    ///         assert source is not target
    ///         source.set_forwarded(target)
    ///         info = exported_state.exported_infos.get(target, None)
    ///         if info is not None:
    ///             self.optimizer.setinfo_from_preamble(source, info,
    ///                                             exported_state.exported_infos)
    ///     label_args = exported_state.virtual_state.make_inputargs(
    ///         targetargs, self.optimizer)
    ///     self.short_preamble_producer = ShortPreambleBuilder(
    ///         label_args, exported_state.short_boxes,
    ///         exported_state.short_inputargs, exported_state.exported_infos,
    ///         self.optimizer)
    ///     for produced_op in exported_state.short_boxes:
    ///         produced_op.produce_op(self, exported_state.exported_infos)
    ///     return label_args
    /// ```
    ///
    /// In RPython the short-preamble setup is done by constructing
    /// `ShortPreambleBuilder` with `label_args` and looping over
    /// `exported_state.short_boxes` calling `produce_op`.  In majit the
    /// equivalent is `import_short_preamble_ops` (a serialized form of
    /// the per-PreambleOp `produce_op` body) plus
    /// `initialize_imported_short_preamble_builder_from_exported_ops`
    /// (a serialized form of the `ShortPreambleBuilder` constructor).
    /// Both consume the same `short_args = label_args + virtuals` slot
    /// space that `export_state` used to build `ExportedShortOp` (see
    /// `collect_exported_short_ops`); the slot indices are the
    /// majit-only stand-in for RPython's Box-keyed `produced_short_boxes`
    /// dict.
    pub fn import_state(
        &self,
        targetargs: &[OpRef],
        exported_state: &ExportedState,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        // assert len(exported_state.next_iteration_args) == len(targetargs)
        assert_eq!(
            exported_state.next_iteration_args.len(),
            targetargs.len(),
            "import_state: next_iteration_args mismatch"
        );
        // for i, target in enumerate(exported_state.next_iteration_args):
        for (i, target) in exported_state.next_iteration_args.iter().enumerate() {
            // source = targetargs[i]
            let source = targetargs[i];
            // assert source is not target — see commit log for the
            // disjoint-namespace invariant from Step 2 Commit D2 that
            // makes this hold by construction in production callers.
            debug_assert!(source != *target, "import_state: source is target");
            // source.set_forwarded(target)
            ctx.replace_op(source, *target);
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!("[jit] import_state_map[{i}]: source={source:?} target={target:?}");
            }
            // info = exported_state.exported_infos.get(target, None)
            // if info is not None:
            if let Some(info) = exported_state.exported_infos.get(target) {
                //     self.optimizer.setinfo_from_preamble(source, info,
                //                                     exported_state.exported_infos)
                self.setinfo_from_preamble(source, info, &exported_state.exported_infos, ctx);
            }
        }
        // label_args = exported_state.virtual_state.make_inputargs(
        //     targetargs, self.optimizer)
        let label_args = match exported_state
            .virtual_state
            .make_inputargs(targetargs, optimizer, ctx, false)
        {
            Ok(args) => args,
            Err(()) => {
                std::panic::panic_any(crate::optimize::InvalidLoop(
                    "Cannot import state, virtual states don't match",
                ));
            }
        };
        // The short-preamble replay part of unroll.py:496-502 is performed
        // by import_short_preamble_state after majit's split-out
        // Optimizer::install_imported_virtuals has completed. RPython's
        // virtual_state.make_inputargs installs virtual PtrInfo before
        // produced_op.produce_op; doing it earlier here replays short ops
        // against incomplete virtual state.
        // return label_args
        label_args
    }

    pub fn import_short_preamble_state(
        &self,
        targetargs: &[OpRef],
        label_args: &[OpRef],
        exported_state: &ExportedState,
        ctx: &mut OptContext,
    ) {
        // self.short_preamble_producer = ShortPreambleBuilder(
        //     label_args, exported_state.short_boxes,
        //     exported_state.short_inputargs, exported_state.exported_infos,
        //     self.optimizer)
        //
        // majit's `ShortPreambleBuilder` constructor is the
        // `initialize_imported_short_preamble_builder_from_exported_ops`
        // call below. The majit serialization keys short ops by slot index
        // over the combined `label_args + virtuals` slot space (the same
        // space `collect_exported_short_ops` built against), so we compute
        // the virtuals tail inline before calling the initializer.
        let virtuals: Vec<OpRef> = exported_state
            .virtual_state
            .state
            .iter()
            .enumerate()
            .filter(|(_, info)| info.is_virtual())
            .filter_map(|(i, _)| targetargs.get(i).copied())
            .collect();
        let mut short_args = label_args.to_vec();
        short_args.extend(virtuals);
        // unroll.py:486-489 ShortPreambleBuilder constructor parity.
        //
        // RPython already has all Box identities before this point, so the
        // ShortPreambleBuilder can be constructed before the produce_op loop.
        // Majit precomputes the equivalent Phase-2 result OpRefs first, then
        // shares that map with both the builder constructor and produce_op.
        // history.py:220 Box `.type` is fixed at construction.  RPython
        // assigns a Box of the correct kind from `produced_op.short_op.res`
        // before `produce_op` runs; majit must allocate the result OpRef
        // with the typed allocator so `opref_type` resolution doesn't
        // depend on a downstream `register_value_type` patch.
        let mut result_map: HashMap<OpRef, OpRef> = HashMap::new();
        for (source, produced) in &exported_state.short_boxes {
            let result_type = produced.preamble_op.result_type();
            let result = match produced.kind {
                crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
                | crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant => {
                    if let Some(slot) = exported_state
                        .short_inputargs
                        .iter()
                        .position(|&arg| arg == *source)
                    {
                        short_args.get(slot).copied()
                    } else {
                        Some(ctx.alloc_op_position_typed(result_type))
                    }
                }
                crate::optimizeopt::shortpreamble::PreambleOpKind::Heap => {
                    match produced.preamble_op.opcode {
                        OpCode::GetfieldGcI
                        | OpCode::GetfieldGcR
                        | OpCode::GetfieldGcF
                        | OpCode::GetarrayitemGcI
                        | OpCode::GetarrayitemGcR
                        | OpCode::GetarrayitemGcF => Some(ctx.alloc_op_position_typed(result_type)),
                        _ => None,
                    }
                }
                crate::optimizeopt::shortpreamble::PreambleOpKind::InputArg
                | crate::optimizeopt::shortpreamble::PreambleOpKind::Guard => None,
            };
            if let Some(result) = result {
                result_map.insert(*source, result);
            }
        }
        let mut imported_constants: HashMap<OpRef, OpRef> = HashMap::new();
        let from_short_boxes = ctx.initialize_imported_short_preamble_builder_from_short_boxes(
            &short_args,
            &exported_state.short_inputargs,
            &exported_state.short_boxes,
            &exported_state.short_box_const_values,
            &result_map,
            &mut imported_constants,
            &exported_state.exported_infos,
        );
        debug_assert!(
            from_short_boxes,
            "initialize_imported_short_preamble_builder_from_short_boxes returned false: \
             the short-preamble import path failed to handle some entry. \
             This signals an unresolvable arg classification (Slot/Const/Produced)."
        );
        // unroll.py:501-502 line-by-line port:
        //
        //   for produced_op in exported_state.short_boxes:
        //       produced_op.produce_op(self, exported_state.exported_infos)
        //
        // `produced_results` accumulates `source pos -> replay identity` so
        // successor entries can resolve `Produced` arg classifications. After
        // Path B (B.6.7) the producer-side `produce_*` methods return
        // `Some(source)` (Phase 1 OpRef = `self.res`); for invented Pure the
        // value is the body-visible OpRef per `replay_pos` in
        // `initialize_imported_short_preamble_builder_from_short_boxes`.
        let mut produced_results: HashMap<OpRef, OpRef> = HashMap::new();
        for (_, produced) in &exported_state.short_boxes {
            let produced_result = produced.produce_op(
                ctx,
                &exported_state.exported_infos,
                &exported_state.short_inputargs,
                &short_args,
                &result_map,
                &mut produced_results,
                &mut imported_constants,
                &exported_state.short_box_const_values,
            );
            debug_assert!(
                produced_result.is_some()
                    || matches!(
                        produced.kind,
                        crate::optimizeopt::shortpreamble::PreambleOpKind::InputArg
                            | crate::optimizeopt::shortpreamble::PreambleOpKind::Guard
                    )
                    || !result_map.contains_key(&produced.preamble_op.pos),
                "ProducedShortOp::produce_op failed for source {:?} kind {:?}",
                produced.preamble_op.pos,
                produced.kind
            );
        }
    }

    /// unroll.py:432-443 `_expand_info` + `unroll.py:548`
    /// `ExportedState.exported_infos` entry producer. Returns the single
    /// `OpInfo` variant RPython's dict would hold for this box, or `None`
    /// for the RPython `if info:` falsy fall-through at unroll.py:440 — the
    /// caller must treat `None` as "no entry in the dict" so downstream
    /// `setinfo_from_preamble_list` takes the `item.set_forwarded(None)`
    /// branch at unroll.py:49.
    ///
    ///   - Ref-typed live box with PtrInfo → `Some(OpInfo::Ptr(...))`.
    ///   - Constant OpRef (value-carrying) → `Some(OpInfo::Ptr(PtrInfo::Constant))`
    ///     for Refs, `Some(OpInfo::FloatConst(f))` for floats,
    ///     `Some(OpInfo::IntBound(IntBound::from_constant(v)))` for ints
    ///     (mirroring RPython's `ConstPtrInfo` / `FloatConstInfo` /
    ///     `IntBound` dispatch).
    ///   - Int-typed box with a preamble-exported `IntBound` →
    ///     `Some(OpInfo::IntBound(...))`.
    ///   - No info → `None`. `OpInfo::Unknown` must not appear in
    ///     `ExportedState.exported_infos`.
    fn collect_exported_info(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
    ) -> Option<crate::optimizeopt::info::OpInfo> {
        use crate::optimizeopt::info::{OpInfo, PtrInfo};
        let resolved = ctx.get_box_replacement(opref);
        // RPython export_state/_expand_info records properties on the Box
        // objects that survive to the next iteration. In majit, a loop box may
        // temporarily forward to a Const during the current iteration even
        // though the next iteration still needs a box identity. Export only
        // true Const objects here; forwarded-to-Const metadata would otherwise
        // over-specialize the imported targetarg and leak one iteration's
        // guard knowledge into the next.
        if opref.is_constant() {
            if let Some(value) = ctx.get_constant(resolved) {
                return match value {
                    // ConstPtrInfo parity: RPython stores Ref constants as
                    // ConstPtrInfo (a `PtrInfo` subclass). `setinfo_from_preamble`
                    // at unroll.py:65-68 dispatches through `is_constant()`.
                    Value::Ref(gcref) => Some(OpInfo::Ptr(PtrInfo::Constant(gcref))),
                    // FloatConstInfo parity: unroll.py:97-98 handles
                    // `isinstance(preamble_info, info.FloatConstInfo)` with
                    // `op.set_forwarded(preamble_info._const)`.
                    Value::Float(f) => Some(OpInfo::FloatConst(f)),
                    // Int constants: RPython uses IntBound with lower==upper.
                    Value::Int(v) => Some(OpInfo::IntBound(
                        crate::optimizeopt::intutils::IntBound::from_constant(v),
                    )),
                    Value::Void => None,
                };
            }
        }
        // unroll.py:432-443 _expand_info uses self.optimizer.getinfo(arg) which
        // dispatches by op.type ('r' → getptrinfo, 'i' → getintbound). The Rust
        // port stores int bounds in a separate table populated earlier by
        // `OptIntBounds::export_arg_int_bounds`, which already filters by
        // `opref_type(resolved) == Some(Int)`. We rely on that filter so the
        // lookup here cannot pull a bound for a ref/float box.
        if let Some(ptr_info) = ctx.peek_ptr_info_via_box(resolved) {
            // Drop `PtrInfo::Constant` synthesized for non-constant OpRefs —
            // the per-iteration constant-forwarding shouldn't leak into the
            // next iteration's imported state (see `get_box_replacement` note).
            let is_const_shadow = matches!(ptr_info, PtrInfo::Constant(_)) && !opref.is_constant();
            if !is_const_shadow {
                return Some(OpInfo::Ptr(ptr_info));
            }
        }
        if let Some(bound) = exported_int_bounds.and_then(|bounds| bounds.get(&resolved).cloned()) {
            return Some(OpInfo::IntBound(bound));
        }
        None
    }

    /// unroll.py:53-98 `setinfo_from_preamble(op, preamble_info, exported_infos)`.
    ///
    /// Thin forwarding wrapper onto `OptContext::setinfo_from_preamble_item`
    /// (mod.rs), which is the shared RPython-literal dispatcher used both by
    /// this top-level import and by the recursive virtual-field walker at
    /// `setinfo_from_preamble_list`. Keeping a single implementation prevents
    /// the two call sites from drifting: any shortcut in one must also apply
    /// to the other or the early-return semantics at unroll.py:54-58 are
    /// bypassed for part of the import path.
    fn setinfo_from_preamble(
        &self,
        opref: OpRef,
        info: &crate::optimizeopt::info::OpInfo,
        exported_infos: &HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
        ctx: &mut OptContext,
    ) {
        ctx.setinfo_from_preamble_item(opref, info, exported_infos);
    }
}

/// unroll.py: export_state — module-level entry point.
pub(crate) fn export_state(
    jump_args: &[OpRef],
    renamed_inputargs: &[OpRef],
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
    exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
) -> ExportedState {
    OptUnroll::new().export_state_with_bounds(
        jump_args,
        renamed_inputargs,
        optimizer,
        ctx,
        exported_int_bounds,
    )
}

/// unroll.py:479-504 import_state — module-level entry point.
pub(crate) fn import_state(
    targetargs: &[OpRef],
    exported_state: &ExportedState,
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
) -> Vec<OpRef> {
    OptUnroll::new().import_state(targetargs, exported_state, optimizer, ctx)
}

pub(crate) fn import_short_preamble_state(
    targetargs: &[OpRef],
    label_args: &[OpRef],
    exported_state: &ExportedState,
    ctx: &mut OptContext,
) {
    OptUnroll::new().import_short_preamble_state(targetargs, label_args, exported_state, ctx)
}

/// RPython unroll.py:479-504 `import_state(targetargs, exported_state)`
/// canonical end-to-end orchestrator.  Bundles the three majit-side
/// sub-steps that mirror the RPython single-function flow:
///
/// 1. `OptUnroll::import_state` — `unroll.py:483-494` forwarding +
///    `virtual_state.make_inputargs` to compute label_args.
/// 2. `Optimizer::install_imported_virtuals` — majit-only counterpart
///    to RPython's inline virtualizable PtrInfo installation that
///    happens during `make_inputargs` itself.  Split out for borrow
///    reasons (`Optimizer` access to `imported_virtuals` / `imported_loop_state`).
/// 3. `OptUnroll::import_short_preamble_state` — `unroll.py:496-502`
///    `ShortPreambleBuilder` construction + `produce_op` loop.
///
/// Production callers should use this end-to-end form.  Test callers
/// that need to inspect intermediate state can still invoke the
/// individual functions above.
pub(crate) fn import_state_full(
    targetargs: &[OpRef],
    exported_state: &ExportedState,
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
) -> Vec<OpRef> {
    let label_args = OptUnroll::new().import_state(targetargs, exported_state, optimizer, ctx);
    optimizer.imported_label_args = Some(label_args.clone());
    if !optimizer.imported_virtuals.is_empty() {
        optimizer.install_imported_virtuals(ctx);
    }
    OptUnroll::new().import_short_preamble_state(targetargs, &label_args, exported_state, ctx);
    label_args
}

/// unroll.py: pick_virtual_state(my_vs, label_vs, target_tokens)
///
/// Given the current virtual state and available target tokens,
/// find a compatible target to jump to. Returns the target index
/// or None if no match.
/// RPython unroll.py: import_state + _generate_virtual.
///
// ── RPython-parity helper functions for 2-phase preamble peeling ──

/// Derive ImportedVirtual entries from ExportedState's VirtualState.
/// Virtual structure is obtained from the VirtualState snapshot.
fn build_imported_virtuals_from_state(
    state: &ExportedState,
) -> Vec<crate::optimizeopt::optimizer::ImportedVirtual> {
    use crate::optimizeopt::virtualstate::{VirtualStateInfo, VirtualStateInfoNode};

    /// virtualstate.py:158-165 AbstractVirtualStructStateInfo.generate_guards
    /// parity: walk `fielddescrs` in parent-local slot order, looking up
    /// the matching field state via descr.get_index() (= field_idx in pyre).
    fn ordered_fields(
        field_descrs: &[majit_ir::DescrRef],
        fields: &[(u32, std::rc::Rc<VirtualStateInfoNode>)],
    ) -> Vec<(majit_ir::DescrRef, VirtualStateInfo)> {
        field_descrs
            .iter()
            .filter_map(|field_descr| {
                let field_idx = field_descr
                    .as_field_descr()
                    .map(|fd| fd.index_in_parent() as u32)?;
                fields
                    .iter()
                    .find(|(idx, _)| *idx == field_idx)
                    .map(|(_, field_value)| (field_descr.clone(), field_value.info.clone()))
            })
            .collect()
    }

    let mut result = Vec::new();
    for (idx, info) in state.virtual_state.state.iter().enumerate() {
        match &info.info {
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                fields,
                field_descrs,
                ..
            } => {
                result.push(crate::optimizeopt::optimizer::ImportedVirtual {
                    inputarg_index: idx,
                    size_descr: descr.clone(),
                    kind: crate::optimizeopt::optimizer::ImportedVirtualKind::Instance {
                        known_class: *known_class,
                    },
                    fields: ordered_fields(field_descrs, fields),
                    head_load_descr_index: None,
                });
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                result.push(crate::optimizeopt::optimizer::ImportedVirtual {
                    inputarg_index: idx,
                    size_descr: descr.clone(),
                    kind: crate::optimizeopt::optimizer::ImportedVirtualKind::Struct,
                    fields: ordered_fields(field_descrs, fields),
                    head_load_descr_index: None,
                });
            }
            _ => {}
        }
    }
    result
}

/// compile.py:310-338: [preamble_no_jump] + Label(label_args) + [body_with_jump]
#[cfg(test)]
fn assemble_peeled_trace(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    start_label_args: &[OpRef],
    extra_label_args: &[OpRef],
    body_num_inputs: usize,
    jump_to_self: bool,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
    constants: &std::collections::HashMap<u32, i64>,
    start_label_descr: Option<DescrRef>,
    loop_label_descr: Option<DescrRef>,
) -> Vec<Op> {
    let mut ctx = assemble_test_context(p1_ops, p2_ops, body_num_inputs);
    assemble_peeled_trace_with_jump_args(
        p1_ops,
        p2_ops,
        label_args,
        start_label_args,
        extra_label_args,
        extra_label_args,
        body_num_inputs,
        0, // inputarg_base — tests/simple cases use shared namespace
        jump_to_self,
        imported_short_aliases,
        constants,
        start_label_descr,
        loop_label_descr,
        &[], // no p1_end_args for simple assembly
        &mut ctx,
    )
}

#[cfg(test)]
fn assemble_test_context(p1_ops: &[Op], p2_ops: &[Op], body_num_inputs: usize) -> OptContext {
    // Test helper — caller provides only the body inputarg count, so seed
    // every slot as Ref (matches the common loop-body shape; fixtures that
    // need typed inputargs construct ctx directly).
    let types = vec![Type::Ref; body_num_inputs];
    let mut ctx = OptContext::with_inputarg_types(p1_ops.len() + p2_ops.len(), &types);
    for op in p1_ops.iter().chain(p2_ops.iter()) {
        if op.pos.is_none() || op.result_type() == Type::Void {
            continue;
        }
        ctx.value_types
            .entry(op.pos.raw())
            .or_insert(op.result_type());
    }
    ctx
}

/// shortpreamble.py:436-439 alias-side `extra_same_as` emission.
///
/// Compound alternates (`invented_name=true`) record their SameAs source in
/// `imported_short_aliases`. Emit one `SameAs(alias.same_as_source)` op at
/// `pos=alias.result` per entry so the body's reference to the alias result
/// has a defining op.
fn emit_alias_same_as_for_imports(
    result: &mut Vec<Op>,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
) {
    for alias in imported_short_aliases {
        let mut op = Op::new(alias.same_as_opcode, &[alias.same_as_source]);
        op.pos = alias.result;
        result.push(op);
    }
}

fn assemble_peeled_trace_with_jump_args(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    start_label_args: &[OpRef],
    extra_label_args: &[OpRef],
    extra_jump_args: &[OpRef],
    body_num_inputs: usize,
    inputarg_base: u32,
    jump_to_self: bool,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
    constants: &std::collections::HashMap<u32, i64>,
    start_label_descr: Option<DescrRef>,
    loop_label_descr: Option<DescrRef>,
    p1_end_args: &[OpRef],
    ctx: &mut crate::optimizeopt::OptContext,
) -> Vec<Op> {
    let mut result =
        Vec::with_capacity(p1_ops.len() + p2_ops.len() + 1 + imported_short_aliases.len());
    let mut filtered_extra_label_args = Vec::new();
    let mut filtered_extra_jump_args = Vec::new();
    // RPython Box parity (compile.py:327, shortpreamble.py:436-439):
    //
    //   loop.operations = ([start_label] + preamble_ops + loop_info.extra_same_as +
    //                      loop_info.extra_before_label + [loop_info.label_op] + loop_ops)
    //
    //   op = preamble_op.op.get_box_replacement()
    //   if preamble_op.invented_name:
    //       self.extra_same_as.append(op)
    //   self.used_boxes.append(op)
    //   self.short_preamble_jump.append(preamble_op.preamble_op)
    //
    // `used_boxes` is appended to `label_op.arglist()` in its entirety, with
    // NO deduplication against the base label args. RPython's Box identity
    // makes every used_box a distinct Python object; even two boxes that
    // happen to carry the same runtime value are separate slots in the
    // label. In majit's flat OpRef model we carry this literally — do not
    // drop entries just because their OpRef coincides with a base label
    // arg, because that silently corrupts the loop arity (the matching
    // `extra_jump_args[idx]` value that the JUMP was supposed to deliver
    // into that slot disappears along with it).
    //
    // The only entries that legitimately drop out here are literal Const
    // boxes. `used_boxes` / `short_preamble_jump` are RPython Box lists;
    // a stale backend-constants entry must not make us drop a live runtime
    // Box from the LABEL/JUMP contract (e.g. an imported short-preamble
    // HeapField result that still has optimizer constant knowledge attached).
    for (idx, &label_arg) in extra_label_args.iter().enumerate() {
        let jump_arg = extra_jump_args.get(idx).copied().unwrap_or(label_arg);
        if label_arg.is_constant() {
            continue;
        }
        filtered_extra_label_args.push(label_arg);
        filtered_extra_jump_args.push(jump_arg);
    }

    let next_free_pos = |mut next: u32| {
        next = next.max(inputarg_base + body_num_inputs as u32);
        while constants.contains_key(&next) {
            next += 1;
        }
        next
    };

    if let Some(start_label_descr) = start_label_descr {
        let mut start_label = Op::new(OpCode::Label, start_label_args);
        start_label.pos = OpRef::NONE;
        start_label.descr = Some(start_label_descr);
        result.push(start_label);
    }

    // Preamble: everything except Jump
    for op in p1_ops {
        if op.opcode == OpCode::Jump {
            break;
        }
        result.push(op.clone());
    }

    // max_pos must account for ALL p1_ops positions, including SameAs
    // ops AFTER the Jump that weren't copied into result. These positions
    // are referenced by the body Label args and must not be reused.
    let result_max = result
        .iter()
        .map(|op| op.pos.raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(inputarg_base + body_num_inputs as u32);
    let p1_all_max = p1_ops
        .iter()
        .map(|op| op.pos.raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut max_pos = result_max.max(p1_all_max);
    max_pos = next_free_pos(max_pos.saturating_add(1));
    emit_alias_same_as_for_imports(&mut result, imported_short_aliases);

    // Label position
    let label_pos = next_free_pos(max_pos);
    let mut full_label_args: Vec<OpRef> = label_args
        .iter()
        .copied()
        .filter(|arg| !is_trace_constant_ref(*arg, constants))
        .collect();

    // Collect preamble-defined OpRefs BEFORE adding extra label args,
    // so we can filter out virtual remnants (removed New ops).
    //
    // resoperation.py:719/727/739 InputArg{Int,Ref,Float}: preamble-defined
    // inputargs carry `box.type` intrinsically (history.py:220). Mint
    // typed variants from `inputarg_types` so this set's OpRefs match the
    // typed mints used at trace start / Phase 2 import under variant-aware
    // Eq.
    let preamble_defs: std::collections::HashSet<OpRef> = {
        let mut s: std::collections::HashSet<OpRef> = (0..body_num_inputs)
            .map(|i| {
                let pos = inputarg_base + i as u32;
                match ctx.inputarg_type_at(i) {
                    Some(tp) => OpRef::input_arg_typed(pos, tp),
                    None => OpRef::from_raw(pos),
                }
            })
            .collect();
        for op in &result {
            if !op.pos.is_none() && op.opcode != OpCode::Jump && op.result_type() != Type::Void {
                s.insert(op.pos);
            }
        }
        s
    };

    // Append non-constant extra label args. The caller (unroll pass)
    // determines which extra values the loop header needs; virtual
    // remnants have already been filtered by the caller.
    //
    // Advance max_pos past every position already in use before allocating
    // fresh SameAs positions. In RPython, Box identity prevents collisions;
    // in the flat OpRef model, the assembly-allocated SameAs position must
    // not overlap base label_args, filtered_extra_label_args, or p2_ops
    // positions/args/fail_args — each of those may be referenced by the
    // body. A single colliding OpRef aliases two distinct values and
    // corrupts the loop.
    for &la in &full_label_args {
        if is_trace_runtime_ref(la, constants) {
            max_pos = max_pos.max(la.raw().saturating_add(1));
        }
    }
    for &la in &filtered_extra_label_args {
        if is_trace_runtime_ref(la, constants) {
            max_pos = max_pos.max(la.raw().saturating_add(1));
        }
    }
    for op in p2_ops.iter() {
        if is_trace_runtime_ref(op.pos, constants) {
            max_pos = max_pos.max(op.pos.raw().saturating_add(1));
        }
        for &arg in op.args.iter() {
            if is_trace_runtime_ref(arg, constants) {
                max_pos = max_pos.max(arg.raw().saturating_add(1));
            }
        }
        if let Some(ref fa) = op.fail_args {
            for &arg in fa.iter() {
                if is_trace_runtime_ref(arg, constants) {
                    max_pos = max_pos.max(arg.raw().saturating_add(1));
                }
            }
        }
    }
    let extra_label_start_idx = full_label_args.len();
    full_label_args.extend(filtered_extra_label_args.iter().copied());

    // RPython compile.py parity: after the loop label, only the loop-header
    // contract is live. When `splice_redirected_tail` glues a redirected
    // tail onto the body, the spliced output may mention OpRefs whose
    // defining op was removed (the section between the splice point and
    // the original Jump). Carry such "body use-before-def" references
    // through the label so the assembled body doesn't contain dangling
    // references. With Phase 2's disjoint OpRef namespace, body op args
    // are pre-resolved through ctx.get_box_replacement, so the carried
    // OpRef can be appended directly to full_label_args — the JUMP's
    // mapped_base_args path picks up the corresponding fresh value on the
    // next iteration. The filter only needs to skip filtered_extra_jump_args.
    let mut carried_source_slots: std::collections::HashSet<OpRef> =
        std::collections::HashSet::new();
    carried_source_slots.extend(filtered_extra_jump_args.iter().copied());
    // `label_set` tracks which OpRefs are already carried by the label so
    // that the body-use-before-def pass doesn't add the same OpRef twice
    // (which would inflate label arity without adding new information).
    // RPython's Box identity makes this a correctness filter: two
    // references to the same Box collapse into one live-in slot. This
    // is NOT the Issue 1 dedup — which drops distinct Boxes that happen
    // to share an OpRef — it is RPython parity: the same Box appears
    // once in the label arglist.
    let mut label_set: std::collections::HashSet<OpRef> = full_label_args.iter().copied().collect();
    {
        let mut seen_body_defs = std::collections::HashSet::new();
        for op in p2_ops {
            // compile.py assembles the loop LABEL from `label_op` plus
            // short-preamble `used_boxes`; the terminal JUMP's target-local
            // payload is not part of that contract. In particular,
            // jump_to_preamble retargets the body JUMP to the preamble start
            // label, so carrying its extra args into the loop LABEL mutates
            // the body arity in a way upstream never does.
            if op.opcode == OpCode::Jump {
                continue;
            }
            let all_refs = op
                .args
                .iter()
                .chain(op.fail_args.as_ref().into_iter().flat_map(|fa| fa.iter()));
            for &arg in all_refs {
                if !is_trace_runtime_ref(arg, constants) {
                    continue; // skip NONE and constants
                }
                if label_set.contains(&arg)
                    || carried_source_slots.contains(&arg)
                    || seen_body_defs.contains(&arg)
                {
                    continue;
                }
                full_label_args.push(arg);
                label_set.insert(arg);
            }
            if op.result_type() != Type::Void && !op.pos.is_none() {
                seen_body_defs.insert(op.pos);
            }
        }
    }

    // RPython Box identity parity: in RPython, the JUMP's Box IS the
    // label's Box — no mapping needed. The flat OpRef model previously
    // required a Phase 1 end_arg → label_arg SameAs bridge here for cases
    // where the two diverged. Probe across 14 benchmarks (both backends)
    // showed this block fires 0 times after Commit D1/D2's disjoint Phase
    // 2 OpRef range eliminated the divergence cases. Removed.
    let mut label_op = Op::new(OpCode::Label, &full_label_args);
    label_op.pos = OpRef::from_raw(label_pos);
    label_op.descr = loop_label_descr;
    result.push(label_op);

    // Body: 2-pass remap (inputarg refs -> label args, body results -> fresh boxes)
    let max_label_arg_pos = full_label_args
        .iter()
        .map(|a| a.raw())
        .max()
        .unwrap_or(label_pos);
    let max_emitted_pos = result
        .iter()
        .map(|op| op.pos.raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(label_pos);
    // Fresh body positions must be higher than ALL existing positions:
    // already-emitted preamble/imported-short ops, label, label args, AND
    // Phase 2 op positions (which may be higher than label positions due
    // to Phase 2 remap or redirect ops).
    //
    // RPython's distinct Box identities prevent an imported short-preamble
    // replay result from colliding with a freshly emitted body result. In
    // majit's flat OpRef model we must keep the body allocator above every
    // position already emitted into `result`.
    let max_p2_pos = p2_ops
        .iter()
        .map(|op| op.pos.raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut next_body_pos = next_free_pos(
        max_emitted_pos
            .max(label_pos)
            .max(max_label_arg_pos)
            .max(max_p2_pos)
            .saturating_add(1),
    );
    let mut body_result_remap: HashMap<OpRef, OpRef> = HashMap::new();
    let visible_before_label: std::collections::HashSet<OpRef> = full_label_args
        .iter()
        .copied()
        .chain(preamble_defs.iter().copied())
        .collect();

    // shortpreamble.py:436-439 + unroll.py:497-504 parity: when
    // `sp.used_boxes[i] != sp.jump_args[i]` (Case A/B end-of-preamble
    // SameAs alias splits identity — preamble JUMP carries `jump_source`,
    // LABEL takes `source_slot`), register the forwarding chain so body
    // ops referencing `jump_source` resolve to `source_slot` via
    // `ctx.get_box_replacement` (`optimizer.py:266 set_forwarded`).
    // RPython's Box identity makes this implicit — the alias's Box is
    // the same Python object that body ops already hold. Pyre's flat
    // OpRef model needs an explicit forwarding registration here.
    for (i, &source_slot) in filtered_extra_label_args.iter().enumerate() {
        if source_slot.is_none() {
            continue;
        }
        let Some(&extended_label_arg) = full_label_args.get(extra_label_start_idx + i) else {
            continue;
        };
        debug_assert_eq!(
            source_slot, extended_label_arg,
            "full_label_args at extra_label_start_idx + i must equal \
             filtered_extra_label_args[i] (line 4338 extend invariant)"
        );
        if let Some(&jump_source) = filtered_extra_jump_args.get(i) {
            if !jump_source.is_none() && jump_source != source_slot {
                ctx.replace_op(jump_source, extended_label_arg);
            }
        }
    }
    for op in p2_ops.iter() {
        // Only map non-Void ops that actually produce a result.
        // Void ops (SetfieldGc, guards, Jump) don't define values at
        // their position — mapping them creates phantom OpRefs.
        if op.pos.raw() != u32::MAX && op.result_type() != Type::Void {
            let fresh = OpRef::from_raw(next_body_pos);
            next_body_pos = next_free_pos(next_body_pos.saturating_add(1));
            body_result_remap.insert(op.pos, fresh);
        }
    }

    let mut seen_body_defs = std::collections::HashSet::new();
    let mut current_inner_label_index: Option<usize> = None;
    let mut defs_since_inner_label: std::collections::HashSet<OpRef> =
        std::collections::HashSet::new();
    for (op_idx, op) in p2_ops.iter().enumerate() {
        let mut new_op = op.clone();
        let mut original_args = op.args.clone();
        if let Some(&mapped_pos) = body_result_remap.get(&op.pos) {
            new_op.pos = mapped_pos;
        }
        // Body op args go through Phase 2 forwarding (set by both the
        // optimizer's regular forwarding-resolve pass at end of Phase 2,
        // optimizer.rs:2236-2262, AND the Case A/B SameAs alias
        // registration above) followed by the shift-back from Phase 2
        // inputarg_base to the body layout (body_result_remap, populated
        // by the dedicated pass below).
        let remap_body_arg = |arg: OpRef,
                              body_result_remap: &HashMap<OpRef, OpRef>,
                              seen_body_defs: &std::collections::HashSet<OpRef>,
                              visible_before_label: &std::collections::HashSet<OpRef>|
         -> OpRef {
            if let Some(&mapped) = body_result_remap.get(&arg) {
                if seen_body_defs.contains(&arg) || !visible_before_label.contains(&arg) {
                    return mapped;
                }
            }
            arg
        };
        for arg in &mut new_op.args {
            *arg = ctx.get_box_replacement(*arg);
            *arg = remap_body_arg(
                *arg,
                &body_result_remap,
                &seen_body_defs,
                &visible_before_label,
            );
        }
        if new_op.opcode == OpCode::Label {
            let mut seen_after_label_defs = std::collections::HashSet::new();
            let mut extra_inner_sources = Vec::new();
            let mut extra_inner_set = std::collections::HashSet::new();
            let label_arg_set: std::collections::HashSet<OpRef> = original_args
                .iter()
                .copied()
                .filter(|arg| !arg.is_none())
                .collect();
            for later_op in p2_ops.iter().skip(op_idx + 1) {
                for arg in later_op
                    .args
                    .iter()
                    .copied()
                    .chain(later_op.fail_args.iter().flatten().copied())
                {
                    if arg.is_none()
                        || constants.contains_key(&arg.raw())
                        || label_arg_set.contains(&arg)
                        || extra_inner_set.contains(&arg)
                        || seen_after_label_defs.contains(&arg)
                    {
                        continue;
                    }
                    let resolved_arg = ctx.get_box_replacement(arg);
                    let available_before_label = visible_before_label.contains(&arg)
                        || visible_before_label.contains(&resolved_arg)
                        || seen_body_defs.contains(&arg);
                    if available_before_label {
                        extra_inner_set.insert(arg);
                        extra_inner_sources.push(arg);
                    }
                }
                if later_op.result_type() != Type::Void && !later_op.pos.is_none() {
                    seen_after_label_defs.insert(later_op.pos);
                }
            }
            // RPython Box parity: do not dedup the inner-label extension
            // against its existing args. Each source_arg collected from
            // the later body is a distinct RPython Box; adding it as a
            // separate slot matches `label_op.initarglist(label_op.getarglist()
            // + sb.used_boxes)` (unroll.py:300) where RPython never filters
            // by value or by box coincidence. The outer collection above
            // already dedups by box identity (`extra_inner_set.contains`),
            // so each source_arg appears at most once — which is the
            // RPython-parity behavior for Box-keyed live-in sets.
            for &source_arg in &extra_inner_sources {
                let resolved = ctx.get_box_replacement(source_arg);
                let mapped_arg = remap_body_arg(
                    resolved,
                    &body_result_remap,
                    &seen_body_defs,
                    &visible_before_label,
                );
                new_op.args.push(mapped_arg);
                original_args.push(source_arg);
            }
        }
        if new_op.opcode == OpCode::Jump {
            // unroll.py:238-242 jump_to_preamble sends the live JUMP after
            // force_box / send_extra_operation rewrites. Body args reference
            // the trace inputarg slots OpRef(0)..OpRef(start_label_args.len());
            // remap those positional refs to start_label_args[i].
            let mapped_base_args: Vec<OpRef> = new_op
                .args
                .iter()
                .map(|&arg| {
                    if jump_to_self {
                        return arg;
                    }
                    let i = arg.raw() as usize;
                    start_label_args.get(i).copied().unwrap_or(arg)
                })
                .collect();
            let target_label_args: Vec<OpRef> = current_inner_label_index
                .and_then(|label_idx| result.get(label_idx).map(|op| op.args.to_vec()))
                .unwrap_or_else(|| full_label_args.clone());
            let target_base_len = if current_inner_label_index.is_some() {
                original_args.len()
            } else {
                label_args.len()
            };
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] assemble_jump: inner_label={:?} original_args={:?} mapped_base_args={:?} label_args={:?} filtered_extra_jump_args={:?}",
                    current_inner_label_index,
                    original_args,
                    mapped_base_args,
                    label_args,
                    filtered_extra_jump_args,
                );
            }
            let mut jump_args = mapped_base_args;
            if jump_to_self {
                // RPython compile.py:334: assert jump.numargs() == label.numargs().
                // Truncate excess JUMP args (from forced virtuals in
                // jump_to_existing_trace) to match the LABEL arity.
                if jump_args.len() > target_label_args.len() {
                    jump_args.truncate(target_label_args.len());
                }
                // Pad if JUMP is shorter than the target label.
                while jump_args.len() < target_label_args.len() {
                    let extra_idx = jump_args.len().saturating_sub(target_base_len);
                    let extra_arg = if current_inner_label_index.is_some() {
                        target_label_args[jump_args.len()]
                    } else {
                        filtered_extra_jump_args
                            .get(extra_idx)
                            .copied()
                            .unwrap_or(target_label_args[jump_args.len()])
                    };
                    // Extra args are assembly-allocated label positions;
                    // body-scoped remaps never contain them. Probe across 14
                    // benchmarks × 2 backends showed the prior lookup chain
                    // returned identity in 100% of fires. Pass through.
                    jump_args.push(extra_arg);
                }
            }
            new_op.args = jump_args.into();
        }
        // RPython resume.py parity: fail_args capture guard-point state
        // (the snapshot), not the body's final state. body_result_remap
        // applies only to values body-defined AND not visible before the
        // label, so snapshot refs to label args stay intact.
        if let Some(ref mut fa) = new_op.fail_args {
            for a in fa.iter_mut() {
                if let Some(&mapped) = body_result_remap.get(a) {
                    if seen_body_defs.contains(a) && !visible_before_label.contains(a) {
                        *a = mapped;
                    }
                }
            }
        }
        if let Some(label_idx) = current_inner_label_index {
            let mut extra_live_args = Vec::new();
            let label_args = &result[label_idx].args;
            for arg in new_op
                .args
                .iter()
                .copied()
                .chain(new_op.fail_args.iter().flatten().copied())
            {
                if arg.is_none()
                    || constants.contains_key(&arg.raw())
                    || defs_since_inner_label.contains(&arg)
                    || label_args.contains(&arg)
                    || extra_live_args.contains(&arg)
                {
                    continue;
                }
                extra_live_args.push(arg);
            }
            if !extra_live_args.is_empty() {
                let existing: std::collections::HashSet<OpRef> =
                    result[label_idx].args.iter().copied().collect();
                result[label_idx].args.extend(
                    extra_live_args
                        .into_iter()
                        .filter(|arg| !existing.contains(arg)),
                );
            }
        }
        // RPython parity: each guard in the assembled trace owns a
        // distinct ResumeGuardDescr with a globally unique fail_index.
        // optimizeopt::store_final_boxes_in_guard (mod.rs:3392-3404,
        // commit 43c64ee0bb) installs a fresh ResumeGuardDescr on every
        // optimizer-routed guard, and Optimizer::_copy_resume_data_from /
        // OptContext::emit_guard_operation share-branch (commit
        // 329297b38a) call Descr::clone_descr to allocate a fresh
        // fail_index for the sharing-path guard. Both phases of the
        // unroll optimizer therefore emit body guards with unique
        // descrs already; no post-process re-stamping is needed.
        result.push(new_op);
        if op.opcode == OpCode::Label {
            current_inner_label_index = Some(result.len() - 1);
            defs_since_inner_label.clear();
        }
        if op.result_type() != Type::Void && !op.pos.is_none() {
            seen_body_defs.insert(op.pos);
            defs_since_inner_label.insert(result.last().unwrap().pos);
        }
    }

    result
}

fn splice_redirected_tail(body_ops: &[Op], redirected_tail_ops: &[Op]) -> Vec<Op> {
    let mut result = Vec::with_capacity(body_ops.len() + redirected_tail_ops.len());
    let split_idx = body_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Jump)
        .unwrap_or(body_ops.len());
    result.extend_from_slice(&body_ops[..split_idx]);
    result.extend_from_slice(redirected_tail_ops);
    result
}

fn replace_terminal_jump(body_ops: &[Op], jump_op: Op) -> Vec<Op> {
    let mut result = Vec::with_capacity(body_ops.len() + 1);
    let split_idx = body_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Jump)
        .unwrap_or(body_ops.len());
    result.extend_from_slice(&body_ops[..split_idx]);
    result.push(jump_op);
    result
}

fn reshape_jump_args_for_preamble(jump_args: &mut Vec<OpRef>, preamble_args: &[OpRef]) {
    if jump_args.len() > preamble_args.len() {
        jump_args.truncate(preamble_args.len());
    }
    while jump_args.len() < preamble_args.len() {
        jump_args.push(preamble_args[jump_args.len()]);
    }
}

fn apply_box_replacements(op: &mut Op, ctx: &crate::optimizeopt::OptContext) {
    for arg in op.args.iter_mut() {
        *arg = ctx.get_box_replacement(*arg);
    }
    if let Some(fail_args) = op.fail_args.as_mut() {
        for arg in fail_args.iter_mut() {
            *arg = ctx.get_box_replacement(*arg);
        }
    }
}

pub fn pick_virtual_state(
    my_vs: &crate::optimizeopt::virtualstate::VirtualState,
    target_states: &[crate::optimizeopt::virtualstate::VirtualState],
) -> Option<usize> {
    for (i, target_vs) in target_states.iter().enumerate() {
        if target_vs.generalization_of(my_vs) {
            return Some(i);
        }
    }
    None
}

pub struct OptUnroll {
    /// Buffer of ops received before the Jump back-edge.
    buffer: Vec<Op>,
    /// Whether a Jump was already seen (avoid double-unrolling).
    seen_jump: bool,
}

impl OptUnroll {
    pub fn new() -> Self {
        OptUnroll {
            buffer: Vec::new(),
            seen_jump: false,
        }
    }

    /// Peel one iteration of the loop body.
    ///
    /// Given the buffered ops (everything before the Jump), this emits:
    /// 1. The peeled (duplicated) body with remapped OpRefs
    /// 2. A Label op marking the loop header
    /// 3. The original body ops
    ///
    /// The caller is responsible for emitting the final Jump.
    /// Emit the peeled iteration, the loop Label, and the body iteration
    /// in that order, returning the `original_pos -> body_pos` map the
    /// caller uses to rewrite the back-edge Jump's args.
    ///
    /// Positions are allocated through `reserve_pos` (which honors the
    /// `inputarg_base + num_inputs` floor) instead of being computed
    /// arithmetically from `new_operations.len()`. The arithmetic form
    /// is fine for production traces (inputargs occupy a disjoint low
    /// range) but collides with the 1024-slot `trace_inputarg_types`
    /// stubs that the unit-test harnesses seed. Using `reserve_pos`
    /// closes that collision and lets the Box.type invariant enforce
    /// itself uniformly at `emit()` / `emit_extra()` /
    /// `propagate_from_pass_range`.
    fn peel_iteration(&self, jump_op: &Op, ctx: &mut OptContext) -> HashMap<OpRef, OpRef> {
        // First pass: reserve peeled-iteration positions, tagged with each
        // source op's result type (Slice 0.5 follow-up — `OpRef.ty()`
        // matches RPython's `box.type` at allocation time).
        let peeled_positions: Vec<OpRef> = self
            .buffer
            .iter()
            .map(|op| ctx.reserve_pos_typed(op.result_type()))
            .collect();
        let mut ref_map: HashMap<OpRef, OpRef> = HashMap::new();
        for (op, &new_pos) in self.buffer.iter().zip(peeled_positions.iter()) {
            ref_map.insert(op.pos, new_pos);
        }

        // Emit peeled iteration with remapped refs.
        for (op, &new_pos) in self.buffer.iter().zip(peeled_positions.iter()) {
            let mut peeled = op.clone();
            peeled.pos = new_pos;
            for arg in &mut peeled.args {
                if let Some(&new_ref) = ref_map.get(arg) {
                    *arg = new_ref;
                }
                // Args referencing ops outside the buffer (e.g., input args)
                // are kept as-is.
            }
            if let Some(ref mut fa) = peeled.fail_args {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = ref_map.get(arg) {
                        *arg = new_ref;
                    }
                }
            }
            if peeled.opcode.is_guard() {
                clone_guard_snapshot_remapped(ctx, &mut peeled, &ref_map);
                // opencoder.py:391-401 parity: trace iteration strips guard
                // descrs, keeping only rd_resume_position. optimizer.py then
                // invents a fresh opcode-appropriate descr at emission time.
                peeled.descr = None;
            }
            ctx.emit(peeled);
        }

        // Emit Label between peeled and original body.
        // The Label's args match the Jump's args, forming the loop header.
        let label_pos = ctx.reserve_pos_typed(OpCode::Label.result_type());
        let mut label_op = Op::new(OpCode::Label, &jump_op.args);
        label_op.pos = label_pos;
        ctx.emit(label_op);

        // Reserve body positions, tagged per source op (Slice 0.5 follow-up).
        let body_positions: Vec<OpRef> = self
            .buffer
            .iter()
            .map(|op| ctx.reserve_pos_typed(op.result_type()))
            .collect();
        let mut orig_ref_map: HashMap<OpRef, OpRef> = HashMap::new();
        for (op, &new_pos) in self.buffer.iter().zip(body_positions.iter()) {
            orig_ref_map.insert(op.pos, new_pos);
        }

        // Emit original body ops with remapped positions and refs.
        for (op, &new_pos) in self.buffer.iter().zip(body_positions.iter()) {
            let mut body_op = op.clone();
            body_op.pos = new_pos;
            for arg in &mut body_op.args {
                if let Some(&new_ref) = orig_ref_map.get(arg) {
                    *arg = new_ref;
                }
            }
            if let Some(ref mut fa) = body_op.fail_args {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = orig_ref_map.get(arg) {
                        *arg = new_ref;
                    }
                }
            }
            if body_op.opcode.is_guard() {
                clone_guard_snapshot_remapped(ctx, &mut body_op, &orig_ref_map);
                // Same opencoder.py guard-descr stripping as the peeled copy.
                // ResumeAtPositionDescr is inline_short_preamble-only
                // (unroll.py:406-409), not normal peeled body guard state.
                body_op.descr = None;
            }
            ctx.emit(body_op);
        }
        orig_ref_map
    }
}

fn fresh_snapshot_key(ctx: &OptContext) -> i32 {
    next_snapshot_pos(&ctx.snapshot_boxes)
}

fn remap_snapshot_boxes(
    boxes: &[SnapshotBox],
    ref_map: &HashMap<OpRef, OpRef>,
) -> Vec<SnapshotBox> {
    boxes
        .iter()
        .map(|&boxref| boxref.map_opref(|opref| ref_map.get(&opref).copied().unwrap_or(opref)))
        .collect()
}

fn clone_guard_snapshot_remapped(
    ctx: &mut OptContext,
    guard: &mut Op,
    ref_map: &HashMap<OpRef, OpRef>,
) {
    let old_pos = guard.rd_resume_position;
    if old_pos < 0 {
        return;
    }
    let Some(snapshot_boxes) = snapshot_get(&ctx.snapshot_boxes, old_pos).cloned() else {
        return;
    };

    // RPython TraceIterator decodes snapshot box references against the
    // current iteration's box cache.  Majit keeps snapshots in side tables,
    // so cloned guards need a cloned snapshot table entry with the same
    // raw-box -> cloned-box mapping used for op args.
    let new_pos = fresh_snapshot_key(ctx);
    snapshot_insert(
        &mut ctx.snapshot_boxes,
        new_pos,
        remap_snapshot_boxes(&snapshot_boxes, ref_map),
    );
    if let Some(vable_boxes) = snapshot_get(&ctx.snapshot_vable_boxes, old_pos).cloned() {
        snapshot_insert(
            &mut ctx.snapshot_vable_boxes,
            new_pos,
            remap_snapshot_boxes(&vable_boxes, ref_map),
        );
    }
    if let Some(vref_boxes) = snapshot_get(&ctx.snapshot_vref_boxes, old_pos).cloned() {
        snapshot_insert(
            &mut ctx.snapshot_vref_boxes,
            new_pos,
            remap_snapshot_boxes(&vref_boxes, ref_map),
        );
    }
    if let Some(frame_pcs) = snapshot_get(&ctx.snapshot_frame_pcs, old_pos).cloned() {
        snapshot_insert(&mut ctx.snapshot_frame_pcs, new_pos, frame_pcs);
    }
    if let Some(frame_sizes) = snapshot_get(&ctx.snapshot_frame_sizes, old_pos).cloned() {
        snapshot_insert(&mut ctx.snapshot_frame_sizes, new_pos, frame_sizes);
    }
    guard.rd_resume_position = new_pos;
}

impl Default for OptUnroll {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptUnroll {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Only peel once per trace, and only for Jump (back-edge).
        if op.opcode == OpCode::Jump && !self.seen_jump {
            self.seen_jump = true;

            if self.buffer.is_empty() {
                // Empty loop body, nothing to peel.
                return OptimizationResult::Emit(op.clone());
            }

            // Perform the peeling: emit peeled body + Label + original body.
            // The returned map forwards original `buffer` positions to
            // their freshly-allocated body positions so we can rewrite
            // the back-edge Jump's args without re-deriving the layout.
            let orig_ref_map = self.peel_iteration(op, ctx);

            // Emit the final Jump with remapped args pointing to the
            // body iteration's ops.
            let mut jump = op.clone();
            for arg in &mut jump.args {
                if let Some(&new_ref) = orig_ref_map.get(arg) {
                    *arg = new_ref;
                }
            }
            // Reserve the Jump's own position so it lands above any
            // inputarg range and above every body op allocated by
            // peel_iteration. Jump is Void-typed.
            jump.pos = ctx.reserve_pos_typed(jump.result_type());
            return OptimizationResult::Emit(jump);
        }

        // For non-Jump ops (or after we've already unrolled), buffer them.
        if !self.seen_jump {
            self.buffer.push(op.clone());
            return OptimizationResult::Remove;
        }

        // After unrolling, pass everything through.
        OptimizationResult::Emit(op.clone())
    }

    fn setup(&mut self) {
        self.buffer.clear();
        self.seen_jump = false;
    }

    fn name(&self) -> &'static str {
        "unroll"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::GcRef;
    use std::collections::HashMap;

    /// Assign sequential positions to ops starting from `base`.
    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef::from_raw(base + i as u32);
        }
    }

    #[test]
    fn test_retrace_box_pool_snapshot_preserves_inputargs_for_p1_full_prefix() {
        let input0 = crate::r#box::BoxRef::new_inputarg(Type::Ref, Some(0));
        let input1 = crate::r#box::BoxRef::new_inputarg(Type::Int, Some(1));
        let emit2 = crate::r#box::BoxRef::new_resop(Type::Ref, 2);
        let emit3 = crate::r#box::BoxRef::new_resop(Type::Int, 3);
        let snapshot = vec![input0.clone(), input1.clone(), emit2.clone(), emit3.clone()];

        let prefix =
            p1_full_prefix_from_box_pool_snapshot(Some(&snapshot)).expect("snapshot exists");

        assert_eq!(prefix, vec![input0, input1, emit2, emit3]);
    }

    #[test]
    fn test_exported_state_high_water_covers_retrace_namespace() {
        let exported = ExportedState::new(
            vec![OpRef::from_raw(52)],
            vec![OpRef::from_raw(109), OpRef::from_const(3)],
            crate::optimizeopt::virtualstate::VirtualState::new(Vec::new()),
            HashMap::new(),
            Vec::new(),
            vec![OpRef::from_raw(14)],
            Vec::new(),
            vec![OpRef::from_raw(23)],
        );

        assert_eq!(exported.opref_high_water(), 110);
    }

    fn run_unroll_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        // See `run_heap_opt` in heap.rs for the 1024-slot Ref seed
        // rationale — the preamble exporter needs an intrinsic type
        // for every renamed inputarg, which production derives from
        // the recorder's trace_inputarg_types.
        opt.trace_inputarg_types = vec![majit_ir::Type::Ref; 1024];
        // Production trace iteration gets rd_resume_position from
        // opencoder.py:399-401.  Direct unit-test guards must seed the
        // corresponding snapshot explicitly before store_final_boxes_in_guard.
        let (ops, snapshots) = super::super::seed_guard_snapshots_with(ops, |guard| {
            // Direct unroll tests use guard brackets as the explicit active
            // snapshot so remapping can verify the TraceIterator cache
            // semantics that RPython gets from opencoder.py.
            guard
                .fail_args
                .as_deref()
                .map(|fail_args| fail_args.iter().copied().collect())
                .unwrap_or_default()
        });
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024)
    }

    // ── Basic peeling ─────────────────────────────────────────────────

    #[test]
    fn test_no_jump_no_unroll() {
        // Without a Jump back-edge, the pass just buffers and nothing is emitted.
        // (In practice, traces always end with Jump or Finish.)
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Finish, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // IntAdd gets buffered, Finish is not a Jump so it gets buffered too.
        // Nothing is emitted because there's no Jump to trigger peeling.
        // The buffered ops are lost (which is correct: no loop = no unrolling).
        assert!(
            result.is_empty(),
            "no Jump means no loop to unroll, ops are buffered but never emitted"
        );
    }

    #[test]
    fn test_empty_loop_body() {
        // Jump with no prior ops: nothing to peel.
        let mut ops = vec![Op::new(OpCode::Jump, &[])];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Jump);
    }

    #[test]
    fn test_jump_to_preamble_preserves_jump_args() {
        let body_ops = vec![
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
                op.pos = OpRef::from_raw(2);
                op
            },
            Op::new(
                OpCode::Jump,
                &[OpRef::from_raw(0), OpRef::from_raw(2), OpRef::from_raw(50)],
            ),
        ];
        let preamble_target = TargetToken::new_preamble(7);

        let result = UnrollOptimizer::jump_to_preamble(&body_ops, &preamble_target);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(
            result[1].args.as_slice(),
            &[OpRef::from_raw(0), OpRef::from_raw(2), OpRef::from_raw(50)]
        );
        assert_eq!(
            result[1].descr.as_ref().map(|descr| descr.repr()),
            Some("LoopTargetDescr(start:7)".to_string())
        );
    }

    #[test]
    fn test_reshape_jump_args_for_preamble_pads_missing_invariants() {
        let mut jump_args = vec![OpRef::from_raw(0)];
        reshape_jump_args_for_preamble(&mut jump_args, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
        assert_eq!(
            jump_args.as_slice(),
            &[OpRef::from_raw(0), OpRef::from_raw(1)]
        );
    }

    #[test]
    fn test_reshape_jump_args_for_preamble_truncates_extra_slots() {
        let mut jump_args = vec![OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)];
        reshape_jump_args_for_preamble(&mut jump_args, &[OpRef::from_raw(10), OpRef::from_raw(11)]);
        assert_eq!(
            jump_args.as_slice(),
            &[OpRef::from_raw(0), OpRef::from_raw(1)]
        );
    }

    #[test]
    fn test_replace_terminal_jump_appends_when_body_prefix_has_no_jump() {
        let body_ops = vec![{
            let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
            op.pos = OpRef::from_raw(2);
            op
        }];
        let mut jump = Op::new(OpCode::Jump, &[OpRef::from_raw(2)]);
        jump.descr = Some(TargetToken::new_preamble(7).as_jump_target_descr());

        let result = replace_terminal_jump(&body_ops, jump);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(result[1].args.as_slice(), &[OpRef::from_raw(2)]);
        assert_eq!(
            result[1].descr.as_ref().map(|descr| descr.repr()),
            Some("LoopTargetDescr(start:7)".to_string())
        );
    }

    #[test]
    fn test_ensure_preamble_target_token_inserts_start_descr_first() {
        let mut unroll = UnrollOptimizer::new();
        let regular = TargetToken {
            token_id: 3,
            is_preamble_target: false,
            virtual_state: Some(crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(majit_ir::Type::Int),
            ])),
            short_preamble: None,
            short_preamble_producer: None,
            jump_target_descr: Arc::new(LoopTargetDescr::new(3, false)),
        };
        unroll.target_tokens.push(regular);

        unroll.ensure_preamble_target_token();

        assert_eq!(unroll.target_tokens.len(), 2);
        assert!(unroll.target_tokens[0].is_preamble_target);
        assert!(unroll.target_tokens[0].virtual_state.is_none());
        assert_eq!(unroll.target_tokens[1].token_id, 3);
    }

    #[test]
    fn test_pick_virtual_state_uses_target_generalization_direction() {
        // NonNull is Ref-typed; target Unknown must also be Ref to match.
        // RPython: NotVirtualStateInfoPtr(LEVEL_UNKNOWN) generalizes
        // NotVirtualStateInfoPtr(LEVEL_NONNULL).
        let my_vs = crate::optimizeopt::virtualstate::VirtualState::new(vec![
            crate::optimizeopt::virtualstate::VirtualStateInfo::NonNull,
        ]);
        let target_states = vec![
            crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(majit_ir::Type::Ref),
            ]),
            crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::KnownClass {
                    class_ptr: GcRef(0x1234),
                },
            ]),
        ];

        assert_eq!(pick_virtual_state(&my_vs, &target_states), Some(0));
    }

    #[test]
    fn test_simple_loop_peeled() {
        // A simple loop: one add op, then Jump.
        // Expected output: peeled_add, Label, original_add, Jump
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(
            result.len(),
            4,
            "expected: peeled_add, Label, original_add, Jump"
        );
        assert_eq!(result[0].opcode, OpCode::IntAdd); // peeled
        assert_eq!(result[1].opcode, OpCode::Label);
        assert_eq!(result[2].opcode, OpCode::IntAdd); // original body
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    #[test]
    fn test_peeled_ops_have_different_positions() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 2 peeled + Label + 2 original + Jump = 6
        assert_eq!(result.len(), 6);

        // All positions should be unique.
        let positions: Vec<OpRef> = result.iter().map(|op| op.pos).collect();
        for (i, pos) in positions.iter().enumerate() {
            for (j, other) in positions.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        pos, other,
                        "positions at index {} and {} should differ",
                        i, j
                    );
                }
            }
        }
    }

    // ── OpRef remapping ───────────────────────────────────────────────

    #[test]
    fn test_internal_refs_remapped_in_peeled_copy() {
        // op0: v0 = IntAdd(v100, v101)  -- uses input args
        // op1: v1 = IntMul(v0, v101)    -- uses result of op0
        // Jump()
        //
        // After peeling:
        // peeled_v0 = IntAdd(v100, v101)     -- input refs unchanged
        // peeled_v1 = IntMul(peeled_v0, v101) -- v0 remapped to peeled_v0
        // Label()
        // body_v0 = IntAdd(v100, v101)
        // body_v1 = IntMul(body_v0, v101)    -- v0 remapped to body_v0
        // Jump()
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(101)]), // references op0
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(result.len(), 6); // 2 peeled + Label + 2 body + Jump

        // Peeled iteration:
        let peeled_add = &result[0];
        let peeled_mul = &result[1];
        assert_eq!(peeled_add.opcode, OpCode::IntAdd);
        assert_eq!(peeled_mul.opcode, OpCode::IntMul);
        // peeled_mul should reference peeled_add's position, not original op0.
        assert_eq!(peeled_mul.args[0], peeled_add.pos);
        // Second arg (input ref) should be unchanged.
        assert_eq!(peeled_mul.args[1], OpRef::from_raw(101));

        // Original body:
        let body_add = &result[3];
        let body_mul = &result[4];
        assert_eq!(body_add.opcode, OpCode::IntAdd);
        assert_eq!(body_mul.opcode, OpCode::IntMul);
        // body_mul should reference body_add's position.
        assert_eq!(body_mul.args[0], body_add.pos);
        assert_eq!(body_mul.args[1], OpRef::from_raw(101));
    }

    #[test]
    fn test_external_refs_preserved() {
        // Refs to ops outside the buffer (input arguments) should not be remapped.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Peeled add should still reference v100 and v101.
        assert_eq!(result[0].args[0], OpRef::from_raw(100));
        assert_eq!(result[0].args[1], OpRef::from_raw(101));

        // Body add should also reference v100 and v101.
        assert_eq!(result[2].args[0], OpRef::from_raw(100));
        assert_eq!(result[2].args[1], OpRef::from_raw(101));
    }

    // ── Guard preservation ────────────────────────────────────────────

    #[test]
    fn test_guards_duplicated_in_peel() {
        // Guards in the preamble serve as type checks.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_guard, peeled_add, Label, body_guard, body_add, Jump
        assert_eq!(result.len(), 6);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .count();
        assert_eq!(
            guard_count, 2,
            "guard should appear in both peeled and body"
        );
    }

    #[test]
    fn test_guard_fail_args_remapped() {
        // Guards with fail_args should have those refs remapped too.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            {
                let mut guard = Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]);
                guard.fail_args = Some(vec![OpRef::from_raw(0)].into()); // refs op0
                guard
            },
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Check peeled guard's fail_args.
        let peeled_guard = &result[1];
        assert_eq!(peeled_guard.opcode, OpCode::GuardTrue);
        let peeled_add_pos = result[0].pos;
        assert_eq!(
            peeled_guard.fail_args.as_ref().unwrap()[0],
            peeled_add_pos,
            "peeled guard's fail_args should reference peeled add"
        );

        // Check body guard's fail_args.
        let body_guard = &result[4]; // after Label (idx 3) and body_add (idx 3)
        assert_eq!(body_guard.opcode, OpCode::GuardTrue);
        let body_add_pos = result[3].pos;
        assert_eq!(
            body_guard.fail_args.as_ref().unwrap()[0],
            body_add_pos,
            "body guard's fail_args should reference body add"
        );
    }

    // ── Jump args remapping ───────────────────────────────────────────

    #[test]
    fn test_jump_args_remapped_to_body() {
        // Jump args should reference the body's ops, not the original positions.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[OpRef::from_raw(0)]), // carries v0 (the add result)
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_add, Label, body_add, Jump
        assert_eq!(result.len(), 4);

        let jump = result.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);

        let body_add_pos = result[2].pos;
        assert_eq!(
            jump.args[0], body_add_pos,
            "Jump arg should reference body add, not original"
        );
    }

    #[test]
    fn test_label_args_match_jump_args() {
        // The Label should carry the same args as the Jump.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[OpRef::from_raw(0), OpRef::from_raw(100)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        let label = result.iter().find(|o| o.opcode == OpCode::Label).unwrap();
        let jump_args = &ops.last().unwrap().args;
        assert_eq!(
            label.args.as_slice(),
            jump_args.as_slice(),
            "Label args should match original Jump args"
        );
    }

    // ── Multiple ops in loop body ─────────────────────────────────────

    #[test]
    fn test_multi_op_loop() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::IntSub, &[OpRef::from_raw(0), OpRef::from_raw(101)]),
            Op::new(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(2)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 4 peeled + Label + 4 body + Jump = 10
        assert_eq!(result.len(), 10);

        // Verify structure: peeled body, then Label, then body, then Jump.
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::IntSub);
        assert_eq!(result[2].opcode, OpCode::IntMul);
        assert_eq!(result[3].opcode, OpCode::GuardTrue);
        assert_eq!(result[4].opcode, OpCode::Label);
        assert_eq!(result[5].opcode, OpCode::IntAdd);
        assert_eq!(result[6].opcode, OpCode::IntSub);
        assert_eq!(result[7].opcode, OpCode::IntMul);
        assert_eq!(result[8].opcode, OpCode::GuardTrue);
        assert_eq!(result[9].opcode, OpCode::Jump);
    }

    // ── Setup resets state ────────────────────────────────────────────

    #[test]
    fn test_setup_resets_state() {
        let mut pass = OptUnroll::new();

        // Simulate some state.
        pass.buffer
            .push(Op::new(OpCode::IntAdd, &[OpRef::from_raw(0)]));
        pass.seen_jump = true;

        pass.setup();

        assert!(pass.buffer.is_empty());
        assert!(!pass.seen_jump);
    }

    // ── Integration with optimizer ────────────────────────────────────

    #[test]
    fn test_unroll_standalone_optimizer() {
        // Run the unroll pass through the optimizer infrastructure.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        opt.trace_inputarg_types = vec![majit_ir::Type::Ref; 1024];
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        // Expect: peeled_add, peeled_guard, Label, body_add, body_guard, Jump = 6
        assert_eq!(result.len(), 6);

        // All ops should have valid (non-NONE) positions.
        for op in &result {
            assert!(
                !op.pos.is_none(),
                "op {:?} should have a valid pos",
                op.opcode
            );
        }
    }

    #[test]
    fn test_unroll_mints_fresh_guard_descrs() {
        // RPython parity: normal trace guards do not keep their encoded descr.
        //
        // opencoder.py:391 strips guard descrs to None during trace iteration;
        // optimizer.py:724-729 invents a fresh opcode-specific descr at guard
        // emission. ResumeAtPositionDescr is only for inline_short_preamble().
        let descr = crate::compile::make_resume_guard_descr_typed(Vec::new());
        let original_index = descr.index();
        let mut ops = vec![
            Op::with_descr(OpCode::GuardTrue, &[OpRef::from_raw(100)], descr.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        let guards: Vec<&Op> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .collect();
        assert_eq!(guards.len(), 2);
        for guard in &guards {
            assert!(guard.descr.is_some(), "guard should have a descriptor");
            assert!(
                guard.descr.as_ref().unwrap().is_resume_guard(),
                "optimizer.py:723: descr must be a ResumeGuardDescr subtype"
            );
        }
        let peel_index = guards[0].descr.as_ref().unwrap().index();
        let body_index = guards[1].descr.as_ref().unwrap().index();
        assert_ne!(peel_index, original_index);
        assert_ne!(body_index, original_index);
        assert_ne!(
            peel_index, body_index,
            "peeled and body guards must own distinct freshly invented descrs"
        );
        assert!(!guards[0].descr.as_ref().unwrap().is_resume_at_position());
        assert!(!guards[1].descr.as_ref().unwrap().is_resume_at_position());
    }

    // ── Chain of references ───────────────────────────────────────────

    #[test]
    fn test_chain_of_refs_correctly_remapped() {
        // v0 = IntAdd(v100, v101)
        // v1 = IntAdd(v0, v100)
        // v2 = IntAdd(v1, v0)
        // Jump(v2)
        //
        // In the peeled copy, all internal refs must point to peeled positions.
        // In the body copy, all internal refs must point to body positions.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(100)]),
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(1), OpRef::from_raw(0)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 3 peeled + Label + 3 body + Jump = 8
        assert_eq!(result.len(), 8);

        // Peeled iteration refs:
        let p0 = result[0].pos;
        let p1 = result[1].pos;
        let _p2 = result[2].pos;
        assert_eq!(result[1].args[0], p0, "peeled v1 should ref peeled v0");
        assert_eq!(result[2].args[0], p1, "peeled v2 should ref peeled v1");
        assert_eq!(result[2].args[1], p0, "peeled v2 should ref peeled v0");

        // Body refs:
        let b0 = result[4].pos;
        let b1 = result[5].pos;
        let b2 = result[6].pos;
        assert_eq!(result[5].args[0], b0, "body v1 should ref body v0");
        assert_eq!(result[6].args[0], b1, "body v2 should ref body v1");
        assert_eq!(result[6].args[1], b0, "body v2 should ref body v0");

        // Jump should reference body v2.
        let jump = &result[7];
        assert_eq!(jump.args[0], b2, "Jump should ref body v2");
    }

    #[test]
    fn test_unroll_optimizer_optimize_trace() {
        let mut unroll_opt = UnrollOptimizer::new();
        // IntAdd operates on Int-typed inputs — seed the inner phase1/2
        // optimizers' trace_inputarg_types via UnrollOptimizer so the
        // intbounds pass sees Int on OpRef::from_raw(0), OpRef::from_raw(1).
        unroll_opt.trace_inputarg_types = vec![majit_ir::Type::Int; 2];
        // Use optimize_trace_with_constants_and_inputs to properly set
        // num_inputs so input args don't collide with op positions.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops, 2);
        let mut constants = std::collections::HashMap::new();
        let (result, _) =
            unroll_opt.optimize_trace_with_constants_and_inputs(&ops, &mut constants, 2);
        // The optimizer processes the trace; result should not be empty
        assert!(!result.is_empty(), "optimize_trace should produce output");
    }

    #[test]
    fn test_unroll_optimizer_count_guards() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(0)]),
        ];
        assert_eq!(UnrollOptimizer::count_guards(&ops), 2);
    }

    #[test]
    fn test_exported_state_reimports_widened_intbounds() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        use crate::optimizeopt::intutils::IntBound;

        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        let mut exported_bounds = std::collections::HashMap::new();
        exported_bounds.insert(OpRef::from_raw(21), IntBound::bounded(10, 20));

        let exported = export_state(
            &[OpRef::from_raw(21)],
            &[],
            &mut optimizer,
            &mut ctx,
            Some(&exported_bounds),
        );

        assert_eq!(
            match &exported.exported_infos[&OpRef::from_raw(21)] {
                crate::optimizeopt::info::OpInfo::IntBound(b) => Some((b.lower, b.upper)),
                _ => None,
            },
            Some((10, 20))
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Ref]);
        let label_args = import_state(&[OpRef::from_raw(0)], &exported, &mut optimizer, &mut ctx2);
        assert_eq!(label_args, vec![OpRef::from_raw(21)]);
        // unroll.py:93-96: IntBound IS imported with widen() and stored
        // directly on the box's _forwarded slot via setintbound.
        // widen() relaxes bounds: lower < MININT/2 → MININT, upper > MAXINT/2 → MAXINT.
        // For [10, 20], both are within MININT/2..MAXINT/2 so widen() preserves them.
        let imported_bound = ctx2.getintbound(OpRef::from_raw(21));
        assert_eq!((imported_bound.lower, imported_bound.upper), (10, 20));
    }

    #[test]
    fn test_export_state_uses_forced_end_args_snapshot() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Ref]);
        ctx.preamble_end_args = Some(vec![OpRef::from_raw(21)]);

        let exported = export_state(&[OpRef::from_raw(0)], &[], &mut optimizer, &mut ctx, None);

        assert_eq!(exported.end_args, vec![OpRef::from_raw(21)]);
    }

    #[test]
    fn test_exported_state_reimports_short_heap_field_facts() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        // optimizer.py:478 ensure_ptr_info_arg0 parity: every FieldDescr
        // must resolve its parent SizeDescr so that the routing between
        // StructPtrInfo and InstancePtrInfo can ask `is_object()`.
        // SimpleFieldDescr stores the parent as Weak<DescrRef>, so the
        // test must keep the parent Arc alive until import_state has run.
        let parent = majit_ir::descr::make_size_descr(16);
        let field_descr = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(0, 0, 8, majit_ir::Type::Int, false)
                .with_signed(true)
                .with_parent_descr(parent.clone(), 0),
        ) as majit_ir::DescrRef;
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcI,
                        &[OpRef::from_raw(10)],
                        field_descr.clone(),
                    );
                    op.pos = OpRef::from_raw(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(
            &[OpRef::from_raw(10), OpRef::from_raw(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Ref, Type::Ref]);
        let targetargs = [OpRef::from_raw(0), OpRef::from_raw(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(&targetargs, &label_args, &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef::from_raw(10), OpRef::from_raw(11)]);
        // RPython PreambleOp parity: PreambleOp stored in PtrInfo._fields.
        // No imported_short_fields for heap fields — PtrInfo is the single
        // source of truth, matching RPython's HeapOp.produce_op → opinfo.setfield.
        let obj_resolved = ctx2.get_box_replacement(OpRef::from_raw(10));
        let pop = ctx2
            .with_ptr_info_mut(obj_resolved, |info| info.take_preamble_field(0))
            .flatten();
        assert!(pop.is_some(), "PreambleOp must be in PtrInfo._fields");
        let pop = pop.unwrap();
        assert_eq!(pop.op, OpRef::from_raw(11)); // Phase 1 source — pop.op
        // forwards via replace_op to the body-visible OpRef.
        drop(parent);
    }

    #[test]
    fn test_import_state_reimports_short_ref_constant_identity() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        let ptr = GcRef(0x1234_5678);
        let field_descr = majit_ir::descr::make_field_descr_full(88, 0, 8, Type::Int, false);
        ctx.seed_constant(OpRef::from_const(23), Value::Ref(ptr));
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcPureI,
                        &[OpRef::from_const(23)],
                        field_descr.clone(),
                    );
                    op.pos = OpRef::from_raw(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(
            &[OpRef::from_raw(12), OpRef::from_raw(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(8, &[Type::Ref, Type::Ref]);
        let targetargs = [OpRef::from_raw(0), OpRef::from_raw(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(&targetargs, &label_args, &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef::from_raw(12), OpRef::from_raw(11)]);
        // unroll.py:454: the imported constant must land in a fresh
        // const-pool slot in the consumer trace, not reuse the producer
        // trace's slot index — majit OpRef is a trace-local index, not
        // a Python identity. The fresh slot is the first one allocated
        // by `reserve_const_ref` in ctx2.
        let fresh_const = OpRef::const_ptr(0);
        assert_eq!(ctx2.get_constant(fresh_const), Some(Value::Ref(ptr)));
        assert_eq!(ctx2.get_constant(OpRef::const_ptr(23)), None);
        assert_eq!(
            ctx2.imported_short_pure_ops,
            vec![crate::optimizeopt::ImportedShortPureOp::new(
                OpCode::GetfieldGcPureI,
                Some(field_descr.clone()),
                vec![crate::optimizeopt::ImportedShortPureArg::Const(
                    Value::Ref(ptr),
                    fresh_const,
                )],
                OpRef::from_raw(11),
                OpRef::from_raw(11),
                false
            )]
        );
    }

    #[test]
    fn test_import_short_loopinvariant_uses_producer_const_snapshot() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        let func = OpRef::from_const(7);
        let func_ptr = 0xCAFE;
        ctx.seed_constant(func, Value::Int(func_ptr));
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::CallLoopinvariantI, &[func]);
                    op.pos = OpRef::from_raw(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(
            &[OpRef::from_raw(10), OpRef::from_raw(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(8, &[Type::Ref, Type::Ref]);
        let targetargs = [OpRef::from_raw(0), OpRef::from_raw(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(&targetargs, &label_args, &exported, &mut ctx2);

        assert_eq!(
            ctx2.imported_loop_invariant_results.get(&func_ptr),
            Some(&OpRef::from_raw(11))
        );
        assert_eq!(ctx2.get_constant(func), None);
        assert_eq!(
            ctx2.get_constant(OpRef::from_const(0)),
            Some(Value::Int(func_ptr))
        );
    }

    #[test]
    fn test_import_short_loopinvariant_result_uses_short_inputarg_slot() {
        let func = OpRef::from_const(7);
        let func_ptr = 0xBEEF;
        let source = OpRef::from_raw(11);
        let phase2_result = OpRef::from_raw(3);
        let exported = ExportedState::new(
            vec![source],
            vec![source],
            crate::optimizeopt::virtualstate::VirtualState::new(Vec::new()),
            HashMap::new(),
            vec![crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::CallLoopinvariantI, &[func]);
                    op.pos = source;
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant,
                label_arg_idx: Some(0),
                invented_name: false,
                same_as_source: None,
            }],
            Vec::new(),
            Vec::new(),
            vec![source],
        );
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            8,
            &[Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        );
        ctx.seed_constant(func, Value::Int(func_ptr));

        import_short_preamble_state(&[OpRef::from_raw(0)], &[phase2_result], &exported, &mut ctx);

        // Path B (B.6.7-loopinv): imported_loop_invariant_results stores the
        // Phase 1 source directly (RPython `shortpreamble.py:120 op = self.res`).
        let _ = phase2_result;
        assert_eq!(
            ctx.imported_loop_invariant_results.get(&func_ptr),
            Some(&source)
        );
    }

    /// B.6.3 invariant: `force_op_from_preamble_op` follows RPython
    /// `unroll.py:26-39` line-by-line. It calls `use_box` (updates
    /// `self.short`) and seeds `potential_extra_ops`. It does NOT touch
    /// `used_boxes` / `short_preamble_jump`. Those grow only via the
    /// orthodox `force_box` → `potential_extra_ops.pop` →
    /// `add_preamble_op` (`shortpreamble.py:432-440`) path.
    #[test]
    fn test_force_op_from_preamble_orthodox_does_not_record_used_boxes() {
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            10,
            &[Type::Ref, Type::Ref, Type::Ref],
        );
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
            &[
                OpRef::from_raw(10),
                OpRef::from_raw(11),
                OpRef::from_raw(12),
            ],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
                    op.pos = OpRef::from_raw(20);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );

        let produced = ctx
            .imported_short_preamble_builder
            .as_ref()
            .unwrap()
            .produced_short_op(OpRef::from_raw(20))
            .unwrap();
        let pop = crate::optimizeopt::info::PreambleOp {
            op: OpRef::from_raw(20),
            invented_name: produced.invented_name,
            preamble_op: produced.preamble_op,
        };
        let forced = ctx.force_op_from_preamble_op(&pop);
        assert_eq!(forced, OpRef::from_raw(20));

        // RPython `unroll.py:32` `use_box` populates `self.short`.
        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.ops.len(), 1);
        // RPython `unroll.py:34-37` seeds `potential_extra_ops` so a later
        // `force_box` will run `add_preamble_op` (shortpreamble.py:432-440).
        assert!(
            ctx.potential_extra_ops.contains_key(&OpRef::from_raw(20)),
            "force_op_from_preamble_op must seed potential_extra_ops"
        );
        // RPython parity: `force_op_from_preamble` does NOT call
        // `add_preamble_op`. `used_boxes` / `short_preamble_jump` /
        // `extra_same_as` stay empty until `force_box` pops the entry.
        assert!(sp.used_boxes.is_empty());
        assert!(sp.jump_args.is_empty());

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(OpRef::from_raw(20), &mut ctx);

        let sp = ctx.build_imported_short_preamble().unwrap();
        // After force_box: orthodox `add_preamble_op` (shortpreamble.py:432-440)
        // populated all three lists in lock-step.
        assert_eq!(sp.used_boxes, vec![OpRef::from_raw(20)]);
        assert_eq!(sp.jump_args, vec![OpRef::from_raw(20)]);
        assert!(
            !ctx.potential_extra_ops.contains_key(&OpRef::from_raw(20)),
            "force_box must consume the potential_extra_ops entry"
        );
    }

    /// B.6.3 invariant (Heap variant): same orthodox boundary as the Pure
    /// case, with the producer's `replace_op(source, value)` forwarding
    /// installed (`shortpreamble.rs::produce_heap_field`).
    /// `force_op_from_preamble_op` returns `preamble_source` (RPython
    /// `unroll.py:38 return preamble_op.op` ≡ `self.res`); the producer's
    /// `get_box_replacement(source)` lookup is consumed inside `force_box`
    /// (`shortpreamble.py:436 op = preamble_op.op.get_box_replacement()`)
    /// when it runs `add_preamble_op`.
    #[test]
    fn test_force_op_from_preamble_returns_source_for_heap_variant() {
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            32,
            &[Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        );
        ctx.initialize_imported_short_preamble_builder(
            &[
                OpRef::from_raw(0),
                OpRef::from_raw(1),
                OpRef::from_raw(2),
                OpRef::from_raw(3),
            ],
            &[
                OpRef::from_raw(10),
                OpRef::from_raw(11),
                OpRef::from_raw(12),
                OpRef::from_raw(13),
            ],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcR,
                        &[OpRef::from_raw(3)],
                        majit_ir::descr::make_field_descr_full(56, 0, 8, Type::Ref, false),
                    );
                    op.pos = OpRef::from_raw(19);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );
        let produced = ctx
            .imported_short_preamble_builder
            .as_ref()
            .unwrap()
            .produced_short_op(OpRef::from_raw(19))
            .unwrap();
        // Path B (B.6.7-heap-field): produce_heap_field no longer installs
        // replace_op, but the test still walks the get_box_replacement
        // chain inside force_box's add_preamble_op, so install a manual
        // forwarding to the body-visible OpRef to exercise that path.
        ctx.replace_op(OpRef::from_raw(19), OpRef::from_raw(14));
        let pop = crate::optimizeopt::info::PreambleOp {
            op: OpRef::from_raw(19),
            invented_name: produced.invented_name,
            preamble_op: produced.preamble_op,
        };
        let forced = ctx.force_op_from_preamble_op(&pop);
        // RPython `unroll.py:38 return preamble_op.op` ≡ self.res.
        // pyre's Phase 1 source IS self.res for the imported short box.
        assert_eq!(forced, OpRef::from_raw(19));

        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.ops.len(), 1);
        // RPython parity: orthodox boundary — used_boxes / jump_args stay
        // empty until force_box runs add_preamble_op.
        assert!(sp.used_boxes.is_empty());
        assert!(sp.jump_args.is_empty());
        // `force_op_from_preamble_op` keys potential_extra_ops by
        // `preamble_source` (non-invented) per `unroll.py:34-37`.
        assert!(
            ctx.potential_extra_ops.contains_key(&OpRef::from_raw(19)),
            "force_op_from_preamble_op must seed potential_extra_ops by source"
        );

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(forced, &mut ctx);

        let sp = ctx.build_imported_short_preamble().unwrap();
        // shortpreamble.py:436 `op = preamble_op.op.get_box_replacement()`.
        // pop.op=19 forwards to body-visible 14 via the producer's replace_op,
        // so used_boxes carries the resolved body-visible OpRef while
        // jump_args carries the unresolved Phase 1 source.
        assert_eq!(sp.used_boxes, vec![OpRef::from_raw(14)]);
        assert_eq!(sp.jump_args, vec![OpRef::from_raw(19)]);
        assert!(
            !ctx.potential_extra_ops.contains_key(&OpRef::from_raw(19)),
            "force_box must consume the potential_extra_ops entry"
        );
    }

    #[test]
    fn test_exported_state_reimports_invented_short_alias_metadata() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(6, 0);
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op =
                        Op::new(OpCode::IntAdd, &[OpRef::from_raw(12), OpRef::from_raw(13)]);
                    op.pos = OpRef::from_raw(30);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: true,
                same_as_source: Some(OpRef::from_raw(14)),
            });

        let exported = export_state(
            &[
                OpRef::from_raw(12),
                OpRef::from_raw(13),
                OpRef::from_raw(14),
            ],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // Generous Ref-typed inputarg pool — the test only consumes slots 0..3
        // as `targetargs` and walks `OpRef::from_raw(K)` for various large K
        // as opaque opref handles. Ref matches the producer shape.
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(6, &vec![Type::Ref; 1024]);
        let targetargs = [OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(&targetargs, &label_args, &exported, &mut ctx2);
        let imported_result = ctx2.imported_short_pure_ops[0].result;
        assert_ne!(imported_result, OpRef::from_raw(30));
        let pop = ctx2.imported_short_pure_ops[0].pop.clone();
        let forced = ctx2.force_op_from_preamble_op(&pop);
        // force_op_from_preamble may return the imported position (not necessarily 30)
        let _ = forced;
        assert_eq!(ctx2.imported_short_pure_ops.len(), 1);
        // RPython parity: extra_same_as is populated lazily by add_preamble_op
        // (called from optimizer.force_box's potential_extra_ops.pop path).
        // shortpreamble.py:432-440 record the SameAs Box at use-box time, not
        // at produce_op time — verify the lazy population fires.
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(forced, &mut ctx2);
        let aliases = ctx2.used_imported_short_aliases();
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0].same_as_source, OpRef::from_raw(14));
        assert_eq!(aliases[0].same_as_opcode, OpCode::SameAsI);
    }

    #[test]
    fn test_assemble_peeled_trace_emits_extra_same_as_before_label() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
            op.pos = OpRef::from_raw(3);
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntMul, &[OpRef::from_raw(50), OpRef::from_raw(0)]);
                op.pos = OpRef::from_raw(1);
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(50)]),
        ];

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::from_raw(10)],
            &[OpRef::from_raw(0)],
            &[],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::from_raw(50),
                same_as_source: OpRef::from_raw(10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &std::collections::HashMap::new(),
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].opcode, OpCode::SameAsI);
        assert_eq!(combined[1].args.as_slice(), &[OpRef::from_raw(10)]);
        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(combined[3].opcode, OpCode::IntMul);
        assert_eq!(combined[3].args[0], combined[1].pos);
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].args[0], combined[1].pos);
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_visible_label_arg_until_body_redef() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
            op.pos = OpRef::from_raw(11);
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntGe, &[OpRef::from_raw(11), OpRef::from_const(0)]);
                op.pos = OpRef::from_raw(4);
                op
            },
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(11), OpRef::from_const(1)]);
                op.pos = OpRef::from_raw(11);
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(11)]),
        ];

        let mut constants = std::collections::HashMap::new();
        constants.insert(OpRef::from_const(0).raw(), 2);
        constants.insert(OpRef::from_const(1).raw(), 1);

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::from_raw(11)],
            &[OpRef::from_raw(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[1].args.as_slice(), &[OpRef::from_raw(11)]);
        assert_eq!(combined[2].opcode, OpCode::IntGe);
        assert_eq!(combined[2].args[0], OpRef::from_raw(11));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_eq!(combined[3].args[0], OpRef::from_raw(11));
        assert_ne!(combined[3].pos, OpRef::from_raw(11));
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].args[0], combined[3].pos);
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_visible_preamble_box_over_body_collision() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::GetfieldGcR, &[OpRef::from_raw(3)]);
            op.pos = OpRef::from_raw(19);
            op.descr = Some(majit_ir::descr::make_field_descr_full(
                56,
                0,
                8,
                Type::Ref,
                false,
            ));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::SetfieldGc,
                    &[OpRef::from_raw(25), OpRef::from_raw(19)],
                );
                op.descr = Some(majit_ir::descr::make_field_descr_full(
                    57,
                    8,
                    8,
                    Type::Ref,
                    false,
                ));
                op
            },
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_const(1)]);
                op.pos = OpRef::from_raw(19);
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(19)]),
        ];

        let mut constants = std::collections::HashMap::new();
        constants.insert(OpRef::from_const(1).raw(), 1);

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::from_raw(0)],
            &[OpRef::from_raw(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[2].opcode, OpCode::SetfieldGc);
        assert_eq!(combined[2].args[1], OpRef::from_raw(19));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_ne!(combined[3].pos, OpRef::from_raw(19));
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].args[0], combined[3].pos);
    }

    #[test]
    fn test_assemble_peeled_trace_extends_label_with_used_boxes() {
        // Production caller passes body ops whose args have already been
        // resolved through ctx.get_box_replacement (optimizer.rs:2236-2262
        // forwarding-resolve pass). We mirror that here by writing the
        // resolved label_arg (OpRef::from_raw(10)) directly into the body op rather
        // than the raw inputarg position OpRef::from_raw(0).
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
            op.pos = OpRef::from_raw(3);
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntMul, &[OpRef::from_raw(50), OpRef::from_raw(10)]);
                op.pos = OpRef::from_raw(1);
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(10), OpRef::from_raw(50)]),
        ];

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::from_raw(10)],
            &[OpRef::from_raw(0)],
            &[OpRef::from_raw(50)],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::from_raw(50),
                same_as_source: OpRef::from_raw(10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &std::collections::HashMap::new(),
            None,
            None,
        );

        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(
            combined[2].args.as_slice(),
            &[OpRef::from_raw(10), combined[1].pos]
        );
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(
            combined[4].args.as_slice(),
            &[OpRef::from_raw(10), combined[1].pos]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_keeps_used_box_with_stale_constant_entry() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::SameAsI, &[OpRef::from_raw(37)]);
            op.pos = OpRef::from_raw(857);
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::GuardValue,
                    &[OpRef::from_raw(857), OpRef::from_const(0)],
                );
                op.fail_args = Some(vec![OpRef::from_raw(857)].into());
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    OpRef::from_raw(10),
                    OpRef::from_raw(853),
                    OpRef::from_raw(857),
                    OpRef::from_raw(850),
                ],
            ),
        ];
        let constants = std::collections::HashMap::from([
            (OpRef::from_raw(857).raw(), 2_i64),
            (OpRef::from_const(0).raw(), 2_i64),
        ]);

        let mut ctx = assemble_test_context(&p1_ops, &p2_ops, 1);
        let combined = assemble_peeled_trace_with_jump_args(
            &p1_ops,
            &p2_ops,
            &[OpRef::from_raw(10)],
            &[OpRef::from_raw(0)],
            &[
                OpRef::from_raw(22),
                OpRef::from_raw(857),
                OpRef::from_raw(150),
            ],
            &[
                OpRef::from_raw(853),
                OpRef::from_raw(857),
                OpRef::from_raw(850),
            ],
            1,
            0,
            true,
            &[],
            &constants,
            None,
            None,
            &[],
            &mut ctx,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(
            combined[1].args.as_slice(),
            &[
                OpRef::from_raw(10),
                OpRef::from_raw(22),
                OpRef::from_raw(857),
                OpRef::from_raw(150)
            ]
        );
        assert_eq!(combined[2].opcode, OpCode::GuardValue);
        assert_eq!(
            combined[2].args.as_slice(),
            &[OpRef::from_raw(857), OpRef::from_const(0)]
        );
        assert_eq!(combined[3].opcode, OpCode::Jump);
        assert_eq!(
            combined[3].args.as_slice(),
            &[
                OpRef::from_raw(10),
                OpRef::from_raw(22),
                OpRef::from_raw(857),
                OpRef::from_raw(150)
            ]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_remaps_extra_label_source_slots() {
        // Production caller pre-resolves body args through
        // ctx.get_box_replacement; OpRef::from_raw(0) (Phase 2 inputarg slot) is
        // pre-replaced with OpRef::from_raw(10) (the corresponding label_arg).
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::GetfieldGcPureI, &[OpRef::from_raw(50)]);
                op.pos = OpRef::from_raw(1);
                op.descr = Some(majit_ir::make_field_descr(
                    0,
                    8,
                    majit_ir::Type::Int,
                    majit_ir::ArrayFlag::Signed,
                ));
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(10), OpRef::from_raw(50)]),
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::from_raw(10)],
            &[OpRef::from_raw(0)],
            &[OpRef::from_raw(50)],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::from_raw(50),
                same_as_source: OpRef::from_raw(10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &std::collections::HashMap::new(),
            None,
            None,
        );

        let label_idx = combined
            .iter()
            .position(|op| op.opcode == OpCode::Label)
            .expect("label");
        let label = &combined[label_idx];
        let extra_label_arg = label.args[1];
        assert_eq!(
            label.args.as_slice(),
            &[OpRef::from_raw(10), extra_label_arg]
        );
        let body_getfield = &combined[label_idx + 1];
        assert_eq!(body_getfield.opcode, OpCode::GetfieldGcPureI);
        assert_eq!(body_getfield.args.as_slice(), &[extra_label_arg]);
    }

    #[test]
    fn test_assemble_peeled_trace_carries_body_value_used_before_local_def() {
        // Production caller pre-resolves body args through
        // ctx.get_box_replacement (optimizer.rs:2236-2262), so
        // OpRef::from_raw(0) (Phase 2 inputarg slot) would already be replaced
        // with OpRef::from_raw(10) (the label_arg for that slot) by the time
        // it reaches the assembler. We mirror that here.
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::GuardTrue, &[OpRef::from_raw(64)]);
                op.fail_args = Some(vec![OpRef::from_raw(64)].into());
                op
            },
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(10), OpRef::from_const(0)]);
                op.pos = OpRef::from_raw(64);
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(64)]),
        ];
        let constants = std::collections::HashMap::from([(OpRef::from_const(0).raw(), 1_i64)]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::from_raw(10)],
            &[OpRef::from_raw(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0].args.as_slice(),
            &[OpRef::from_raw(10), OpRef::from_raw(64)]
        );
        assert_eq!(combined[1].opcode, OpCode::GuardTrue);
        assert_eq!(combined[1].args.as_slice(), &[OpRef::from_raw(64)]);
        assert_eq!(
            combined[1]
                .fail_args
                .as_ref()
                .expect("guard fail args")
                .as_slice(),
            &[OpRef::from_raw(64)]
        );
        assert_eq!(combined[2].opcode, OpCode::IntAdd);
        assert_ne!(combined[2].pos, OpRef::from_raw(64));
    }

    #[test]
    fn test_assemble_peeled_trace_does_not_route_jump_by_preamble_descr() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let p2_ops = vec![{
            let mut jump = Op::new(
                OpCode::Jump,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(2),
                    OpRef::from_raw(3),
                    OpRef::from_raw(4),
                ],
            );
            jump.descr = Some(start_descr.clone());
            jump
        }];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[
                OpRef::from_raw(10),
                OpRef::from_raw(11),
                OpRef::from_raw(12),
                OpRef::from_raw(13),
                OpRef::from_raw(14),
            ],
            &[
                OpRef::from_raw(100),
                OpRef::from_raw(101),
                OpRef::from_raw(102),
            ],
            &[OpRef::from_raw(13), OpRef::from_raw(14)],
            5,
            false,
            &[],
            &std::collections::HashMap::new(),
            Some(start_descr),
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[2].opcode, OpCode::Jump);
        // RPython parity: unroll.py:238-242 `jump_to_preamble` retargets the
        // Jump before compile.py assembles the trace, and this assembly helper
        // must not derive a different arg contract by inspecting that descr.
        // The caller is responsible for constructing the preamble-shaped Jump.
        assert_eq!(
            combined[2].args.as_slice(),
            &[
                OpRef::from_raw(100),
                OpRef::from_raw(101),
                OpRef::from_raw(102),
                OpRef::from_raw(3),
                OpRef::from_raw(4)
            ]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_preamble_jump_arity_when_body_label_is_shorter() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
                op.pos = OpRef::from_raw(2);
                op
            },
            {
                let mut jump = Op::new(OpCode::Jump, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
                jump.descr = Some(start_descr.clone());
                jump
            },
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::from_raw(0)],
            &[OpRef::from_raw(100), OpRef::from_raw(101)],
            &[],
            2,
            false,
            &[],
            &std::collections::HashMap::new(),
            Some(start_descr),
            Some(loop_descr),
        );

        let jump = combined.last().expect("assembled jump");
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(
            jump.args.as_slice(),
            &[OpRef::from_raw(100), OpRef::from_raw(101)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_does_not_extend_body_label_with_preamble_jump_only_args() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(10), OpRef::from_const(0)]);
                op.pos = OpRef::from_raw(20);
                op
            },
            {
                let mut jump = Op::new(
                    OpCode::Jump,
                    &[
                        OpRef::from_raw(10),
                        OpRef::from_raw(50),
                        OpRef::from_raw(60),
                    ],
                );
                jump.descr = Some(start_descr.clone());
                jump
            },
        ];
        let constants = std::collections::HashMap::from([(OpRef::from_const(0).raw(), 1_i64)]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::from_raw(10)],
            &[
                OpRef::from_raw(100),
                OpRef::from_raw(101),
                OpRef::from_raw(102),
            ],
            &[],
            1,
            false,
            &[],
            &constants,
            Some(start_descr),
            Some(loop_descr),
        );

        let body_label = combined
            .iter()
            .find(|op| {
                op.opcode == OpCode::Label
                    && op.descr.as_ref().map(|descr| descr.repr())
                        == Some("LoopTargetDescr(1)".to_string())
            })
            .expect("body label");
        assert_eq!(body_label.args.as_slice(), &[OpRef::from_raw(10)]);

        let jump = combined.last().expect("assembled jump");
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(
            jump.args.as_slice(),
            &[
                OpRef::from_raw(10),
                OpRef::from_raw(50),
                OpRef::from_raw(60)
            ]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_skips_constant_slots_for_new_body_positions() {
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::New, &[]);
                op.pos = OpRef::from_raw(1);
                op
            },
            Op::new(
                OpCode::SetfieldGc,
                &[OpRef::from_raw(1), OpRef::from_raw(0)],
            ),
            Op::new(OpCode::Jump, &[OpRef::from_raw(0)]),
        ];
        let constants = std::collections::HashMap::from([(2_u32, 606_i64), (4_u32, 611_i64)]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::from_raw(10)],
            &[OpRef::from_raw(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[1].opcode, OpCode::New);
        assert_ne!(combined[1].pos, OpRef::from_raw(2));
        assert_ne!(combined[1].pos, OpRef::from_raw(4));
        assert_eq!(combined[2].opcode, OpCode::SetfieldGc);
        assert_eq!(combined[2].args[0], combined[1].pos);
    }

    #[test]
    fn test_assemble_peeled_trace_skips_constant_extra_label_args() {
        // Production caller pre-resolves the Jump's body inputarg ref
        // (OpRef::from_raw(0)) to label_args[0] = OpRef::from_raw(10) before it reaches the
        // assembler.
        //
        // `extra_label_args` holds one literal Const entry and one Box.
        // The filter keeps Boxes (they still back runtime slots) and drops
        // literal Const entries (their value is emitted inline at use
        // sites). Mirrors `assemble_peeled_trace_with_jump_args`'s
        // `label_arg.is_constant()` predicate.
        let const_extra = OpRef::from_const(7);
        let p2_ops = vec![Op::new(OpCode::Jump, &[OpRef::from_raw(10)])];
        let constants = std::collections::HashMap::from([(const_extra.raw(), 606_i64)]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::from_raw(10)],
            &[OpRef::from_raw(0)],
            &[const_extra, OpRef::from_raw(8)],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0].args.as_slice(),
            &[OpRef::from_raw(10), OpRef::from_raw(8)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_passes_through_resolved_body_inputs() {
        // Production caller pre-resolves Phase 2 body op args through
        // ctx.get_box_replacement (optimizer.rs:2236-2262), so by the time
        // they reach assemble_peeled_trace they already match label_args.
        // The assembler is therefore a passthrough for inputarg references
        // — no source_slot input_remap needed. This test verifies that
        // pre-resolved body args survive intact.
        let mut constants = std::collections::HashMap::new();
        constants.insert(OpRef::from_const(0).raw(), 1);
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[OpRef::from_raw(200), OpRef::from_const(0)],
                );
                op.pos = OpRef::from_raw(20);
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(200)]),
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::from_raw(200), OpRef::from_raw(300)],
            &[OpRef::from_raw(0)],
            &[],
            6,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0].args.as_slice(),
            &[OpRef::from_raw(200), OpRef::from_raw(300)]
        );
        assert_eq!(combined[1].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].args[0], OpRef::from_raw(200));
        assert_eq!(combined[2].opcode, OpCode::Jump);
        assert_eq!(
            combined[2].args.as_slice(),
            &[OpRef::from_raw(200), OpRef::from_raw(300)]
        );
    }

    #[test]
    fn test_splice_redirected_tail_replaces_terminal_jump() {
        let body_ops = vec![
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
                op.pos = OpRef::from_raw(3);
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(0)]),
        ];
        let redirected_tail = vec![
            {
                let mut op = Op::new(OpCode::GuardTrue, &[OpRef::from_raw(3)]);
                op.fail_args = Some(vec![OpRef::from_raw(3)].into());
                op
            },
            Op::new(OpCode::Jump, &[OpRef::from_raw(3), OpRef::from_raw(4)]),
        ];

        let spliced = splice_redirected_tail(&body_ops, &redirected_tail);
        assert_eq!(spliced.len(), 3);
        assert_eq!(spliced[0].opcode, OpCode::IntAdd);
        assert_eq!(spliced[1].opcode, OpCode::GuardTrue);
        assert_eq!(spliced[2].opcode, OpCode::Jump);
        assert_eq!(
            spliced[2].args.as_slice(),
            &[OpRef::from_raw(3), OpRef::from_raw(4)]
        );
    }

    #[test]
    fn test_closing_loop_contract_arity_uses_actual_jump_contract() {
        let ops = vec![Op::new(
            OpCode::Jump,
            &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
        )];

        assert_eq!(closing_loop_contract_arity(&ops, 5), 3);
    }
}
