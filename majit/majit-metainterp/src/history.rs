/// The Trace data structure — a completed sequence of IR operations.
///
/// A Trace is the output of the Trace and the input to the
/// optimizer and backend. It represents a linear sequence of operations
/// that forms a loop (ending with JUMP) or an exit (ending with FINISH).
///
/// Reference: rpython/jit/metainterp/history.py TreeLoop
use majit_ir::{DescrRef, InputArg, Op, OpCode, OpRef, Type, Value};

use crate::r#box::BoxRef;

/// RPython `History` parity name.
///
/// The current Rust port still fuses RPython's `History` recording role into
/// `TraceCtx`; keep the `History` item so line-by-line ports can refer to the
/// RPython role explicitly while the eventual `MetaInterp.history` field split
/// is still pending.
pub type History = crate::trace_ctx::TraceCtx;

/// Cut position for a materialized `TreeLoop`.
///
/// RPython's byte-stream opencoder uses the full 5-tuple cut point, but the
/// already-materialized `TreeLoop` only needs the op index into `ops`.
/// Keeping this separate avoids reusing byte-cursor `_pos` as if it were
/// always a `Vec<Op>` index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeLoopCutPosition {
    pub op_index: usize,
}

impl TreeLoopCutPosition {
    pub fn new(op_index: usize) -> Self {
        Self { op_index }
    }
}

/// A completed trace ready for optimization and compilation.
#[derive(Clone, Debug)]
pub struct TreeLoop {
    /// Input arguments to the trace (loop header variables).
    pub inputargs: Vec<InputArg>,
    /// The recorded operations, in execution order.
    pub ops: Vec<Op>,
    /// opencoder.py parity: per-guard snapshots captured during tracing.
    /// Indexed by the guard op's `rd_resume_position`.
    pub snapshots: Vec<crate::recorder::Snapshot>,
    /// Epic H H-3.0a: per-position BoxRef pool inherited from the
    /// `recorder::Trace` that produced this loop. `box_pool[i]` is the
    /// `AbstractValue` mirror of the operation at `op_count == i`
    /// (inputargs followed by ops, in record order). Empty for tests
    /// that synthesize a `TreeLoop` directly without going through the
    /// recorder.
    ///
    /// RPython parity: PyPy's `AbstractResOp` / `AbstractInputArg`
    /// objects created during tracing flow unchanged into the optimizer
    /// (same Python objects, same `_forwarded` slot). The Rust pool
    /// preserves that identity by carrying the same `Rc<Box>` allocations
    /// from the recorder through to the optimizer.
    pub box_pool: crate::r#box::BoxPool,
}

impl TreeLoop {
    #[inline]
    fn is_runtime_opref(opref: OpRef) -> bool {
        !opref.is_none() && !opref.is_constant()
    }

    /// Create a new trace from input arguments and operations.
    pub fn new(inputargs: Vec<InputArg>, ops: Vec<Op>) -> Self {
        TreeLoop {
            inputargs,
            ops,
            snapshots: Vec::new(),
            box_pool: crate::r#box::BoxPool::new(),
        }
    }

    /// Create a new trace with snapshots.
    pub fn with_snapshots(
        inputargs: Vec<InputArg>,
        ops: Vec<Op>,
        snapshots: Vec<crate::recorder::Snapshot>,
    ) -> Self {
        TreeLoop {
            inputargs,
            ops,
            snapshots,
            box_pool: crate::r#box::BoxPool::new(),
        }
    }

    /// H-3.0a: build with explicit BoxRef pool inherited from a recorder.
    /// Production path: `recorder::Trace::get_trace` uses this so the
    /// optimizer receives the same `Rc<Box>` allocations that were
    /// created during tracing.
    pub fn with_box_pool(
        inputargs: Vec<InputArg>,
        ops: Vec<Op>,
        snapshots: Vec<crate::recorder::Snapshot>,
        box_pool: impl Into<crate::r#box::BoxPool>,
    ) -> Self {
        TreeLoop {
            inputargs,
            ops,
            snapshots,
            box_pool: box_pool.into(),
        }
    }

    /// Number of operations in the trace.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Number of input arguments.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Whether this trace ends with a JUMP (i.e., is a loop).
    pub fn is_loop(&self) -> bool {
        self.ops.last().is_some_and(|op| op.opcode == OpCode::Jump)
    }

    /// Whether this trace ends with FINISH.
    pub fn is_finished(&self) -> bool {
        self.ops
            .last()
            .is_some_and(|op| op.opcode == OpCode::Finish)
    }

    /// Iterate over all operations.
    pub fn iter_ops(&self) -> impl Iterator<Item = &Op> {
        self.ops.iter()
    }

    /// Iterate over all guard operations.
    pub fn iter_guards(&self) -> impl Iterator<Item = &Op> {
        self.ops.iter().filter(|op| op.opcode.is_guard())
    }

    /// Number of guard operations.
    pub fn num_guards(&self) -> usize {
        self.ops.iter().filter(|op| op.opcode.is_guard()).count()
    }

    /// Get the final operation (Jump or Finish).
    pub fn get_final_op(&self) -> Option<&Op> {
        self.ops.last().filter(|op| op.opcode.is_final())
    }

    /// Get the Label position (if this is a peeled loop).
    pub fn find_label(&self) -> Option<usize> {
        self.ops.iter().position(|op| op.opcode == OpCode::Label)
    }

    /// Split at Label: returns (preamble_ops, body_ops).
    /// If no Label, returns (all_ops, empty).
    pub fn split_at_label(&self) -> (&[Op], &[Op]) {
        match self.find_label() {
            Some(pos) => (&self.ops[..pos], &self.ops[pos..]),
            None => (&self.ops, &[]),
        }
    }

    /// Get the input arg types.
    pub fn inputarg_types(&self) -> Vec<majit_ir::Type> {
        self.inputargs.iter().map(|ia| ia.tp).collect()
    }

    /// opencoder.py:848-850 Trace.get_iter() — produce a TraceIterator over
    /// the recorded ops with fresh per-iteration boxes.
    ///
    /// `start_index = 0` reproduces the legacy positional layout: inputargs
    /// allocated at `[OpRef::from_raw(0), …, OpRef::from_raw(num_inputargs - 1))`, op results
    /// at `[OpRef::from_raw(num_inputargs), …)`. Phase 2 / bridge callers that need
    /// disjoint OpRef namespaces must construct `TraceIterator::new`
    /// directly with a higher `start_index`.
    pub fn get_iter(&self) -> crate::opencoder::TraceIterator<'_> {
        let inputarg_types = self.inputarg_types();
        crate::opencoder::TraceIterator::new(
            &self.ops,
            0,
            self.ops.len(),
            None,
            &inputarg_types,
            0,
            None,
        )
    }

    /// history.py:552-608 check_consistency — full structural validation.
    ///
    /// Verifies:
    /// - No constants in inputargs
    /// - No duplicate inputargs
    /// - Every op arg is either a Const or was defined earlier
    /// - Guards have descrs (when `check_descr` is true)
    /// - fail_args entries are non-Const and defined
    /// - Non-guard ops have no fail_args (when `check_descr` is true)
    /// - Overflow ops are followed by GuardNoOverflow/GuardOverflow
    /// - LABEL resets the defined-set to its arglist
    /// - JUMP target (if any) is present
    pub fn check_consistency(&self) -> bool {
        self.check_consistency_impl(true)
    }

    fn check_consistency_impl(&self, check_descr: bool) -> bool {
        if self.ops.is_empty() {
            return true;
        }
        let mut seen = std::collections::HashSet::new();
        let mut op_positions = std::collections::HashSet::new();
        // history.py:564-565: inputargs must not contain constants
        for ia in &self.inputargs {
            let ia_ref = OpRef::input_arg_typed(ia.index, ia.tp);
            if ia_ref.is_constant() {
                return false;
            }
            // history.py:566-568: no duplicate inputargs
            if !seen.insert(ia_ref) {
                return false;
            }
        }

        // history.py:573-603: walk operations
        for (num, op) in self.ops.iter().enumerate() {
            // PyPy's operation objects are unique by allocation. Rust traces
            // carry that identity in OpRef positions, so duplicate positions
            // are structurally invalid before considering dataflow.
            if !op.pos.is_none() && !op_positions.insert(op.pos) {
                return false;
            }
            // history.py:576-578: ovf ops must be followed by guard_overflow
            if op.opcode.is_ovf() {
                if let Some(next_op) = self.ops.get(num + 1) {
                    if !next_op.opcode.is_guard_overflow() {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            // history.py:579-581: each arg must be Const or in seen
            for arg in &op.args {
                if arg.is_none() {
                    return false;
                }
                if !arg.is_constant() && !seen.contains(arg) {
                    return false;
                }
            }
            // history.py:582-593: guard checks
            if op.opcode.is_guard() {
                if check_descr && op.descr.is_none() {
                    return false;
                }
                // history.py:588-591: fail_args validation
                if let Some(ref fa) = op.fail_args {
                    for arg in fa.iter() {
                        if arg.is_none() {
                            continue;
                        }
                        if arg.is_constant() {
                            return false;
                        }
                        if !seen.contains(arg) {
                            return false;
                        }
                    }
                }
            } else if check_descr {
                // history.py:592-593: non-guard ops must have no fail_args
                if op.fail_args.is_some() {
                    return false;
                }
            }
            // history.py:594-595: if op produces a value, add to seen
            if op.opcode.result_type() != Type::Void {
                if !op.pos.is_none() {
                    if !seen.insert(op.pos) {
                        return false;
                    }
                }
            }
            // history.py:596-602: LABEL resets seen
            if op.opcode == OpCode::Label {
                seen.clear();
                for arg in &op.args {
                    if arg.is_none() || arg.is_constant() {
                        return false;
                    }
                    if !seen.insert(*arg) {
                        return false;
                    }
                }
            }
        }

        let last = self.ops.last().unwrap();
        if !last.opcode.is_final() {
            return false;
        }
        // history.py:605-608: if a JUMP has a target, it must be TargetToken.
        if last.opcode == OpCode::Jump {
            if let Some(descr) = last.descr.as_ref() {
                if descr.as_loop_target_descr().is_none() {
                    return false;
                }
            }
        }

        true
    }

    /// opencoder.py CutTrace parity — create a new trace by cutting at the
    /// given position. `original_boxes` become the new inputargs; any OpRef
    /// referenced after the cut but defined before it (and not in
    /// `original_boxes`) is re-emitted as a prefix operation (transitive
    /// closure of dependencies).
    pub fn cut_trace_from(
        &self,
        start: TreeLoopCutPosition,
        original_boxes: &[OpRef],
        original_box_types: &[majit_ir::Type],
    ) -> TreeLoop {
        self.cut_trace_from_with_consts(start, original_boxes, original_box_types, &[])
    }

    /// Like `cut_trace_from`, but with pre-allocated constant OpRefs for each
    /// original inputarg.  Escaped original inputargs are remapped to these
    /// pool-managed constants (already GC-rooted), preventing both stale
    /// pointers and entry-contract mismatches at compiled-code entry.
    pub fn cut_trace_from_with_consts(
        &self,
        start: TreeLoopCutPosition,
        original_boxes: &[OpRef],
        original_box_types: &[majit_ir::Type],
        inputarg_consts: &[OpRef],
    ) -> TreeLoop {
        use std::collections::{HashMap, HashSet, VecDeque};

        let num_original_inputargs = self.inputargs.len() as u32;
        let cut_ops = &self.ops[start.op_index..];

        // Phase 1: Build initial remap from original_boxes → new inputargs.
        // Each new inputarg carries the type recorded in `original_box_types[i]`.
        let mut remap: HashMap<OpRef, OpRef> = HashMap::new();
        let original_set: HashSet<OpRef> = original_boxes.iter().copied().collect();
        for (i, &old_ref) in original_boxes.iter().enumerate() {
            remap.insert(
                old_ref,
                OpRef::input_arg_typed(i as u32, original_box_types[i]),
            );
        }

        // Collect all OpRefs defined by post-cut ops.
        let defined_after_cut: HashSet<OpRef> = cut_ops
            .iter()
            .filter(|op| !op.pos.is_none())
            .map(|op| op.pos)
            .collect();

        // Phase 2: Find escaped refs — referenced after cut, defined before
        // cut, not in original_boxes. Use BFS for transitive closure: an
        // escaped op's own args may also be escaped.
        let is_pre_cut_ref = |r: &OpRef| -> bool {
            Self::is_runtime_opref(*r)
                && !original_set.contains(r)
                && !defined_after_cut.contains(r)
        };

        let mut escaped_set: HashSet<OpRef> = HashSet::new();
        let mut queue: VecDeque<OpRef> = VecDeque::new();

        // Seed with refs used by post-cut ops (args only, not fail_args).
        // RPython CutTrace parity: pre-cut refs in fail_args map to
        // OpRef::NONE (resume data handles materialization). Only regular
        // op args seed escaped refs for prefix re-emission.
        for op in cut_ops {
            for arg in &op.args {
                if is_pre_cut_ref(arg) && escaped_set.insert(*arg) {
                    queue.push_back(*arg);
                }
            }
        }

        // BFS: transitively collect dependencies of escaped ops.
        while let Some(esc_ref) = queue.pop_front() {
            if esc_ref.raw() < num_original_inputargs {
                // Original inputarg of the full trace — must become a new
                // inputarg (handled in phase 3 below).
                continue;
            }
            let op_idx = (esc_ref.raw() - num_original_inputargs) as usize;
            if let Some(op) = self.ops.get(op_idx) {
                for arg in &op.args {
                    if is_pre_cut_ref(arg) && escaped_set.insert(*arg) {
                        queue.push_back(*arg);
                    }
                }
            }
        }

        // Phase 3: Partition escaped refs.
        //  - "orig_inputarg_escaped": refs to the full trace's original inputargs
        //    that weren't in original_boxes → must become new inputargs.
        //  - "op_escaped": refs to pre-cut ops → re-emit as prefix operations.
        let mut orig_inputarg_escaped: Vec<OpRef> = Vec::new();
        let mut op_escaped: Vec<OpRef> = Vec::new();
        for &r in &escaped_set {
            if r.raw() < num_original_inputargs {
                orig_inputarg_escaped.push(r);
            } else {
                op_escaped.push(r);
            }
        }
        orig_inputarg_escaped.sort_by_key(|r| r.raw());
        op_escaped.sort_by_key(|r| r.raw()); // preserve original order

        // Phase 4: Build new inputargs.
        // If concrete initial values are available, escaped original inputargs
        // become typed constants (avoiding entry-contract mismatch at runtime).
        // Otherwise, they become additional inputargs (original behavior).
        let mut new_ia_boxes = original_boxes.to_vec();
        let mut new_ia_types = original_box_types.to_vec();
        for &r in &orig_inputarg_escaped {
            if let Some(&const_opref) = inputarg_consts.get(r.raw() as usize) {
                // Remap to the pre-allocated pool constant (already GC-rooted).
                remap.insert(r, const_opref);
            } else {
                // No pool constant available: fall back to new inputarg.
                let tp = self.inputargs[r.raw() as usize].tp;
                remap.insert(r, OpRef::input_arg_typed(new_ia_boxes.len() as u32, tp));
                new_ia_boxes.push(r);
                new_ia_types.push(tp);
            }
        }
        let new_inputargs_count = new_ia_boxes.len() as u32;

        let new_inputargs: Vec<InputArg> = new_ia_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| InputArg {
                index: i as u32,
                tp,
            })
            .collect();

        // Build a fresh `box_pool` mirroring the new namespace. Inputargs
        // go first (fresh `BoxRef::new_inputarg` with type/position),
        // followed by prefix re-emitted ops and post-cut ops (fresh
        // `BoxRef::new_resop` with result type). The optimizer reads
        // PtrInfo / IntBound / Const exclusively through these BoxRefs,
        // so retrace baselines must arrive with the pool seeded. Total
        // length: `new_ia_boxes.len() + op_escaped.len() + cut_ops.len()`.
        let mut box_pool: crate::r#box::BoxPool =
            Vec::with_capacity(new_ia_boxes.len() + op_escaped.len() + cut_ops.len()).into();
        for (i, &tp) in new_ia_types.iter().enumerate() {
            box_pool.push(BoxRef::new_inputarg(tp, Some(i as u32)));
        }
        for &r in op_escaped.iter() {
            let op_idx = (r.raw() - num_original_inputargs) as usize;
            let result_tp = self.ops[op_idx].opcode.result_type();
            let position = box_pool.len() as u32;
            box_pool.push(BoxRef::new_resop(result_tp, position));
        }
        for op in cut_ops.iter() {
            let position = box_pool.len() as u32;
            box_pool.push(BoxRef::new_resop(op.opcode.result_type(), position));
        }

        // Phase 5: Re-emit escaped ops as prefix, assigning fresh OpRefs.
        // Result type comes from the original op's opcode so the new OpRef
        // variant matches RPython's IntOp/FloatOp/RefOp dispatch.
        let mut next_ref = new_inputargs_count;
        for &r in &op_escaped {
            let op_idx = (r.raw() - num_original_inputargs) as usize;
            let result_tp = self.ops[op_idx].opcode.result_type();
            remap.insert(r, OpRef::op_typed(next_ref, result_tp));
            next_ref += 1;
        }

        // Also assign fresh refs for post-cut ops (shifted by prefix count).
        let prefix_count = op_escaped.len() as u32;
        for (i, op) in cut_ops.iter().enumerate() {
            if !op.pos.is_none() {
                remap.insert(
                    op.pos,
                    OpRef::op_typed(
                        new_inputargs_count + prefix_count + i as u32,
                        op.opcode.result_type(),
                    ),
                );
            }
        }

        let remap_ref = |r: &OpRef| -> OpRef {
            if !Self::is_runtime_opref(*r) {
                *r
            } else if let Some(&new_ref) = remap.get(r) {
                new_ref
            } else {
                OpRef::NONE
            }
        };

        // Build prefix ops (re-emitted escaped definitions).
        let mut prefix_ops: Vec<Op> = Vec::with_capacity(op_escaped.len());
        for (pi, &r) in op_escaped.iter().enumerate() {
            let op_idx = (r.raw() - num_original_inputargs) as usize;
            let orig_op = &self.ops[op_idx];
            let mut new_op = orig_op.clone();
            new_op.pos =
                OpRef::op_typed(new_inputargs_count + pi as u32, new_op.opcode.result_type());
            for arg in new_op.args.iter_mut() {
                *arg = remap_ref(arg);
            }
            // Prefix ops don't need fail_args (they're not guards).
            new_op.fail_args = None;
            prefix_ops.push(new_op);
        }

        // Phase 6: Remap post-cut ops.
        let mut new_ops: Vec<Op> = Vec::with_capacity(prefix_ops.len() + cut_ops.len());
        new_ops.extend(prefix_ops);
        for (i, op) in cut_ops.iter().enumerate() {
            let mut new_op = op.clone();
            new_op.pos = OpRef::op_typed(
                new_inputargs_count + prefix_count + i as u32,
                new_op.opcode.result_type(),
            );
            for arg in new_op.args.iter_mut() {
                *arg = remap_ref(arg);
            }
            if let Some(ref mut fa) = new_op.fail_args {
                for arg in fa.iter_mut() {
                    *arg = remap_ref(arg);
                }
            }
            new_ops.push(new_op);
        }

        // opencoder.py parity: carry snapshots through cut_trace_from.
        // RPython's CutTrace wraps the original trace and iterates from the
        // cut point — the TraceIterator._cache remaps old Box positions to
        // new InputArgs automatically. In pyre, snapshots store raw OpRef
        // indices that must be explicitly remapped to match the post-cut
        // OpRef namespace.
        let remapped_snapshots: Vec<crate::recorder::Snapshot> = self
            .snapshots
            .iter()
            .map(|snap| {
                let remap_tagged =
                    |t: &crate::recorder::SnapshotTagged| -> crate::recorder::SnapshotTagged {
                        match t {
                            crate::recorder::SnapshotTagged::Box(old_ref, tp) => {
                                if let Some(&new_ref) = remap.get(old_ref) {
                                    crate::recorder::SnapshotTagged::Box(new_ref, *tp)
                                } else if !Self::is_runtime_opref(*old_ref) {
                                    // Constants and NONE pass through unchanged.
                                    t.clone()
                                } else {
                                    // opencoder.py:287-288: _get(i) asserts
                                    // _cache[i] is not None. An unmapped pre-cut
                                    // Box has no entry in the post-cut namespace.
                                    // Map to NONE so _number_boxes emits
                                    // UNINITIALIZED rather than a stale TAGBOX.
                                    crate::recorder::SnapshotTagged::Box(OpRef::NONE, *tp)
                                }
                            }
                            other => other.clone(),
                        }
                    };
                crate::recorder::Snapshot {
                    frames: snap
                        .frames
                        .iter()
                        .map(|f| crate::recorder::SnapshotFrame {
                            jitcode_index: f.jitcode_index,
                            pc: f.pc,
                            boxes: f.boxes.iter().map(&remap_tagged).collect(),
                        })
                        .collect(),
                    vable_boxes: snap.vable_boxes.iter().map(&remap_tagged).collect(),
                    vref_boxes: snap.vref_boxes.iter().map(&remap_tagged).collect(),
                }
            })
            .collect();
        TreeLoop::with_box_pool(new_inputargs, new_ops, remapped_snapshots, box_pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;

    #[derive(Debug)]
    struct DummyGuardDescr;
    impl majit_ir::Descr for DummyGuardDescr {}

    fn iarg(pos: u32) -> OpRef {
        OpRef::input_arg_int(pos)
    }

    fn iop(pos: u32) -> OpRef {
        OpRef::int_op(pos)
    }

    fn vop(pos: u32) -> OpRef {
        OpRef::void_op(pos)
    }

    #[test]
    fn test_empty_trace() {
        let trace = TreeLoop::new(vec![], vec![]);
        assert_eq!(trace.num_ops(), 0);
        assert_eq!(trace.num_inputargs(), 0);
        assert!(!trace.is_loop());
        assert!(!trace.is_finished());
    }

    #[test]
    fn test_trace_with_jump() {
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
            ),
            Op::new(OpCode::Jump, &[OpRef::int_op(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.is_loop());
        assert!(!trace.is_finished());
        assert_eq!(trace.num_ops(), 2);
        assert_eq!(trace.num_inputargs(), 1);
    }

    #[test]
    fn test_trace_with_finish() {
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
            ),
            Op::new(OpCode::Finish, &[OpRef::int_op(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.is_loop());
        assert!(trace.is_finished());
    }

    #[test]
    fn test_inputarg_types() {
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let trace = TreeLoop::new(inputargs, vec![]);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
    }

    // ══════════════════════════════════════════════════════════════════
    // History / TreeLoop parity tests
    // Local parity coverage for history.py TreeLoop structure.
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_structure_inputargs_and_ops() {
        // TreeLoop has inputargs and operations as primary fields.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
            Op::new(OpCode::IntSub, &[OpRef::int_op(2), OpRef::input_arg_int(0)]),
            Op::new(OpCode::Jump, &[OpRef::int_op(3), OpRef::input_arg_int(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.num_ops(), 3);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_guards_can_have_fail_args() {
        // Guards in a trace carry fail_args.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef::input_arg_int(0)]);
        guard.fail_args = Some(smallvec::smallvec![
            OpRef::input_arg_int(0),
            OpRef::input_arg_int(1)
        ]);

        let ops = vec![
            guard,
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
            Op::new(OpCode::Jump, &[OpRef::int_op(2), OpRef::input_arg_int(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        let fa = guards[0].fail_args.as_ref().unwrap();
        assert_eq!(fa.len(), 2);
        assert_eq!(fa[0], OpRef::input_arg_int(0));
        assert_eq!(fa[1], OpRef::input_arg_int(1));
    }

    #[test]
    fn test_trace_iter_guards_filters_correctly() {
        // iter_guards returns only guard ops.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(2)]),
            Op::new(OpCode::IntSub, &[OpRef::int_op(2), OpRef::input_arg_int(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef::int_op(3)]),
            Op::new(OpCode::Jump, &[OpRef::int_op(3), OpRef::input_arg_int(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
        assert_eq!(guards[1].opcode, OpCode::GuardFalse);
    }

    #[test]
    fn test_trace_not_loop_not_finished() {
        // A trace without Jump or Finish is neither loop nor finished.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
        )];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.is_loop());
        assert!(!trace.is_finished());
    }

    #[test]
    fn test_trace_loop_vs_finish_exclusive() {
        // A trace cannot be both a loop and finished.
        let inputargs = vec![InputArg::new_int(0)];

        let loop_trace = TreeLoop::new(
            inputargs.clone(),
            vec![
                Op::new(
                    OpCode::IntAdd,
                    &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
                ),
                Op::new(OpCode::Jump, &[OpRef::int_op(1)]),
            ],
        );
        assert!(loop_trace.is_loop());
        assert!(!loop_trace.is_finished());

        let finish_trace = TreeLoop::new(
            inputargs,
            vec![
                Op::new(
                    OpCode::IntAdd,
                    &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
                ),
                Op::new(OpCode::Finish, &[OpRef::int_op(1)]),
            ],
        );
        assert!(!finish_trace.is_loop());
        assert!(finish_trace.is_finished());
    }

    #[test]
    fn test_trace_mixed_type_inputargs() {
        // Traces support mixed-type input arguments (int, ref, float).
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let ops = vec![Op::new(
            OpCode::Jump,
            &[
                OpRef::input_arg_int(0),
                OpRef::input_arg_ref(1),
                OpRef::input_arg_float(2),
            ],
        )];
        let trace = TreeLoop::new(inputargs, ops);

        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_multiple_guards_with_different_fail_args() {
        // Multiple guards can have different fail_args.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        let mut g0 = Op::new(OpCode::GuardTrue, &[OpRef::input_arg_int(0)]);
        g0.fail_args = Some(smallvec::smallvec![OpRef::input_arg_int(0)]);

        let mut g1 = Op::new(OpCode::GuardFalse, &[OpRef::input_arg_int(1)]);
        g1.fail_args = Some(smallvec::smallvec![
            OpRef::input_arg_int(0),
            OpRef::input_arg_int(1)
        ]);

        let ops = vec![
            g0,
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
            g1,
            Op::new(OpCode::Jump, &[OpRef::int_op(2), OpRef::input_arg_int(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);

        assert_eq!(guards[0].fail_args.as_ref().unwrap().len(), 1);
        assert_eq!(guards[1].fail_args.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_trace_guard_without_fail_args() {
        // Guards without explicitly set fail_args have None.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::input_arg_int(0)]),
            Op::new(OpCode::Jump, &[OpRef::input_arg_int(0)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        assert!(guards[0].fail_args.is_none());
    }

    #[test]
    fn test_trace_ops_have_correct_opcodes() {
        // iter_ops preserves op order and opcodes.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
            Op::new(OpCode::IntMul, &[OpRef::int_op(2), OpRef::input_arg_int(0)]),
            Op::new(OpCode::IntSub, &[OpRef::int_op(3), OpRef::input_arg_int(1)]),
            Op::new(OpCode::Jump, &[OpRef::int_op(4), OpRef::input_arg_int(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let opcodes: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::IntAdd, OpCode::IntMul, OpCode::IntSub, OpCode::Jump]
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // History breadth tests — deeper parity with test_history.py
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_ops_with_descrs() {
        // Ops can carry descriptors (field descrs, call descrs).
        use majit_ir::DescrRef;
        use std::sync::Arc;

        #[derive(Debug)]
        struct TestDescr(u32);
        impl majit_ir::Descr for TestDescr {
            fn index(&self) -> u32 {
                self.0
            }
        }

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let descr: DescrRef = Arc::new(TestDescr(42));
        let ops = vec![
            Op::with_descr(OpCode::CallI, &[OpRef::input_arg_int(0)], descr.clone()),
            Op::with_descr(OpCode::GuardTrue, &[OpRef::input_arg_int(0)], descr.clone()),
            Op::new(
                OpCode::Jump,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        // Call op has descr
        assert!(trace.ops[0].descr.is_some());
        assert_eq!(trace.ops[0].descr.as_ref().unwrap().index(), 42);
        // Guard op has descr
        assert!(trace.ops[1].descr.is_some());
        assert_eq!(trace.ops[1].descr.as_ref().unwrap().index(), 42);
        // Jump op has no descr
        assert!(trace.ops[2].descr.is_none());
    }

    #[test]
    fn test_trace_iteration_order_matches_recording() {
        // Iteration order must match the order in which ops were recorded.
        let inputargs = vec![InputArg::new_int(0)];
        let expected_opcodes = vec![
            OpCode::IntAdd,
            OpCode::IntSub,
            OpCode::IntMul,
            OpCode::IntNeg,
            OpCode::IntLt,
            OpCode::Jump,
        ];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
            ),
            Op::new(OpCode::IntSub, &[OpRef::int_op(1), OpRef::input_arg_int(0)]),
            Op::new(OpCode::IntMul, &[OpRef::int_op(2), OpRef::input_arg_int(0)]),
            Op::new(OpCode::IntNeg, &[OpRef::int_op(3)]),
            Op::new(OpCode::IntLt, &[OpRef::int_op(4), OpRef::input_arg_int(0)]),
            Op::new(OpCode::Jump, &[OpRef::int_op(4)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let actual: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(actual, expected_opcodes);
    }

    #[test]
    fn test_trace_is_immutable_snapshot() {
        // After creation, Trace fields are only accessible as immutable references.
        // Verify that cloning a trace produces an independent copy.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
            ),
            Op::new(OpCode::Jump, &[OpRef::int_op(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        let trace2 = trace.clone();

        assert_eq!(trace.num_ops(), trace2.num_ops());
        assert_eq!(trace.num_inputargs(), trace2.num_inputargs());
        assert_eq!(trace.is_loop(), trace2.is_loop());
    }

    #[test]
    fn test_trace_stress_100_ops() {
        // Stress test: a trace with 100+ operations.
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();
        let mut prev = OpRef::input_arg_int(0);
        for i in 0..100 {
            let mut op = Op::new(OpCode::IntAdd, &[prev, OpRef::input_arg_int(0)]);
            op.pos = OpRef::int_op(i + 1);
            ops.push(op);
            prev = OpRef::int_op(i + 1);
        }
        ops.push(Op::new(OpCode::Jump, &[prev]));
        let trace = TreeLoop::new(inputargs, ops);

        assert_eq!(trace.num_ops(), 101); // 100 IntAdd + 1 Jump
        assert!(trace.is_loop());

        // Verify first and last ops
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[99].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[100].opcode, OpCode::Jump);

        // All intermediate ops should be IntAdd
        for op in &trace.ops[..100] {
            assert_eq!(op.opcode, OpCode::IntAdd);
        }
    }

    #[test]
    fn test_trace_guard_fail_args_reference_valid_refs() {
        // fail_args must reference valid input or op refs.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        let add_op = Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        );
        let mut guard_op = Op::new(OpCode::GuardTrue, &[OpRef::int_op(2)]);
        // fail_args referencing input args (0, 1) and the add result (2)
        guard_op.fail_args = Some(smallvec::smallvec![
            OpRef::input_arg_int(0),
            OpRef::input_arg_int(1),
            OpRef::int_op(2)
        ]);

        let ops = vec![
            add_op,
            guard_op,
            Op::new(OpCode::Jump, &[OpRef::int_op(2), OpRef::input_arg_int(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guard = trace.iter_guards().next().unwrap();
        let fa = guard.fail_args.as_ref().unwrap();
        // All referenced OpRefs are valid: 0, 1 are inputargs; 2 is the add op
        assert!(fa.iter().all(|r| r.raw() <= 2));
        assert_eq!(fa.len(), 3);
    }

    #[test]
    fn test_trace_many_guards_with_varying_fail_args() {
        // Multiple guards with varying fail_args sizes.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        let mut g0 = Op::new(OpCode::GuardTrue, &[OpRef::input_arg_int(0)]);
        g0.fail_args = Some(smallvec::smallvec![]);

        let mut g1 = Op::new(OpCode::GuardFalse, &[OpRef::input_arg_int(1)]);
        g1.fail_args = Some(smallvec::smallvec![OpRef::input_arg_int(0)]);

        let add = Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        );

        let mut g2 = Op::new(OpCode::GuardTrue, &[OpRef::input_arg_int(0)]);
        g2.fail_args = Some(smallvec::smallvec![
            OpRef::input_arg_int(0),
            OpRef::input_arg_int(1),
            OpRef::int_op(2)
        ]);

        let ops = vec![
            g0,
            g1,
            add,
            g2,
            Op::new(
                OpCode::Jump,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 3);
        assert_eq!(guards[0].fail_args.as_ref().unwrap().len(), 0);
        assert_eq!(guards[1].fail_args.as_ref().unwrap().len(), 1);
        assert_eq!(guards[2].fail_args.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_trace_clone_independence() {
        // Modifications to a cloned trace do not affect the original.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
            ),
            Op::new(OpCode::Jump, &[OpRef::int_op(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        let mut trace2 = trace.clone();

        trace2.ops.push(Op::new(
            OpCode::IntSub,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
        ));
        assert_eq!(trace.num_ops(), 2);
        assert_eq!(trace2.num_ops(), 3);
    }

    #[test]
    fn test_trace_only_guards_in_iter_guards() {
        // iter_guards must skip all non-guard ops, even in a complex trace.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            ),
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(2)]),
            Op::new(OpCode::IntMul, &[OpRef::int_op(2), OpRef::int_op(3)]),
            Op::new(OpCode::IntNeg, &[OpRef::int_op(4)]),
            Op::new(OpCode::GuardFalse, &[OpRef::int_op(5)]),
            Op::new(OpCode::IntLt, &[OpRef::int_op(4), OpRef::int_op(5)]),
            Op::new(OpCode::GuardNoException, &[]),
            Op::new(OpCode::Jump, &[OpRef::int_op(4), OpRef::int_op(5)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guard_opcodes: Vec<_> = trace.iter_guards().map(|op| op.opcode).collect();
        assert_eq!(
            guard_opcodes,
            vec![
                OpCode::GuardTrue,
                OpCode::GuardFalse,
                OpCode::GuardNoException
            ]
        );
    }

    #[test]
    fn test_check_consistency_valid() {
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        op0.pos = iop(2);
        let ops = vec![op0, Op::new(OpCode::Jump, &[iop(2)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_no_final() {
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[OpRef::int_op(0), OpRef::int_op(1)],
        )];
        let trace = TreeLoop::new(vec![], ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_undefined_arg() {
        // history.py:579-581: arg not in seen → invalid
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg(0), iop(99)]),
            Op::new(OpCode::Finish, &[]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_none_arg_invalid() {
        // history.py:579-582: regular op args must be Const or known boxes;
        // None is only accepted in fail_args.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg(0), OpRef::NONE]),
            Op::new(OpCode::Finish, &[]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_const_arg_ok() {
        // history.py:580: constants are always valid args
        let inputargs = vec![InputArg::new_int(0)];
        let const_ref = OpRef::const_int(0);
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), const_ref]);
        op0.pos = iop(1);
        let ops = vec![op0, Op::new(OpCode::Finish, &[iop(1)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_ovf_not_followed_by_guard() {
        // history.py:576-578: ovf must be followed by guard_overflow
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut op0 = Op::new(OpCode::IntAddOvf, &[iarg(0), iarg(1)]);
        op0.pos = iop(2);
        let ops = vec![op0, Op::new(OpCode::Finish, &[iop(2)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_ovf_followed_by_guard() {
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut op0 = Op::new(OpCode::IntAddOvf, &[iarg(0), iarg(1)]);
        op0.pos = iop(2);
        let mut guard = Op::new(OpCode::GuardNoOverflow, &[]);
        guard.descr = Some(std::sync::Arc::new(DummyGuardDescr));
        let ops = vec![op0, guard, Op::new(OpCode::Finish, &[iop(2)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_guard_without_descr_invalid() {
        // history.py:583-584: every guard needs a descr when check_descr=True,
        // including overflow guards.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut op0 = Op::new(OpCode::IntAddOvf, &[iarg(0), iarg(1)]);
        op0.pos = iop(2);
        let ops = vec![
            op0,
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Finish, &[iop(2)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_fail_args_const_invalid() {
        // history.py:590: fail_args must not contain constants
        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = Op::new(OpCode::GuardTrue, &[iarg(0)]);
        guard.fail_args = Some(smallvec::smallvec![OpRef::const_int(0)]);
        guard.descr = Some(std::sync::Arc::new(DummyGuardDescr));
        let ops = vec![guard, Op::new(OpCode::Finish, &[])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_fail_args_undefined_invalid() {
        // history.py:591: fail_args entries must be in seen
        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = Op::new(OpCode::GuardTrue, &[iarg(0)]);
        guard.fail_args = Some(smallvec::smallvec![iop(99)]);
        guard.descr = Some(std::sync::Arc::new(DummyGuardDescr));
        let ops = vec![guard, Op::new(OpCode::Finish, &[])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_label_resets_seen() {
        // history.py:596-602: LABEL resets the seen set to its args
        let inputargs = vec![InputArg::new_int(0)];
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), iarg(0)]);
        op0.pos = iop(1);
        // LABEL introduces a fresh scope with iarg(0) only
        let label = Op::new(OpCode::Label, &[iarg(0)]);
        // iop(1) was defined before label, so it's no longer in seen
        let ops = vec![op0, label, Op::new(OpCode::Jump, &[iop(1)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_label_valid() {
        let inputargs = vec![InputArg::new_int(0)];
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), iarg(0)]);
        op0.pos = iop(1);
        let label = Op::new(OpCode::Label, &[iop(1)]);
        let ops = vec![op0, label, Op::new(OpCode::Jump, &[iop(1)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_duplicate_op_position_invalid() {
        let inputargs = vec![InputArg::new_int(0)];
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), iarg(0)]);
        op0.pos = iop(1);
        let mut op1 = Op::new(OpCode::IntSub, &[iop(1), iarg(0)]);
        op1.pos = iop(1);
        let ops = vec![op0, op1, Op::new(OpCode::Finish, &[iop(1)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_jump_descr_must_be_target_token() {
        let inputargs = vec![InputArg::new_int(0)];
        let mut jump = Op::new(OpCode::Jump, &[iarg(0)]);
        jump.descr = Some(std::sync::Arc::new(DummyGuardDescr));
        let trace = TreeLoop::new(inputargs, vec![jump]);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_split_at_label() {
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)]),
            Op::new(OpCode::Label, &[OpRef::int_op(0)]),
            Op::new(OpCode::IntMul, &[OpRef::int_op(0), OpRef::int_op(1)]),
            Op::new(OpCode::Jump, &[OpRef::int_op(0)]),
        ];
        let trace = TreeLoop::new(vec![], ops);
        let (preamble, body) = trace.split_at_label();
        assert_eq!(preamble.len(), 1);
        assert_eq!(body.len(), 3); // Label + IntMul + Jump
    }

    #[test]
    fn test_num_guards() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(0)]),
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::int_op(0)]),
            Op::new(OpCode::GuardClass, &[OpRef::int_op(0), OpRef::int_op(1)]),
            Op::new(OpCode::Finish, &[]),
        ];
        let trace = TreeLoop::new(vec![], ops);
        assert_eq!(trace.num_guards(), 3);
    }

    #[test]
    fn test_get_final_op() {
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)]),
            Op::new(OpCode::Finish, &[OpRef::int_op(0)]),
        ];
        let trace = TreeLoop::new(vec![], ops);
        let final_op = trace.get_final_op().unwrap();
        assert_eq!(final_op.opcode, OpCode::Finish);
    }

    #[test]
    fn test_get_iter() {
        // opencoder.py:848-850 Trace.get_iter() — produce a TraceIterator
        // that walks the trace producing fresh boxes per visited op.
        // The trace must reference its own inputargs at OpRef positions
        // [0, num_inputargs); a malformed trace that references raw 0
        // without a matching inputarg would cache-miss in `_get`.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut add = Op::new(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        add.pos = iop(2);
        let ops = vec![add, Op::new(OpCode::Jump, &[iop(2)])];
        let trace = TreeLoop::new(inputargs, ops);
        let mut iter = trace.get_iter();
        assert!(!iter.done());
        // Walk one op via TraceIterator.next() — opencoder.py:362-406.
        let r = iter.next().unwrap();
        assert_eq!(r.pos, iop(2));
        assert_eq!(r.args[0], iarg(0));
        assert_eq!(r.args[1], iarg(1));
        assert_eq!(iter.pos, 1);
    }

    #[test]
    fn test_inputarg_types_all() {
        let inputargs = vec![
            InputArg {
                index: 0,
                tp: Type::Int,
            },
            InputArg {
                index: 1,
                tp: Type::Ref,
            },
            InputArg {
                index: 2,
                tp: Type::Float,
            },
        ];
        let trace = TreeLoop::new(inputargs, vec![Op::new(OpCode::Finish, &[])]);
        let types = trace.inputarg_types();
        assert_eq!(types, vec![Type::Int, Type::Ref, Type::Float]);
    }

    // ══════════════════════════════════════════════════════════════════
    // cut_trace_from tests — opencoder.py CutTrace parity
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_cut_trace_from_no_escaped_refs() {
        // Simple cut: all post-cut refs are either in original_boxes
        // or defined after the cut.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut ops = Vec::new();
        // Pre-cut ops (2 inputargs → first op is BoxInt at position 2)
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        op0.pos = iop(2);
        ops.push(op0);
        // Post-cut ops
        let mut op1 = Op::new(OpCode::IntMul, &[iarg(0), iarg(1)]);
        op1.pos = iop(3);
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iop(3)]);
        op2.pos = vop(4);
        ops.push(op2);
        let trace = TreeLoop::new(inputargs, ops);

        let start = TreeLoopCutPosition::new(1); // cut after op0
        let original_boxes = vec![iarg(0), iarg(1)];
        let original_box_types = vec![Type::Int, Type::Int];

        let cut = trace.cut_trace_from(start, &original_boxes, &original_box_types);
        assert_eq!(cut.inputargs.len(), 2);
        assert_eq!(cut.ops.len(), 2); // IntMul + Jump
        assert_eq!(cut.ops[0].opcode, OpCode::IntMul);
        assert_eq!(cut.ops[0].args[0], iarg(0)); // remapped from iarg(0)
        assert_eq!(cut.ops[0].args[1], iarg(1)); // remapped from iarg(1)
        assert_eq!(cut.ops[1].opcode, OpCode::Jump);
        assert_eq!(cut.ops[1].args[0], iop(2)); // remapped from iop(3) → new idx 2
    }

    #[test]
    fn test_cut_trace_from_with_escaped_op() {
        // An op defined before the cut point is used after the cut.
        // It should be re-emitted as a prefix operation.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut ops = Vec::new();
        // op0: v2 = int_add(v0, v1) — before cut
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        op0.pos = iop(2);
        ops.push(op0);
        // op1: v3 = int_mul(v2, v0) — after cut, references v2 (escaped!)
        let mut op1 = Op::new(OpCode::IntMul, &[iop(2), iarg(0)]);
        op1.pos = iop(3);
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iop(3)]);
        op2.pos = vop(4);
        ops.push(op2);
        let trace = TreeLoop::new(inputargs, ops);

        let start = TreeLoopCutPosition::new(1); // cut after op0
        // original_boxes only has v0 — v2 is escaped
        let original_boxes = vec![iarg(0)];
        let original_box_types = vec![Type::Int];

        let cut = trace.cut_trace_from(start, &original_boxes, &original_box_types);
        // v1 = OpRef::input_arg_int(1) is an original trace inputarg NOT in original_boxes.
        // It's referenced by the escaped int_add op → added as extra inputarg.
        // Result: inputargs = [v0, v1], prefix = [int_add], post-cut = [int_mul, jump]
        assert_eq!(cut.inputargs.len(), 2); // v0 + escaped v1
        assert_eq!(cut.ops.len(), 3); // prefix(int_add) + int_mul + jump
    }

    #[test]
    fn test_cut_trace_from_constants_preserved() {
        // Tagged constant OpRefs should not be remapped.
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();
        // pre-cut: noop
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), iarg(0)]);
        op0.pos = iop(1);
        ops.push(op0);
        // post-cut: uses a constant
        let const_ref = OpRef::const_int(0);
        let mut op1 = Op::new(OpCode::IntAdd, &[iarg(0), const_ref]);
        op1.pos = iop(2);
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iop(2)]);
        op2.pos = vop(3);
        ops.push(op2);
        let trace = TreeLoop::new(inputargs, ops);

        let start = TreeLoopCutPosition::new(1);
        let original_boxes = vec![iarg(0)];
        let original_box_types = vec![Type::Int];

        let cut = trace.cut_trace_from(start, &original_boxes, &original_box_types);
        assert_eq!(cut.ops.len(), 2);
        // Constant ref should be preserved as-is
        assert_eq!(cut.ops[0].args[1], const_ref);
    }

    #[test]
    fn test_cut_trace_from_transitive_escaped() {
        // Escaped op depends on another escaped op (transitive closure).
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();
        // v1 = int_add(v0, v0) — before cut
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg(0), iarg(0)]);
        op0.pos = iop(1);
        ops.push(op0);
        // v2 = int_mul(v1, v0) — before cut
        let mut op1 = Op::new(OpCode::IntMul, &[iop(1), iarg(0)]);
        op1.pos = iop(2);
        ops.push(op1);
        // v3 = int_sub(v2, v0) — after cut, references v2 (escaped, depends on v1)
        let mut op2 = Op::new(OpCode::IntSub, &[iop(2), iarg(0)]);
        op2.pos = iop(3);
        ops.push(op2);
        let mut op3 = Op::new(OpCode::Jump, &[iop(3)]);
        op3.pos = vop(4);
        ops.push(op3);
        let trace = TreeLoop::new(inputargs, ops);

        let start = TreeLoopCutPosition::new(2); // cut after op0 and op1
        let original_boxes = vec![iarg(0)];
        let original_box_types = vec![Type::Int];

        let cut = trace.cut_trace_from(start, &original_boxes, &original_box_types);
        // 1 inputarg, 2 prefix ops (v1=int_add, v2=int_mul), 2 post-cut ops
        assert_eq!(cut.inputargs.len(), 1);
        assert_eq!(cut.ops.len(), 4);
        assert_eq!(cut.ops[0].opcode, OpCode::IntAdd); // re-emitted v1
        assert_eq!(cut.ops[1].opcode, OpCode::IntMul); // re-emitted v2
        assert_eq!(cut.ops[2].opcode, OpCode::IntSub);
        assert_eq!(cut.ops[3].opcode, OpCode::Jump);
        // Verify remapping chain: v2's arg should reference re-emitted v1
        assert_eq!(cut.ops[1].args[0], iop(1)); // v1 → prefix idx 0 → BoxInt at position 1
    }

    // ══════════════════════════════════════════════════════════════════
    // History / TreeLoop parity tests
    // Local parity coverage for history.py/opencoder.py trace materialization.
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_has_inputargs_ops_structure() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Int);

        assert_eq!(trace.num_ops(), 3);
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_trace_guards_have_fail_args() {
        use crate::recorder::Trace;
        use majit_ir::{DescrRef, FailDescr};
        use std::sync::Arc;

        #[derive(Debug)]
        struct TestFailDescr(u32);
        impl majit_ir::Descr for TestFailDescr {
            fn index(&self) -> u32 {
                self.0
            }
            fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
                Some(self)
            }
        }
        impl FailDescr for TestFailDescr {
            fn fail_index(&self) -> u32 {
                self.0
            }
            fn fail_arg_types(&self) -> &[Type] {
                &[]
            }
        }

        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let cmp = rec.record_op(OpCode::IntLt, &[i0, i1]);
        let descr: DescrRef = Arc::new(TestFailDescr(0));
        rec.record_guard_with_fail_args(OpCode::GuardTrue, &[cmp], Some(descr), &[i0, i1]);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);

        let fail_args = guards[0].fail_args.as_ref().unwrap();
        assert_eq!(fail_args.len(), 2);
        assert_eq!(fail_args[0], i0);
        assert_eq!(fail_args[1], i1);
    }

    #[test]
    fn test_trace_iter_ops() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.record_op(OpCode::IntSub, &[iop(1), i0]);
        rec.close_loop(&[iop(2)]);

        let trace = rec.get_trace();
        let opcodes: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::IntAdd, OpCode::IntSub, OpCode::Jump]);
    }

    #[test]
    fn test_trace_mixed_types() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let r0 = rec.record_input_arg(Type::Ref);
        let f0 = rec.record_input_arg(Type::Float);

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1, r0, f0]);

        let trace = rec.get_trace();
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_pos_matches_opref() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let ref0 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let ref1 = rec.record_op(OpCode::IntMul, &[ref0, i1]);
        let ref2 = rec.record_op(OpCode::IntSub, &[ref1, ref0]);

        rec.close_loop(&[ref2, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].pos, ref0);
        assert_eq!(trace.ops[1].pos, ref1);
        assert_eq!(trace.ops[2].pos, ref2);
    }

    #[test]
    fn test_recorder_get_trace_for_tree_loop() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.num_ops(), 2);
        assert!(trace.is_loop());
    }
}

// ── TraceCtx recording API (History role) ───────────────────────────────
//
// Moved from `trace_ctx.rs` — these are the **History role** of `TraceCtx`,
// mirroring RPython's `history.py` `History` class: operation recording,
// trace position / cut management, call descriptor construction, guard
// emission, and all the typed call-recording convenience wrappers
// (`pyjitpl.py:2455+ self.history.record2(...)` call sites).

use crate::call_descr::{
    EffectInfoSlot, make_call_descr_from_target_slot, make_call_may_force_descr,
};
use crate::constant_pool::ConstantPool;
use crate::jitdriver::JitDriverStaticData;
use crate::recorder::{Trace, TracePosition};
use crate::trace_ctx::TraceCtx;

use majit_backend::JitCellToken;

impl TraceCtx {
    /// history.py: get_trace_position — current recorder position.
    ///
    /// Combines the recorder's 3-tuple (`_pos` / `_count` / `_index`) with
    /// the TraceCtx-owned snapshot side table length so callers see the
    /// full opencoder.py:567-568 5-tuple.
    pub fn get_trace_position(&self) -> TracePosition {
        let mut pos = self.recorder.get_position();
        pos.snapshot_data_len = self.snapshots.len();
        pos
    }

    /// history.py: cut — restore recorder to a saved position.
    ///
    /// Does NOT truncate `self.snapshots` — matches the pre-Task #70
    /// `recorder::Trace::cut` behavior where snapshots grew monotonically
    /// even across rewinds. Downstream code only indexes new snapshot ids
    /// minted after each cut, so stale entries are harmless; truncating
    /// regresses bench (tested under Task #70).
    pub fn cut_trace(&mut self, pos: TracePosition) {
        self.recorder.cut(pos);
    }

    /// pyjitpl.py:3499-3512 `MetaInterp.replace_box(oldbox, newbox)` —
    /// trace-context portion.
    ///
    /// ```text
    ///  def replace_box(self, oldbox, newbox):
    ///      for frame in self.framestack:
    ///          frame.replace_active_box_in_frame(oldbox, newbox)
    ///      boxes = self.virtualref_boxes
    ///      for i in range(len(boxes)):
    ///          if boxes[i] is oldbox:
    ///              boxes[i] = newbox
    ///      if (self.jitdriver_sd.virtualizable_info is not None or
    ///          self.jitdriver_sd.greenfield_info is not None):
    ///          boxes = self.virtualizable_boxes
    ///          for i in range(len(boxes)):
    ///              if boxes[i] is oldbox:
    ///                  boxes[i] = newbox
    ///      self.heapcache.replace_box(oldbox, newbox)
    /// ```
    ///
    /// pyre splits `MetaInterp.replace_box` across two layers:
    ///
    ///   * `TraceCtx::replace_box` (this method) handles the
    ///     `virtualizable_boxes` + `heap_cache` walks. This is what
    ///     `is_nonstandard_virtualizable` Step 4 calls.
    ///
    ///   * `MetaInterp::replace_box` (in pyjitpl.rs) is the structural
    ///     mirror of the full RPython entry point; it adds the
    ///     `virtualref_boxes` walk and the framestack walk on top of
    ///     this `TraceCtx::replace_box`.
    pub fn replace_box(&mut self, oldbox: OpRef, newbox: OpRef) {
        // pyjitpl.py:3506-3511 virtualizable_boxes walk.
        if let Some(boxes) = self.virtualizable_boxes.as_mut() {
            for slot in boxes.iter_mut() {
                if *slot == oldbox {
                    *slot = newbox;
                }
            }
        }
        // pyjitpl.py:3512 self.heapcache.replace_box(oldbox, newbox).
        self.heap_cache.replace_box(oldbox, newbox);
    }

    /// Record a regular IR operation.
    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        Self::do_record_op(&mut self.recorder, &self.constants, opcode, args)
    }

    /// Record an operation with a descriptor (e.g., calls).
    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        Self::do_record_op_with_descr(&mut self.recorder, &self.constants, opcode, args, descr)
    }

    /// Record a guard with auto-generated FailDescr.
    ///
    /// `num_live` is the number of live integer values (for the FailDescr).
    /// opencoder.py:819 parity: capture a snapshot of the interpreter
    /// frame state. Returns a snapshot_id for use as rd_resume_position.
    pub fn capture_resumedata(&mut self, snapshot: crate::recorder::Snapshot) -> i32 {
        let id = self.snapshots.len() as i32;
        self.snapshots.push(snapshot);
        id
    }

    /// Look up a captured snapshot by id.
    pub fn get_snapshot(&self, id: i32) -> Option<&crate::recorder::Snapshot> {
        if id >= 0 {
            self.snapshots.get(id as usize)
        } else {
            None
        }
    }

    /// Set rd_resume_position on the last recorded guard.
    pub fn set_last_guard_resume_position(&mut self, snapshot_id: i32) {
        self.recorder.set_last_op_resume_position(snapshot_id);
    }

    /// PRE-EXISTING-ADAPTATION: low-level / single-frame snapshot helper
    /// used by callers that record guards without a populated framestack
    /// to walk.
    ///
    /// RPython's `pyjitpl.py:2586 capture_resumedata` walks the
    /// `framestack`, encoding one `SnapshotFrame` per `MIFrame` (with
    /// the real `jitcode_index`, `pc`, plus `virtualizable_boxes` and
    /// `virtualref_boxes` when configured).  Pyre's segmented low-level
    /// driver (`jitdriver.rs::force_finish_trace`), the standalone
    /// walker (`jitcode_dispatch.rs::record_guard_with_current_snapshot`),
    /// and the recorder-level unit tests in `jitdriver.rs::tests` /
    /// `pyjitpl::tests` don't have a populated MIFrame at the guard
    /// point — they only know the live `OpRef` set.  Caller supplies the
    /// `jitcode_index` and `pc` of the frame the guard belongs to so
    /// downstream layout matching (`jit_state.rs::*` keys on these
    /// fields) sees real coordinates rather than the previous
    /// `0/0` placeholder.  `vable_boxes` and `vref_boxes` are empty:
    /// callers using this path don't manage virtualizables or virtual
    /// refs.
    ///
    /// Convergence (Task #89): once `S::Sym` is lifted into
    /// `MIFrame::populate_for_guard`, both call sites can route through
    /// the standard `capture_resumedata(snapshot)` flow built from the
    /// live framestack and this helper dissolves.
    ///
    /// Strict-types parity (`history.py:802`): every `OpRef` must
    /// resolve to a known `Box.type`; constants must have a recorded
    /// value.  Misses are bookkeeping bugs and panic, not silent
    /// fallbacks.
    pub fn capture_snapshot_for_last_guard(
        &mut self,
        active_boxes: &[OpRef],
        jitcode_index: u32,
        pc: u32,
    ) {
        let boxes: Vec<crate::recorder::SnapshotTagged> = active_boxes
            .iter()
            .map(|opref| {
                let tp = self
                    .get_opref_type(*opref)
                    .expect("capture_snapshot_for_last_guard: active OpRef missing Box.type");
                if opref.is_constant() {
                    let value = self.constant_value(*opref).expect(
                        "capture_snapshot_for_last_guard: constant OpRef missing recorded value",
                    );
                    crate::recorder::SnapshotTagged::Const(value, tp)
                } else {
                    crate::recorder::SnapshotTagged::Box(*opref, tp)
                }
            })
            .collect();
        let snapshot_id = self.capture_resumedata(crate::recorder::Snapshot {
            frames: vec![crate::recorder::SnapshotFrame {
                jitcode_index,
                pc,
                boxes,
            }],
            vable_boxes: Vec::new(),
            vref_boxes: Vec::new(),
        });
        self.set_last_guard_resume_position(snapshot_id);
    }

    /// Mutate `op.fail_args` on a recorded op identified by `opref`.
    ///
    /// 1:1 port of RPython's `Op.setfailargs([...])`
    /// (`resoperation.py`).  Production guard recording goes through
    /// the snapshot path (`record_guard_typed` + `capture_resumedata`
    /// + `set_last_guard_resume_position`); the optimizer's
    /// `store_final_boxes_in_guard` (`optimizeopt/mod.rs:3200`) then
    /// derives `op.fail_args` from the snapshot via
    /// `op.store_final_boxes(liveboxes)` (`mod.rs:3392`).  This setter
    /// is for tests and other callers that construct synthetic guard
    /// shapes outside the standard `capture_resumedata` flow —
    /// matching how RPython's `test_resume.py` /
    /// `test_optimizeopt.py` build `ResOperation`s with `setfailargs`
    /// directly rather than through `history.record_default_val`.
    pub fn set_fail_args(&mut self, opref: OpRef, fail_args: &[OpRef]) {
        self.recorder.set_op_fail_args(opref, fail_args);
    }

    /// Look up a constant value by its OpRef (>= 10_000).
    pub fn constant_value(&self, opref: OpRef) -> Option<i64> {
        self.constants.as_ref().get(&opref.raw()).copied()
    }

    /// `pyjitpl.py:2548 generate_guard()` parity: tracer-stage guards
    /// carry `descr=None`. The optimizer's `store_final_boxes_in_guard`
    /// (mod.rs:3417 with codex #1 fix) mints the descr via
    /// `invent_fail_descr_for_op`-style dispatch. `num_live` was a
    /// placeholder used by the prior `make_resume_guard_descr(num_live)`
    /// stamping — kept on the signature for caller compatibility but
    /// no longer used.
    pub fn record_guard(&mut self, opcode: OpCode, args: &[OpRef], num_live: usize) -> OpRef {
        let _ = num_live;
        Self::do_record_guard(&mut self.recorder, &self.constants, opcode, args, None)
    }

    /// `pyjitpl.py:2548 generate_guard()` parity: tracer-stage typed
    /// guards carry `descr=None`. The `fail_arg_types` are stamped onto
    /// `op.fail_arg_types` directly so the optimizer's
    /// `store_final_boxes_in_guard` (mod.rs:3433) can mint a fresh
    /// `ResumeGuardDescr` carrying those types via the
    /// `op.descr.is_none()` branch (RPython
    /// `invent_fail_descr_for_op`-style fallthrough,
    /// compile.py:938-941).
    ///
    /// Like `record_guard_typed` predecessors, this records **no**
    /// inline `op.fail_args` — the caller attaches a snapshot via
    /// `capture_resumedata` + `set_last_guard_resume_position`; the
    /// snapshot's frame boxes feed the eventual `liveboxes` written
    /// into `op.fail_args` by `op.store_final_boxes(liveboxes)`
    /// (`pyjitpl.py:2558-2602`).
    pub fn record_guard_typed(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        fail_arg_types: Vec<Type>,
    ) -> OpRef {
        let opref = Self::do_record_guard(&mut self.recorder, &self.constants, opcode, args, None);
        self.recorder.set_last_op_fail_arg_types(fail_arg_types);
        opref
    }

    // ── Step 2e.2a: split-borrow helpers ──────────────────────────────
    //
    // Private `do_*` helpers take `(&mut Trace, &ConstantPool, ...)` so the
    // caller performs an explicit two-field borrow of `self.recorder` and
    // `self.constants`. The `_constants` parameter is currently unused —
    // `recorder::Trace` carries raw `OpRef` values and needs no constant
    // resolution at record time — but the signature shape matches
    // `TraceRecordBuffer::record_op_oprefs` / `record_guard_oprefs` /
    // `close_loop_oprefs` / `finish_oprefs` (opencoder.rs:2420-2551), all
    // of which consume `&ConstantPool` to resolve tagged-constant `OpRef`
    // into wire bytes.
    //
    // Step 2e.2b swaps the `recorder` field type from `Trace` to
    // `TraceRecordBuffer`. Contrary to an earlier note here, this is NOT
    // a simple helper-body replacement. TRB returns RPython-orthodox
    // `_index`-based positions (box-yielding count, opencoder.py:664-670
    // `record_op` returns `pos = self._index`), while
    // `recorder::Trace::record_op` (recorder.rs:159-169) returns
    // `OpRef::from_raw(op_count)` — every op (void or not) gets a unique index.
    // TRB's `_untag` (opencoder.rs:717-770) resolves `TAGBOX(v)` via
    // `_cache[v]`, and `_cache` is indexed by `_index`, so callers that
    // store an OpRef and later pass it as an arg must have stored an
    // `_index`-based value. Across pyre, `op.pos.raw()` is used as a HashMap
    // key (compile.rs, blackhole.rs, optimizeopt/*, pyjitpl/mod.rs) under
    // the pyre-legacy "all ops unique" invariant; a straight swap would
    // corrupt those maps. The swap therefore has to land together with
    // caller-side OpRef convention migration.
    //
    // See `step2e_traceposition_parity_2026_04_22` memory +
    // `rpython-trace-jitcode-hidden-candle.md` plan (Step 3–5) for the
    // multi-session route.

    pub(crate) fn do_record_op(
        recorder: &mut Trace,
        _constants: &ConstantPool,
        opcode: OpCode,
        args: &[OpRef],
    ) -> OpRef {
        recorder.record_op(opcode, args)
    }

    pub(crate) fn do_record_op_with_descr(
        recorder: &mut Trace,
        _constants: &ConstantPool,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        recorder.record_op_with_descr(opcode, args, descr)
    }

    pub(crate) fn do_record_guard(
        recorder: &mut Trace,
        _constants: &ConstantPool,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
    ) -> OpRef {
        recorder.record_guard(opcode, args, descr)
    }

    pub(crate) fn do_close_loop(
        recorder: &mut Trace,
        _constants: &ConstantPool,
        jump_args: &[OpRef],
    ) {
        recorder.close_loop(jump_args);
    }

    pub(crate) fn do_close_loop_with_descr(
        recorder: &mut Trace,
        _constants: &ConstantPool,
        jump_args: &[OpRef],
        descr: Option<DescrRef>,
    ) {
        recorder.close_loop_with_descr(jump_args, descr);
    }

    pub(crate) fn do_finish(
        recorder: &mut Trace,
        _constants: &ConstantPool,
        finish_args: &[OpRef],
        descr: DescrRef,
    ) {
        recorder.finish(finish_args, descr);
    }

    // ── Step 2e.2b.glue: public TraceCtx wrappers over self.recorder ──
    //
    // These methods centralize every `ctx.recorder.X()` external call
    // pattern, so the eventual TRB swap can be threaded through the
    // `do_*` helpers above. TRB already has matching byte-stream entry
    // points (`record_op_oprefs` / `close_loop_oprefs` / `finish_oprefs`
    // at opencoder.rs:2536-2642), but the `TreeLoop`-shaped result
    // produced by `recorder::Trace::get_trace()` has no RPython analogue
    // — upstream (opencoder.py:848) exposes `get_iter()` and the
    // optimizer walks the iterator directly, with no intermediate
    // `Vec<Op>` materialization. Step 2e.2b must either port pyre's
    // consumers onto an iterator-walk shape or introduce a documented
    // pyre-ADAPTATION materializer (`TRB -> TreeLoop`) with a comment
    // pointing at the specific RPython call that it stands in for.

    /// pyjitpl.py:3188-3190 `history.record1(rop.JUMP, ..., descr=ptoken)` —
    /// close the loop with an implicit no-descr JUMP.
    pub fn close_loop(&mut self, jump_args: &[OpRef]) {
        Self::do_close_loop(&mut self.recorder, &self.constants, jump_args);
    }

    /// pyjitpl.py:3188-3190 close-loop variant with an explicit JUMP
    /// descriptor (tentative target token recorded before compile_trace).
    pub fn close_loop_with_descr(&mut self, jump_args: &[OpRef], descr: Option<DescrRef>) {
        Self::do_close_loop_with_descr(&mut self.recorder, &self.constants, jump_args, descr);
    }

    /// pyjitpl.py:1637 `history.record1(rop.FINISH, ..., descr=token)` —
    /// finalize a non-looping trace with explicit FailDescr.
    pub fn finish(&mut self, finish_args: &[OpRef], descr: DescrRef) {
        Self::do_finish(&mut self.recorder, &self.constants, finish_args, descr);
    }

    /// Consume the TraceCtx and return the completed `TreeLoop`.
    ///
    /// Pyre analog of RPython's `MetaInterp.history.trace` access — after
    /// tracing ends, downstream callers (optimizer, bridge export) see
    /// the loop as `TreeLoop { inputargs, ops, snapshots }`. Snapshots
    /// come from the TraceCtx-owned side table (Task #70 moved them off
    /// `recorder::Trace`); the recorder contributes only inputargs + ops.
    pub fn into_tree_loop(self) -> crate::history::TreeLoop {
        // H-3.0a: forward the recorder's BoxRef pool so the optimizer
        // sees the same `Rc<Box>` allocations created during tracing —
        // RPython parity for `AbstractValue` object identity.
        let (inputargs, ops, box_pool) = self.recorder.into_parts();
        crate::history::TreeLoop::with_box_pool(inputargs, ops, self.snapshots, box_pool)
    }

    /// Snapshot slice accessor — Pyre-level parity with
    /// `MetaInterp.history.trace.snapshots()`.
    pub fn snapshots(&self) -> &[crate::recorder::Snapshot] {
        &self.snapshots
    }

    /// Op slice accessor — returns the raw recorded operations. After
    /// Step 2e.2b this materializes via `ByteTraceIter::next` walking
    /// the byte stream.
    pub fn ops(&self) -> &[Op] {
        self.recorder.ops()
    }

    /// `num_inputargs()` — alias for `num_inputs()` keeping RPython
    /// `Trace.num_inputargs` name parity in external call sites.
    pub fn num_inputargs(&self) -> usize {
        self.recorder.num_inputargs()
    }

    fn infer_arg_types(&self, args: &[OpRef]) -> Vec<Type> {
        args.iter()
            .map(|&arg| self.get_opref_type(arg).unwrap_or(Type::Int))
            .collect()
    }

    /// Record a void-returning function call (CallN).
    ///
    /// Automatically registers the function pointer as a constant and
    /// creates a CallDescr. The interpreter doesn't need to manage
    /// function pointer constants or CallDescr implementations.
    pub fn call_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types = self.infer_arg_types(args);
        self.call_void_typed(func_ptr, args, &arg_types);
    }

    /// Record an integer-returning function call (CallI).
    ///
    /// Same convenience as `call_void` but returns an OpRef for the result.
    pub fn call_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a FINISH op with a single result value.
    /// pyjitpl.py:1637 history.record1(rop.FINISH, ..., descr=token)
    pub fn record_finish(&mut self, result: OpRef, _tp: Type) {
        Self::do_record_op(
            &mut self.recorder,
            &self.constants,
            OpCode::Finish,
            &[result],
        );
    }

    /// pyjitpl.py:2789-2791 `blackhole_if_trace_too_long` check:
    /// `length > warmrunnerstate.trace_limit`.  `num_ops` is the non-inputarg
    /// op count (= `history.length()`); `trace_limit` is cached from warmstate
    /// at trace start.
    pub fn is_too_long(&self) -> bool {
        self.recorder.num_ops() > self.trace_limit
    }

    /// Current cached trace limit snapshot (for diagnostics + force_finish
    /// segmenting heuristic).
    pub fn trace_limit(&self) -> usize {
        self.trace_limit
    }

    /// Called by `setup_tracing` to snapshot `warmstate.trace_limit` onto
    /// this per-trace context.
    pub fn set_trace_limit(&mut self, limit: usize) {
        self.trace_limit = limit;
    }

    /// pyjitpl.py:1618 force_finish_trace flag.
    pub fn force_finish_trace(&self) -> bool {
        self.force_finish
    }

    /// pyjitpl.py:2898 `metainterp.resumekey_original_loop_token` accessor.
    /// Returns `Some` when this is a bridge trace, `None` for a loop-entry
    /// trace.  Read by `prepare_trace_segmenting` (pyjitpl.py:2825-2834) to
    /// decide whether to set `FORCE_BRIDGE_SEGMENTING` on the source token.
    pub fn resumekey_original_loop_token(&self) -> Option<&std::sync::Arc<JitCellToken>> {
        self.resumekey_original_loop_token.as_ref()
    }

    /// Stash the source loop token at bridge-tracing entry
    /// (`start_retrace_from_guard`) so the segmenting setter can find it
    /// later.
    pub fn set_resumekey_original_loop_token(&mut self, token: std::sync::Arc<JitCellToken>) {
        self.resumekey_original_loop_token = Some(token);
    }

    /// Set force_finish_trace flag.
    pub fn set_force_finish(&mut self, val: bool) {
        self.force_finish = val;
    }

    /// Get the result type of an OpRef from the recorded trace.
    ///
    /// resoperation.py:567 / history.py:220 Box.type parity: every typed
    /// Box (`InputArg{Int,Ref,Float}`, `IntOp`/`RefOp`/`FloatOp`,
    /// `Const{Int,Ref,Float}`) carries `box.type` intrinsically on the
    /// object itself. pyre encodes that on the typed `OpRef` variant via
    /// `opref.ty()`, so the variant tag IS the authoritative answer when
    /// it is present. Trust it first; the inputarg / constant_pool /
    /// recorded-op fallbacks remain for the transitional `Untyped`
    /// variant produced by legacy `OpRef::from_raw` / `OpRef::from_const`
    /// callers (closing them is tracked under the typed-OpRef migration).
    ///
    /// Box.type is always one of `'i'` / `'r'` / `'f'`. Void is NOT a
    /// valid Box type — only value-producing ops have Boxes. pyre's
    /// recorder assigns `pos` to every op (including void ops like
    /// `SetfieldGc` and guards), so a stale lookup of a void op's pos
    /// would otherwise return `Type::Void`; filter that out and return
    /// `None` so callers fall back to a safe default rather than letting
    /// Void leak into `livebox_types` / `fail_arg_types`.
    pub fn get_opref_type(&self, opref: OpRef) -> Option<Type> {
        // resoperation.py:29 / history.py:220: typed Box's `.type` is
        // intrinsic. Trust the variant tag before consulting any side
        // table.
        if let Some(tp) = opref.ty() {
            return (tp != Type::Void).then_some(tp);
        }
        if (opref.raw() as usize) < self.recorder.num_inputargs() {
            return Some(self.recorder.inputarg_types()[opref.raw() as usize]);
        }
        // ConstantPool: check constant_type first, then numbering
        // type overrides (mark_type for resume-data-only Ref constants).
        if opref.is_constant() {
            if let Some(tp) = self.constants.constant_type(opref) {
                return Some(tp);
            }
            if let Some(&tp) = self.constants.numbering_type_overrides().get(&opref.raw()) {
                return Some(tp);
            }
        }
        // Untyped-OpRef fallback: `opref.ty()` returned None above, so a
        // variant-aware `get_op_by_pos` would never match a typed
        // `op.pos`. Look the op up by raw position only — once the
        // Untyped variant retirement (#171) completes, the entire
        // fallback chain disappears together with this branch.
        self.recorder
            .get_op_by_raw_pos(opref.raw())
            .map(|op| op.result_type())
            .filter(|tp| *tp != Type::Void)
    }

    /// The green key hash (loop header PC) for this trace.
    pub fn green_key(&self) -> u64 {
        self.green_key
    }

    /// `staticdata.profiler` accessor — RPython parity for
    /// `self.metainterp.staticdata.profiler` (pyjitpl.py:2581 etc.).
    ///
    /// Cross-crate tracers (`pyre-jit-trace`) reach the shared atomic
    /// counter sink through this method instead of holding the Arc
    /// directly; the borrow shape stays `&self` because every counter
    /// op on [`crate::jitprof::JitProfiler`] is an `AtomicUsize`
    /// fetch_add.
    pub fn profiler(&self) -> &crate::jitprof::JitProfiler {
        &self.metainterp_sd.profiler
    }

    /// Root portal merge-point green key for this trace.
    ///
    /// Mirrors RPython's `current_merge_points[0]`: this stays anchored to the
    /// original loop/portal merge point even if `green_key` is later retargeted

    /// Record a promote: emit GuardValue to specialize on a runtime value.
    ///
    /// In RPython this is `jit.promote(x)` — it records a `GUARD_VALUE`
    /// that asserts the runtime value equals the constant captured during
    /// tracing. After the guard, the optimizer treats the value as constant.
    ///
    /// `opref` is the traced value, `runtime_value` is the current concrete
    /// value seen at trace time.
    pub fn promote_int(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_int(runtime_value);
        self.record_guard(OpCode::GuardValue, &[opref, const_ref], num_live);
        const_ref
    }

    /// Record a ref-typed promote (GUARD_VALUE for GC references).
    pub fn promote_ref(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_ref(runtime_value);
        self.record_guard(OpCode::GuardValue, &[opref, const_ref], num_live);
        const_ref
    }

    /// Record a float-typed promote (GUARD_VALUE for floats).
    ///
    /// pyjitpl.py:1515 opimpl_float_guard_value = _opimpl_guard_value
    pub fn promote_float(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_float(runtime_value);
        self.record_guard(OpCode::GuardValue, &[opref, const_ref], num_live);
        const_ref
    }

    /// Record a call to an elidable (pure) function.
    ///
    /// In RPython, `@jit.elidable` marks a function whose result depends
    /// only on its arguments and has no side effects. The optimizer can
    /// constant-fold calls where all args are constants, or CSE identical calls.
    ///
    /// This records a CALL_PURE_I (or CALL_PURE_R/CALL_PURE_N) which the
    /// optimizer's pure pass can eliminate.
    pub fn call_elidable_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_elidable_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a void-returning call to a may-force function (e.g., one that
    /// may trigger GC or exceptions).
    ///
    /// In RPython this is `call_may_force` — a call that may force virtualizable
    /// frames or raise exceptions. Must be followed by `GUARD_NOT_FORCED`.
    pub fn call_may_force_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_may_force_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning call to a may-force function.
    pub fn call_may_force_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_may_force_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a void-returning call to a may-force function.
    pub fn call_may_force_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types = self.infer_arg_types(args);
        self.call_may_force_void_typed(func_ptr, args, &arg_types);
    }

    /// Record a call with GIL release (for C extensions / external libs).
    ///
    /// In RPython this is `call_release_gil`. The GIL is released before the
    /// call and reacquired after. Used for long-running C functions.
    pub fn call_release_gil_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_release_gil_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a call to a loop-invariant function.
    ///
    /// The result is cached for the duration of one loop iteration.
    /// In RPython, `@jit.loop_invariant` marks such functions.
    pub fn call_loopinvariant_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_loopinvariant_int_typed(func_ptr, args, &arg_types)
    }

    /// Record GUARD_NOT_FORCED (must follow a call_may_force).
    pub fn guard_not_forced(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotForced, &[], num_live)
    }

    // ── CALL_MAY_FORCE with virtualizable synchronization ─────────

    fn call_may_force_with_jitstate_sync_impl<S, R>(
        &mut self,
        state: &S,
        num_live: usize,
        record_call: impl FnOnce(&mut Self) -> R,
    ) -> (R, crate::jit_state::ResidualVirtualizableSync)
    where
        S: crate::jit_state::JitState,
    {
        state.sync_virtualizable_before_residual_call(self);
        let result = record_call(self);
        let sync = state.sync_virtualizable_after_residual_call(self);
        if !sync.forced {
            self.guard_not_forced(num_live);
        }
        (result, sync)
    }

    /// Callback-based virtualizable sync for CALL_MAY_FORCE.
    ///
    /// Uses JitState's `sync_virtualizable_before/after_residual_call`
    /// methods to emit the appropriate SETFIELD/GETFIELD ops. This is
    /// the preferred API for interpreters that implement the JitState
    /// virtualizable sync hooks.
    ///
    /// Returns `(call_result, sync)` where `sync` reports any updated field
    /// OpRefs and whether the residual call forced the standard virtualizable.
    pub fn call_may_force_with_jitstate_sync_int<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> (OpRef, crate::jit_state::ResidualVirtualizableSync) {
        self.call_may_force_with_jitstate_sync_impl(state, num_live, |ctx| {
            ctx.call_may_force_int_typed(func_ptr, args, arg_types)
        })
    }

    /// Ref-returning variant of [`call_may_force_with_jitstate_sync_int`].
    pub fn call_may_force_with_jitstate_sync_ref<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> (OpRef, crate::jit_state::ResidualVirtualizableSync) {
        self.call_may_force_with_jitstate_sync_impl(state, num_live, |ctx| {
            ctx.call_may_force_ref_typed(func_ptr, args, arg_types)
        })
    }

    /// Float-returning variant of [`call_may_force_with_jitstate_sync_int`].
    pub fn call_may_force_with_jitstate_sync_float<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> (OpRef, crate::jit_state::ResidualVirtualizableSync) {
        self.call_may_force_with_jitstate_sync_impl(state, num_live, |ctx| {
            ctx.call_may_force_float_typed(func_ptr, args, arg_types)
        })
    }

    /// Void-returning variant of [`call_may_force_with_jitstate_sync_int`].
    pub fn call_may_force_with_jitstate_sync_void<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> crate::jit_state::ResidualVirtualizableSync {
        let (_, sync) = self.call_may_force_with_jitstate_sync_impl(state, num_live, |ctx| {
            ctx.call_may_force_void_typed(func_ptr, args, arg_types)
        });
        sync
    }

    /// Record GUARD_NO_EXCEPTION (check no pending exception).
    pub fn guard_no_exception(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNoException, &[], num_live)
    }

    /// Record GUARD_NOT_INVALIDATED (check loop not invalidated).
    pub fn guard_not_invalidated(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotInvalidated, &[], num_live)
    }

    // ── Generic typed call ──────────────────────────────────────────

    /// Record a function call with explicit argument and return types.
    ///
    /// `opcode` selects the call family (CallI/R/F/N, CallPureI/R/F/N, etc.).
    /// Synthesizes the per-opcode default `EffectInfo`
    /// (`call_descr::default_effect_for_opcode`).
    ///
    /// **Prefer `call_typed_with_effect`** for line-by-line PyPy parity:
    /// `pyjitpl.py:1995-2068 do_residual_call` threads the codewriter-
    /// analyzed `calldescr` through `record_nospec` so the trace IR
    /// retains `oopspecindex`, `read/write_descrs_*`, `can_invalidate`,
    /// `can_collect`, and `call_release_gil_target` exactly as written.
    /// This helper is the no-EI shortcut for callers that genuinely
    /// have no per-callee analysis available — equivalent to PyPy's
    /// `effectinfo.MOST_GENERAL` fallback for unanalyzed callees.
    /// The codewriter's `CallControl::getcalldescr`
    /// (`majit-translate/src/jit_codewriter/call.rs`) does port
    /// call.py:210-335 in full (raise / random-effects / write /
    /// collect / virtualizable / quasi-immut analyzers); the remaining
    /// gap is plumbing the per-callsite EI it produces back to runtime
    /// trace recording — Task #64 (analyzer-rollout) is that plumbing
    /// work, not a missing analyzer.
    pub fn call_typed(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = crate::call_descr::make_call_descr_for_opcode(opcode, arg_types, ret_type);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2659 `_record_helper_varargs` parity: route every
        // CALL family record through `heapcache.invalidate_caches_varargs`
        // so the elidable / loopinvariant / arraycopy / arraymove
        // fast-paths inside `clear_caches_varargs` (heapcache.py:341-376)
        // run exactly once per call. The previous escape-only path
        // (`_escape_argboxes + invalidate_caches_for_escaped`) skipped
        // those branches.
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        result
    }

    pub fn call_void_typed(&mut self, func_ptr: *const (), args: &[OpRef], arg_types: &[Type]) {
        let _ = self.call_typed(OpCode::CallN, func_ptr, args, arg_types, Type::Void);
    }

    /// `call_typed` variant that preserves the caller-supplied `EffectInfo`
    /// instead of re-deriving the default for the opcode. Mirrors
    /// `pyjitpl.py:1995-2068 do_residual_call` parity: PyPy passes the
    /// original `calldescr` through `record_nospec` so the trace IR
    /// retains `oopspec`, `read/write_descrs_*`, `can_invalidate`,
    /// `can_collect`, and `call_release_gil_target` exactly as written
    /// by the codewriter / write-analyzer.
    pub fn call_typed_with_effect(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2659 `_record_helper_varargs` parity (see
        // `call_typed` for the full rationale).
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        result
    }

    pub fn call_void_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let _ = self.call_typed_with_effect(
            OpCode::CallN,
            func_ptr,
            args,
            arg_types,
            Type::Void,
            effect_info,
        );
    }

    /// Pure-call analog of [`call_typed_with_effect`] that mirrors
    /// `pyjitpl.py:1941-1958 MIFrame.execute_varargs(opnum, argboxes,
    /// descr, exc=False, pure=True)` for `EF_ELIDABLE_CANNOT_RAISE`
    /// callees: records the initial `Call{I,R,F,N}` op, then patches
    /// it via [`record_result_of_call_pure`] so the trace ends up with
    /// `CallPure*` (or a `Const` when all args fold) AND the
    /// `call_pure_results` cache is populated for cross-trace
    /// constant folding by the optimizer's pure pass
    /// (`pyjitpl.py:2397 + compile.py:221 take_call_pure_results`).
    ///
    /// `concrete_arg_values` must be parallel to `args` and start with
    /// the funcbox's concrete value (i.e., one entry for the funcbox
    /// followed by one per real arg) — same shape as
    /// `_build_allboxes(funcbox, argboxes, descr)` in
    /// `pyjitpl.py:1960-1993`. `concrete_result` is the value returned
    /// by executing the helper with the concrete operand values; the
    /// caller is responsible for invoking the helper (the runtime
    /// tracer already has the concrete operands available before the
    /// recorded trace runs).
    ///
    /// Caller must guarantee the EI is `EF_ELIDABLE_CANNOT_RAISE`
    /// (`check_can_raise()` false, `check_is_elidable()` true) — this
    /// helper does NOT emit `GuardNoException`.  For elidable-can-raise
    /// callees the caller must thread the call through the
    /// (yet-unwritten) variant that handles `handle_possible_exception`.
    pub fn call_typed_with_effect_pure(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
        concrete_arg_values: &[Value],
        concrete_result: Value,
    ) -> OpRef {
        debug_assert!(
            effect_info.check_is_elidable() && !effect_info.check_can_raise(false),
            "call_typed_with_effect_pure requires EF_ELIDABLE_CANNOT_RAISE"
        );
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(func_ref);
        call_args.extend_from_slice(args);
        debug_assert_eq!(
            call_args.len(),
            concrete_arg_values.len(),
            "concrete_arg_values must include the funcbox concrete value as the first entry"
        );
        // pyjitpl.py:1943: patch_pos = self.metainterp.history.get_trace_position()
        let patch_pos = self.get_trace_position();
        // pyjitpl.py:1944-1945: op = execute_and_record_varargs(opnum, ...)
        let op = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2659 _record_helper_varargs heap invalidation parity.
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        // pyjitpl.py:1947-1948: record_result_of_call_pure patches CALL → CALL_PURE
        // and populates call_pure_results.
        self.record_result_of_call_pure(
            op,
            &call_args,
            concrete_arg_values,
            descr,
            patch_pos,
            opcode,
            concrete_result,
        )
    }

    /// pyjitpl.py:3553-3579: record_result_of_call_pure.
    ///
    /// Patch a CALL into a CALL_PURE. Called after a pure call executes
    /// during tracing with no exception.
    ///
    /// `concrete_arg_values` contains the execution-time values for ALL
    /// args (pyjitpl.py:3572 `[executor.constant_from_op(a) for a in
    /// normargboxes]`). Used as the full cache key.
    pub fn record_result_of_call_pure(
        &mut self,
        op: OpRef,
        argboxes: &[OpRef],
        concrete_arg_values: &[Value],
        descr: DescrRef,
        patch_pos: TracePosition,
        opcode: OpCode,
        result_value: Value,
    ) -> OpRef {
        let resbox_as_const = result_value;
        // pyjitpl.py:3557-3561: COND_CALL_VALUE ignores the 'value' arg
        let is_cond_value = opcode.is_cond_call_value();
        let norm_start = if is_cond_value { 1 } else { 0 };
        let normargboxes = &argboxes[norm_start..];
        let norm_values = &concrete_arg_values[norm_start..];
        // pyjitpl.py:3562-3565: check if all args are Const
        let all_const = normargboxes
            .iter()
            .all(|arg| self.constants.get_value(*arg).is_some());
        if all_const {
            // pyjitpl.py:3566-3569: all-constants → cut the CALL
            self.recorder.cut(patch_pos);
            let const_opref = match resbox_as_const {
                Value::Int(v) => self.constants.get_or_insert(v),
                Value::Float(v) => self
                    .constants
                    .get_or_insert_typed(v.to_bits() as i64, Type::Float),
                Value::Ref(r) => self
                    .constants
                    .get_or_insert_typed(r.as_usize() as i64, Type::Ref),
                Value::Void => self.constants.get_or_insert(0),
            };
            return const_opref;
        }
        // pyjitpl.py:3572-3573: constant_from_op(a) for ALL args
        let arg_consts: Vec<Value> = norm_values.to_vec();
        self.call_pure_results.insert(arg_consts, resbox_as_const);
        // pyjitpl.py:3574-3575: COND_CALL_VALUE remains as-is
        if is_cond_value {
            return op;
        }
        // pyjitpl.py:3576-3579: cut CALL, re-record as CALL_PURE
        let ret_type = match resbox_as_const {
            Value::Int(_) => Type::Int,
            Value::Ref(_) => Type::Ref,
            Value::Float(_) => Type::Float,
            Value::Void => Type::Void,
        };
        let pure_opcode = OpCode::call_pure_for_type(ret_type);
        self.recorder.cut(patch_pos);
        self.recorder
            .record_op_with_descr(pure_opcode, argboxes, descr)
    }

    /// pyjitpl.py:2397 + compile.py:221: take call_pure_results for
    /// passing to the optimizer.
    pub fn take_call_pure_results(&mut self) -> std::collections::HashMap<Vec<Value>, Value> {
        std::mem::take(&mut self.call_pure_results)
    }

    // ── conditional_call / record_known_result (jtransform.py:1665, 292) ──

    /// RPython pyjitpl.py opimpl_conditional_call_ir_v: emit CondCallN.
    ///
    /// `slot` carries the per-callee `EffectInfo` classification produced
    /// by the macro-time analyzer-equivalent at
    /// `pyre-jit/src/jit/codewriter.rs::register_helper_fn_pointers`,
    /// mirroring `call.py:282-303 getcalldescr`'s analyzer chain output.
    pub fn cond_call_void_typed(
        &mut self,
        condition: i64,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        slot: EffectInfoSlot,
    ) {
        let cond_ref = self.constants.get_or_insert(condition);
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr_from_target_slot(arg_types, Type::Void, slot);
        let mut call_args = vec![cond_ref, func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallN, &call_args, descr);
    }

    /// RPython pyjitpl.py opimpl_conditional_call_value_ir_i: emit CondCallValueI.
    pub fn cond_call_value_int_typed(
        &mut self,
        value: i64,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        slot: EffectInfoSlot,
    ) -> OpRef {
        let value_ref = self.constants.get_or_insert(value);
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr_from_target_slot(arg_types, Type::Int, slot);
        let mut call_args = vec![value_ref, func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallValueI, &call_args, descr)
    }

    /// RPython pyjitpl.py opimpl_conditional_call_value_ir_r: emit CondCallValueR.
    ///
    /// `blackhole.py:1271-1276 bhimpl_conditional_call_value_ir_r` declares
    /// `@arguments("cpu", "r", "i", "I", "R", "d", returns="r")` — the
    /// leading `value` is a Ref-typed argbox, so the recorded op's first
    /// arg must be a `ConstPtr` rather than a `ConstInt`.  Routing the
    /// raw pointer-as-i64 through `get_or_insert` would produce a
    /// `ConstInt` slot that aliases with any int constant of the same
    /// numeric value (`history.py:220` `ConstInt` vs `:307 ConstPtr`
    /// pin distinct types at construction).
    pub fn cond_call_value_ref_typed(
        &mut self,
        value: i64,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        slot: EffectInfoSlot,
    ) -> OpRef {
        let value_ref = self.constants.get_or_insert_typed(value, Type::Ref);
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr_from_target_slot(arg_types, Type::Ref, slot);
        let mut call_args = vec![value_ref, func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallValueR, &call_args, descr)
    }

    /// RPython pyjitpl.py opimpl_record_known_result_i / _r: emit RecordKnownResult.
    ///
    /// Mirrors `blackhole.py:620-628 bhimpl_record_known_result_{i,r}_ir_v`'s
    /// `(cpu, res, func, args_i, args_r, calldescr)` signature: the
    /// trailing `d` argcode carries the per-callee calldescr that
    /// `jtransform.py:299-310 rewrite_op_jit_record_known_result` builds
    /// from `getcalldescr`.  `OptPure.optimize_record_known_result`
    /// (`optimizeopt/pure.py:211-220`, ported at
    /// `optimizeopt/pure.rs:1028-1036`) keys its `known_result_call_pure`
    /// table off `descr_identity`, so a missing descr would let two
    /// distinct elidable callees with matching argument shapes collide
    /// at the later `CALL_PURE_*` lookup.
    ///
    /// `result_type` is the result kind of the underlying `CALL_PURE_*`
    /// the recorded entry will later match.  `jtransform.py:296` uses
    /// `op.args[0]` as a "fake result var, which is correct with
    /// regards to the concretetype, the only thing that getcalldescr
    /// accesses": the calldescr's result type follows the known-result
    /// box's concretetype (int or ref), even though the recorded
    /// `record_known_result_*_ir_v` op itself produces no result
    /// register.  `CallDescrKey` (`call_descr.rs:54`) hashes
    /// `result_type` into the descr identity, so passing `Type::Void`
    /// here would never match the `Type::Int` / `Type::Ref` descr that
    /// `getcalldescr` (`jit_codewriter/call.rs:2799+`) builds for the
    /// matching `CALL_PURE_*` op.
    ///
    /// `slot` is the per-callee classification chosen at producer time
    /// (`call.py:282-303 getcalldescr`'s `extraeffect` selection); see
    /// `make_call_descr_from_target_slot` for the resolution rule.
    pub fn record_known_result_typed(
        &mut self,
        result_value: i64,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        result_type: Type,
        slot: EffectInfoSlot,
    ) {
        // `blackhole.py:620-628` declares the two opcodes as
        //   @arguments("cpu", "i", "i", "I", "R", "d")  # _i_ir_v
        //   @arguments("cpu", "r", "i", "I", "R", "d")  # _r_ir_v
        // so the leading known-result argbox is Ref-typed for
        // `record_known_result_r_ir_v` and Int-typed for `_i_ir_v`.
        // Use `get_or_insert_typed(result_value, result_type)` so the
        // recorded constant lands as `ConstPtr` for Ref results,
        // matching `history.py:307 ConstPtr` and preventing alias with
        // `ConstInt` slots of the same raw value.
        let result_ref = self
            .constants
            .get_or_insert_typed(result_value, result_type);
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr_from_target_slot(arg_types, result_type, slot);
        let mut call_args = vec![result_ref, func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::RecordKnownResult, &call_args, descr);
    }

    pub fn call_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallI, func_ptr, args, arg_types, Type::Int)
    }

    /// `call_int_typed` preserving the caller-supplied `EffectInfo`.
    /// Mirrors `pyjitpl.py:1995-2068 do_residual_call` parity (see
    /// `call_typed_with_effect` for the full rationale): PyPy passes
    /// the original `calldescr` through `record_nospec` so the trace
    /// IR retains `oopspec`, `read/write_descrs_*`, `can_invalidate`,
    /// `can_collect`, and `call_release_gil_target` exactly as written
    /// by the codewriter / write-analyzer.
    pub fn call_int_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_typed_with_effect(
            OpCode::CallI,
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
        )
    }

    pub fn call_elidable_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureI, func_ptr, args, arg_types, Type::Int)
    }

    // ── Ref/Float call variants ─────────────────────────────────────

    /// Record a ref-returning function call (CallR).
    pub fn call_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning function call (CallF).
    pub fn call_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_float_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning elidable (pure) call (CallPureR).
    pub fn call_elidable_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_elidable_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning elidable (pure) call (CallPureF).
    pub fn call_elidable_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_elidable_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallR, func_ptr, args, arg_types, Type::Ref)
    }

    /// `call_ref_typed` preserving the caller-supplied `EffectInfo`.
    /// See `call_int_typed_with_effect` for the parity rationale.
    pub fn call_ref_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_typed_with_effect(
            OpCode::CallR,
            func_ptr,
            args,
            arg_types,
            Type::Ref,
            effect_info,
        )
    }

    pub fn call_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallF, func_ptr, args, arg_types, Type::Float)
    }

    /// `call_float_typed` preserving the caller-supplied `EffectInfo`.
    /// See `call_int_typed_with_effect` for the parity rationale.
    pub fn call_float_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_typed_with_effect(
            OpCode::CallF,
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
        )
    }

    pub fn call_elidable_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureR, func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_elidable_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureF, func_ptr, args, arg_types, Type::Float)
    }

    /// Shared body for typed-helper MayForce calls.  Records
    /// `[func_ref] + args`, matching `pyjitpl.py:1995-2002 _build_allboxes(
    /// funcbox, argboxes, descr)`'s `[funcbox] + reordered_argboxes` shape
    /// (here `args` already excludes the funcbox — `func_ptr` is passed
    /// separately and prepended once).
    ///
    /// Release-GIL calls do NOT route through here: the upstream
    /// `pyjitpl.py:3671-3681 direct_call_release_gil` records the distinct
    /// `[savebox, funcbox_real] + argboxes[1:]` shape with
    /// `funcbox_real` resolved from
    /// `effectinfo.call_release_gil_target` (the *real* C function
    /// address, potentially distinct from the wrapper at `argboxes[0]`
    /// per `call.py:252-258`).  That shape is implemented by
    /// [`Self::record_release_gil_typed_with_effect`].
    fn call_family_typed(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_may_force_descr(arg_types, ret_type);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2070-2072 `do_residual_call` may-force branch:
        // `self.metainterp.heapcache.invalidate_caches_varargs(opnum1,
        // descr, allboxes)` after recording the may-force op. The
        // canonical helper marks args escaped and runs the heap-array
        // invalidation arm in one pass (heapcache.py:341-376).
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        result
    }

    pub fn call_may_force_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
        );
    }

    /// `call_family_typed` variant preserving the caller-supplied
    /// `EffectInfo`. Mirrors `pyjitpl.py:1995-2068 do_residual_call`
    /// parity (see `call_typed_with_effect` for the full rationale).
    /// Routes through a fresh `MetaCallDescr` (`make_call_descr_with_effect`)
    /// instead of the static-`EffectInfo` `MetaCallMayForceDescr`, so
    /// `oopspecindex`, `read/write_descrs_*`, and
    /// `call_release_gil_target` survive into the trace IR.
    fn call_family_typed_with_effect(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2070-2072 (see `call_family_typed` for rationale).
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        result
    }

    pub fn call_may_force_void_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let _ = self.call_family_typed_with_effect(
            OpCode::call_may_force_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
            effect_info,
        );
    }

    /// pyjitpl.py:3671-3681 direct_call_release_gil parity. RPython:
    /// ```python
    /// realfuncaddr, saveerr = effectinfo.call_release_gil_target
    /// funcbox = ConstInt(adr2int(realfuncaddr))
    /// savebox = ConstInt(saveerr)
    /// opnum   = rop.call_release_gil_for_descr(calldescr)
    /// return self.history.record_nospec(opnum,
    ///     [savebox, funcbox] + argboxes[1:], ..., calldescr)
    /// ```
    ///
    /// Pyre's typed-helper API takes `args` *without* a leading funcbox
    /// (the funcbox is the `func_ptr` parameter), so `args` is already
    /// the upstream `argboxes[1:]` shape. The trace op shape becomes
    /// `[savebox, realfuncaddr] + args`. The body reads
    /// `(realfuncaddr, saveerr)` directly off `effect_info.call_release_gil_target`
    /// matching `pyjitpl.py:3675` line-by-line; the descr is guaranteed
    /// to carry a real C address by the time we read it because either
    /// (a) the emit-side `assembler.rs::resolve_call_release_gil_target`
    /// substituted the resolved `target.concrete_ptr` into a sentinel
    /// `(1, 0)` slot before the descr was materialized, or (b) a
    /// producer-side typed caller (`trace_ctx.rs::call_release_gil_int_typed`,
    /// `_float_typed`) populated the slot directly from `func_ptr`.
    ///
    /// Routes heapcache invalidation through `invalidate_caches_varargs`
    /// (heapcache.py:309-340) instead of the escape-only path used by
    /// `call_family_typed_with_effect`. RPython
    /// `heapcache.py:341-376 clear_caches_varargs` enumerates the
    /// plain CALL_* / CALL_LOOPINVARIANT_* / COND_CALL_* opcodes and
    /// EXCLUDES the `CALL_RELEASE_GIL_*` family — release-gil falls
    /// through to `reset_keep_likely_virtuals` because the optimizer
    /// cannot selectively invalidate across a GIL-release boundary.
    /// Pyre's `clear_caches_varargs` (`heapcache.rs:1097`) mirrors
    /// the upstream enumeration with an explicit
    /// `!is_call_release_gil()` guard.
    pub fn call_release_gil_void_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let _ = self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
            effect_info,
        );
    }

    /// Shared release-gil recorder for the `i / r / f / v` typed
    /// variants. Mirrors `pyjitpl.py:3671-3681 direct_call_release_gil`:
    ///
    /// ```python
    /// realfuncaddr, saveerr = effectinfo.call_release_gil_target
    /// funcbox = ConstInt(adr2int(realfuncaddr))
    /// savebox = ConstInt(saveerr)
    /// opnum   = rop.call_release_gil_for_descr(calldescr)
    /// return self.history.record_nospec(opnum,
    ///     [savebox, funcbox] + argboxes[1:], ..., calldescr)
    /// ```
    ///
    /// Pyre's typed-helper API takes `args` *without* a leading funcbox
    /// (the funcbox is the `func_ptr` parameter), so `args` is already
    /// the upstream `argboxes[1:]` shape. The trace op shape becomes
    /// `[savebox, realfuncaddr] + args`.
    ///
    /// The Cranelift / Dynasm consumers (`compiler.rs:9807` etc.)
    /// require this shape uniformly; emitting the legacy `[func, args]`
    /// shape for int/float typed release-gil silently mis-routes the
    /// first real arg as the function pointer.
    fn record_release_gil_typed_with_effect(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        // pyjitpl.py:3675-3677:
        //   realfuncaddr, saveerr = effectinfo.call_release_gil_target
        //   funcbox = ConstInt(adr2int(realfuncaddr))
        //   savebox = ConstInt(saveerr)
        // Pyre's emit-side `resolve_call_release_gil_target`
        // (`jitcode/assembler.rs`) substitutes the resolved
        // `target.concrete_ptr` into the `realfuncaddr` slot for the
        // sentinel-(1, 0) descrs emitted by the macro DSL wrappers
        // before the descr is materialized.  Caller-side
        // `trace_ctx::call_release_gil_*_typed` already populates the
        // slot with `func_ptr` directly (`trace_ctx.rs:3852, 3874`).
        // Either way the descr carries a real C address by the time
        // we read it here.
        //
        // PyPy's `call.py:252-258 _call_aroundstate_target_` allows
        // the wrapper at `direct_call`'s `args[0]` and the real GIL-
        // release target to be intentionally distinct values, so the
        // recorded `funcbox` (built from `realfuncaddr`) is NOT
        // required to equal `func_ptr` — only `realfuncaddr != 0` is
        // structurally guaranteed.
        let (realfuncaddr, saveerr) = effect_info.call_release_gil_target;
        debug_assert!(
            realfuncaddr != 0,
            "release_gil call_release_gil_target unset — emit-side resolve_call_release_gil_target should have populated realfuncaddr",
        );
        let _ = func_ptr;
        let savebox = self.constants.get_or_insert(saveerr as i64);
        let funcbox = self.constants.get_or_insert(realfuncaddr as i64);

        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let mut call_args = Vec::with_capacity(2 + args.len());
        call_args.push(savebox);
        call_args.push(funcbox);
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // heapcache.py:341-376 clear_caches_varargs: release-gil falls
        // through to `reset_keep_likely_virtuals`, the most aggressive
        // arm. Pass the descr's effectinfo so the elidable / arraycopy /
        // arraymove fast-paths are honoured if a future descr
        // classification activates them.
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        result
    }

    /// `call_loopinvariant_void_typed` preserving the caller-supplied
    /// `EffectInfo`. Mirrors `pyjitpl.py:2087-2110` for `tp == 'v'`.
    ///
    /// Upstream's loop-invariant cache (`heapcache.py:629-639
    /// call_loopinvariant_known_result` / `call_loopinvariant_now_known`)
    /// stores the *result* op, but `_record_helper_varargs`
    /// (`pyjitpl.py:2655-2663`) returns `None` for void calls — so the
    /// cached "known result" lookup at `pyjitpl.py:2088` returns `None`
    /// and the `if res is not None: return res` early-out always misses.
    /// `pyjitpl.py:2109` still calls `call_loopinvariant_now_known(allboxes,
    /// descr, res)` with `res = None`, which evicts whatever prior typed
    /// result shared the (descr, arg0) slot.  The void-overload
    /// `call_loopinvariant_now_known_void` (heapcache.rs) stores
    /// `loopinvariant_result = None` so the next typed lookup correctly
    /// misses the stale slot.
    pub fn call_loopinvariant_void_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let opcode = OpCode::call_loopinvariant_for_type(Type::Void);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, Type::Void, effect_info);
        let descr_index = descr.index();
        let arg0_int = func_ptr as usize as i64;
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let _ = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2659 `_record_helper_varargs` parity (see
        // `call_typed`): every CALL family record routes through the
        // canonical heap_cache.invalidate_caches_varargs.
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        // pyjitpl.py:2109 `call_loopinvariant_now_known(allboxes, descr, res)`
        // with `res = None` for void.  Evicts any prior typed entry sharing
        // this (descr, arg0) key so subsequent typed loop-invariant lookups
        // do not return a stale OpRef.
        self.heap_cache
            .call_loopinvariant_now_known_void(descr_index, arg0_int);
    }

    pub fn call_may_force_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_may_force_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
        )
    }

    /// Record a float-returning may-force call (CallMayForceF).
    pub fn call_may_force_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_may_force_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_may_force_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
        )
    }

    // call_release_gil_void / _typed intentionally absent:
    // production void release-GIL calls are emitted via the canonical
    // `JitCodeBuilder::call_release_gil_void_canonical_via_target`
    // (jitcode/assembler.rs) which writes the upstream-shaped
    // `[savebox, funcbox]+args` operand layout directly. The legacy
    // `call_family_typed`-based void helper produced a `[func]+args`
    // layout that did not match `cranelift::compiler.rs:9807`'s
    // expectation, so it was removed once the only caller migrated.
    //
    // call_release_gil_ref / _typed intentionally absent:
    // resoperation.py:1243-1244 (`# no such thing`) excludes
    // CALL_RELEASE_GIL_R from the upstream opcode table.

    /// Record a float-returning GIL-release call (CallReleaseGilF).
    pub fn call_release_gil_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_release_gil_float_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning loop-invariant call (CallLoopinvariantR).
    pub fn call_loopinvariant_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_loopinvariant_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning loop-invariant call (CallLoopinvariantF).
    pub fn call_loopinvariant_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_loopinvariant_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_release_gil_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        // pyjitpl.py:3671-3681 direct_call_release_gil shape:
        // `[savebox, funcbox_real] + argboxes[1:]`.  Pyre dispatches the
        // C callee directly via `bh_call_*_dispatch` (no asm helper),
        // so `realfuncaddr` IS `func_ptr` and `saveerr=0`.  Populating
        // the field at the call site keeps the descr's IR carrying the
        // real `(realfuncaddr, saveerr)` pair just like upstream's
        // `effectinfo.call_release_gil_target`. effectinfo.py:149-155
        // requires every readonly/write descr set be `None` for
        // `EF_RANDOM_EFFECTS`; spread MOST_GENERAL instead of
        // `EffectInfo::default()` whose `Some(Vec::new())` bitstrings
        // would silently misrepresent the wildcard.
        let effect_info = majit_ir::EffectInfo {
            call_release_gil_target: (func_ptr as usize as u64, 0),
            ..majit_ir::EffectInfo::MOST_GENERAL
        };
        self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
        )
    }

    pub fn call_release_gil_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        // effectinfo.py:149-155: `EF_RANDOM_EFFECTS` keeps every
        // readonly/write descr set as `None`; spread MOST_GENERAL for
        // the wildcard rather than `EffectInfo::default()`'s
        // `Some(Vec::new())` bitstrings.
        let effect_info = majit_ir::EffectInfo {
            call_release_gil_target: (func_ptr as usize as u64, 0),
            ..majit_ir::EffectInfo::MOST_GENERAL
        };
        self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
        )
    }

    pub fn call_loopinvariant_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Void);
    }

    pub fn call_loopinvariant_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Int)
    }

    pub fn call_loopinvariant_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_loopinvariant_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Float)
    }

    /// pyjitpl.py:2081-2104: emit a loop-invariant CALL_LOOPINVARIANT_*
    /// with the heapcache lookup/store envelope.
    ///
    /// `heapcache.py:629-639 call_loopinvariant_known_result` /
    /// `call_loopinvariant_now_known` keys the slot by descr **object
    /// identity** (`if self.loop_invariant_descr is not descr: return
    /// None`) and `allboxes[0].getint()`.  `MetaCallDescr` is interned
    /// through `GcCache._cache_call`'s local equivalent, so
    /// `descr.index()` returns a stable per-instance `heapcache_index`
    /// that supplies the `is`-equivalent identity key.  `func_ptr as
    /// i64` is the typed-helper analogue of `funcbox.getint()`
    /// (`pyjitpl.py:2002 _build_allboxes`'s slot 0).
    ///
    /// An earlier revision used `signature_hash(arg_types, ret_type)`
    /// as a structural surrogate.  That hash violated `is` semantics
    /// (collision + same-signature over-merge across distinct upstream
    /// descrs) and was removed once the interned `MetaCallDescr`
    /// landed.
    fn call_loopinvariant_impl(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let opcode = OpCode::call_loopinvariant_for_type(ret_type);
        let descr = crate::call_descr::make_call_descr_for_opcode(opcode, arg_types, ret_type);
        // RPython `heapcache.py:629-639` keys by descriptor identity
        // and `allboxes[0].getint()`. `MetaCallDescr` is cached through
        // the local equivalent of `GcCache._cache_call`, so `index()`
        // is a stable identity key for this heapcache slot while
        // `get_descr_index()` keeps its opencoder meaning.
        let descr_index = descr.index();
        let arg0_int = func_ptr as usize as i64;
        // heapcache: check loop-invariant cache
        if let Some((cached, _resvalue)) = self
            .heap_cache
            .call_loopinvariant_lookup(descr_index, arg0_int)
        {
            // Legacy trace_ctx helper does not yet thread the concrete
            // resvalue from this call site; the cached symbolic OpRef
            // is enough for the consumers of this method.
            return cached;
        }
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2659 `_record_helper_varargs` parity (mirror
        // `call_typed` at trace_ctx.rs:3083). Routes
        // heapcache.invalidate_caches_varargs through the recorded
        // CALL_LOOPINVARIANT_* so escape / clear_caches_varargs paths
        // run exactly once per recorded op (heapcache.py:211).
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        // Concrete resvalue is unknown to this legacy helper; pass 0.
        self.heap_cache
            .call_loopinvariant_cache(descr_index, arg0_int, result, 0);
        result
    }

    // ── Slice 4 Slice 1c.0: typed (i/r/f) `_with_effect` recorders ──
    //
    // Mirrors the void-family `_with_effect` wrappers (call_*_void_typed_with_effect)
    // for the int/ref/float result kinds.  Pyre's canonical typed
    // residual_call recording arms (pyjitpl/dispatch.rs BC_RESIDUAL_CALL_*_{I,R,F})
    // route through these helpers instead of the legacy `_typed`
    // siblings that re-derive the EffectInfo from the opcode policy.
    // RPython `pyjitpl.py:1995-2068 do_residual_call` threads the
    // calldescr's EffectInfo through `record_nospec` for every result
    // kind; pyre's void path already honours that — these wrappers
    // close the i/r/f gap.

    pub fn call_may_force_int_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_family_typed_with_effect(
            OpCode::call_may_force_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
        )
    }

    pub fn call_may_force_ref_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_family_typed_with_effect(
            OpCode::call_may_force_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
            effect_info,
        )
    }

    pub fn call_may_force_float_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_family_typed_with_effect(
            OpCode::call_may_force_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
        )
    }

    /// `pyjitpl.py:3671-3681 direct_call_release_gil` — Int result.
    /// Routes through the shared `record_release_gil_typed_with_effect`
    /// emitting `[savebox, realfuncaddr] + args` per the void sibling.
    /// `resoperation.py:1243-1244 # no such thing` excludes the Ref
    /// flavour, so no `_ref_typed_with_effect` counterpart exists.
    pub fn call_release_gil_int_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
        )
    }

    pub fn call_release_gil_float_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
        )
    }

    pub fn call_loopinvariant_int_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
        concrete_resvalue: i64,
    ) -> OpRef {
        self.call_loopinvariant_impl_with_effect(
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
            concrete_resvalue,
        )
    }

    pub fn call_loopinvariant_ref_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
        concrete_resvalue: i64,
    ) -> OpRef {
        self.call_loopinvariant_impl_with_effect(
            func_ptr,
            args,
            arg_types,
            Type::Ref,
            effect_info,
            concrete_resvalue,
        )
    }

    pub fn call_loopinvariant_float_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
        concrete_resvalue: i64,
    ) -> OpRef {
        self.call_loopinvariant_impl_with_effect(
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
            concrete_resvalue,
        )
    }

    /// pyjitpl.py:2087-2090 parity: heapcache lookup-only for the
    /// loop-invariant `_with_effect` family. Returns `Some((cached_opref,
    /// cached_resvalue))` if `heapcache.call_loopinvariant_known_result`
    /// (heapcache.py:629-634) has a hit, otherwise `None`.
    ///
    /// Callers use this to short-circuit BOTH the concrete C call and the
    /// trace record on a hit — RPython does the lookup before
    /// `execute_varargs` (`pyjitpl.py:2088 if res is not None: return res`)
    /// and never executes the call when the cache returns a result.
    pub fn call_loopinvariant_lookup_with_effect(
        &self,
        func_ptr: *const (),
        arg_types: &[Type],
        ret_type: Type,
        effect_info: &majit_ir::EffectInfo,
    ) -> Option<(OpRef, i64)> {
        let descr = crate::call_descr::make_call_descr_with_effect(
            arg_types,
            ret_type,
            effect_info.clone(),
        );
        let descr_index = descr.index();
        let arg0_int = func_ptr as usize as i64;
        self.heap_cache
            .call_loopinvariant_known_result(descr_index, arg0_int)
    }

    /// `call_loopinvariant_impl` variant preserving the caller-supplied
    /// `EffectInfo`. Mirrors the heapcache lookup/store envelope of the
    /// non-`_with_effect` sibling but produces the descr through
    /// `make_call_descr_with_effect` so `oopspecindex`,
    /// `read/write_descrs_*`, and `can_invalidate` survive into the
    /// trace IR.
    fn call_loopinvariant_impl_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
        concrete_resvalue: i64,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let opcode = OpCode::call_loopinvariant_for_type(ret_type);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let descr_index = descr.index();
        let arg0_int = func_ptr as usize as i64;
        if let Some((cached, _resvalue)) = self
            .heap_cache
            .call_loopinvariant_lookup(descr_index, arg0_int)
        {
            return cached;
        }
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2659 `_record_helper_varargs` parity (mirror
        // `call_typed_with_effect` at trace_ctx.rs:3122). Routes
        // heapcache.invalidate_caches_varargs through the recorded
        // CALL_LOOPINVARIANT_* so escape / clear_caches_varargs paths
        // run exactly once per recorded op — without this, the
        // loop-invariant helper's record skipped mark_escaped_varargs
        // (heapcache.py:211).
        if let Some(call_descr) = descr.as_call_descr() {
            self.heap_cache.invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
            );
        }
        // pyjitpl.py:2109 call_loopinvariant_now_known(allboxes, descr, res):
        // store the concrete result so the next iteration's
        // `call_loopinvariant_known_result` returns it without re-executing
        // the C call.
        self.heap_cache
            .call_loopinvariant_cache(descr_index, arg0_int, result, concrete_resvalue);
        result
    }

    // ── CALL_ASSEMBLER ────────────────────────────────────────────

    fn call_assembler_typed(
        &mut self,
        opcode: OpCode,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        // Test/dispatch callers pass a stack-synthesised `JitCellToken`
        // (no `Arc` identity). Route through the number-only factory; the
        // keepalive walker recovers the real Arc via
        // `jitcell_token_by_number` (transitional fallback in
        // `record_loop_or_bridge`) until callers thread Arc identity.
        let descr = crate::call_descr::make_call_assembler_descr_by_number(
            target.number,
            arg_types,
            ret_type,
            target.virtualizable_arg_index,
        );
        self.record_op_with_descr(opcode, args, descr)
    }

    /// Emit CALL_ASSEMBLER_<type> by token number with explicit arg types.
    /// resoperation.py:1251 `call_assembler_for_descr`: opcode is selected
    /// from `result_type` per `OpCode::call_assembler_for_type`.
    fn call_assembler_typed_by_number(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
        result_type: Type,
    ) -> OpRef {
        let descr = crate::call_descr::make_call_assembler_descr_by_number(
            target_number,
            arg_types,
            result_type,
            self.driver_descriptor
                .as_ref()
                .and_then(JitDriverStaticData::virtualizable_arg_index),
        );
        let opcode = OpCode::call_assembler_for_type(result_type);
        let result = self.record_op_with_descr(opcode, args, descr);
        // pyjitpl.py:2072 heapcache.invalidate_caches_varargs(opnum1, descr,
        // allboxes).  RPython uses CALL_MAY_FORCE_* as the opnum for
        // invalidation, but CALL_ASSEMBLER_* falls through to
        // reset_keep_likely_virtuals in heapcache.py:372-376 — same path as
        // CALL_MAY_FORCE_*.
        self.heap_cache
            .invalidate_caches_varargs(opcode, None, args);
        result
    }

    pub fn call_assembler_void_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_assembler_typed_by_number(target_number, args, arg_types, Type::Void);
    }

    pub fn call_assembler_int_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_by_number(target_number, args, arg_types, Type::Int)
    }

    pub fn call_assembler_ref_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_by_number(target_number, args, arg_types, Type::Ref)
    }

    pub fn call_assembler_float_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_by_number(target_number, args, arg_types, Type::Float)
    }

    /// rewrite.py:665-695 handle_call_assembler parity.
    /// Emit CALL_ASSEMBLER with only the frame reference as arg; the backend
    /// expands to the full callee inputarg layout via `VableExpansion`.
    pub fn call_assembler_with_vable_expansion(
        &mut self,
        target_number: u64,
        frame_arg: OpRef,
        result_type: Type,
        expansion: majit_ir::VableExpansion,
    ) -> OpRef {
        self.call_assembler_with_vable_expansion_args(
            target_number,
            &[frame_arg],
            &[Type::Ref],
            result_type,
            expansion,
        )
    }

    /// pyjitpl.py:3589-3609 direct_assembler_call parity.
    /// Emit CALL_ASSEMBLER with multiple red args + VableExpansion.
    /// The backend reads some fields from args[0] (frame) and uses
    /// arg_overrides/const_overrides for callee-specific values.
    pub fn call_assembler_with_vable_expansion_args(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
        result_type: Type,
        expansion: majit_ir::VableExpansion,
    ) -> OpRef {
        let opcode = match result_type {
            Type::Int => OpCode::CallAssemblerI,
            Type::Ref => OpCode::CallAssemblerR,
            Type::Float => OpCode::CallAssemblerF,
            Type::Void => OpCode::CallAssemblerN,
        };
        let descr = crate::call_descr::make_call_assembler_descr_with_vable_by_number(
            target_number,
            arg_types,
            result_type,
            expansion,
        );
        self.record_op_with_descr(opcode, args, descr)
    }

    /// RPython `direct_assembler_call` red-args-only emission
    /// (pyjitpl.py:3589-3609). Takes the JitDriver reds directly and emits
    /// a CALL_ASSEMBLER with no `VableExpansion` — the callee's compiled
    /// loop reconstructs each virtualizable field via its GETFIELD_GC /
    /// GETARRAYITEM_GC preamble emitted by
    /// `patch_new_loop_to_load_virtualizable_fields` (compile.py:425-461).
    ///
    /// `virtualizable_arg_index` of the emitted descriptor comes from the
    /// active `JitDriverStaticData`, matching RPython's
    /// `rewrite.py:684 jd.index_of_virtualizable` lookup.
    ///
    /// Covered by `call_assembler_red_only_ref_emits_no_vable_expansion`
    /// to verify the emitted descriptor shape.
    pub fn call_assembler_red_only_ref(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let descr = crate::call_descr::make_call_assembler_descr_by_number(
            target_number,
            arg_types,
            Type::Ref,
            self.driver_descriptor
                .as_ref()
                .and_then(JitDriverStaticData::virtualizable_arg_index),
        );
        self.record_op_with_descr(OpCode::CallAssemblerR, args, descr)
    }

    /// Emit CALL_ASSEMBLER_N (void), inferring arg types from the current boxes.
    pub fn call_assembler_void(&mut self, target: &JitCellToken, args: &[OpRef]) {
        let arg_types = self.infer_arg_types(args);
        self.call_assembler_void_typed(target, args, &arg_types);
    }

    /// Emit CALL_ASSEMBLER_I, inferring arg types from the current boxes.
    pub fn call_assembler_int(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_assembler_int_typed(target, args, &arg_types)
    }

    /// Emit CALL_ASSEMBLER_R, inferring arg types from the current boxes.
    pub fn call_assembler_ref(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_assembler_ref_typed(target, args, &arg_types)
    }

    /// Emit CALL_ASSEMBLER_F, inferring arg types from the current boxes.
    pub fn call_assembler_float(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_assembler_float_typed(target, args, &arg_types)
    }

    pub fn call_assembler_void_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Void),
            target,
            args,
            arg_types,
            Type::Void,
        );
    }

    pub fn call_assembler_int_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Int),
            target,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_assembler_ref_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Ref),
            target,
            args,
            arg_types,
            Type::Ref,
        )
    }

    pub fn call_assembler_float_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Float),
            target,
            args,
            arg_types,
            Type::Float,
        )
    }

    // ── Exception handling ──────────────────────────────────────────

    /// Record GUARD_EXCEPTION: assert that the pending exception matches
    /// the given class, and produce a ref to the exception value.
    pub fn guard_exception(&mut self, exc_class: OpRef, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardException, &[exc_class], num_live)
    }

    /// Record SAVE_EXCEPTION: capture the pending exception value as a ref.
    pub fn save_exception(&mut self) -> OpRef {
        self.record_op(OpCode::SaveException, &[])
    }

    /// Record SAVE_EXC_CLASS: capture the pending exception's class as an int.
    pub fn save_exc_class(&mut self) -> OpRef {
        self.record_op(OpCode::SaveExcClass, &[])
    }

    /// Record RESTORE_EXCEPTION: restore exception state from saved
    /// class and value refs.
    pub fn restore_exception(&mut self, exc_class: OpRef, exc_value: OpRef) {
        self.record_op(OpCode::RestoreException, &[exc_class, exc_value]);
    }

    // ── Object allocation ───────────────────────────────────────────

    /// Record NEW: allocate a new object described by `descr`.
    pub fn record_new(&mut self, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::New, &[], descr)
    }

    /// Record NEW_WITH_VTABLE: allocate a new object with an explicit vtable pointer.
    pub fn record_new_with_vtable(&mut self, vtable: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewWithVtable, &[vtable], descr)
    }

    /// Record NEW_ARRAY: allocate a new array with the given length.
    pub fn record_new_array(&mut self, length: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewArray, &[length], descr)
    }

    /// Record NEW_ARRAY_CLEAR: allocate a zero-initialized array.
    pub fn record_new_array_clear(&mut self, length: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewArrayClear, &[length], descr)
    }

    // ── Virtual references ────────────────────────────────────────

    /// Record VIRTUAL_REF_R: create a virtual reference (ref-typed result).
    ///
    /// `virtual_obj` is the real object being wrapped.
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index
    /// (pyjitpl.py:1805-1806 parity).
    ///
    /// The optimizer replaces this with a virtual struct, so if the vref
    /// never escapes, no allocation happens.
    pub fn virtual_ref_r(&mut self, virtual_obj: OpRef, cindex: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefR, &[virtual_obj, cindex])
    }

    /// Record VIRTUAL_REF_I: create a virtual reference (int-typed result).
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index.
    pub fn virtual_ref_i(&mut self, virtual_obj: OpRef, cindex: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefI, &[virtual_obj, cindex])
    }

    /// Record VIRTUAL_REF_FINISH: finalize a virtual reference.
    ///
    /// `vref` is the virtual reference to finalize.
    /// `virtual_obj` is the real object (or NULL/0 if the frame is being left normally).
    pub fn virtual_ref_finish(&mut self, vref: OpRef, virtual_obj: OpRef) {
        self.record_op(OpCode::VirtualRefFinish, &[vref, virtual_obj]);
    }

    /// Record FORCE_TOKEN: capture the current JIT frame address.
    pub fn force_token(&mut self) -> OpRef {
        self.record_op(OpCode::ForceToken, &[])
    }

    // ── Overflow-checked arithmetic ────────────────────────────────

    /// Record overflow-checked integer add + GuardNoOverflow.
    ///
    /// Returns the result OpRef. On overflow at trace time, the caller
    /// should abort tracing.
    pub fn int_add_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntAddOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record overflow-checked integer sub + GuardNoOverflow.
    pub fn int_sub_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntSubOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record overflow-checked integer mul + GuardNoOverflow.
    pub fn int_mul_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntMulOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    // ── String operations ───────────────────────────────────────────

    /// Record NEWSTR: allocate a new string with given length.
    pub fn newstr(&mut self, length: OpRef) -> OpRef {
        self.record_op(OpCode::Newstr, &[length])
    }

    /// Record STRLEN: get string length.
    pub fn strlen(&mut self, string: OpRef) -> OpRef {
        self.record_op(OpCode::Strlen, &[string])
    }

    /// Record STRGETITEM: read character at index.
    pub fn strgetitem(&mut self, string: OpRef, index: OpRef) -> OpRef {
        self.record_op(OpCode::Strgetitem, &[string, index])
    }

    /// Record STRSETITEM: write character at index.
    pub fn strsetitem(&mut self, string: OpRef, index: OpRef, value: OpRef) {
        self.record_op(OpCode::Strsetitem, &[string, index, value]);
    }

    /// Record COPYSTRCONTENT: copy characters between strings.
    pub fn copystrcontent(
        &mut self,
        src: OpRef,
        dst: OpRef,
        src_start: OpRef,
        dst_start: OpRef,
        length: OpRef,
    ) {
        self.record_op(
            OpCode::Copystrcontent,
            &[src, dst, src_start, dst_start, length],
        );
    }

    /// Record STRHASH: compute string hash.
    pub fn strhash(&mut self, string: OpRef) -> OpRef {
        self.record_op(OpCode::Strhash, &[string])
    }
}

#[cfg(test)]
mod history_record_tests {
    use crate::jitdriver::JitDriverStaticData;
    use crate::recorder::Trace;
    use crate::trace_ctx::TraceCtx;
    use majit_backend::JitCellToken;
    use majit_ir::{OpCode, OpRef, Type};

    extern "C" fn dummy_call_target() {}

    fn make_ctx_with_mixed_inputs() -> (TraceCtx, [OpRef; 3]) {
        let mut recorder = Trace::new();
        let r = recorder.record_input_arg(Type::Ref);
        let f = recorder.record_input_arg(Type::Float);
        let i = recorder.record_input_arg(Type::Int);
        (
            TraceCtx::new(
                recorder,
                0,
                std::sync::Arc::new(crate::MetaInterpStaticData::new()),
            ),
            [r, f, i],
        )
    }

    fn take_single_call_descr(ctx: TraceCtx, jump_args: &[OpRef]) -> (Vec<Type>, OpCode) {
        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let trace = recorder.get_trace();
        let call_op = &trace.ops[0];
        let arg_types = call_op
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("call op should carry CallDescr")
            .arg_types()
            .to_vec();
        (arg_types, call_op.opcode)
    }

    fn take_single_call_op(ctx: TraceCtx, jump_args: &[OpRef]) -> majit_ir::Op {
        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let mut trace = recorder.get_trace();
        trace.ops.remove(0)
    }

    #[test]
    fn call_may_force_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_may_force_ref_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallMayForceR);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_void_infers_mixed_arg_types_from_boxes() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        ctx.call_void(dummy_call_target as *const (), &args);
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallN);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_ref_infers_mixed_arg_types_from_boxes() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_ref(dummy_call_target as *const (), &args);
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallR);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_release_gil_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_release_gil_float_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallReleaseGilF);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_loopinvariant_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_loopinvariant_int_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallLoopinvariantI);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_assembler_typed_preserves_mixed_arg_types_and_target_token() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let mut token = JitCellToken::new(777);
        token.virtualizable_arg_index = Some(1);
        let _ = ctx.call_assembler_ref_typed(&token, &args, &[Type::Ref, Type::Float, Type::Int]);
        let op = take_single_call_op(ctx, &args);
        assert_eq!(op.opcode, OpCode::CallAssemblerR);
        assert_eq!(op.args.as_slice(), &args);
        let call_descr = op
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("call op should carry CallDescr");
        let loop_token = op
            .descr
            .as_ref()
            .and_then(|descr| descr.as_loop_token_descr())
            .expect("call op should carry loop-token metadata");
        assert_eq!(call_descr.arg_types(), &[Type::Ref, Type::Float, Type::Int]);
        assert_eq!(call_descr.call_target_token(), Some(777));
        assert_eq!(call_descr.call_virtualizable_index(), Some(1));
        assert_eq!(loop_token.loop_token_number(), 777);
        assert_eq!(loop_token.call_virtualizable_index(), Some(1));
    }

    #[test]
    fn call_assembler_red_only_ref_emits_no_vable_expansion() {
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let frame = OpRef::input_arg_ref(0);
        ctx.set_driver_descriptor(JitDriverStaticData::with_virtualizable(
            Vec::new(),
            vec![("frame", Type::Ref)],
            Some("frame"),
        ));

        let _ = ctx.call_assembler_red_only_ref(999, &[frame], &[Type::Ref]);

        let op = take_single_call_op(ctx, &[frame]);
        assert_eq!(op.opcode, OpCode::CallAssemblerR);
        assert_eq!(op.args.as_slice(), &[frame]);
        let call_descr = op
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("red-only CA should still carry a CallDescr");
        assert_eq!(call_descr.arg_types(), &[Type::Ref]);
        assert_eq!(call_descr.call_target_token(), Some(999));
        assert_eq!(call_descr.call_virtualizable_index(), Some(0));
        assert!(
            call_descr.vable_expansion().is_none(),
            "red-only emission must not attach a VableExpansion",
        );
    }
}
