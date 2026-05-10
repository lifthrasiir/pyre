/// The Trace data structure — a completed sequence of IR operations.
///
/// A Trace is the output of the Trace and the input to the
/// optimizer and backend. It represents a linear sequence of operations
/// that forms a loop (ending with JUMP) or an exit (ending with FINISH).
///
/// Reference: rpython/jit/metainterp/history.py TreeLoop
use majit_ir::{InputArg, Op, OpCode, OpRef};

use crate::r#box::BoxRef;

/// RPython `History` parity alias for `TreeLoop`.
///
/// The Rust implementation keeps the historical `TreeLoop` name for naming
/// consistency with `rpython/jit/metainterp/pyjitpl.py` internals, while
/// `History` is retained for direct RPython call-site parity.
pub type History = TreeLoop;

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

    /// history.py: check_consistency()
    /// Verify that the trace structure is valid.
    pub fn check_consistency(&self) -> bool {
        if self.ops.is_empty() {
            return true;
        }
        // Last op must be Jump or Finish
        let last = self.ops.last().unwrap();
        if !last.opcode.is_final() {
            return false;
        }
        // No duplicate positions
        let mut seen = std::collections::HashSet::new();
        for op in &self.ops {
            if !op.pos.is_none() {
                if !seen.insert(op.pos) {
                    return false; // duplicate position
                }
            }
        }
        true
    }

    /// Get all OpRefs used as arguments but not defined by any op.
    /// These are the "free variables" — inputs from outside the trace.
    pub fn free_vars(&self) -> Vec<OpRef> {
        let defined: std::collections::HashSet<OpRef> = self
            .ops
            .iter()
            .filter(|op| !op.pos.is_none())
            .map(|op| op.pos)
            .collect();
        let mut free = std::collections::HashSet::new();
        for op in &self.ops {
            for arg in &op.args {
                if !arg.is_none() && !defined.contains(arg) {
                    free.insert(*arg);
                }
            }
        }
        let mut result: Vec<OpRef> = free.into_iter().collect();
        result.sort_by_key(|r| r.raw());
        result
    }

    /// Count operations by opcode category.
    pub fn count_by_category(&self) -> (usize, usize, usize, usize) {
        let mut guards = 0;
        let mut pure = 0;
        let mut calls = 0;
        let mut other = 0;
        for op in &self.ops {
            if op.opcode.is_guard() {
                guards += 1;
            } else if op.opcode.is_always_pure() {
                pure += 1;
            } else if op.opcode.is_call() {
                calls += 1;
            } else {
                other += 1;
            }
        }
        (guards, pure, calls, other)
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
    // Ported from rpython/jit/metainterp/test/test_history.py
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
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::input_arg_int(0), OpRef::int_op(1)]),
            Op::new(OpCode::Jump, &[OpRef::input_arg_int(0)]),
        ];
        let trace = TreeLoop::new(vec![InputArg::new_int(0)], ops);
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
    fn test_free_vars() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::int_op(100), OpRef::int_op(101)]),
            Op::new(OpCode::Finish, &[OpRef::int_op(0)]),
        ];
        ops[0].pos = OpRef::int_op(0);
        ops[1].pos = OpRef::int_op(1);
        let trace = TreeLoop::new(vec![], ops);
        let free = trace.free_vars();
        // OpRef::int_op(100) and OpRef::int_op(101) are free (not defined)
        assert!(free.contains(&OpRef::int_op(100)));
        assert!(free.contains(&OpRef::int_op(101)));
        assert!(!free.contains(&OpRef::int_op(0))); // defined by op[0]
    }

    #[test]
    fn test_count_by_category() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(0)]),
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)]),
            Op::new(OpCode::IntMul, &[OpRef::int_op(0), OpRef::int_op(1)]),
            Op::new(OpCode::CallI, &[OpRef::int_op(0)]),
            Op::new(OpCode::Finish, &[]),
        ];
        let trace = TreeLoop::new(vec![], ops);
        let (guards, pure, calls, other) = trace.count_by_category();
        assert_eq!(guards, 1);
        assert_eq!(pure, 2);
        assert_eq!(calls, 1);
        assert_eq!(other, 1); // Finish
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
}
