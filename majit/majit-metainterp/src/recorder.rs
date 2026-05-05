/// Trace recorder — records IR operations during interpreter execution.
///
/// The recorder is the bridge between the interpreter and the JIT.
/// When tracing is active, every interpreter operation is fed to the
/// recorder, which builds a linear sequence of IR operations (a trace).
///
/// Upstream counterpart: `rpython/jit/metainterp/history.py:714 class History`
/// (`record_nospec` / `record_same_as` / `record_aborted` etc.).  Pyre
/// callers reach the recorder via `MetaInterp.history.record*` mirroring
/// `pyjitpl.py:2455+ self.history.record2(...)`.
use majit_ir::{DescrRef, InputArg, Op, OpCode, OpRef, Type};

use crate::history::TreeLoop;

/// opencoder.py:567-568 `cut_point()` — RPython 5-tuple
/// `(_pos, _count, _index, len(_snapshot_data), len(_snapshot_array_data))`.
///
/// The byte-stream recorder (`TraceRecordBuffer`) fills in every field
/// from its byte cursor / counter state.  The legacy `Vec<Op>` recorder
/// (`recorder::Trace`, being migrated away in Step 2e.2b) maps `_pos` to
/// the ops-Vec cursor (number of ops currently stored). `_count` mirrors
/// the total number of recorded ops, while `_index` mirrors the number of
/// box-yielding positions (inputargs + non-void ops), matching
/// opencoder.py's split counters even though the legacy `Vec<Op>` recorder
/// still assigns `OpRef` positions in total-op order. Snapshot lens come
/// from the recorder-owned `snapshots` side table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracePosition {
    /// opencoder.py:475 `self._pos` — byte cursor for TRB, ops-Vec cursor
    /// for `recorder::Trace`.
    pub _pos: usize,
    /// opencoder.py:497 `self._count` — total op count (including voids).
    pub _count: u32,
    /// opencoder.py:498 `self._index` — count of box-yielding (non-void)
    /// ops; equals `_count` in `recorder::Trace`.
    pub _index: u32,
    /// opencoder.py:567 `len(self._snapshot_data)`.
    pub snapshot_data_len: usize,
    /// opencoder.py:567 `len(self._snapshot_array_data)`.
    pub snapshot_array_data_len: usize,
}

/// opencoder.py Snapshot parity: per-guard snapshot of the interpreter
/// frame state, encoded as tagged references to boxes.
///
/// RPython stores snapshots inline in the trace byte stream
/// (`_snapshot_data` / `_snapshot_array_data`).  Pyre owns them on
/// `TraceCtx` as a `Vec<Snapshot>` side-table (Step 2e.2b will migrate
/// this to the byte-stream form already carried by `TraceRecordBuffer`).
/// Each snapshot captures the live variables of each frame in the call
/// stack at the guard point.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// Frames in the snapshot, outermost first.
    pub frames: Vec<SnapshotFrame>,
    /// Virtualizable box references (tagged).
    pub vable_boxes: Vec<SnapshotTagged>,
    /// VirtualRef box references (tagged).
    pub vref_boxes: Vec<SnapshotTagged>,
}

/// One frame in a snapshot — corresponds to one MIFrame/JitCode position.
#[derive(Clone, Debug)]
pub struct SnapshotFrame {
    /// Index of the jitcode (or 0 for the root portal).
    pub jitcode_index: u32,
    /// Program counter within the jitcode.
    pub pc: u32,
    /// Tagged references to the live boxes in this frame.
    pub boxes: Vec<SnapshotTagged>,
}

/// opencoder.py:603 trace-snapshot encode parity: tagged reference to a
/// live value at a recorder snapshot site.  The recorder only sees Box
/// (live in deadframe fail_args) and Const (compile-time constant)
/// payloads; TAGVIRTUAL belongs to the resume-numbering layer
/// (resume.py:_number_boxes) and is synthesized later from
/// PtrInfo::is_virtual on the live Box's OpRef.  Keeping a `Virtual`
/// variant here would let a virtual index reach
/// `translate_trace_iter_opref`'s _cache as a Box position, silently
/// remapping it (recorder.rs has no Box at that integer).  The variant
/// is therefore intentionally absent — adding a virtual-tagged source
/// requires a dedicated resume enum, not this snapshot type.
///
/// TAGBOX(n)    → value lives in fail_args[n] (deadframe slot n)
/// TAGCONST(v)  → compile-time constant (i64 value)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotTagged {
    /// Value from deadframe fail_args slot.
    /// RPython: InputArgRef/InputArgInt carry type ('r'/'i'/'f') in
    /// the Box class itself. Pyre stores the typed `OpRef` first so
    /// the variant tag (InputArg{Int,Float,Ref} / IntOp/FloatOp/RefOp)
    /// reaches `_number_boxes` (resume.py:210-216 `box.type == 'r' vs
    /// 'i'`). The trailing `Type` is the lockstep authority for the
    /// transitional `OpRef::Untyped` case (test fixtures, residual
    /// from_raw consumers): production must keep both in sync, and
    /// consumers prefer `opref.ty()` when it is `Some`.
    Box(majit_ir::OpRef, majit_ir::Type),
    /// Compile-time constant value with type.
    /// RPython resume.py:157: Const boxes carry their type (INT/REF/FLOAT)
    /// for correct TAGINT/TAGCONST encoding in rd_numb.
    Const(i64, majit_ir::Type),
}

#[derive(Clone)]
pub struct Trace {
    /// Recorded operations.
    ops: Vec<Op>,
    /// Input arguments to the trace (live variables at the loop header).
    inputargs: Vec<InputArg>,
    /// Next OpRef index to assign.
    op_count: u32,
    /// opencoder.py parity: count of box-yielding positions
    /// (inputargs + non-void ops).
    box_count: u32,
}

impl Trace {
    /// Create a new, empty trace recorder.
    ///
    /// opencoder.py Trace.__init__ — trace_limit is enforced at the
    /// MetaInterp / TraceCtx level by consulting warmstate.trace_limit,
    /// not stored on the recorder.
    pub fn new() -> Self {
        Trace {
            ops: Vec::with_capacity(256),
            inputargs: Vec::new(),
            op_count: 0,
            box_count: 0,
        }
    }

    /// Create a trace recorder pre-configured for retracing from a guard.
    ///
    /// The recorder starts with `num_inputs` int-typed input args,
    /// matching the guard's fail_args.
    pub fn with_num_inputs(num_inputs: usize) -> Self {
        Self::with_input_types(&vec![Type::Int; num_inputs])
    }

    /// Create a trace recorder pre-configured for retracing from a guard
    /// with explicit input arg types.
    pub fn with_input_types(input_types: &[Type]) -> Self {
        let mut recorder = Self::new();
        for tp in input_types {
            recorder.record_input_arg(*tp);
        }
        recorder
    }

    /// Register an input argument of the given type.
    /// Returns an OpRef that can be used as an argument to subsequent operations.
    /// Input arguments are numbered starting from 0; the OpRef index matches
    /// the input argument index.
    ///
    /// resoperation.py:719/727/739 — InputArgInt/InputArgFloat/InputArgRef
    /// each pin `type = 'i'/'f'/'r'` at construction.
    pub fn record_input_arg(&mut self, tp: Type) -> OpRef {
        assert!(
            self.ops.is_empty(),
            "input args must be registered before any operations"
        );
        let index = self.inputargs.len() as u32;
        self.inputargs.push(InputArg::from_type(tp, index));
        let opref = match tp {
            Type::Int => OpRef::input_arg_int(self.op_count),
            Type::Float => OpRef::input_arg_float(self.op_count),
            Type::Ref => OpRef::input_arg_ref(self.op_count),
            Type::Void => panic!("input args cannot be Void"),
        };
        self.op_count += 1;
        self.box_count += 1;
        opref
    }

    /// Record a regular (non-guard) operation.
    /// Returns the OpRef for this operation's result.
    ///
    /// `AbstractResOp` + IntOp/FloatOp/RefOp mixins (resoperation.py:564-638)
    /// pin the result type at construction. Void-result ops keep the
    /// `AbstractResOp.type = 'v'` default (resoperation.py:260).
    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        assert!(!opcode.is_guard(), "use record_guard for guard operations");
        let opref = OpRef::op_typed(self.op_count, opcode.result_type());
        let mut op = Op::new(opcode, args);
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        if opcode.result_type() != Type::Void {
            self.box_count += 1;
        }
        opref
    }

    /// Record an operation with a descriptor (e.g., field access, call).
    /// Returns the OpRef for this operation's result.
    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        assert!(!opcode.is_guard(), "use record_guard for guard operations");
        let opref = OpRef::op_typed(self.op_count, opcode.result_type());
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        if opcode.result_type() != Type::Void {
            self.box_count += 1;
        }
        opref
    }

    /// Record a guard operation.
    /// `pyjitpl.py:2548 generate_guard()` parity: tracer-stage guards
    /// carry `descr=None`. The optimizer creates the FailDescr later in
    /// `store_final_boxes_in_guard` / `invent_fail_descr_for_op`
    /// (compile.py:722-730 / 924-942). Tests that need a pre-stamped
    /// descr pass `Some(descr)`.
    /// Returns the OpRef for this guard.
    pub fn record_guard(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
    ) -> OpRef {
        assert!(opcode.is_guard(), "opcode {:?} is not a guard", opcode);
        let opref = OpRef::op_typed(self.op_count, opcode.result_type());
        let mut op = match descr {
            Some(d) => Op::with_descr(opcode, args, d),
            None => Op::new(opcode, args),
        };
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        if opcode.result_type() != Type::Void {
            self.box_count += 1;
        }
        opref
    }

    /// Record a guard with explicit fail_args — values stored in the dead
    /// frame on guard failure. Mirrors rpython setfailargs().
    /// See `record_guard` for the descr=None convention.
    pub fn record_guard_with_fail_args(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
        fail_args: &[OpRef],
    ) -> OpRef {
        assert!(opcode.is_guard(), "opcode {:?} is not a guard", opcode);
        let opref = OpRef::op_typed(self.op_count, opcode.result_type());
        let mut op = match descr {
            Some(d) => Op::with_descr(opcode, args, d),
            None => Op::new(opcode, args),
        };
        op.pos = opref;
        op.fail_args = Some(smallvec::SmallVec::from_slice(fail_args));
        self.ops.push(op);
        self.op_count += 1;
        if opcode.result_type() != Type::Void {
            self.box_count += 1;
        }
        opref
    }

    /// Set rd_resume_position on the last recorded op.
    /// Called after record_guard* to associate a snapshot.
    pub fn set_last_op_resume_position(&mut self, snapshot_id: i32) {
        if let Some(op) = self.ops.last_mut() {
            op.rd_resume_position = snapshot_id;
        }
    }

    /// Set fail_args on a recorded op identified by `opref`.
    ///
    /// Mirrors RPython's `Op.setfailargs([...])` (`resoperation.py`)
    /// — the post-record mutation API used by tests and any path that
    /// constructs a synthetic guard shape outside the standard
    /// `capture_resumedata` flow.  Production guard recording uses the
    /// snapshot path (`record_guard_typed` + `capture_resumedata` +
    /// `set_last_op_resume_position`); `op.fail_args` is then derived
    /// from the snapshot via `op.store_final_boxes(liveboxes)` in the
    /// optimizer's `store_final_boxes_in_guard`.
    pub fn set_op_fail_args(&mut self, opref: OpRef, fail_args: &[OpRef]) {
        let op = self
            .ops
            .iter_mut()
            .rev()
            .find(|op| op.pos == opref)
            .unwrap_or_else(|| panic!("set_op_fail_args: no op with pos {:?}", opref));
        op.fail_args = Some(smallvec::SmallVec::from_slice(fail_args));
    }

    /// Set `fail_arg_types` on the last recorded op. Used by
    /// `trace_ctx::record_guard_typed` to record types for fail args
    /// without stamping a tracer-stage descr (codex #3 /
    /// pyjitpl.py:2548 generate_guard parity).
    pub fn set_last_op_fail_arg_types(&mut self, types: Vec<Type>) {
        if let Some(op) = self.ops.last_mut() {
            op.fail_arg_types = Some(types);
        }
    }

    /// Close the loop: add a JUMP operation back to the start.
    /// `jump_args` are the values of the input arguments at the end of the loop.
    pub fn close_loop(&mut self, jump_args: &[OpRef]) {
        self.close_loop_with_descr(jump_args, None);
    }

    /// Close the loop with an explicit JUMP descriptor.
    ///
    /// RPython pyjitpl.py:3188-3190 records the tentative JUMP with
    /// `descr=ptoken` before compile_trace(). Plain loop recording keeps
    /// `descr=None` until optimization rewrites it.
    pub fn close_loop_with_descr(&mut self, jump_args: &[OpRef], descr: Option<DescrRef>) {
        // RPython parity: Jump args may differ from InputArgs count when
        // virtualizable arrays change depth. The optimizer (OptUnroll preamble
        // peeling) bridges the gap by creating a Label with the extended count.
        let opref = OpRef::op_typed(self.op_count, OpCode::Jump.result_type());
        let mut op = match descr {
            Some(descr) => Op::with_descr(OpCode::Jump, jump_args, descr),
            None => Op::new(OpCode::Jump, jump_args),
        };
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        if OpCode::Jump.result_type() != Type::Void {
            self.box_count += 1;
        }
    }

    /// Finish the trace (non-looping): add a FINISH operation.
    /// `finish_args` are the values returned from the trace.
    pub fn finish(&mut self, finish_args: &[OpRef], descr: DescrRef) {
        let opref = OpRef::op_typed(self.op_count, OpCode::Finish.result_type());
        let mut op = Op::with_descr(OpCode::Finish, finish_args, descr);
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        if OpCode::Finish.result_type() != Type::Void {
            self.box_count += 1;
        }
    }

    /// Return the completed trace.
    /// The recorder is consumed; no further operations can be recorded.
    ///
    /// Snapshots (pyre-only Vec<Snapshot> side table) live on `TraceCtx`;
    /// callers that need them assemble via `TreeLoop::with_snapshots`
    /// directly — see `TraceCtx::into_tree_loop`.
    pub fn get_trace(self) -> TreeLoop {
        TreeLoop::new(self.inputargs, self.ops)
    }

    /// opencoder.py:567-568 `cut_point()` — the recorder's local slice of
    /// the 5-tuple. `snapshot_data_len` / `snapshot_array_data_len` come
    /// from `TraceCtx` (which owns the pyre-only Vec<Snapshot> side
    /// table); callers should use `TraceCtx::get_trace_position` for a
    /// fully-populated position.
    pub fn get_position(&self) -> TracePosition {
        TracePosition {
            _pos: self.ops.len(),
            _count: self.op_count,
            _index: self.box_count,
            snapshot_data_len: 0,
            snapshot_array_data_len: 0,
        }
    }

    /// opencoder.py:570-575 `cut_at(end)` — restore the recorder to a
    /// previously saved position.
    ///
    /// Discards all operations recorded after `pos`. Used to undo a
    /// tentative JUMP after compile_trace succeeds or fails
    /// (pyjitpl.py:3195 finally: `self.history.cut(cut_at)`).
    pub fn cut(&mut self, pos: TracePosition) {
        self.ops.truncate(pos._pos);
        self.op_count = pos._count;
        self.box_count = pos._index;
    }

    /// history.py:725 `length`: number of non-inputarg ops recorded so far.
    /// Compared against `warmstate.trace_limit` by
    /// `MetaInterp.blackhole_if_trace_too_long` (pyjitpl.py:2791).
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Number of input arguments registered.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Input argument types in loop-header order.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.inputargs.iter().map(|arg| arg.tp).collect()
    }

    /// Number of guards recorded so far.
    pub fn num_guards(&self) -> usize {
        self.ops.iter().filter(|op| op.opcode.is_guard()).count()
    }

    /// Access the recorded operations.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Access the recorded input arguments.
    pub fn inputargs(&self) -> &[InputArg] {
        &self.inputargs
    }

    /// Get an operation by its OpRef position.
    pub fn get_op_by_pos(&self, pos: OpRef) -> Option<&Op> {
        self.ops.iter().find(|op| op.pos == pos)
    }

    /// Get an operation by its raw u32 position, ignoring the variant
    /// tag. Use this when iterating a numeric position range without
    /// knowing each op's RPython `box.type` upfront — the typed lookup
    /// `get_op_by_pos` requires the variant to match (variant-aware Eq).
    pub fn get_op_by_raw_pos(&self, raw: u32) -> Option<&Op> {
        self.ops.iter().find(|op| op.pos.raw() == raw)
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{DescrRef, FailDescr, Type};
    use std::sync::Arc;

    fn iarg(pos: u32) -> OpRef {
        OpRef::input_arg_int(pos)
    }

    fn farg(pos: u32) -> OpRef {
        OpRef::input_arg_float(pos)
    }

    fn rarg(pos: u32) -> OpRef {
        OpRef::input_arg_ref(pos)
    }

    fn iop(pos: u32) -> OpRef {
        OpRef::int_op(pos)
    }

    fn vop(pos: u32) -> OpRef {
        OpRef::void_op(pos)
    }

    /// A minimal FailDescr implementation for testing.
    #[derive(Debug)]
    struct TestFailDescr {
        index: u32,
    }

    impl majit_ir::Descr for TestFailDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
            Some(self)
        }
    }

    impl FailDescr for TestFailDescr {
        fn fail_index(&self) -> u32 {
            self.index
        }
        fn fail_arg_types(&self) -> &[Type] {
            &[]
        }
    }

    fn make_fail_descr(index: u32) -> DescrRef {
        Arc::new(TestFailDescr { index })
    }

    #[test]
    fn test_record_simple_loop() {
        // Trace: i0 -> i1 = int_add(i0, i0) -> jump(i1)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(i0, iarg(0));

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(i1, iop(1));

        rec.close_loop(&[i1]);

        let trace = rec.get_trace();
        assert!(trace.is_loop());
        assert_eq!(trace.num_inputargs(), 1);
        assert_eq!(trace.num_ops(), 2); // IntAdd + Jump
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::Jump);
        assert_eq!(trace.ops[1].args[0], i1);
    }

    #[test]
    fn test_record_with_guard() {
        // i0 -> guard_true(i0) -> i1 = int_add(i0, i0) -> jump(i1)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        let _g = rec.record_guard(OpCode::GuardTrue, &[i0], Some(descr));

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_ops(), 3); // GuardTrue + IntAdd + Jump
        assert!(trace.ops[0].opcode.is_guard());
        assert!(trace.ops[0].descr.is_some());
    }

    #[test]
    fn test_record_finish() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(99);
        rec.finish(&[i1], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());
        assert!(!trace.is_loop());
    }

    #[test]
    fn test_record_multiple_inputargs() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let r0 = rec.record_input_arg(Type::Ref);
        let f0 = rec.record_input_arg(Type::Float);
        assert_eq!(i0, iarg(0));
        assert_eq!(r0, rarg(1));
        assert_eq!(f0, farg(2));
        assert_eq!(rec.num_inputargs(), 3);

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1, r0, f0]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
    }

    #[test]
    fn test_opref_assignment() {
        // OpRef indices are monotonically increasing, with input args taking
        // the first slots.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        assert_eq!(i0, iarg(0));
        assert_eq!(i1, iarg(1));

        let i2 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        assert_eq!(i2, iop(2));

        let descr = make_fail_descr(0);
        let g0 = rec.record_guard(OpCode::GuardTrue, &[i2], Some(descr));
        assert_eq!(g0, vop(3));

        let i3 = rec.record_op(OpCode::IntSub, &[i2, i0]);
        assert_eq!(i3, iop(4));

        rec.close_loop(&[i3, i1]);
        let trace = rec.get_trace();

        // The op's .pos should match
        assert_eq!(trace.ops[0].pos, iop(2)); // IntAdd
        assert_eq!(trace.ops[1].pos, vop(3)); // GuardTrue
        assert_eq!(trace.ops[2].pos, iop(4)); // IntSub
        assert_eq!(trace.ops[3].pos, vop(5)); // Jump
    }

    #[test]
    #[should_panic(expected = "use record_guard")]
    fn test_record_op_with_guard_opcode() {
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        // Should panic: guard opcodes must use record_guard
        rec.record_op(OpCode::GuardTrue, &[iarg(0)]);
    }

    #[test]
    fn test_num_ops_counts_non_inputargs() {
        // history.py:725 length() = trace._count - len(inputargs).
        // In pyre that's ops.len() since inputargs aren't stored in ops.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_ops(), 0);

        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 1);
    }

    #[test]
    fn test_record_op_with_descr() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(42);
        let i1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr);
        assert_eq!(i1, iop(1));

        // Verify the op has a descriptor
        assert!(rec.ops[0].descr.is_some());
    }

    #[test]
    fn test_complex_trace() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let i2 = rec.record_op(OpCode::IntLt, &[i0, i1]);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[i2], Some(descr));

        let i3 = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        rec.close_loop(&[i3, i1]);

        let trace = rec.get_trace();
        assert!(trace.is_loop());
        assert_eq!(trace.num_ops(), 4);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
    }

    // ══════════════════════════════════════════════════════════════════
    // Opencoder parity tests
    // Ported from rpython/jit/metainterp/test/test_opencoder.py
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_simple_iterator() {
        // Parity: test_simple_iterator
        // Record two INT_ADD ops and verify trace structure matches.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add0 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let add1 = rec.record_op(OpCode::IntAdd, &[add0, i0]);

        rec.close_loop(&[add1, i1]);
        let trace = rec.get_trace();

        // Verify the trace has the correct number of ops (2 + Jump).
        assert_eq!(trace.num_ops(), 3);
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[2].opcode, OpCode::Jump);

        // First add uses input args i0, i1.
        assert_eq!(trace.ops[0].args[0], i0);
        assert_eq!(trace.ops[0].args[1], i1);

        // Second add references the result of first add and i0.
        assert_eq!(trace.ops[1].args[0], add0);
        assert_eq!(trace.ops[1].args[1], i0);
    }

    #[test]
    fn test_inputargs_preserved() {
        // Parity: Trace([i0, i1], ...) preserves input args.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_inputargs(), 2);

        rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let add = rec.record_op(OpCode::IntAdd, &[iop(2), i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Int);
    }

    #[test]
    fn test_op_references_chain() {
        // Parity: ops that reference previous ops form correct chains.
        // i0 -> add = int_add(i0, i0) -> sub = int_sub(add, i0) -> jump(sub)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[0].pos, iop(1)); // after 1 inputarg
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[1].args[0], add); // references the add result
        assert_eq!(trace.ops[1].args[1], i0); // references the input arg
    }

    #[test]
    fn test_guard_with_fail_args() {
        // Parity: guards can carry fail_args describing live values at guard.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let descr = make_fail_descr(0);
        let guard =
            rec.record_guard_with_fail_args(OpCode::GuardTrue, &[add], Some(descr), &[i0, i1, add]);

        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        rec.close_loop(&[sub, i1]);

        let trace = rec.get_trace();
        // Find the guard op.
        let guard_op = &trace.ops[1]; // after IntAdd
        assert_eq!(guard_op.pos, guard);
        assert!(guard_op.opcode.is_guard());
        let fail_args = guard_op.fail_args.as_ref().unwrap();
        assert_eq!(fail_args.len(), 3);
        assert_eq!(fail_args[0], i0);
        assert_eq!(fail_args[1], i1);
        assert_eq!(fail_args[2], add);
    }

    #[test]
    fn test_multiple_guards() {
        // Parity: multiple guards in one trace.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr0 = make_fail_descr(0);
        rec.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], Some(descr0), &[i0, i1]);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        let descr1 = make_fail_descr(1);
        rec.record_guard_with_fail_args(OpCode::GuardFalse, &[add], Some(descr1), &[i0, add]);

        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        rec.close_loop(&[sub, i1]);

        let trace = rec.get_trace();
        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
        assert_eq!(guards[1].opcode, OpCode::GuardFalse);

        // First guard's fail_args
        let fa0 = guards[0].fail_args.as_ref().unwrap();
        assert_eq!(fa0.len(), 2);
        assert_eq!(fa0[0], i0);
        assert_eq!(fa0[1], i1);

        // Second guard's fail_args
        let fa1 = guards[1].fail_args.as_ref().unwrap();
        assert_eq!(fa1.len(), 2);
        assert_eq!(fa1[0], i0);
        assert_eq!(fa1[1], add);
    }

    #[test]
    fn test_close_loop_jump_targets_inputargs() {
        // Parity: close_loop produces a JUMP whose args correspond to inputargs.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        let jump = trace.ops.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(jump.args.len(), trace.num_inputargs());
        assert_eq!(jump.args[0], add);
        assert_eq!(jump.args[1], i1);
    }

    #[test]
    fn test_finish_produces_finish_op() {
        // Parity: finish() produces a FINISH op with the given args.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(42);
        rec.finish(&[add], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());
        assert!(!trace.is_loop());

        let finish_op = trace.ops.last().unwrap();
        assert_eq!(finish_op.opcode, OpCode::Finish);
        assert_eq!(finish_op.args.len(), 1);
        assert_eq!(finish_op.args[0], add);
        assert!(finish_op.descr.is_some());
    }

    #[test]
    fn test_trace_length_tracking() {
        // Parity: num_ops() tracks the count accurately.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_ops(), 0);

        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 1);

        rec.record_op(OpCode::IntSub, &[iop(1), i0]);
        assert_eq!(rec.num_ops(), 2);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[iop(2)], Some(descr));
        assert_eq!(rec.num_ops(), 3);

        rec.close_loop(&[iop(2)]);
        // After close_loop, Jump is added.
        assert_eq!(rec.num_ops(), 4);
    }

    #[test]
    fn test_trace_position_splits_count_and_index_for_void_ops() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let before_guard = rec.get_position();
        assert_eq!(before_guard._count, 2);
        assert_eq!(before_guard._index, 2);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[i1], Some(descr));
        rec.close_loop(&[i1]);
        rec.finish(&[i1], make_fail_descr(1));

        let pos = rec.get_position();
        assert_eq!(pos._count, 5);
        assert_eq!(pos._index, 2);
    }

    #[test]
    fn test_with_num_inputs_creates_inputargs() {
        // Parity: Trace::with_num_inputs pre-creates input args.
        let rec = Trace::with_num_inputs(3);
        assert_eq!(rec.num_inputargs(), 3);
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn test_with_num_inputs_oprefs() {
        // Input args from with_num_inputs get InputArgInt(0), InputArgInt(1), ...
        let mut rec = Trace::with_num_inputs(3);

        // The input args consumed positions 0..2, so next op gets BoxInt(3).
        let add = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        assert_eq!(add, iop(3));

        rec.close_loop(&[iarg(0), iarg(1), iarg(2)]);
        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.ops[0].pos, iop(3));
    }

    #[test]
    fn test_guard_descr_preserved() {
        // Parity: guard descriptors are preserved through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(77);
        rec.record_guard(OpCode::GuardNoException, &[], Some(descr));

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();
        let guard = &trace.ops[0];
        assert!(guard.descr.is_some());
        let d = guard.descr.as_ref().unwrap();
        assert_eq!(d.index(), 77);
    }

    #[test]
    fn test_op_with_descr_preserved() {
        // Parity: op descriptors (e.g., for calls) are preserved.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let call_descr = make_fail_descr(55);
        let result = rec.record_op_with_descr(OpCode::CallI, &[i0], call_descr);

        rec.close_loop(&[result]);
        let trace = rec.get_trace();
        let call_op = &trace.ops[0];
        assert_eq!(call_op.opcode, OpCode::CallI);
        assert!(call_op.descr.is_some());
        assert_eq!(call_op.descr.as_ref().unwrap().index(), 55);
    }

    #[test]
    fn test_empty_fail_args() {
        // Parity: guard with empty fail_args is valid.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        rec.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], Some(descr), &[]);

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();
        let guard = &trace.ops[0];
        let fail_args = guard.fail_args.as_ref().unwrap();
        assert!(fail_args.is_empty());
    }

    #[test]
    #[should_panic(expected = "opcode")]
    fn test_record_guard_with_non_guard_opcode() {
        // Parity: record_guard rejects non-guard opcodes.
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::IntAdd, &[iarg(0)], Some(descr));
    }

    #[test]
    #[should_panic(expected = "input args must be registered before any operations")]
    fn test_inputarg_after_ops() {
        // Parity: input args must come before any operations.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        // This should panic.
        rec.record_input_arg(Type::Int);
    }

    // ══════════════════════════════════════════════════════════════════
    // History / TreeLoop parity tests
    // Ported from rpython/jit/metainterp/test/test_history.py
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_has_inputargs_ops_structure() {
        // Parity: TreeLoop has inputargs and operations.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub, i1]);
        let trace = rec.get_trace();

        // inputargs
        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Int);

        // ops: IntAdd, IntSub, Jump
        assert_eq!(trace.num_ops(), 3);
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_trace_guards_have_fail_args() {
        // Parity: guards in a trace carry fail_args.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let cmp = rec.record_op(OpCode::IntLt, &[i0, i1]);
        let descr = make_fail_descr(0);
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
        // Parity: can iterate over all ops.
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
        // Parity: traces can have mixed-type inputargs (int, ref, float).
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
        // Parity: each op's .pos matches the OpRef returned by record_op.
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

    // ══════════════════════════════════════════════════════════════════
    // Opencoder breadth tests — deeper parity with test_opencoder.py
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_recorder_const_int_via_constant_oprefs() {
        // Constants live in a dedicated pool and keep stable OpRefs.
        // Recording ops that reference constants should preserve them.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        // Simulate a pooled constant reference.
        let const_ref = OpRef::from_const(0);
        let add = rec.record_op(OpCode::IntAdd, &[i0, const_ref]);

        rec.close_loop(&[add]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].args[1], const_ref);
        assert!(trace.ops[0].args[1].is_constant());
    }

    #[test]
    fn test_recorder_const_deduplication_by_opref() {
        // Two ops referencing the same constant OpRef share the same value.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let const_ref = OpRef::from_const(1);
        let add1 = rec.record_op(OpCode::IntAdd, &[i0, const_ref]);
        let add2 = rec.record_op(OpCode::IntAdd, &[add1, const_ref]);

        rec.close_loop(&[add2]);
        let trace = rec.get_trace();

        // Both ops reference the same constant
        assert_eq!(trace.ops[0].args[1], trace.ops[1].args[1]);
        assert_eq!(trace.ops[0].args[1], const_ref);
    }

    #[test]
    fn test_recorder_descriptors_preserved_on_ops() {
        // Descriptors on non-guard ops should survive through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr1 = make_fail_descr(10);
        let descr2 = make_fail_descr(20);

        let call1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr1);
        let call2 = rec.record_op_with_descr(OpCode::CallI, &[i1], descr2);

        rec.close_loop(&[call1, call2]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].descr.as_ref().unwrap().index(), 10);
        assert_eq!(trace.ops[1].descr.as_ref().unwrap().index(), 20);
    }

    #[test]
    fn test_recorder_100_plus_ops_stress() {
        // Recording 200 ops should work without issue.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut prev = i0;
        for _ in 0..200 {
            prev = rec.record_op(OpCode::IntAdd, &[prev, i0]);
        }
        assert_eq!(rec.num_ops(), 200);

        rec.close_loop(&[prev]);
        let trace = rec.get_trace();
        // 200 IntAdd + 1 Jump
        assert_eq!(trace.num_ops(), 201);
        assert!(trace.is_loop());

        // Verify chain: each op references the previous op's result.
        // i0 = OpRef::from_raw(0), first IntAdd = OpRef::from_raw(1), second = OpRef::from_raw(2), ...
        for (i, op) in trace.ops[..200].iter().enumerate() {
            assert_eq!(op.opcode, OpCode::IntAdd);
            if i > 0 {
                // The previous IntAdd produced BoxInt(i) (offset by inputarg).
                assert_eq!(op.args[0], iop(i as u32));
            }
        }
    }

    #[test]
    fn test_recorder_mixed_type_input_args() {
        // Mixed-type inputs (Int, Float, Ref) produce correct types.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let f0 = rec.record_input_arg(Type::Float);
        let r0 = rec.record_input_arg(Type::Ref);
        let i1 = rec.record_input_arg(Type::Int);

        assert_eq!(i0, iarg(0));
        assert_eq!(f0, farg(1));
        assert_eq!(r0, rarg(2));
        assert_eq!(i1, iarg(3));

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, f0, r0, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Float);
        assert_eq!(trace.inputargs[2].tp, Type::Ref);
        assert_eq!(trace.inputargs[3].tp, Type::Int);
    }

    #[test]
    fn test_recorder_guard_with_many_fail_args() {
        // Guard with 10+ fail_args.
        let mut rec = Trace::new();
        let mut inputs = Vec::new();
        for _ in 0..12 {
            inputs.push(rec.record_input_arg(Type::Int));
        }

        // Record some ops
        let add = rec.record_op(OpCode::IntAdd, &[inputs[0], inputs[1]]);

        // Guard with all 12 inputs + 1 computed value as fail_args (13 total)
        let mut fail_args: Vec<OpRef> = inputs.clone();
        fail_args.push(add);

        let descr = make_fail_descr(0);
        let guard =
            rec.record_guard_with_fail_args(OpCode::GuardTrue, &[add], Some(descr), &fail_args);

        rec.close_loop(&inputs);
        let trace = rec.get_trace();

        // Find the guard
        let guard_op = trace.iter_guards().next().unwrap();
        assert_eq!(guard_op.pos, guard);
        let fa = guard_op.fail_args.as_ref().unwrap();
        assert_eq!(fa.len(), 13);

        // Verify all fail_args match what we specified
        for (i, &expected) in fail_args.iter().enumerate() {
            assert_eq!(fa[i], expected);
        }
    }

    #[test]
    fn test_recorder_guard_descr_index_survives() {
        // Each guard's descriptor index must survive through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let d0 = make_fail_descr(100);
        let d1 = make_fail_descr(200);
        let d2 = make_fail_descr(300);

        rec.record_guard(OpCode::GuardTrue, &[i0], Some(d0));
        rec.record_guard(OpCode::GuardFalse, &[i0], Some(d1));
        rec.record_guard(OpCode::GuardNoException, &[], Some(d2));

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 3);
        assert_eq!(guards[0].descr.as_ref().unwrap().index(), 100);
        assert_eq!(guards[1].descr.as_ref().unwrap().index(), 200);
        assert_eq!(guards[2].descr.as_ref().unwrap().index(), 300);
    }

    #[test]
    fn test_recorder_ops_interleaved_with_guards() {
        // Interleaved ops and guards: verify ordering is correct.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let d0 = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[add], Some(d0));
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        let d1 = make_fail_descr(1);
        rec.record_guard(OpCode::GuardFalse, &[sub], Some(d1));
        let mul = rec.record_op(OpCode::IntMul, &[sub, i1]);

        rec.close_loop(&[mul, i1]);
        let trace = rec.get_trace();

        let expected = vec![
            OpCode::IntAdd,
            OpCode::GuardTrue,
            OpCode::IntSub,
            OpCode::GuardFalse,
            OpCode::IntMul,
            OpCode::Jump,
        ];
        let actual: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_recorder_finish_with_descr() {
        // finish() records a FINISH op with the given descriptor.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(999);
        rec.finish(&[add], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());

        let finish = trace.ops.last().unwrap();
        assert_eq!(finish.opcode, OpCode::Finish);
        assert_eq!(finish.descr.as_ref().unwrap().index(), 999);
    }

    #[test]
    fn test_recorder_no_ops_just_inputargs_and_jump() {
        // Minimal trace: input args directly jump back.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        rec.close_loop(&[i0, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.num_ops(), 1); // Only Jump
        assert_eq!(trace.ops[0].opcode, OpCode::Jump);
        assert_eq!(trace.ops[0].args[0], i0);
        assert_eq!(trace.ops[0].args[1], i1);
    }

    #[test]
    fn test_get_position_and_cut() {
        let mut rec = Trace::with_num_inputs(2);
        let pos0 = rec.get_position();
        assert_eq!(pos0._pos, 0);
        assert_eq!(pos0._count, 2); // 2 inputargs
        assert_eq!(pos0._index, 2);
        assert_eq!(pos0.snapshot_data_len, 0);
        assert_eq!(pos0.snapshot_array_data_len, 0);

        let _a = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        let pos1 = rec.get_position();
        assert_eq!(pos1._pos, 1);
        assert_eq!(pos1._count, 3);
        assert_eq!(pos1._index, 3);

        let _b = rec.record_op(OpCode::IntSub, &[iarg(0), iarg(1)]);
        let _c = rec.record_op(OpCode::IntMul, &[iarg(0), iarg(1)]);
        assert_eq!(rec.num_ops(), 3);

        // Cut back to pos1 — should discard IntSub and IntMul
        rec.cut(pos1);
        assert_eq!(rec.num_ops(), 1);
        assert_eq!(rec.get_position(), pos1);

        // Can record more ops after cut
        let d = rec.record_op(OpCode::IntNeg, &[iarg(0)]);
        assert_eq!(d, iop(3)); // continues from pos1._count
        assert_eq!(rec.num_ops(), 2);

        // Cut back to pos0 — should discard everything
        rec.cut(pos0);
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn test_get_position_tracks_count_and_index_separately() {
        let mut rec = Trace::with_num_inputs(1);
        let i0 = iarg(0);

        let _add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let pos_after_add = rec.get_position();
        assert_eq!(pos_after_add._count, 2);
        assert_eq!(pos_after_add._index, 2);

        let descr = make_fail_descr(1);
        rec.record_guard(OpCode::GuardTrue, &[iop(1)], Some(descr));
        let pos_after_guard = rec.get_position();
        assert_eq!(pos_after_guard._pos, 2);
        assert_eq!(pos_after_guard._count, 3);
        assert_eq!(pos_after_guard._index, 2);
    }
}
