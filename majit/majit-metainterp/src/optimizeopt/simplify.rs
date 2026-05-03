/// Late-stage simplification pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/simplify.py.
///
/// This is the simplest optimization pass. It runs after the other passes
/// and cleans up operations that are no longer needed:
///
/// 1. Converts CALL_PURE_* to the corresponding CALL_* (purity has already
///    been exploited by OptPure).
/// 2. Converts CALL_LOOPINVARIANT_* to the corresponding CALL_*.
/// 3. Removes hint operations (RECORD_EXACT_CLASS, RECORD_EXACT_VALUE_*,
///    RECORD_KNOWN_RESULT, VIRTUAL_REF_FINISH, QUASIIMMUT_FIELD,
///    ASSERT_NOT_NONE) that were consumed by earlier passes.
/// 4. Rewrites VIRTUAL_REF to SAME_AS_R (the virtualisation is resolved).
use majit_ir::{Op, OpCode};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

pub struct OptSimplify;

impl OptSimplify {
    pub fn new() -> Self {
        OptSimplify
    }

    /// Convert a CALL_PURE_* or CALL_LOOPINVARIANT_* to the corresponding CALL_*.
    fn rewrite_call(op: &Op) -> Op {
        let new_opcode = OpCode::call_for_type(op.result_type());
        let mut new_op = Op::new(new_opcode, &op.args);
        new_op.descr = op.descr.clone();
        new_op.pos = op.pos;
        new_op
    }
}

impl Default for OptSimplify {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptSimplify {
    fn propagate_forward(&mut self, op: &Op, _ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            // CALL_PURE_* -> CALL_*
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN => {
                OptimizationResult::Emit(Self::rewrite_call(op))
            }

            // CALL_LOOPINVARIANT_* -> CALL_*
            OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN => OptimizationResult::Emit(Self::rewrite_call(op)),

            // VIRTUAL_REF -> SAME_AS_R (just forward the first arg)
            OpCode::VirtualRefR => {
                let mut new_op = Op::new(OpCode::SameAsR, &[op.arg(0)]);
                new_op.pos = op.pos;
                OptimizationResult::Emit(new_op)
            }

            // simplify.py: GUARD_FUTURE_CONDITION — record in patchguardop,
            // then remove. Unroll uses patchguardop to attach resume data to
            // extra guards from short preamble.
            OpCode::GuardFutureCondition => {
                _ctx.patchguardop = Some(op.clone());
                OptimizationResult::Remove
            }

            // Hint operations that are simply removed
            OpCode::VirtualRefFinish
            | OpCode::QuasiimmutField
            | OpCode::AssertNotNone
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueR
            | OpCode::RecordExactValueI
            | OpCode::RecordKnownResult => OptimizationResult::Remove,

            _ => OptimizationResult::PassOn,
        }
    }

    fn name(&self) -> &'static str {
        "simplify"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::OpRef;

    fn run_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(OptSimplify::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024)
    }

    #[test]
    fn test_call_pure_to_call() {
        for (pure_op, expected_op) in [
            (OpCode::CallPureI, OpCode::CallI),
            (OpCode::CallPureR, OpCode::CallR),
            (OpCode::CallPureF, OpCode::CallF),
            (OpCode::CallPureN, OpCode::CallN),
        ] {
            let ops = vec![Op::new(pure_op, &[OpRef::from_raw(0), OpRef::from_raw(1)])];
            let result = run_pass(&ops);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].opcode, expected_op);
            assert_eq!(
                result[0].args.as_slice(),
                &[OpRef::from_raw(0), OpRef::from_raw(1)]
            );
        }
    }

    #[test]
    fn test_call_loopinvariant_to_call() {
        for (loopinv_op, expected_op) in [
            (OpCode::CallLoopinvariantI, OpCode::CallI),
            (OpCode::CallLoopinvariantR, OpCode::CallR),
            (OpCode::CallLoopinvariantF, OpCode::CallF),
            (OpCode::CallLoopinvariantN, OpCode::CallN),
        ] {
            let ops = vec![Op::new(loopinv_op, &[OpRef::from_raw(0)])];
            let result = run_pass(&ops);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].opcode, expected_op);
        }
    }

    #[test]
    fn test_virtual_ref_to_same_as() {
        let ops = vec![Op::new(
            OpCode::VirtualRefR,
            &[OpRef::from_raw(0), OpRef::from_raw(1)],
        )];
        let result = run_pass(&ops);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SameAsR);
        assert_eq!(result[0].args.as_slice(), &[OpRef::from_raw(0)]);
    }

    #[test]
    fn test_removed_ops() {
        let removed_opcodes = [
            OpCode::VirtualRefFinish,
            OpCode::QuasiimmutField,
            OpCode::AssertNotNone,
            OpCode::RecordExactClass,
            OpCode::RecordExactValueR,
            OpCode::RecordExactValueI,
            OpCode::RecordKnownResult,
        ];
        for opcode in removed_opcodes {
            let arity = opcode.arity().unwrap_or(0) as usize;
            let args: Vec<OpRef> = (0..arity).map(|i| OpRef::from_raw(i as u32)).collect();
            let ops = vec![Op::new(opcode, &args)];
            let result = run_pass(&ops);
            assert!(result.is_empty(), "{:?} should be removed", opcode);
        }
    }

    #[test]
    fn test_passthrough() {
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
        ];
        let result = run_pass(&ops);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
    }

    #[test]
    fn test_preserves_args_on_call_rewrite() {
        let ops = vec![Op::new(
            OpCode::CallPureI,
            &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
        )];
        let result = run_pass(&ops);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallI);
        assert_eq!(
            result[0].args.as_slice(),
            &[OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)]
        );
    }

    #[test]
    fn test_mixed_ops() {
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::CallPureI, &[OpRef::from_raw(0)]),
            Op::new(
                OpCode::RecordExactClass,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
            ),
            Op::new(OpCode::IntSub, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
        ];
        let result = run_pass(&ops);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::CallI);
        assert_eq!(result[2].opcode, OpCode::IntSub);
    }

    #[test]
    fn test_guard_future_condition_removed() {
        let ops = vec![
            Op::new(OpCode::GuardFutureCondition, &[]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = run_pass(&ops);
        // GUARD_FUTURE_CONDITION should be removed
        assert!(
            !result
                .iter()
                .any(|o| o.opcode == OpCode::GuardFutureCondition)
        );
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_assert_not_none_removed() {
        let ops = vec![
            Op::new(OpCode::AssertNotNone, &[OpRef::from_raw(100)]),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = run_pass(&ops);
        assert!(!result.iter().any(|o| o.opcode == OpCode::AssertNotNone));
    }

    #[test]
    fn test_virtual_ref_r_to_same_as() {
        let ops = vec![
            Op::new(
                OpCode::VirtualRefR,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = run_pass(&ops);
        assert!(result.iter().any(|o| o.opcode == OpCode::SameAsR));
    }

    #[test]
    fn test_record_exact_class_removed() {
        let ops = vec![
            Op::new(
                OpCode::RecordExactClass,
                &[OpRef::from_raw(100), OpRef::from_raw(200)],
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = run_pass(&ops);
        assert!(!result.iter().any(|o| o.opcode == OpCode::RecordExactClass));
    }
}
