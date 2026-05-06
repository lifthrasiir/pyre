/// earlyforce.py: OptEarlyForce — force virtual args before escaping.
///
/// RPython earlyforce.py forces ALL arguments of most operations to ensure
/// virtual objects are materialized before they can escape. Exempt ops:
/// SETFIELD_GC, SETARRAYITEM_GC, SETARRAYITEM_RAW, QUASIIMMUT_FIELD,
/// SAME_AS_*, raw_free calls.
///
/// earlyforce.py:32: self.optimizer.optearlyforce = self
/// The pass registers itself so force_at_the_end_of_preamble can route
/// forced operations starting from earlyforce.next (= heap).
use majit_ir::{Op, OpCode};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

pub struct OptEarlyForce;

impl OptEarlyForce {
    pub fn new() -> Self {
        OptEarlyForce
    }

    /// earlyforce.py:7-11: is_raw_free check.
    /// Raw free calls should not force their arguments.
    fn is_raw_free(op: &Op) -> bool {
        if !op.opcode.is_call() {
            return false;
        }
        if let Some(ref descr) = op.descr {
            if let Some(cd) = descr.as_call_descr() {
                let ei = cd.get_extra_info();
                return ei.oopspecindex == majit_ir::OopSpecIndex::RawFree;
            }
        }
        false
    }

    /// earlyforce.py:15-29: should we force args for this op?
    /// RPython exempt set: SETFIELD_GC, SETARRAYITEM_GC, SETARRAYITEM_RAW,
    /// QUASIIMMUT_FIELD, SAME_AS_I/R/F, and raw_free. Note that
    /// SETFIELD_RAW is NOT exempt in RPython.
    fn should_force_args(op: &Op) -> bool {
        !matches!(
            op.opcode,
            OpCode::SetfieldGc
                | OpCode::SetarrayitemGc
                | OpCode::SetarrayitemRaw
                | OpCode::QuasiimmutField
                | OpCode::SameAsI
                | OpCode::SameAsR
                | OpCode::SameAsF
        ) && !Self::is_raw_free(op)
    }
}

impl Default for OptEarlyForce {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptEarlyForce {
    /// earlyforce.py:15-29: propagate_forward.
    /// Force all virtual args of non-exempt operations, then emit.
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if Self::should_force_args(op) {
            // earlyforce.py:28: self.optimizer.force_box(arg, self)
            // In Rust, we can't call Optimizer.force_box (borrow conflict).
            // Instead, force directly through PtrInfo.force_box_impl,
            // which uses ctx.current_pass_idx (== earlyforce_idx) for
            // emit_extra routing. This matches RPython's optforce=self.
            for i in 0..op.num_args() {
                let arg = ctx.get_box_replacement(op.arg(i));
                if ctx.get_ptr_info(arg).is_some_and(|info| info.is_virtual()) {
                    // optimizer.py:345-364: force_box path.
                    // potential_extra_ops are handled by Optimizer.force_box,
                    // but earlyforce only needs the virtual materialization.
                    if let Some(tracked) = ctx.take_potential_extra_op(arg) {
                        if let Some(builder) = ctx.active_short_preamble_producer_mut() {
                            builder.add_preamble_op_from_pop(&tracked, arg);
                        } else if let Some(builder) = ctx.imported_short_preamble_builder.as_mut() {
                            builder.add_preamble_op_from_pop(&tracked, arg);
                        }
                    }
                    let mut info = ctx.take_ptr_info(arg).unwrap();
                    let _forced = info.force_box(arg, ctx);
                }
            }
        }
        // earlyforce.py:29: return self.emit(op)
        OptimizationResult::PassOn
    }

    fn name(&self) -> &'static str {
        "earlyforce"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::OpRef;

    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            let pos = i as u32;
            op.pos = OpRef::op_typed(pos, op.result_type());
        }
    }

    #[test]
    fn test_earlyforce_resolves_call_may_force_args() {
        let mut ops = vec![Op::new(
            OpCode::CallMayForceN,
            &[OpRef::int_op(100), OpRef::int_op(101)],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallMayForceN);
    }

    #[test]
    fn test_earlyforce_passthrough_non_call() {
        let mut ops = vec![Op::new(
            OpCode::IntAdd,
            &[OpRef::int_op(100), OpRef::int_op(101)],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_earlyforce_call_assembler_handled() {
        let mut ops = vec![Op::new(OpCode::CallAssemblerI, &[OpRef::int_op(100)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallAssemblerI);
    }

    #[test]
    fn test_earlyforce_guard_not_forced() {
        // GUARD_NOT_FORCED should have its fail_args resolved.
        let mut guard = Op::new(OpCode::GuardNotForced, &[]);
        guard.fail_args = Some(Default::default());
        let mut ops = vec![guard];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNotForced);
        assert!(result[0].fail_args.is_some());
    }

    #[test]
    fn test_earlyforce_all_call_may_force_types() {
        for opcode in [
            OpCode::CallMayForceI,
            OpCode::CallMayForceR,
            OpCode::CallMayForceF,
            OpCode::CallMayForceN,
        ] {
            let mut ops = vec![Op::new(opcode, &[OpRef::int_op(100)])];
            assign_positions(&mut ops);

            let mut opt = Optimizer::new();
            opt.add_pass(Box::new(OptEarlyForce::new()));
            let result = opt.optimize_with_constants_and_inputs(
                &ops,
                &mut std::collections::HashMap::new(),
                1024,
            );
            assert_eq!(result.len(), 1, "{opcode:?} should be handled");
        }
    }

    #[test]
    fn test_earlyforce_exempt_setfield() {
        // SETFIELD_GC should NOT force args (earlyforce.py:18)
        let mut ops = vec![Op::new(
            OpCode::SetfieldGc,
            &[OpRef::int_op(100), OpRef::int_op(101)],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
    }
}
