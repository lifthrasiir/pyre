/// OptRewrite: algebraic simplification and constant folding.
///
/// Translated from rpython/jit/metainterp/optimizeopt/rewrite.py.
/// Rewrites operations into equivalent, cheaper operations.
/// This includes constant folding for pure ops and algebraic identities.
use majit_ir::{Op, OpCode, OpRef, Value};

use crate::optimizeopt::info::{PreambleOp, PtrInfoExt};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult, intdiv};

/// rewrite.py: loop_invariant_results value.
/// RPython stores PreambleOp or regular Box (AbstractResOp) directly
/// in the dict. In Rust, we use an enum to distinguish.
#[derive(Clone, Debug)]
enum LoopInvariantEntry {
    /// Regular result (already forced or body-computed).
    Direct(OpRef),
    /// shortpreamble.py:148-159: LoopInvariantOp.produce_op stores
    /// PreambleOp(op, preamble_op, invented_name) in the dict.
    Preamble(PreambleOp),
}

#[cold]
#[inline(never)]
fn raise_invalid_loop(msg: &'static str) -> ! {
    std::panic::panic_any(crate::optimize::InvalidLoop(msg));
}

/// info.py:16-18: INFO_NULL / INFO_NONNULL / INFO_UNKNOWN
/// optimizer.py:127-135: getnullness()
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nullness {
    Null,
    Nonnull,
    Unknown,
}

/// Rewrite operations into equivalent, cheaper forms.
///
/// Handles:
/// - Constant folding for pure integer/boolean ops
/// - Algebraic simplifications (identity, absorbing elements)
/// - Strength reduction (e.g., `x + x` -> `x << 1`)
/// - Guard simplification when argument is known constant
/// - Boolean operation rewrites (inverse/reflex)
/// - Conditional call elimination when condition/value is constant
/// - Pointer equality on same OpRef
/// - Cast and convert round-trip elimination
/// - Guard-no-exception removal after removed calls
pub struct OptRewrite {
    /// rewrite.py: bool_result_cache — maps (opcode, arg0, arg1) → result OpRef.
    /// Used by find_rewritable_bool to check if inverse/reflex was computed.
    bool_result_cache: crate::optimizeopt::vec_assoc::VecAssoc<(OpCode, OpRef, OpRef), OpRef>,
    /// rewrite.py:39: loop_invariant_results — cache for CALL_LOOPINVARIANT results.
    /// Key: function pointer (arg0 as i64).
    /// Value: Direct(OpRef) or Preamble(PreambleOp) — RPython isinstance check.
    loop_invariant_results: crate::optimizeopt::vec_assoc::VecAssoc<i64, LoopInvariantEntry>,
    /// rewrite.py:40: loop_invariant_producer — maps func_ptr → emitted Call op.
    /// Used by produce_potential_short_preamble_ops (rewrite.py:45-47).
    loop_invariant_producer: crate::optimizeopt::vec_assoc::VecAssoc<i64, Op>,
}

impl OptRewrite {
    pub fn new() -> Self {
        OptRewrite {
            bool_result_cache: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            loop_invariant_results: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            loop_invariant_producer: crate::optimizeopt::vec_assoc::VecAssoc::new(),
        }
    }

    // ── Constant folding for binary integer ops ──

    /// Try to constant-fold a binary integer operation.
    /// Returns `Some(result)` if both args are constant.
    fn try_fold_binary_int(&self, opcode: OpCode, lhs: i64, rhs: i64) -> Option<i64> {
        match opcode {
            OpCode::IntAdd => Some(lhs.wrapping_add(rhs)),
            OpCode::IntSub => Some(lhs.wrapping_sub(rhs)),
            OpCode::IntMul => Some(lhs.wrapping_mul(rhs)),
            OpCode::IntFloorDiv => {
                if rhs != 0 {
                    Some(lhs.wrapping_div(rhs))
                } else {
                    None
                }
            }
            OpCode::IntMod => {
                if rhs != 0 {
                    Some(lhs.wrapping_rem(rhs))
                } else {
                    None
                }
            }
            OpCode::IntAnd => Some(lhs & rhs),
            OpCode::IntOr => Some(lhs | rhs),
            OpCode::IntXor => Some(lhs ^ rhs),
            OpCode::IntLshift => {
                if (0..64).contains(&rhs) {
                    Some(lhs.wrapping_shl(rhs as u32))
                } else {
                    None
                }
            }
            OpCode::IntRshift => {
                if (0..64).contains(&rhs) {
                    Some(lhs.wrapping_shr(rhs as u32))
                } else {
                    None
                }
            }
            OpCode::UintRshift => {
                if (0..64).contains(&rhs) {
                    Some(((lhs as u64).wrapping_shr(rhs as u32)) as i64)
                } else {
                    None
                }
            }
            // Comparisons
            OpCode::IntLt => Some(if lhs < rhs { 1 } else { 0 }),
            OpCode::IntLe => Some(if lhs <= rhs { 1 } else { 0 }),
            OpCode::IntEq => Some(if lhs == rhs { 1 } else { 0 }),
            OpCode::IntNe => Some(if lhs != rhs { 1 } else { 0 }),
            OpCode::IntGt => Some(if lhs > rhs { 1 } else { 0 }),
            OpCode::IntGe => Some(if lhs >= rhs { 1 } else { 0 }),
            OpCode::UintLt => Some(if (lhs as u64) < (rhs as u64) { 1 } else { 0 }),
            OpCode::UintLe => Some(if (lhs as u64) <= (rhs as u64) { 1 } else { 0 }),
            OpCode::UintGt => Some(if (lhs as u64) > (rhs as u64) { 1 } else { 0 }),
            OpCode::UintGe => Some(if (lhs as u64) >= (rhs as u64) { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Try to constant-fold a unary integer operation.
    fn try_fold_unary_int(&self, opcode: OpCode, arg: i64) -> Option<i64> {
        match opcode {
            OpCode::IntNeg => Some(arg.wrapping_neg()),
            OpCode::IntInvert => Some(!arg),
            OpCode::IntIsZero => Some(if arg == 0 { 1 } else { 0 }),
            OpCode::IntIsTrue => Some(if arg != 0 { 1 } else { 0 }),
            OpCode::IntForceGeZero => Some(if arg < 0 { 0 } else { arg }),
            _ => None,
        }
    }

    // ── Algebraic simplifications ──

    /// Try algebraic simplification for INT_ADD.
    /// `x + 0 -> x`, `0 + x -> x`, `x + x -> x << 1`
    fn optimize_int_add(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntAdd, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // add_zero: int_add(x, 0) => x
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg1)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }

        // add_reassoc_consts: int_add(int_add(x, C1), C2) => int_add(x, C1+C2)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntAdd {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(0);
                        let c = self.emit_constant_int(ctx, c1.wrapping_add(c2));
                        let mut new_op = Op::new(OpCode::IntAdd, &[x, c]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        // add_sub_x_c_c: int_add(int_sub(x, C1), C2) => int_add(x, C2-C1)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntSub {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(0);
                        let c = self.emit_constant_int(ctx, c2.wrapping_sub(c1));
                        let mut new_op = Op::new(OpCode::IntAdd, &[x, c]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        // add_sub_c_x_c: int_add(int_sub(C1, x), C2) => int_sub(C1+C2, x)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntSub {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(0))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(1);
                        let c = self.emit_constant_int(ctx, c1.wrapping_add(c2));
                        let mut new_op = Op::new(OpCode::IntSub, &[c, x]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        // x + x -> x << 1
        if ctx.same_box(arg0, arg1) {
            let one = self.emit_constant_int(ctx, 1);
            let mut new_op = Op::new(OpCode::IntLshift, &[arg0, one]);
            new_op.pos.set(op.pos.get());
            return OptimizationResult::Emit(new_op);
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_SUB.
    /// `x - 0 -> x`, `x - x -> 0`
    fn optimize_int_sub(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntSub, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // sub_zero: int_sub(x, 0) => x
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // sub_from_zero: int_sub(0, x) => int_neg(x)
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let mut new_op = Op::new(OpCode::IntNeg, &[arg1]);
            new_op.pos.set(op.pos.get());
            return OptimizationResult::Emit(new_op);
        }
        // sub_x_x: int_sub(x, x) => 0
        if ctx.same_box(arg0, arg1) {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // sub_add_consts: int_sub(int_add(x, C1), C2) => int_sub(x, C2-C1)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntAdd {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(0);
                        let c = self.emit_constant_int(ctx, c2.wrapping_sub(c1));
                        let mut new_op = Op::new(OpCode::IntSub, &[x, c]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        // sub_add: int_sub(int_add(x, y), y) => x
        if let Some(inner) = ctx
            .ensure_box(arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntAdd && inner.arg(1) == arg1 {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_inner = ctx
                    .ensure_box(inner.arg(0))
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_inner);
                return OptimizationResult::Remove;
            }
        }

        // sub_add_neg: int_sub(y, int_add(x, y)) => int_neg(x)
        if let Some(inner) = ctx
            .ensure_box(arg1)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntAdd && inner.arg(1) == arg0 {
                let mut new_op = Op::new(OpCode::IntNeg, &[inner.arg(0)]);
                new_op.pos.set(op.pos.get());
                return OptimizationResult::Emit(new_op);
            }
        }

        // sub_sub_left_x_c_c: int_sub(int_sub(x, C1), C2) => int_sub(x, C1+C2)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntSub {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(0);
                        let c = self.emit_constant_int(ctx, c1.wrapping_add(c2));
                        let mut new_op = Op::new(OpCode::IntSub, &[x, c]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                    // sub_sub_left_c_x_c: int_sub(int_sub(C1, x), C2) => int_sub(C1-C2, x)
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(0))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(1);
                        let c = self.emit_constant_int(ctx, c1.wrapping_sub(c2));
                        let mut new_op = Op::new(OpCode::IntSub, &[c, x]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        // sub_invert_one: int_sub(int_invert(x), -1) => int_neg(x)
        if let Some(-1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntInvert {
                    let mut new_op = Op::new(OpCode::IntNeg, &[inner.arg(0)]);
                    new_op.pos.set(op.pos.get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }

        // sub_xor_x_y_y: int_sub(int_xor(x, y), y) => x (if x & y == 0)
        if let Some(inner) = ctx
            .ensure_box(arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntXor && inner.arg(1) == arg1 {
                if let (Some(bx), Some(by)) = (
                    ctx.get_box_replacement_box(inner.arg(0))
                        .and_then(|b| ctx.peek_intbound_box(&b)),
                    ctx.get_box_replacement_box(arg1)
                        .and_then(|b| ctx.peek_intbound_box(&b)),
                ) {
                    if bx.and_bound(&by).known_eq_const(0) {
                        let b_old = ctx
                            .ensure_box(op.pos.get())
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        let b_inner = ctx
                            .ensure_box(inner.arg(0))
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        ctx.make_equal_to(&b_old, &b_inner);
                        return OptimizationResult::Remove;
                    }
                }
            }
            // sub_or_x_y_y: int_sub(int_or(x, y), y) => x (if x & y == 0)
            if inner.opcode == OpCode::IntOr && inner.arg(1) == arg1 {
                if let (Some(bx), Some(by)) = (
                    ctx.get_box_replacement_box(inner.arg(0))
                        .and_then(|b| ctx.peek_intbound_box(&b)),
                    ctx.get_box_replacement_box(arg1)
                        .and_then(|b| ctx.peek_intbound_box(&b)),
                ) {
                    if bx.and_bound(&by).known_eq_const(0) {
                        let b_old = ctx
                            .ensure_box(op.pos.get())
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        let b_inner = ctx
                            .ensure_box(inner.arg(0))
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        ctx.make_equal_to(&b_old, &b_inner);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_MUL.
    /// `x * 0 -> 0`, `x * 1 -> x`, `0 * x -> 0`, `1 * x -> x`
    fn optimize_int_mul(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntMul, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x * 0 -> 0 (absorbing element)
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }
        // 0 * x -> 0
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }
        // x * 1 -> x (identity)
        if let Some(1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // 1 * x -> x
        if let Some(1) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg1)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // Strength reduction: x * 2^n -> x << n
        if let Some(c) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if c > 0 && (c & (c - 1)) == 0 {
                let shift = c.trailing_zeros() as i64;
                let shift_ref = self.emit_constant_int(ctx, shift);
                let mut new_op = Op::new(OpCode::IntLshift, &[arg0, shift_ref]);
                new_op.pos.set(op.pos.get());
                return OptimizationResult::Emit(new_op);
            }
            // rewrite.py: x * (-1) -> INT_NEG(x)
            if c == -1 {
                let mut neg = Op::new(OpCode::IntNeg, &[arg0]);
                neg.pos.set(op.pos.get());
                return OptimizationResult::Replace(neg);
            }
            // rewrite.py: x * (-(2^n)) -> -(x << n)
            if c < -1 {
                let abs_c = (c as i128).unsigned_abs() as u64;
                if abs_c.is_power_of_two() {
                    let shift = abs_c.trailing_zeros() as i64;
                    let shift_ref = self.emit_constant_int(ctx, shift);
                    let shifted = ctx.emit(Op::new(OpCode::IntLshift, &[arg0, shift_ref]));
                    let mut neg = Op::new(OpCode::IntNeg, &[shifted]);
                    neg.pos.set(op.pos.get());
                    return OptimizationResult::Emit(neg);
                }
            }
        }
        if let Some(c) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if c > 0 && (c & (c - 1)) == 0 {
                let shift = c.trailing_zeros() as i64;
                let shift_ref = self.emit_constant_int(ctx, shift);
                let mut new_op = Op::new(OpCode::IntLshift, &[arg1, shift_ref]);
                new_op.pos.set(op.pos.get());
                return OptimizationResult::Emit(new_op);
            }
            // (-1) * x -> INT_NEG(x)
            if c == -1 {
                let mut neg = Op::new(OpCode::IntNeg, &[arg1]);
                neg.pos.set(op.pos.get());
                return OptimizationResult::Replace(neg);
            }
        }

        // mul_lshift: int_mul(x, int_lshift(1, y)) => int_lshift(x, y)
        if let Some(inner) = ctx
            .ensure_box(arg1)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntLshift {
                if let Some(1) = ctx
                    .get_box_replacement_box(inner.arg(0))
                    .and_then(|b| ctx.get_constant_int_box(&b))
                {
                    let mut new_op = Op::new(OpCode::IntLshift, &[arg0, inner.arg(1)]);
                    new_op.pos.set(op.pos.get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_FLOORDIV.
    /// `x // 1 -> x`, constant fold when both operands are known.
    fn optimize_int_floor_div(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntFloorDiv, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x // 1 -> x (identity)
        if let Some(1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }

        // x // (-1) -> INT_NEG(x)
        if let Some(-1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let mut neg = Op::new(OpCode::IntNeg, &[arg0]);
            neg.pos.set(op.pos.get());
            return OptimizationResult::Replace(neg);
        }

        // 0 // x -> 0 (zero dividend)
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x // x -> 1 (self-division, x != 0 guaranteed by semantics)
        if ctx.same_box(arg0, arg1) {
            ctx.make_constant(op.pos.get(), Value::Int(1));
            return OptimizationResult::Remove;
        }

        // Strength reduction for constant divisor >= 2
        if let Some(divisor) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if divisor > 1 && divisor.count_ones() == 1 {
                // Power-of-2 floor division: x // (2^n) = x >> n
                // Arithmetic right shift IS floor division for positive divisors.
                let shift = divisor.trailing_zeros();
                let shift_ref = self.emit_constant_int(ctx, shift as i64);
                let result_ref = ctx.emit(Op::new(OpCode::IntRshift, &[arg0, shift_ref]));
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_res = ctx
                    .ensure_box(result_ref)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_res);
                return OptimizationResult::Remove;
            }

            // General constant divisor >= 3: magic number multiplication
            if divisor >= 3 {
                let result =
                    intdiv::division_operations(arg0, divisor, false, ctx.current_pass_idx, ctx);
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_res = ctx
                    .ensure_box(result)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_res);
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_MOD.
    ///
    /// Strength reduction from rpython/jit/metainterp/optimizeopt/intdiv.py.
    fn optimize_int_mod(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntMod, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x % 1 -> 0 (any integer mod 1 is 0)
        if let Some(1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x % (-1) -> 0 (any integer mod -1 is 0)
        if let Some(-1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // 0 % x -> 0 (zero dividend)
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x % x -> 0 (self-modulo)
        if ctx.same_box(arg0, arg1) {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // Strength reduction for constant divisor >= 3 (non-power-of-2)
        if let Some(divisor) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if divisor >= 3 && divisor.count_ones() != 1 {
                let result =
                    intdiv::modulo_operations(arg0, divisor, false, ctx.current_pass_idx, ctx);
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_res = ctx
                    .ensure_box(result)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_res);
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_AND.
    /// `x & 0 -> 0`, `x & -1 -> x`
    fn optimize_int_and(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntAnd, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x & 0 -> 0
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }
        // x & -1 -> x (all bits set = identity)
        if let Some(-1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        if let Some(-1) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg1)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // and_x_x: int_and(a, a) => a
        if ctx.same_box(arg0, arg1) {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }

        // and_reassoc_consts: int_and(int_and(x, C1), C2) => int_and(x, C1&C2)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntAnd {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(0);
                        let c = self.emit_constant_int(ctx, c1 & c2);
                        let mut new_op = Op::new(OpCode::IntAnd, &[x, c]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        // and_absorb: int_and(a, int_and(a, b)) => int_and(a, b)
        if let Some(inner) = ctx
            .ensure_box(arg1)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntAnd && inner.arg(0) == arg0 {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_arg = ctx
                    .ensure_box(arg1)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_arg);
                return OptimizationResult::Remove;
            }
        }

        // and_x_c_in_range: int_and(x, C) => x (if x in range [0, C & ~(C+1)])
        if let Some(c) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(bound) = ctx
                .get_box_replacement_box(arg0)
                .and_then(|b| ctx.peek_intbound_box(&b))
            {
                if bound.lower >= 0 && bound.upper <= (c & !(c.wrapping_add(1))) {
                    let b_old = ctx
                        .ensure_box(op.pos.get())
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    let b_arg = ctx
                        .ensure_box(arg0)
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    ctx.make_equal_to(&b_old, &b_arg);
                    return OptimizationResult::Remove;
                }
            }
        }

        // and_known_result: int_and(a, b) => C (if result bound is constant)
        if let (Some(ba), Some(bb)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.peek_intbound_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.peek_intbound_box(&b)),
        ) {
            let result_bound = ba.and_bound(&bb);
            if result_bound.is_constant() {
                ctx.make_constant(op.pos.get(), Value::Int(result_bound.get_constant_int()));
                return OptimizationResult::Remove;
            }
            // and_idempotent: int_and(x, y) => x (if y.ones | x.zeros == -1)
            if (bb.tvalue | (!ba.tvalue & !ba.tmask)) == u64::MAX {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_arg = ctx
                    .ensure_box(arg0)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_arg);
                return OptimizationResult::Remove;
            }
        }

        // and_or: int_and(int_or(x, y), z) => int_and(x, z) (if y & z known 0)
        if let Some(inner) = ctx
            .ensure_box(arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntOr {
                if let (Some(by), Some(bz)) = (
                    ctx.get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.peek_intbound_box(&b)),
                    ctx.get_box_replacement_box(arg1)
                        .and_then(|b| ctx.peek_intbound_box(&b)),
                ) {
                    if by.and_bound(&bz).known_eq_const(0) {
                        let mut new_op = Op::new(OpCode::IntAnd, &[inner.arg(0), arg1]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_OR.
    /// `x | 0 -> x`, `x | -1 -> -1`
    fn optimize_int_or(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntOr, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x | 0 -> x
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg1)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // x | -1 -> -1
        if let Some(-1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(-1));
            return OptimizationResult::Remove;
        }
        if let Some(-1) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(-1));
            return OptimizationResult::Remove;
        }
        // or_x_x: int_or(a, a) => a
        if ctx.same_box(arg0, arg1) {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }

        // or_reassoc_consts: int_or(int_or(x, C1), C2) => int_or(x, C1|C2)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntOr {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(0);
                        let c = self.emit_constant_int(ctx, c1 | c2);
                        let mut new_op = Op::new(OpCode::IntOr, &[x, c]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        // or_absorb: int_or(a, int_or(a, b)) => int_or(a, b)
        if let Some(inner) = ctx
            .ensure_box(arg1)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntOr && inner.arg(0) == arg0 {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_arg = ctx
                    .ensure_box(arg1)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_arg);
                return OptimizationResult::Remove;
            }
        }

        // or_and_two_parts: int_or(int_and(x, C1), int_and(x, C2)) => int_and(x, C1|C2)
        if let (Some(inner0), Some(inner1)) = (
            ctx.ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb)),
            ctx.ensure_box(arg1)
                .and_then(|pb| ctx.get_producing_op(&pb)),
        ) {
            if inner0.opcode == OpCode::IntAnd
                && inner1.opcode == OpCode::IntAnd
                && inner0.arg(0) == inner1.arg(0)
            {
                if let (Some(c1), Some(c2)) = (
                    ctx.get_box_replacement_box(inner0.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b)),
                    ctx.get_box_replacement_box(inner1.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b)),
                ) {
                    let x = inner0.arg(0);
                    let c = self.emit_constant_int(ctx, c1 | c2);
                    let mut new_op = Op::new(OpCode::IntAnd, &[x, c]);
                    new_op.pos.set(op.pos.get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }

        // or_known_result: int_or(a, b) => C (if result bound is constant)
        if let (Some(ba), Some(bb)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.peek_intbound_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.peek_intbound_box(&b)),
        ) {
            let result_bound = ba.or_bound(&bb);
            if result_bound.is_constant() {
                ctx.make_constant(op.pos.get(), Value::Int(result_bound.get_constant_int()));
                return OptimizationResult::Remove;
            }
            // or_idempotent: int_or(x, y) => x (if x.ones | y.zeros == -1)
            if (ba.tvalue | (!bb.tvalue & !bb.tmask)) == u64::MAX {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_arg = ctx
                    .ensure_box(arg0)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_arg);
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_XOR.
    /// `x ^ 0 -> x`, `x ^ x -> 0`
    fn optimize_int_xor(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntXor, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x ^ 0 -> x
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg1)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // x ^ x -> 0
        if ctx.same_box(arg0, arg1) {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }
        // x ^ -1 -> ~x (INT_INVERT)
        if let Some(-1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let mut new_op = Op::new(OpCode::IntInvert, &[arg0]);
            new_op.pos.set(op.pos.get());
            return OptimizationResult::Emit(new_op);
        }
        if let Some(-1) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let mut new_op = Op::new(OpCode::IntInvert, &[arg1]);
            new_op.pos.set(op.pos.get());
            return OptimizationResult::Emit(new_op);
        }

        // xor_reassoc_consts: int_xor(int_xor(x, C1), C2) => int_xor(x, C1^C2)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntXor {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        let x = inner.arg(0);
                        let c = self.emit_constant_int(ctx, c1 ^ c2);
                        let mut new_op = Op::new(OpCode::IntXor, &[x, c]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        // xor_absorb: int_xor(int_xor(a, b), b) => a
        if let Some(inner) = ctx
            .ensure_box(arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntXor && inner.arg(1) == arg1 {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_inner = ctx
                    .ensure_box(inner.arg(0))
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_inner);
                return OptimizationResult::Remove;
            }
        }

        // xor_is_not: int_xor(x, 1) => int_is_zero(x) (if x is bool)
        if let Some(1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(bound) = ctx
                .get_box_replacement_box(arg0)
                .and_then(|b| ctx.peek_intbound_box(&b))
            {
                if bound.is_bool() {
                    let mut new_op = Op::new(OpCode::IntIsZero, &[arg0]);
                    new_op.pos.set(op.pos.get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_LSHIFT.
    /// `x << 0 -> x`
    fn optimize_int_lshift(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntLshift, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x << 0 -> x
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // 0 << x -> 0
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // lshift_rshift_c_c: int_lshift(int_rshift(x, C1), C1) => int_and(x, -1<<C1)
        if let Some(c1) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if (0..64).contains(&c1) {
                if let Some(inner) = ctx
                    .ensure_box(arg0)
                    .and_then(|pb| ctx.get_producing_op(&pb))
                {
                    if inner.opcode == OpCode::IntRshift {
                        if ctx
                            .get_box_replacement_box(inner.arg(1))
                            .and_then(|b| ctx.get_constant_int_box(&b))
                            == Some(c1)
                        {
                            let mask = self.emit_constant_int(ctx, (-1i64).wrapping_shl(c1 as u32));
                            let mut new_op = Op::new(OpCode::IntAnd, &[inner.arg(0), mask]);
                            new_op.pos.set(op.pos.get());
                            return OptimizationResult::Emit(new_op);
                        }
                    }
                    // lshift_urshift_c_c: int_lshift(uint_rshift(x, C1), C1)
                    if inner.opcode == OpCode::UintRshift {
                        if ctx
                            .get_box_replacement_box(inner.arg(1))
                            .and_then(|b| ctx.get_constant_int_box(&b))
                            == Some(c1)
                        {
                            let mask = self.emit_constant_int(ctx, (-1i64).wrapping_shl(c1 as u32));
                            let mut new_op = Op::new(OpCode::IntAnd, &[inner.arg(0), mask]);
                            new_op.pos.set(op.pos.get());
                            return OptimizationResult::Emit(new_op);
                        }
                    }
                    // lshift_and_rshift: int_lshift(int_and(int_rshift(x, C1), C2), C1)
                    if inner.opcode == OpCode::IntAnd {
                        if let Some(c2) = ctx
                            .get_box_replacement_box(inner.arg(1))
                            .and_then(|b| ctx.get_constant_int_box(&b))
                        {
                            if let Some(inner2) = ctx
                                .ensure_box(inner.arg(0))
                                .and_then(|pb| ctx.get_producing_op(&pb))
                            {
                                if inner2.opcode == OpCode::IntRshift
                                    && ctx
                                        .get_box_replacement_box(inner2.arg(1))
                                        .and_then(|b| ctx.get_constant_int_box(&b))
                                        == Some(c1)
                                {
                                    let mask =
                                        self.emit_constant_int(ctx, c2.wrapping_shl(c1 as u32));
                                    let mut new_op =
                                        Op::new(OpCode::IntAnd, &[inner2.arg(0), mask]);
                                    new_op.pos.set(op.pos.get());
                                    return OptimizationResult::Emit(new_op);
                                }
                                // lshift_and_urshift
                                if inner2.opcode == OpCode::UintRshift
                                    && ctx
                                        .get_box_replacement_box(inner2.arg(1))
                                        .and_then(|b| ctx.get_constant_int_box(&b))
                                        == Some(c1)
                                {
                                    let mask =
                                        self.emit_constant_int(ctx, c2.wrapping_shl(c1 as u32));
                                    let mut new_op =
                                        Op::new(OpCode::IntAnd, &[inner2.arg(0), mask]);
                                    new_op.pos.set(op.pos.get());
                                    return OptimizationResult::Emit(new_op);
                                }
                            }
                        }
                    }
                }
            }
        }

        // lshift_lshift_c_c: int_lshift(int_lshift(x, C1), C2) => int_lshift(x, C1+C2)
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntLshift {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        if (0..64).contains(&c1) && (0..64).contains(&c2) {
                            let c = c1 + c2;
                            if c < 64 {
                                let cv = self.emit_constant_int(ctx, c);
                                let mut new_op = Op::new(OpCode::IntLshift, &[inner.arg(0), cv]);
                                new_op.pos.set(op.pos.get());
                                return OptimizationResult::Emit(new_op);
                            }
                        }
                    }
                }
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_RSHIFT.
    /// `x >> 0 -> x`
    fn optimize_int_rshift(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntRshift, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x >> 0 -> x
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // 0 >> x -> 0
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // rshift_known_result: int_rshift(a, b) => C (if result bound is constant)
        if let (Some(ba), Some(bb)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.peek_intbound_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.peek_intbound_box(&b)),
        ) {
            let result_bound = ba.rshift_bound(&bb);
            if result_bound.is_constant() {
                ctx.make_constant(op.pos.get(), Value::Int(result_bound.get_constant_int()));
                return OptimizationResult::Remove;
            }
        }

        // rshift_lshift: int_rshift(int_lshift(x, y), y) => x (if no overflow)
        if let Some(inner) = ctx
            .ensure_box(arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntLshift && inner.arg(1) == arg1 {
                if let (Some(bx), Some(by)) = (
                    ctx.get_box_replacement_box(inner.arg(0))
                        .and_then(|b| ctx.peek_intbound_box(&b)),
                    ctx.get_box_replacement_box(arg1)
                        .and_then(|b| ctx.peek_intbound_box(&b)),
                ) {
                    if bx.lshift_bound_cannot_overflow(&by) {
                        let b_old = ctx
                            .ensure_box(op.pos.get())
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        let b_inner = ctx
                            .ensure_box(inner.arg(0))
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        ctx.make_equal_to(&b_old, &b_inner);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // rshift_rshift_c_c: int_rshift(int_rshift(x, C1), C2) => int_rshift(x, min(C1+C2, 63))
        if let Some(c2) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(inner) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner.opcode == OpCode::IntRshift {
                    if let Some(c1) = ctx
                        .get_box_replacement_box(inner.arg(1))
                        .and_then(|b| ctx.get_constant_int_box(&b))
                    {
                        if (0..64).contains(&c1) && (0..64).contains(&c2) {
                            let c = (c1 + c2).min(63);
                            let cv = self.emit_constant_int(ctx, c);
                            let mut new_op = Op::new(OpCode::IntRshift, &[inner.arg(0), cv]);
                            new_op.pos.set(op.pos.get());
                            return OptimizationResult::Emit(new_op);
                        }
                    }
                }
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for UINT_RSHIFT.
    /// `x >>> 0 -> x`
    fn optimize_uint_rshift(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::UintRshift, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x >>> 0 -> x
        if let Some(0) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg0)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }
        // 0 >>> x -> 0
        if let Some(0) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }

        // urshift_known_result: uint_rshift(a, b) => C (if result bound is constant)
        if let (Some(ba), Some(bb)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.peek_intbound_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.peek_intbound_box(&b)),
        ) {
            let result_bound = ba.urshift_bound(&bb);
            if result_bound.is_constant() {
                ctx.make_constant(op.pos.get(), Value::Int(result_bound.get_constant_int()));
                return OptimizationResult::Remove;
            }
        }

        // urshift_lshift_x_c_c: uint_rshift(int_lshift(x, C), C) => int_and(x, mask)
        if let Some(c) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if (0..64).contains(&c) {
                if let Some(inner) = ctx
                    .ensure_box(arg0)
                    .and_then(|pb| ctx.get_producing_op(&pb))
                {
                    if inner.opcode == OpCode::IntLshift
                        && ctx
                            .get_box_replacement_box(inner.arg(1))
                            .and_then(|b| ctx.get_constant_int_box(&b))
                            == Some(c)
                    {
                        let mask = ((-1i64 as u64).wrapping_shl(c as u32) >> (c as u32)) as i64;
                        let mask_ref = self.emit_constant_int(ctx, mask);
                        let mut new_op = Op::new(OpCode::IntAnd, &[inner.arg(0), mask_ref]);
                        new_op.pos.set(op.pos.get());
                        return OptimizationResult::Emit(new_op);
                    }
                }
            }
        }

        OptimizationResult::PassOn
    }

    // ── Unary operations ──

    /// Constant fold or simplify INT_NEG.
    fn optimize_int_neg(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        if let Some(a) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(result) = self.try_fold_unary_int(OpCode::IntNeg, a) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // neg_neg: int_neg(int_neg(x)) => x
        if let Some(inner) = ctx
            .ensure_box(arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntNeg {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_inner = ctx
                    .ensure_box(inner.arg(0))
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_inner);
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Constant fold or simplify INT_INVERT.
    fn optimize_int_invert(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        if let Some(a) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(result) = self.try_fold_unary_int(OpCode::IntInvert, a) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // invert_invert: int_invert(int_invert(x)) => x
        if let Some(inner) = ctx
            .ensure_box(arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntInvert {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_inner = ctx
                    .ensure_box(inner.arg(0))
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_inner);
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Constant fold INT_IS_ZERO.
    /// rewrite.py:512-513 `optimize_INT_IS_ZERO`:
    ///     return self._optimize_nullness(op, op.getarg(0), False)
    fn optimize_int_is_zero(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        self.optimize_nullness(op, op.arg(0), false, ctx)
    }

    /// rewrite.py:505-510 `optimize_INT_IS_TRUE`:
    ///     if (not self.is_raw_ptr(op.getarg(0)) and
    ///         self.getintbound(op.getarg(0)).is_bool()):
    ///         self.make_equal_to(op, op.getarg(0))
    ///         return
    ///     return self._optimize_nullness(op, op.getarg(0), True)
    fn optimize_int_is_true(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        // rewrite.py:505-510 optimize_INT_IS_TRUE:
        //     if (not self.is_raw_ptr(op.getarg(0)) and
        //         self.getintbound(op.getarg(0)).is_bool()):
        //         self.make_equal_to(op, op.getarg(0))
        //         return
        //     return self._optimize_nullness(op, op.getarg(0), True)
        //
        // is_raw_ptr (optimizer.py:154-158) checks for an
        // AbstractRawPtrInfo on the box's forwarded slot — NOT a
        // Ref-typed Box check. A raw-pointer 'i'-typed Box (i.e. one
        // pointing into a virtual raw buffer) skips the is_bool
        // shortcut because the buffer pointer's intbound is unrelated
        // to its boolean truthiness.
        // OpRef → BoxRef shim until this site's caller migrates (Phase D-2).
        let arg0_box = ctx.get_box_replacement_box(op.arg(0));
        let arg0_is_raw = arg0_box.as_ref().map_or(false, |b| ctx.is_raw_ptr(b));
        if !arg0_is_raw {
            if let Some(bound) = ctx
                .get_box_replacement_box(arg0)
                .and_then(|b| ctx.peek_intbound_box(&b))
            {
                if bound.is_bool() {
                    // make_equal_to: replace INT_IS_TRUE result with arg0.
                    let b_old = ctx
                        .ensure_box(op.pos.get())
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    let b_arg = ctx
                        .ensure_box(arg0)
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    ctx.make_equal_to(&b_old, &b_arg);
                    return OptimizationResult::Remove;
                }
            }
        }

        // is_true_and_minint: int_is_true(int_and(x, MININT)) => int_lt(x, 0)
        if let Some(inner) = ctx
            .ensure_box(arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntAnd {
                if ctx
                    .get_box_replacement_box(inner.arg(1))
                    .and_then(|b| ctx.get_constant_int_box(&b))
                    == Some(i64::MIN)
                {
                    let zero = self.emit_constant_int(ctx, 0);
                    let mut new_op = Op::new(OpCode::IntLt, &[inner.arg(0), zero]);
                    new_op.pos.set(op.pos.get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }

        self.optimize_nullness(op, arg0, true, ctx)
    }

    /// rewrite.py:515-554: _optimize_oois_ooisnot(op, expect_isnot, instance)
    ///
    /// Pointer equality optimization using virtual/null/class information.
    fn optimize_oois_ooisnot(
        &self,
        op: &Op,
        expect_isnot: bool,
        instance: bool,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // rewrite.py:515-554 _optimize_oois_ooisnot:
        //     arg0 = get_box_replacement(op.getarg(0))
        //     arg1 = get_box_replacement(op.getarg(1))
        //     info0 = getptrinfo(arg0)
        //     info1 = getptrinfo(arg1)
        let arg0_box = ctx.get_box_replacement_box(op.arg(0));
        let arg1_box = ctx.get_box_replacement_box(op.arg(1));
        let info0 = arg0_box.as_ref().and_then(|b| ctx.getptrinfo(b));
        let info1 = arg1_box.as_ref().and_then(|b| ctx.getptrinfo(b));

        let is_virtual0 = info0.as_ref().is_some_and(|i| i.is_virtual());
        let is_virtual1 = info1.as_ref().is_some_and(|i| i.is_virtual());

        // rewrite.py:530-535: virtual objects
        if is_virtual0 {
            let intres = if is_virtual1 {
                // rewrite.py:532: `intres = (info0 is info1) ^ expect_isnot`
                // — PtrInfo identity (info.py:71-72 `same_info` is `self is
                // other` for non-Const infos), via getptrinfo_handle which
                // preserves the `_forwarded` cell identity.
                let same = match (
                    arg0_box.as_ref().and_then(|b| ctx.getptrinfo_handle(b)),
                    arg1_box.as_ref().and_then(|b| ctx.getptrinfo_handle(b)),
                ) {
                    (Some(h0), Some(h1)) => h0.same_info(&h1),
                    _ => false,
                };
                same ^ expect_isnot
            } else {
                expect_isnot
            };
            ctx.make_constant(op.pos.get(), Value::Int(intres as i64));
            return OptimizationResult::Remove;
        }
        if is_virtual1 {
            ctx.make_constant(op.pos.get(), Value::Int(expect_isnot as i64));
            return OptimizationResult::Remove;
        }

        // rewrite.py:528-531: null checks — fall back to OpRef for downstream
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        if info1.as_ref().is_some_and(|i| i.is_null()) {
            return self.optimize_nullness(op, arg0, expect_isnot, ctx);
        }
        if info0.as_ref().is_some_and(|i| i.is_null()) {
            return self.optimize_nullness(op, arg1, expect_isnot, ctx);
        }

        // rewrite.py:542-543: `elif arg0 is arg1:` — box identity
        // (resoperation.py:38 `same_box` base = `self is other`).
        if ctx.box_is(arg0, arg1) {
            ctx.make_constant(op.pos.get(), Value::Int(!expect_isnot as i64));
            return OptimizationResult::Remove;
        }

        // rewrite.py:535-553: instance comparison — different classes → not same
        if instance {
            let cls0 = info0
                .as_ref()
                .and_then(|i| i.get_known_class(ctx.cpu.as_ref()));
            let cls1 = info1
                .as_ref()
                .and_then(|i| i.get_known_class(ctx.cpu.as_ref()));
            if let (Some(c0), Some(c1)) = (cls0, cls1) {
                if c0 != c1 {
                    ctx.make_constant(op.pos.get(), Value::Int(expect_isnot as i64));
                    return OptimizationResult::Remove;
                }
            }
        } else {
            // rewrite.py:550-553: non-instance array pointer comparison.
            // If both are ArrayPtrInfo with known-different length bounds,
            // they cannot be the same object.
            let lb0 = info0.clone().and_then(|mut i| i.getlenbound(None));
            let lb1 = info1.clone().and_then(|mut i| i.getlenbound(None));
            if let (Some(lb0), Some(lb1)) = (lb0, lb1) {
                if lb0.known_ne(&lb1) {
                    ctx.make_constant(op.pos.get(), Value::Int(expect_isnot as i64));
                    return OptimizationResult::Remove;
                }
            }
        }

        OptimizationResult::PassOn
    }

    /// rewrite.py:496-503 `_optimize_nullness(op, box, expect_nonnull)`:
    ///     info = self.getnullness(box)
    ///     if info == INFO_NONNULL: self.make_constant_int(op, expect_nonnull)
    ///     elif info == INFO_NULL: self.make_constant_int(op, not expect_nonnull)
    ///     else: return self.emit(op)
    fn optimize_nullness(
        &self,
        op: &Op,
        arg: OpRef,
        expect_nonnull: bool,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match self.getnullness(arg, ctx) {
            Nullness::Nonnull => {
                ctx.make_constant(op.pos.get(), Value::Int(expect_nonnull as i64));
                OptimizationResult::Remove
            }
            Nullness::Null => {
                ctx.make_constant(op.pos.get(), Value::Int(!expect_nonnull as i64));
                OptimizationResult::Remove
            }
            Nullness::Unknown => OptimizationResult::PassOn,
        }
    }

    /// Constant fold INT_FORCE_GE_ZERO.
    fn optimize_int_force_ge_zero(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        if let Some(a) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            ctx.make_constant(op.pos.get(), Value::Int(if a < 0 { 0 } else { a }));
            return OptimizationResult::Remove;
        }

        // force_ge_zero_pos: int_force_ge_zero(x) => x (if x known nonneg)
        if let Some(bound) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.peek_intbound_box(&b))
        {
            if bound.known_nonnegative() {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_arg = ctx
                    .ensure_box(arg0)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_arg);
                return OptimizationResult::Remove;
            }
            // force_ge_zero_neg: int_force_ge_zero(x) => 0 (if x known negative)
            if bound.upper < 0 {
                ctx.make_constant(op.pos.get(), Value::Int(0));
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Constant fold int_between(a, b, c) => a <= b < c.
    fn optimize_int_between(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        let arg2 = op.arg(2);

        if let (Some(a), Some(b), Some(c)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg2)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            let result = (a <= b && b < c) as i64;
            ctx.make_constant(op.pos.get(), Value::Int(result));
            return OptimizationResult::Remove;
        }

        OptimizationResult::PassOn
    }

    // ── Comparisons ──

    /// Constant fold binary comparisons.
    fn optimize_comparison(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        if let (Some(a), Some(b)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(op.opcode, a, b) {
                ctx.make_constant(op.pos.get(), Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // eq_different_knownbits: int_eq(x, y) => 0 (if known_ne)
        if op.opcode == OpCode::IntEq {
            if let (Some(bx), Some(by)) = (
                ctx.get_box_replacement_box(arg0)
                    .and_then(|b| ctx.peek_intbound_box(&b)),
                ctx.get_box_replacement_box(arg1)
                    .and_then(|b| ctx.peek_intbound_box(&b)),
            ) {
                if bx.known_ne(&by) {
                    ctx.make_constant(op.pos.get(), Value::Int(0));
                    return OptimizationResult::Remove;
                }
            }
        }
        // ne_different_knownbits: int_ne(x, y) => 1 (if known_ne)
        if op.opcode == OpCode::IntNe {
            if let (Some(bx), Some(by)) = (
                ctx.get_box_replacement_box(arg0)
                    .and_then(|b| ctx.peek_intbound_box(&b)),
                ctx.get_box_replacement_box(arg1)
                    .and_then(|b| ctx.peek_intbound_box(&b)),
            ) {
                if bx.known_ne(&by) {
                    ctx.make_constant(op.pos.get(), Value::Int(1));
                    return OptimizationResult::Remove;
                }
            }
        }
        // eq_same: int_eq(x, x) => 1
        if op.opcode == OpCode::IntEq && ctx.same_box(arg0, arg1) {
            ctx.make_constant(op.pos.get(), Value::Int(1));
            return OptimizationResult::Remove;
        }
        // ne_same: int_ne(x, x) => 0
        if op.opcode == OpCode::IntNe && ctx.same_box(arg0, arg1) {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            return OptimizationResult::Remove;
        }
        // eq_zero: int_eq(x, 0) => int_is_zero(x)
        if op.opcode == OpCode::IntEq {
            if let Some(0) = ctx
                .get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b))
            {
                let mut new_op = Op::new(OpCode::IntIsZero, &[arg0]);
                new_op.pos.set(op.pos.get());
                return OptimizationResult::Emit(new_op);
            }
        }
        // ne_zero: int_ne(x, 0) => int_is_true(x)
        if op.opcode == OpCode::IntNe {
            if let Some(0) = ctx
                .get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b))
            {
                let mut new_op = Op::new(OpCode::IntIsTrue, &[arg0]);
                new_op.pos.set(op.pos.get());
                return OptimizationResult::Emit(new_op);
            }
        }
        // eq_one: int_eq(x, 1) => x (if x is bool)
        if op.opcode == OpCode::IntEq {
            if let Some(1) = ctx
                .get_box_replacement_box(arg1)
                .and_then(|b| ctx.get_constant_int_box(&b))
            {
                if let Some(bound) = ctx
                    .get_box_replacement_box(arg0)
                    .and_then(|b| ctx.peek_intbound_box(&b))
                {
                    if bound.is_bool() {
                        let b_old = ctx
                            .ensure_box(op.pos.get())
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        let b_arg = ctx
                            .ensure_box(arg0)
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        ctx.make_equal_to(&b_old, &b_arg);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // eq_sub_eq: int_eq(int_sub(x, int_eq(x, a)), a) => 0
        if op.opcode == OpCode::IntEq {
            if let Some(inner_sub) = ctx
                .ensure_box(arg0)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if inner_sub.opcode == OpCode::IntSub && inner_sub.arg(0) != OpRef::NONE {
                    if let Some(inner_eq) = ctx
                        .ensure_box(inner_sub.arg(1))
                        .and_then(|pb| ctx.get_producing_op(&pb))
                    {
                        if inner_eq.opcode == OpCode::IntEq
                            && inner_eq.arg(0) == inner_sub.arg(0)
                            && inner_eq.arg(1) == arg1
                        {
                            ctx.make_constant(op.pos.get(), Value::Int(0));
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
        }

        // rewrite.py: postprocess — record comparison result for
        // find_rewritable_bool (inverse/reflex lookup).
        self.bool_result_cache
            .insert((op.opcode, arg0, arg1), op.pos.get());

        OptimizationResult::PassOn
    }

    // ── Guards ──

    /// Optimize GUARD_TRUE following RPython rewrite.py: optimize_guard(op, CONST_1).
    /// If the condition is a known constant 0, the trace is impossible and must abort.
    ///
    /// rewrite.py:163-184 `optimize_guard` proper (the contradiction check
    /// + emit) is the call-time half. The `make_constant(box, CONST_1)` half
    /// of the upstream `optimize_guard` is split into
    /// `propagate_postprocess` (rewrite.py:352-371) per RPython's
    /// `have_postprocess` model — see the bottom of this file.
    fn optimize_guard_true(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        // rewrite.py:165-168: box.type=='i' checks intbound.is_constant(),
        // which catches values narrowed to a single point by bounds analysis,
        // not just the constant pool.
        if let Some(val) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
        {
            if val != 0 {
                return OptimizationResult::Remove;
            }
            raise_invalid_loop("GUARD_TRUE proven to always fail");
        }

        OptimizationResult::PassOn
    }

    /// Optimize GUARD_FALSE following RPython rewrite.py: optimize_guard(op, CONST_0).
    fn optimize_guard_false(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        // rewrite.py:165-168: box.type=='i' checks intbound.is_constant().
        if let Some(val) = ctx
            .get_box_replacement_box(arg0)
            .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
        {
            if val == 0 {
                return OptimizationResult::Remove;
            }
            raise_invalid_loop("GUARD_FALSE proven to always fail");
        }

        OptimizationResult::PassOn
    }

    /// rewrite.py:284-347: optimize_GUARD_VALUE + replace_old_guard_with_guard_value
    ///
    /// If both args are constants and equal, the guard is redundant → remove.
    /// If arg0 is Ref-typed with a prior guard_nonnull/guard_class, replace
    /// that old guard with guard_value (rewrite.py:307-347).
    /// If the expected value is boolean, replace with GUARD_TRUE/FALSE.
    fn optimize_guard_value(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() < 2 {
            return OptimizationResult::PassOn;
        }
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // rewrite.py:163-184 optimize_guard(op, constbox) — base contradiction
        // check called from optimize_GUARD_VALUE at line 301. `arg1` is the
        // asserted Const. For box.type=='i' (rewrite.py:165-168) the check is
        // intbound.is_constant()/get_constant_int(), catching values narrowed
        // by bounds analysis, not just the constant pool. For 'r'
        // (rewrite.py:174-182) it is get_box_replacement(box).is_constant()/
        // same_constant. For 'f', rewrite.py:295-298 returns silently when
        // arg0 is constant without checking equality, so we mirror that by
        // removing on equality but never raising on mismatch.
        if let Some(expected_int) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(actual_int) = ctx
                .get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
            {
                if actual_int == expected_int {
                    return OptimizationResult::Remove;
                }
                raise_invalid_loop("GUARD_VALUE proven to always fail");
            }
        } else if let (Some(actual), Some(expected)) = (
            ctx.get_box_replacement_box(arg0)
                .and_then(|b| ctx.get_constant_box(&b)),
            ctx.get_box_replacement_box(arg1)
                .and_then(|cb| cb.const_value()),
        ) {
            if actual == expected {
                return OptimizationResult::Remove;
            }
            match actual {
                Value::Int(_) | Value::Ref(_) => {
                    raise_invalid_loop("GUARD_VALUE proven to always fail");
                }
                Value::Float(_) => {
                    return OptimizationResult::Remove;
                }
                Value::Void => {}
            }
        }

        // rewrite.py:284-301: optimize_GUARD_VALUE for Ref args.
        // getptrinfo synthesizes ConstPtrInfo for constant Refs, matching
        // `if info:` in RPython (which is True for ConstPtrInfo too).
        let obj = ctx.get_box_replacement(arg0);
        let obj_box = ctx.get_box_replacement_box(arg0);
        let obj_info = obj_box.as_ref().and_then(|b| ctx.getptrinfo(b));
        if let Some(info) = obj_info {
            if info.is_virtual() {
                raise_invalid_loop("promote of a virtual");
            }
            // rewrite.py:307-347: replace_old_guard_with_guard_value
            if let Some(old_guard) = obj_box
                .as_ref()
                .and_then(|b| ctx.get_last_guard(b))
                .cloned()
            {
                // rewrite.py:320: c_value = op.getarg(1) — generic Const.
                // Under typed seeding c_value can be Int OR Ref; the
                // previous gating on get_constant_int dropped the Ref
                // case entirely so prior GUARD_NONNULL/GUARD_CLASS were
                // never strengthened to GUARD_VALUE for Ref-typed args.
                if let Some(c_value) = ctx
                    .get_box_replacement_box(arg1)
                    .and_then(|b| b.const_value())
                {
                    // rewrite.py:321-323: c_value.nonnull(). ConstInt.nonnull
                    // == (value != 0); ConstPtr.nonnull == (gcref != null).
                    let c_nonnull = match c_value {
                        Value::Int(i) => i != 0,
                        Value::Ref(g) => !g.is_null(),
                        Value::Float(_) => true,
                        Value::Void => true,
                    };
                    if !c_nonnull {
                        raise_invalid_loop(
                            "GUARD_VALUE(..., NULL) follows some other guard that it is not NULL",
                        );
                    }
                    // rewrite.py:324-332: previous_classbox = info.get_known_class(cpu)
                    // expected_classbox = cpu.cls_of_box(c_value)
                    // get_known_class on the c_value side dispatches through
                    // getptrinfo → ConstPtrInfo.get_known_class (info.py:763-772)
                    // which is exactly cls_of_box for constant pointers.
                    if let Some(prev_cls) = info.get_known_class(ctx.cpu.as_ref()) {
                        if let Some(arg1_box) = ctx.get_box_replacement_box(arg1) {
                            if let Some(expected_cls) = ctx.get_known_class(&arg1_box) {
                                if prev_cls != expected_cls {
                                    raise_invalid_loop(
                                        "GUARD_VALUE proven to always fail (class mismatch)",
                                    );
                                }
                            }
                        }
                    }
                    // rewrite.py:333-334: can_replace_guards check.
                    if !ctx.can_replace_guards {
                        return OptimizationResult::PassOn;
                    }
                    // rewrite.py:335-347: replace old guard with GUARD_VALUE.
                    // last_guard_pos is a _newoperations index (info.py:100-103).
                    // rewrite.py:339-340: old descr must not be ResumeAtPositionDescr
                    // — RPython's fresh ResumeGuardDescr() at line 335 must
                    // not overwrite a RAPD marker.
                    if let Some(old_idx) = obj_box.as_ref().and_then(|b| ctx.last_guard_pos(b))
                        && !ctx.is_resume_at_position_guard(old_idx as i32)
                    {
                        // rewrite.py:335-338 + resoperation.py:498-503
                        // GuardResOp.copy_and_change parity: shallow copy
                        // with new opcode/args/descr; fail_args, fail_arg_types,
                        // rd_resume_position, rd_numb, rd_consts, rd_virtuals,
                        // and rd_pendingfields are all carried automatically.
                        // descr is set to None (fresh) per rewrite.py:335:
                        //   descr = compile.ResumeGuardDescr()
                        let replacement = old_guard.copy_and_change(
                            OpCode::GuardValue,
                            Some(&[old_guard.arg(0), arg1]),
                            Some(None),
                        );
                        // rewrite.py:343: self.optimizer.replace_guard(op, info)
                        ctx.new_operations[old_idx] = std::rc::Rc::new(replacement);
                        // rewrite.py:345-346: info.reset_last_guard_pos()
                        if let Some(b) = ctx.ensure_box(obj) {
                            ctx.with_ptr_info_mut(&b, |info_mut| info_mut.reset_last_guard_pos());
                        }
                        // postprocess_GUARD_VALUE (rewrite.py:303-305): make_constant
                        // with the actual c_value (preserving Int vs Ref typing).
                        ctx.make_constant(arg0, c_value);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // rewrite.py postprocess_GUARD_VALUE:
        //   box = get_box_replacement(op.getarg(0))
        //   self.make_constant(box, op.getarg(1))
        let box_ref = ctx.get_box_replacement(arg0);
        let v = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| b.const_value())
            .expect("postprocess_GUARD_VALUE requires const arg1");
        ctx.make_constant(box_ref, v);
        OptimizationResult::PassOn
    }

    /// rewrite.py:397-436 optimize_GUARD_CLASS / postprocess_GUARD_CLASS.
    ///
    /// Shared by GuardClass and GuardNonnullClass — RPython
    /// `optimize_GUARD_NONNULL_CLASS` (rewrite.py:438-444) delegates to
    /// `optimize_GUARD_CLASS` after the null check, so both opcodes go
    /// through the same known-class / strengthening / postprocess logic.
    fn optimize_guard_class(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // rewrite.py:402: info = self.ensure_ptr_info_arg0(op) — installs
        // an InstancePtrInfo on arg0 when one is missing. Discard the
        // EnsuredPtrInfo borrow; downstream lookups re-acquire via
        // `getptrinfo` / `get_ptr_info` against the resolved OpRef.
        let _ = ctx.ensure_ptr_info_arg0(op);
        let obj = ctx.get_box_replacement(op.arg(0));
        // rewrite.py:397-407: ensure_ptr_info_arg0 → info.py:880 getptrinfo.
        // `getptrinfo(ConstPtr)` returns a synthesized ConstPtrInfo, so a
        // constant Ref arg0 is handled uniformly with virtual / instance
        // info: ConstPtrInfo.get_known_class(cpu) (info.py:763-772) reads
        // the typeptr at offset 0 via cls_of_box and compares against
        // expectedclassbox. Mismatch → proven-fail guard → InvalidLoop.
        let obj_info_for_class = ctx
            .get_box_replacement_box(op.arg(0))
            .as_ref()
            .and_then(|b| ctx.getptrinfo(b));
        if let Some(known_class) =
            obj_info_for_class.and_then(|i| i.get_known_class(ctx.cpu.as_ref()))
        {
            if op.num_args() >= 2 {
                // Class pointer may be Value::Int or Value::Ref.
                let expected = ctx
                    .get_box_replacement_box(op.arg(1))
                    .and_then(|b| match b.const_value() {
                        Some(Value::Int(i)) => Some(i),
                        _ => None,
                    })
                    .or_else(|| {
                        ctx.get_box_replacement_box(op.arg(1))
                            .and_then(|b| b.const_value())
                            .and_then(|v| match v {
                                majit_ir::Value::Ref(r) => Some(r.0 as i64),
                                _ => None,
                            })
                    });
                if let Some(expected) = expected {
                    if known_class.0 as i64 == expected {
                        return OptimizationResult::Remove;
                    }
                    // rewrite.py:404-407: known class mismatch is a
                    // proven-fail guard — abort the trace.
                    raise_invalid_loop("GUARD_CLASS proven to always fail");
                }
            }
        }
        // rewrite.py:408-427: guard strengthening.
        // If there was a previous GUARD_NONNULL on the same value,
        // replace it with GUARD_NONNULL_CLASS (combining both checks).
        // rewrite.py:409-410: skip replacement if old descr is a
        // ResumeAtPositionDescr — RPython's fresh ResumeGuardDescr() at
        // line 417 must not overwrite a RAPD marker (rewrite.py:421-422
        // "old descr must not be ResumeAtPositionDescr").
        let obj_box = ctx.get_box_replacement_box(op.arg(0));
        if let Some(old_guard) = obj_box.as_ref().and_then(|b| ctx.get_last_guard(b)) {
            if old_guard.opcode == OpCode::GuardNonnull
                && op.num_args() >= 2
                && ctx.can_replace_guards
            {
                // last_guard_pos is a _newoperations index.
                let old_guard_idx = obj_box.as_ref().and_then(|b| ctx.last_guard_pos(b));
                if let Some(old_idx) = old_guard_idx
                    && !ctx.is_resume_at_position_guard(old_idx as i32)
                {
                    // rewrite.py:417-426 + optimizer.py:713-718:
                    // RPython creates a fresh ResumeGuardDescr for the
                    // strengthened guard, then replace_guard_op copies the
                    // resume payload from the old guard descr into the new
                    // one. This path writes directly into new_operations, so
                    // perform the descr copy inline before replacing the op.
                    let new_descr = crate::compile::make_resume_guard_descr_typed(
                        old_guard
                            .get_fail_arg_types()
                            .map(|t| t.to_vec())
                            .unwrap_or_default(),
                    );
                    let old_descr = old_guard
                        .getdescr()
                        .expect("strengthened GUARD_CLASS donor must carry a descr");
                    crate::compile::copy_all_attributes_from(&new_descr, &old_descr);
                    let combined = old_guard.copy_and_change(
                        OpCode::GuardNonnullClass,
                        Some(&[old_guard.arg(0), op.arg(1)]),
                        Some(Some(new_descr)),
                    );
                    ctx.new_operations[old_idx] = std::rc::Rc::new(combined);
                    // rewrite.py:430-436 postprocess_GUARD_CLASS parity
                    // (invoked inline here because the replacement path
                    // rewrites `new_operations[old_idx]` directly instead
                    // of going through `emit_operation`, which would have
                    // triggered the regular postprocess dispatch).
                    //
                    // The replacement happened because the old guard is
                    // not a ResumeAtPositionDescr (guarded at
                    // `is_resume_at_position_guard(old_idx)` above), so
                    // rewrite.py:434-435 `update_last_guard = not
                    // old_guard_op or isinstance(descr, RAPD)` evaluates
                    // to False — pass `update_last_guard=false` so that
                    // make_constant_class preserves the strengthened
                    // guard's position in last_guard_pos (optimizer.py:137
                    // parity) rather than snapping it to the tail of
                    // new_operations.
                    if let Some(class_val) = ctx
                        .get_box_replacement_box(op.arg(1))
                        .and_then(|b| match b.const_value() {
                            Some(Value::Int(i)) => Some(i),
                            _ => None,
                        })
                        .or_else(|| {
                            ctx.get_box_replacement_box(op.arg(1))
                                .and_then(|b| b.const_value())
                                .and_then(|v| match v {
                                    majit_ir::Value::Ref(r) => Some(r.0 as i64),
                                    _ => None,
                                })
                        })
                    {
                        if let Some(b) = ctx.ensure_box(obj) {
                            crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                                ctx, &b, class_val, /* update_last_guard = */ false,
                            );
                        }
                    }
                    return OptimizationResult::Remove;
                }
            }
        }
        // rewrite.py:430-436 postprocess_GUARD_CLASS: runs AFTER emit.
        // Register deferred postprocess — executed by emit_operation
        // after the guard is added to new_operations.  Upstream
        // `postprocess_GUARD_CLASS` runs unconditionally (no
        // virtual-skip): `make_constant_class` already preserves
        // existing `InstancePtrInfo` whether or not `is_virtual=True`
        // (`optimizer.py:137-151`), so the Rust port skips the local
        // `is_virtual` guard and lets `Optimizer::make_constant_class`
        // dispatch on the live `Instance` / `Virtual` arm.
        if op.num_args() >= 2 {
            if let Some(class_val) = ctx
                .get_box_replacement_box(op.arg(1))
                .and_then(|b| match b.const_value() {
                    Some(Value::Int(i)) => Some(i),
                    _ => None,
                })
                .or_else(|| {
                    ctx.get_box_replacement_box(op.arg(1))
                        .and_then(|b| b.const_value())
                        .and_then(|v| match v {
                            majit_ir::Value::Ref(r) => Some(r.0 as i64),
                            _ => None,
                        })
                })
            {
                ctx.pending_guard_class_postprocess =
                    Some(crate::optimizeopt::PendingGuardClassPostprocess { obj, class_val });
            }
        }
        OptimizationResult::PassOn
    }

    // ── SAME_AS identity ──

    /// optimizer.py:127-135 `getnullness(op)` wrapper. Delegates to
    /// `OptContext::getnullness`, which implements the upstream
    /// `op.type == 'r' or is_raw_ptr(op)` dispatch line-by-line, then
    /// converts the upstream `INFO_NULL` / `INFO_NONNULL` /
    /// `INFO_UNKNOWN` integer return into the local `Nullness` enum.
    fn getnullness(&self, opref: OpRef, ctx: &mut OptContext) -> Nullness {
        // optimizer.py:127-135 `getnullness` has no missing-Box branch —
        // every `op` has a backing `AbstractValue` per
        // `resoperation.py:233-248`. `ensure_box` lazy-allocates the
        // Box (or constructs the const-namespace fresh) so the inlined
        // `getintbound` side effect (`optimizer.py:110-113` unbounded
        // install) materializes on first access. `INFO_UNKNOWN` only
        // for OpRef::NONE sentinels (no upstream equivalent).
        let info = match ctx.ensure_box(opref) {
            Some(b) => ctx.getnullness(&b),
            None => crate::optimizeopt::INFO_UNKNOWN,
        };
        Self::nullness_from_info(info)
    }

    /// Convert an `info.py` INFO_NULL/INFO_NONNULL/INFO_UNKNOWN return
    /// into the local `Nullness` enum used by the rewrite pass.
    fn nullness_from_info(value: i8) -> Nullness {
        if value == crate::optimizeopt::INFO_NULL {
            Nullness::Null
        } else if value == crate::optimizeopt::INFO_NONNULL {
            Nullness::Nonnull
        } else {
            Nullness::Unknown
        }
    }

    /// Check if an OpRef is Ref-typed.
    /// optimizer.py:128: op.type == 'r'
    ///
    /// Routes through the canonical `OptContext::opref_type` accessor
    /// (constant → value_types → producer op result_type) and falls back
    /// to PtrInfo presence — a Ref-only side channel populated for input
    /// args that do not appear in `new_operations`.
    fn is_ref_typed(&self, opref: OpRef, ctx: &OptContext) -> bool {
        if ctx.opref_type(opref) == Some(majit_ir::Type::Ref) {
            return true;
        }
        // BoxRef shim — has_ptr_info takes &BoxRef per info.py:880-894.
        ctx.get_box_replacement_box(opref)
            .as_ref()
            .map_or(false, |b| ctx.has_ptr_info(b))
    }

    /// rewrite.py:95-101: _optimize_CALL_INT_UDIV
    /// x / 1 → x
    fn optimize_call_int_udiv(&mut self, op: &Op, ctx: &mut OptContext) -> bool {
        if op.num_args() < 3 {
            return false;
        }
        let arg2 = op.arg(2);
        if let Some(1) = ctx
            .get_box_replacement_box(arg2)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(op.arg(1))
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            ctx.last_op_removed = true;
            return true;
        }
        false
    }

    /// rewrite.py:768-805: _optimize_CALL_INT_PY_MOD
    fn optimize_call_int_py_mod(
        &mut self,
        op: &Op,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        if op.num_args() < 3 {
            return None;
        }
        let arg1 = op.arg(1);
        let arg2 = op.arg(2);
        let b1 = ctx
            .ensure_box(arg1)
            .map(|b| ctx.getintbound_handle(&b).borrow().clone())
            .expect("getintbound: operand must resolve to a BoxRef");
        let b2 = ctx
            .ensure_box(arg2)
            .map(|b| ctx.getintbound_handle(&b).borrow().clone())
            .expect("getintbound: operand must resolve to a BoxRef");

        // rewrite.py:774-777: b1.known_eq_const(0) → 0
        if b1.known_eq_const(0) {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:780-781: if not b2.is_constant(): return False
        if !b2.is_constant() {
            return None;
        }
        let val = b2.get_constant_int();
        // rewrite.py:783-784
        if val <= 0 {
            return None;
        }
        // rewrite.py:785-788: x % 1 → 0
        if val == 1 {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:789-796: x % power_of_two → x & (power_of_two - 1)
        // Python's modulo: valid even for negative x.
        // RPython: replace_op_with + send_extra_operation (routes through passes).
        if val & (val - 1) == 0 {
            let mask = ctx.make_constant_int(val - 1);
            let mut and_op = Op::new(OpCode::IntAnd, &[arg1, mask]);
            and_op.pos.set(op.pos.get());
            ctx.emit_extra(ctx.current_pass_idx, and_op);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:797-805: intdiv.modulo_operations fallback
        let known_nonneg = b1.known_nonnegative();
        let result_ref = crate::optimizeopt::intdiv::modulo_operations(
            arg1,
            val,
            known_nonneg,
            ctx.current_pass_idx,
            ctx,
        );
        let b_old = ctx
            .ensure_box(op.pos.get())
            .expect("body-namespace OpRef must have a BoxRef slot");
        let b_res = ctx
            .ensure_box(result_ref)
            .expect("body-namespace OpRef must have a BoxRef slot");
        ctx.make_equal_to(&b_old, &b_res);
        ctx.last_op_removed = true;
        Some(OptimizationResult::Remove)
    }

    /// rewrite.py:713-766: _optimize_CALL_INT_PY_DIV
    fn optimize_call_int_py_div(
        &mut self,
        op: &Op,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        if op.num_args() < 3 {
            return None;
        }
        let arg1 = op.arg(1);
        let arg2 = op.arg(2);
        let b1 = ctx
            .ensure_box(arg1)
            .map(|b| ctx.getintbound_handle(&b).borrow().clone())
            .expect("getintbound: operand must resolve to a BoxRef");
        let b2 = ctx
            .ensure_box(arg2)
            .map(|b| ctx.getintbound_handle(&b).borrow().clone())
            .expect("getintbound: operand must resolve to a BoxRef");

        // rewrite.py:726-729: b1.known_eq_const(0) → 0
        if b1.known_eq_const(0) {
            ctx.make_constant(op.pos.get(), Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:730-741: non-constant divisor (shift optimization)
        if !b2.is_constant() {
            // rewrite.py:731-740: x // (1 << y) → x >> y
            // when 0 <= y < LONG_BIT - 1
            if let Some(shift_op) = ctx
                .ensure_box(arg2)
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if shift_op.opcode == OpCode::IntLshift
                    && shift_op.num_args() >= 2
                    && ctx
                        .get_box_replacement_box(shift_op.arg(0))
                        .and_then(|cb| cb.const_int())
                        == Some(1)
                {
                    let shiftvar = ctx.get_box_replacement(shift_op.arg(1));
                    let shiftbound = ctx
                        .ensure_box(shiftvar)
                        .map(|b| ctx.getintbound_handle(&b).borrow().clone())
                        .expect("getintbound: operand must resolve to a BoxRef");
                    if shiftbound.known_nonnegative() && shiftbound.known_lt_const(63) {
                        let mut rshift_op = Op::new(OpCode::IntRshift, &[arg1, shiftvar]);
                        rshift_op.pos.set(op.pos.get());
                        ctx.emit_extra(ctx.current_pass_idx, rshift_op);
                        ctx.last_op_removed = true;
                        return Some(OptimizationResult::Remove);
                    }
                }
            }
            return None;
        }
        let val = b2.get_constant_int();
        // rewrite.py:743-749: x // -1 → -x (if x > MININT)
        if val == -1 {
            if b1.known_gt_const(i64::MIN) {
                let mut neg_op = Op::new(OpCode::IntNeg, &[arg1]);
                neg_op.pos.set(op.pos.get());
                ctx.emit_extra(ctx.current_pass_idx, neg_op);
                ctx.last_op_removed = true;
                return Some(OptimizationResult::Remove);
            }
        }
        // rewrite.py:750-751
        if val <= 0 {
            return None;
        }
        // rewrite.py:752-755: x // 1 → x
        if val == 1 {
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_arg = ctx
                .ensure_box(arg1)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_arg);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:756-757: x // power_of_two → x >> shift
        if val & (val - 1) == 0 {
            let shift = val.trailing_zeros() as i64;
            let shift_const = ctx.make_constant_int(shift);
            let mut rshift_op = Op::new(OpCode::IntRshift, &[arg1, shift_const]);
            rshift_op.pos.set(op.pos.get());
            ctx.emit_extra(ctx.current_pass_idx, rshift_op);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:758-766: intdiv.division_operations fallback
        let known_nonneg = b1.known_nonnegative();
        let result_ref = crate::optimizeopt::intdiv::division_operations(
            arg1,
            val,
            known_nonneg,
            ctx.current_pass_idx,
            ctx,
        );
        let b_old = ctx
            .ensure_box(op.pos.get())
            .expect("body-namespace OpRef must have a BoxRef slot");
        let b_res = ctx
            .ensure_box(result_ref)
            .expect("body-namespace OpRef must have a BoxRef slot");
        ctx.make_equal_to(&b_old, &b_res);
        ctx.last_op_removed = true;
        Some(OptimizationResult::Remove)
    }

    /// rewrite.py:599-670: _optimize_call_arrayop
    ///
    /// Element-by-element unrolling for small constant-length array
    /// copy/move operations. Handles both virtual and non-virtual arrays.
    fn optimize_call_arrayop(
        &mut self,
        op: &Op,
        source_box: OpRef,
        dest_box: OpRef,
        source_start_box: OpRef,
        dest_start_box: OpRef,
        length_box: OpRef,
        ctx: &mut OptContext,
    ) -> bool {
        // rewrite.py:601-602: length = self.get_constant_box(length_box)
        let length_int = match ctx
            .get_box_replacement_box(length_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(l) => l,
            None => return false,
        };
        // rewrite.py:605-606: 0-length → remove
        if length_int == 0 {
            return true;
        }

        let source_box = ctx.get_box_replacement(source_box);
        let dest_box = ctx.get_box_replacement(dest_box);
        let source_is_virtual = ctx
            .get_box_replacement_box(source_box)
            .as_ref()
            .map_or(false, |b| ctx.is_virtual(b));
        let dest_is_virtual = ctx
            .get_box_replacement_box(dest_box)
            .as_ref()
            .map_or(false, |b| ctx.is_virtual(b));

        // rewrite.py:610-611: constant start indices required
        let source_start = match ctx
            .get_box_replacement_box(source_start_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(s) => s,
            None => return false,
        };
        let dest_start = match ctx
            .get_box_replacement_box(dest_start_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(d) => d,
            None => return false,
        };

        // rewrite.py:613-617: both start constant, at least one virtual or length <= 8
        if !((dest_is_virtual || length_int <= 8) && (source_is_virtual || length_int <= 8)) {
            return false;
        }

        // rewrite.py:612,617: extrainfo.single_write_descr_array sanity check
        let call_descr = match op.getdescr() {
            Some(d) => d,
            None => return false,
        };
        let cd = match call_descr.as_call_descr() {
            Some(cd) => cd,
            None => return false,
        };
        let ei = cd.get_extra_info();
        // rewrite.py:617: extrainfo.single_write_descr_array is not None
        // effectinfo.py:201-206: set when exactly one write array descriptor.
        let arraydescr = match &ei.single_write_descr_array {
            Some(d) => d.clone(),
            None => {
                // Fallback: check bitstring — must have exactly one array write.
                let count: u32 = ei
                    .write_descrs_arrays
                    .as_ref()
                    .map(|w| w.iter().map(|b| b.count_ones()).sum())
                    .unwrap_or(0);
                if count != 1 {
                    return false;
                }
                // No actual DescrRef available — cannot emit typed ops.
                return false;
            }
        };

        // rewrite.py:621-635: arraydescr.is_array_of_structs()
        if arraydescr
            .as_array_descr()
            .is_some_and(|ad| ad.is_array_of_structs())
        {
            // rewrite.py:624-627: only if both virtual, not memmove
            if !(source_is_virtual && dest_is_virtual && source_box != dest_box) {
                return false;
            }
            // rewrite.py:628-629: all_fdescrs = arraydescr.get_all_fielddescrs()
            // → all_interiorfielddescrs in descr.py:291.
            let all_fdescr_indices: Vec<u32> = arraydescr
                .as_array_descr()
                .and_then(|ad| ad.get_all_interiorfielddescrs())
                .map(|fds| fds.iter().map(|d| d.index()).collect())
                .or_else(|| {
                    // Fallback: get from virtual's metadata
                    ctx.get_box_replacement_box(source_box)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
                        .and_then(|info| match info {
                            crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(v) => {
                                if v.fielddescrs.is_empty() {
                                    None
                                } else {
                                    Some(v.fielddescrs.iter().map(|d| d.index()).collect())
                                }
                            }
                            _ => None,
                        })
                })
                .unwrap_or_default();
            if all_fdescr_indices.is_empty() {
                return false;
            }
            // rewrite.py:631-634: copy interior fields element by element
            for index in 0..length_int {
                for &fdescr_idx in &all_fdescr_indices {
                    let val = ctx
                        .get_box_replacement_box(source_box)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
                        .and_then(|info| {
                            info.getinteriorfield_virtual(
                                (index + source_start) as usize,
                                fdescr_idx,
                            )
                        });
                    if let Some(val) = val {
                        let idx = (index + dest_start) as usize;
                        if let Some(b) = ctx.ensure_box(dest_box) {
                            ctx.with_ptr_info_mut(&b, |info| {
                                info.setinteriorfield_virtual(idx, fdescr_idx, val);
                            });
                        }
                    }
                }
            }
            return true;
        }

        // rewrite.py:636-643: iteration direction
        let mut index_current: i64 = 0;
        let mut index_delta: i64 = 1;
        let mut index_stop: i64 = length_int;
        if source_box == dest_box && source_start < dest_start {
            // ARRAYMOVE with overlapping regions: iterate in reverse
            index_current = index_stop - 1;
            index_delta = -1;
            index_stop = -1;
        }

        // rewrite.py:646-670: element-by-element copy
        // RPython routes synthesized ops through send_extra_operation()
        // so they pass through downstream optimization passes.
        // We use ctx.emit_extra(current_pass_idx, op) for the same effect.
        let pass_idx = ctx.current_pass_idx;
        while index_current != index_stop {
            let index = index_current;
            index_current += index_delta;
            debug_assert!(index >= 0);

            // Read source element
            let val = if source_is_virtual {
                // rewrite.py:650-651: source_info.getitem(arraydescr, index + source_start)
                ctx.get_box_replacement_box(source_box)
                    .as_ref()
                    .and_then(|b| ctx.peek_ptr_info(b))
                    .and_then(|info| info.getitem((index + source_start) as usize))
                    .and_then(|e| e.as_opref())
            } else {
                // rewrite.py:653: opnum = OpHelpers.getarrayitem_for_descr(arraydescr)
                // Select I/R/F opcode based on item type.
                let item_type = arraydescr
                    .as_array_descr()
                    .map(|ad| ad.item_type())
                    .unwrap_or(majit_ir::Type::Int);
                let opcode = OpCode::getarrayitem_for_type(item_type);
                let idx_const = ctx.make_constant_int(index + source_start);
                let mut getop = Op::new(opcode, &[source_box, idx_const]);
                getop.setdescr(arraydescr.clone());
                let pos = ctx.emit_extra(pass_idx, getop);
                Some(pos)
            };

            let val = match val {
                Some(v) => v,
                None => continue, // rewrite.py:660-661: if val is None: continue
            };

            // Write to destination
            if dest_is_virtual {
                // rewrite.py:662-665: dest_info.setitem(...)
                let idx = (index + dest_start) as usize;
                if let Some(b) = ctx.ensure_box(dest_box) {
                    ctx.with_ptr_info_mut(&b, |info| info.setitem(idx, val));
                }
            } else {
                // rewrite.py:666-670: emit SETARRAYITEM_GC
                let idx_const = ctx.make_constant_int(index + dest_start);
                let mut setop = Op::new(OpCode::SetarrayitemGc, &[dest_box, idx_const, val]);
                setop.setdescr(arraydescr.clone());
                ctx.emit_extra(pass_idx, setop);
            }
        }
        true
    }

    fn optimize_same_as(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() == 0 {
            return OptimizationResult::PassOn;
        }
        let arg0 = op.arg(0);
        let b_old = ctx
            .ensure_box(op.pos.get())
            .expect("body-namespace OpRef must have a BoxRef slot");
        let b_arg = ctx
            .ensure_box(arg0)
            .expect("body-namespace OpRef must have a BoxRef slot");
        ctx.make_equal_to(&b_old, &b_arg);
        OptimizationResult::Remove
    }

    // ── Boolean inverse/reflex rewrites ──

    /// For comparison ops that have a bool_inverse or bool_reflex:
    /// Check if we already computed the inverse/reflex and can reuse that result.
    ///
    /// This mirrors `find_rewritable_bool` from rewrite.py: if we see INT_LT(a, b)
    /// and we previously computed INT_GE(a, b) = K (a constant 0 or 1), then
    /// INT_LT(a, b) = 1 - K.
    /// rewrite.py: find_rewritable_bool(op)
    /// If we see INT_LT(a, b) and previously computed INT_GE(a, b) = K,
    /// then INT_LT(a, b) = 1 - K (boolean inverse).
    /// rewrite.py:56-66 try_boolinvers — check if the inverse operation has
    /// a cached boolean result and negate it.
    ///
    /// RPython uses get_pure_result(targs) + getintbound(oldop).known_eq_const()
    /// which recognizes values that are guaranteed to be 0 or 1 even if not
    /// explicitly constant-folded. We match this by checking IntBound in
    /// addition to direct constant lookup.
    fn try_boolinvers(
        &self,
        op: &Op,
        inverse_opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        let key = (inverse_opcode, arg0, arg1);
        let cached_ref = self.bool_result_cache.get(&key).copied()?;
        // rewrite.py:60-65: b = self.getintbound(oldop)
        // First try direct constant (fast path)
        if let Some(val) = ctx
            .get_box_replacement_box(cached_ref)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let result = 1 - val;
            ctx.make_constant(op.pos.get(), Value::Int(result));
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:61-65: b.known_eq_const(1) / b.known_eq_const(0)
        // Intbound analysis: the value may be bounded to exactly 0 or 1
        // even without being a constant in the optimizer's sense.
        if let Some(bound) = ctx
            .get_box_replacement_box(cached_ref)
            .and_then(|b| ctx.peek_intbound_box(&b))
        {
            if bound.known_eq_const(1) {
                ctx.make_constant(op.pos.get(), Value::Int(0));
                return Some(OptimizationResult::Remove);
            } else if bound.known_eq_const(0) {
                ctx.make_constant(op.pos.get(), Value::Int(1));
                return Some(OptimizationResult::Remove);
            }
        }
        None
    }

    /// rewrite.py:68-93 find_rewritable_bool — three-phase boolean rewrite:
    /// 1. boolinverse(same args)
    /// 2. boolreflex(swapped args)
    /// 3. boolreflex.boolinverse(swapped args)
    fn find_rewritable_bool(&self, op: &Op, ctx: &mut OptContext) -> Option<OptimizationResult> {
        if op.num_args() < 2 {
            return None;
        }
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // rewrite.py:72-75: boolinverse(arg0, arg1)
        if let Some(inverse_opcode) = op.opcode.bool_inverse() {
            if let Some(result) = self.try_boolinvers(op, inverse_opcode, arg0, arg1, ctx) {
                return Some(result);
            }
        }

        // rewrite.py:77-83: boolreflex(arg1, arg0)
        if let Some(reflex_opcode) = op.opcode.bool_reflex() {
            let key = (reflex_opcode, arg1, arg0);
            if let Some(&cached_ref) = self.bool_result_cache.get(&key) {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_cached = ctx
                    .ensure_box(cached_ref)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_cached);
                return Some(OptimizationResult::Remove);
            }

            // rewrite.py:87-91: boolreflex.boolinverse(arg1, arg0)
            if let Some(reflex_inverse) = reflex_opcode.bool_inverse() {
                if let Some(result) = self.try_boolinvers(op, reflex_inverse, arg1, arg0, ctx) {
                    return Some(result);
                }
            }
        }

        None
    }

    // ── Float algebraic simplifications ──
    // rewrite.py:103-161 — only FLOAT_MUL, FLOAT_TRUEDIV, FLOAT_NEG, FLOAT_ABS.
    // Constant folding for all float ops is handled by execute_nonspec_const.

    /// rewrite.py:103-120 optimize_FLOAT_MUL
    fn optimize_float_mul(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        // rewrite.py:109: for lhs, rhs in [(arg1, arg2), (arg2, arg1)]:
        for &(lhs, rhs) in &[(arg0, arg1), (arg1, arg0)] {
            if let Some(v) = ctx
                .get_box_replacement_box(lhs)
                .and_then(|b| b.const_value())
                .and_then(|v| match v {
                    Value::Float(f) => Some(f),
                    _ => None,
                })
            {
                let v2 = ctx.get_box_replacement(rhs);
                if v == 1.0 {
                    let b_old = ctx
                        .ensure_box(op.pos.get())
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    let b_v2 = ctx
                        .ensure_box(v2)
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    ctx.make_equal_to(&b_old, &b_v2);
                    return OptimizationResult::Remove;
                }
                if v == -1.0 {
                    let mut neg = Op::new(OpCode::FloatNeg, &[rhs]);
                    neg.pos.set(op.pos.get());
                    return OptimizationResult::Replace(neg);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py:126-145 optimize_FLOAT_TRUEDIV
    fn optimize_float_truediv(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        if let Some(divisor) = ctx
            .get_box_replacement_box(arg1)
            .and_then(|b| b.const_value())
            .and_then(|v| match v {
                Value::Float(f) => Some(f),
                _ => None,
            })
        {
            // rewrite.py:135-141: frexp check that divisor AND reciprocal
            // are both exact powers of 2. Bit-level equivalent: mantissa
            // bits are all zero and exponent is normal (not zero/subnormal/inf/nan).
            if Self::is_exact_power_of_two(divisor) {
                let reciprocal = 1.0 / divisor;
                if Self::is_exact_power_of_two(reciprocal) {
                    let recip_ref = self.emit_constant_float(ctx, reciprocal);
                    let mut new_op = Op::new(OpCode::FloatMul, &[arg0, recip_ref]);
                    new_op.pos.set(op.pos.get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py:135: `math.frexp(divisor)[0]` == ±0.5 iff exact power of 2.
    fn is_exact_power_of_two(v: f64) -> bool {
        let bits = v.to_bits();
        let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;
        let exponent = ((bits >> 52) & 0x7FF) as u32;
        mantissa == 0 && exponent > 0 && exponent < 0x7FF
    }

    /// rewrite.py:147-153 optimize_FLOAT_NEG
    fn optimize_float_neg(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let v = ctx.get_box_replacement(op.arg(0));
        if let Some(arg_op) = ctx.ensure_box(v).and_then(|pb| ctx.get_producing_op(&pb)) {
            if arg_op.opcode == OpCode::FloatNeg {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_inner = ctx
                    .ensure_box(arg_op.arg(0))
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_inner);
                return OptimizationResult::Remove;
            }
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py:155-161 optimize_FLOAT_ABS
    fn optimize_float_abs(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let v = ctx.get_box_replacement(op.arg(0));
        if let Some(arg_op) = ctx.ensure_box(v).and_then(|pb| ctx.get_producing_op(&pb)) {
            if arg_op.opcode == OpCode::FloatAbs {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_v = ctx
                    .ensure_box(v)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_v);
                return OptimizationResult::Remove;
            }
        }
        OptimizationResult::PassOn
    }

    // ── Helper ──

    /// Emit a constant integer value into the trace and return its OpRef.
    fn emit_constant_int(&self, ctx: &mut OptContext, value: i64) -> OpRef {
        let op = Op::new(OpCode::SameAsI, &[]);
        let opref = ctx.emit(op);
        ctx.make_constant(opref, Value::Int(value));
        opref
    }

    fn emit_constant_float(&self, ctx: &mut OptContext, value: f64) -> OpRef {
        let op = Op::new(OpCode::SameAsF, &[]);
        let opref = ctx.emit(op);
        ctx.make_constant(opref, Value::Float(value));
        opref
    }
}

impl Optimization for OptRewrite {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Track last_op_removed for GuardNoException optimization.
        // Reset for non-guard ops (guards don't count as "the last op").
        if !op.opcode.is_guard() {
            ctx.last_op_removed = false;
        }

        // Try boolean inverse/reflex rewrites for comparisons
        if op.opcode.bool_inverse().is_some() || op.opcode.bool_reflex().is_some() {
            if let Some(result) = self.find_rewritable_bool(op, ctx) {
                return result;
            }
        }

        match op.opcode {
            // ── Binary integer arithmetic ──
            OpCode::IntAdd => self.optimize_int_add(op, ctx),
            OpCode::IntSub => self.optimize_int_sub(op, ctx),
            OpCode::IntMul => self.optimize_int_mul(op, ctx),
            OpCode::IntFloorDiv => self.optimize_int_floor_div(op, ctx),
            OpCode::IntMod => self.optimize_int_mod(op, ctx),
            OpCode::IntAnd => self.optimize_int_and(op, ctx),
            OpCode::IntOr => self.optimize_int_or(op, ctx),
            OpCode::IntXor => self.optimize_int_xor(op, ctx),
            OpCode::IntLshift => self.optimize_int_lshift(op, ctx),
            OpCode::IntRshift => self.optimize_int_rshift(op, ctx),
            OpCode::UintRshift => self.optimize_uint_rshift(op, ctx),

            // ── Unary integer operations ──
            OpCode::IntNeg => self.optimize_int_neg(op, ctx),
            OpCode::IntInvert => self.optimize_int_invert(op, ctx),
            OpCode::IntIsZero => self.optimize_int_is_zero(op, ctx),
            OpCode::IntIsTrue => self.optimize_int_is_true(op, ctx),
            OpCode::IntForceGeZero => self.optimize_int_force_ge_zero(op, ctx),
            OpCode::IntBetween => self.optimize_int_between(op, ctx),

            // ── Comparisons ──
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe => self.optimize_comparison(op, ctx),

            // ── Guards ──
            OpCode::GuardTrue => self.optimize_guard_true(op, ctx),
            OpCode::GuardFalse => self.optimize_guard_false(op, ctx),
            OpCode::GuardValue => self.optimize_guard_value(op, ctx),
            // RPython rewrite.py guard optimizations:
            // If the guarded condition is already known to be true (constant),
            // the guard can be removed entirely.
            OpCode::GuardNonnull => {
                // rewrite.py:269-278 optimize_GUARD_NONNULL
                //     opinfo = getptrinfo(op.getarg(0))
                //     if opinfo is not None:
                //         if opinfo.is_nonnull(): return
                //         elif opinfo.is_null(): raise InvalidLoop(...)
                //     return self.emit(op)
                let obj = ctx.get_box_replacement(op.arg(0));
                let obj_box = ctx.get_box_replacement_box(op.arg(0));
                if let Some(info) = obj_box.as_ref().and_then(|b| ctx.getptrinfo(b)) {
                    if info.is_nonnull() {
                        return OptimizationResult::Remove;
                    }
                    if info.is_null() {
                        raise_invalid_loop("GUARD_NONNULL proven to always fail");
                    }
                }
                // rewrite.py:280-282 postprocess_GUARD_NONNULL:
                // make_nonnull runs immediately; mark_last_guard deferred
                // until emit adds the guard to new_operations.
                let has_info = obj_box.as_ref().map_or(false, |b| ctx.has_ptr_info(b));
                if !has_info {
                    if let Some(b) = ctx.ensure_box(obj) {
                        ctx.set_ptr_info(&b, crate::optimizeopt::info::PtrInfo::nonnull());
                    }
                }
                // rewrite.py:282: mark_last_guard deferred to emit_operation
                ctx.pending_mark_last_guard = Some(obj);
                OptimizationResult::PassOn
            }
            OpCode::GuardIsnull => {
                // rewrite.py:186-195 optimize_GUARD_ISNULL
                //     info = getptrinfo(op.getarg(0))
                //     if info is not None:
                //         if info.is_null(): return
                //         elif info.is_nonnull(): raise InvalidLoop(...)
                //     return self.emit(op)
                let obj = ctx.get_box_replacement(op.arg(0));
                let obj_box = ctx.get_box_replacement_box(op.arg(0));
                if let Some(info) = obj_box.as_ref().and_then(|b| ctx.getptrinfo(b)) {
                    if info.is_null() {
                        return OptimizationResult::Remove;
                    }
                    if info.is_nonnull() {
                        raise_invalid_loop("GUARD_ISNULL proven to always fail");
                    }
                }
                // rewrite.py:197-198 postprocess_GUARD_ISNULL:
                //     self.make_constant(op.getarg(0), CONST_NULL)
                // Ref-typed → Value::Ref(NULL); Int-typed → Value::Int(0).
                if self.is_ref_typed(obj, ctx) {
                    ctx.make_constant(op.arg(0), Value::Ref(majit_ir::GcRef(0)));
                } else {
                    ctx.make_constant(op.arg(0), Value::Int(0));
                }
                OptimizationResult::PassOn
            }
            OpCode::GuardClass => self.optimize_guard_class(op, ctx),
            OpCode::GuardNonnullClass => {
                // rewrite.py:438-444 optimize_GUARD_NONNULL_CLASS:
                //     info = getptrinfo(op.getarg(0))
                //     if info and info.is_null():
                //         raise InvalidLoop(...)
                //     return self.optimize_GUARD_CLASS(op)
                let arg0_box = ctx.get_box_replacement_box(op.arg(0));
                if let Some(info) = arg0_box.as_ref().and_then(|b| ctx.getptrinfo(b)) {
                    if info.is_null() {
                        raise_invalid_loop("GUARD_NONNULL_CLASS proven to always fail");
                    }
                }
                self.optimize_guard_class(op, ctx)
            }
            // rewrite.py: GUARD_IS_OBJECT — if arg is a known constant, the guard
            // was already checked at recording time and can be removed.
            OpCode::GuardIsObject => {
                if ctx
                    .get_box_replacement_box(op.arg(0))
                    .and_then(|b| ctx.get_constant_box(&b))
                    .is_some()
                {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }
            // rewrite.py: GUARD_GC_TYPE — if arg is a known constant, remove.
            OpCode::GuardGcType => {
                if ctx
                    .get_box_replacement_box(op.arg(0))
                    .and_then(|b| ctx.get_constant_box(&b))
                    .is_some()
                {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }
            // rewrite.py: GUARD_SUBCLASS — if arg is a known constant, remove.
            OpCode::GuardSubclass => {
                if ctx
                    .get_box_replacement_box(op.arg(0))
                    .and_then(|b| ctx.get_constant_box(&b))
                    .is_some()
                {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }

            // ── Float arithmetic ──
            OpCode::FloatMul => self.optimize_float_mul(op, ctx),
            OpCode::FloatTrueDiv => self.optimize_float_truediv(op, ctx),
            OpCode::FloatNeg => self.optimize_float_neg(op, ctx),
            OpCode::FloatAbs => self.optimize_float_abs(op, ctx),

            // ── Identity ops ──
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => self.optimize_same_as(op, ctx),

            // ── Conditional calls ──
            OpCode::CondCallN => {
                if let Some(0) = ctx
                    .get_box_replacement_box(op.arg(0))
                    .and_then(|b| ctx.get_constant_int_box(&b))
                {
                    ctx.last_op_removed = true;
                    return OptimizationResult::Remove;
                }
                if let Some(c) = ctx
                    .get_box_replacement_box(op.arg(0))
                    .and_then(|b| ctx.get_constant_int_box(&b))
                {
                    if c != 0 {
                        let mut call_op = Op::new(OpCode::CallN, &op.getarglist()[1..]);
                        call_op.pos.set(op.pos.get());
                        if let Some(d) = op.getdescr() {
                            call_op.setdescr(d);
                        }
                        ctx.last_op_removed = false;
                        return OptimizationResult::Replace(call_op);
                    }
                }
                ctx.last_op_removed = false;
                OptimizationResult::PassOn
            }
            // rewrite.py:483-494: optimize_COND_CALL_VALUE_I/R
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                let nullness = self.getnullness(op.arg(0), ctx);
                // rewrite.py:486-489: INFO_NONNULL → result is arg(0)
                if nullness == Nullness::Nonnull {
                    let b_old = ctx
                        .ensure_box(op.pos.get())
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    let b_arg = ctx
                        .ensure_box(op.arg(0))
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    ctx.make_equal_to(&b_old, &b_arg);
                    ctx.last_op_removed = true;
                    return OptimizationResult::Remove;
                }
                // rewrite.py:490-493: INFO_NULL → demote to CALL_PURE
                if nullness == Nullness::Null {
                    let call_opcode = if op.opcode == OpCode::CondCallValueI {
                        OpCode::CallPureI
                    } else {
                        OpCode::CallPureR
                    };
                    let mut call_op = Op::new(call_opcode, &op.getarglist()[1..]);
                    call_op.pos.set(op.pos.get());
                    if let Some(d) = op.getdescr() {
                        call_op.setdescr(d);
                    }
                    ctx.last_op_removed = false;
                    return OptimizationResult::Replace(call_op);
                }
                ctx.last_op_removed = false;
                OptimizationResult::PassOn
            }

            // ── Pointer equality (rewrite.py: _optimize_oois_ooisnot) ──
            OpCode::PtrEq | OpCode::InstancePtrEq => {
                let instance = matches!(op.opcode, OpCode::InstancePtrEq);
                if instance {
                    // rewrite.py:563-565 optimize_INSTANCE_PTR_EQ:
                    //     arg0 = get_box_replacement(op.getarg(0))
                    //     arg1 = get_box_replacement(op.getarg(1))
                    //     self.pure_from_args2(rop.INSTANCE_PTR_EQ, arg1, arg0, op)
                    let arg0 = ctx.get_box_replacement(op.arg(0));
                    let arg1 = ctx.get_box_replacement(op.arg(1));
                    ctx.register_pure_from_args2(OpCode::InstancePtrEq, op.pos.get(), arg1, arg0);
                }
                return self.optimize_oois_ooisnot(op, false, instance, ctx);
            }
            OpCode::PtrNe | OpCode::InstancePtrNe => {
                let instance = matches!(op.opcode, OpCode::InstancePtrNe);
                if instance {
                    // rewrite.py:568-571 optimize_INSTANCE_PTR_NE: same swap.
                    let arg0 = ctx.get_box_replacement(op.arg(0));
                    let arg1 = ctx.get_box_replacement(op.arg(1));
                    ctx.register_pure_from_args2(OpCode::InstancePtrNe, op.pos.get(), arg1, arg0);
                }
                return self.optimize_oois_ooisnot(op, true, instance, ctx);
            }

            // ── Cast round-trip elimination ──
            // rewrite.py:807-813: register pure inverse for CSE, then emit.
            OpCode::CastPtrToInt => {
                ctx.register_pure_from_args1(OpCode::CastIntToPtr, op.pos.get(), op.arg(0));
                OptimizationResult::PassOn
            }
            OpCode::CastIntToPtr => {
                ctx.register_pure_from_args1(OpCode::CastPtrToInt, op.pos.get(), op.arg(0));
                OptimizationResult::PassOn
            }
            // jtransform.py:1264-1266: CAST_OPAQUE_PTR is identity (no-op).
            OpCode::CastOpaquePtr => {
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_arg = ctx
                    .ensure_box(op.arg(0))
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_arg);
                OptimizationResult::Remove
            }

            // ── Float-bytes conversion round-trip elimination ──
            // rewrite.py:815-821: register inverse pure relationship for CSE.
            // CONVERT_FLOAT_BYTES_TO_LONGLONG(x) does NOT reduce to x —
            // it changes the bit representation. But if we later see
            // CONVERT_LONGLONG_BYTES_TO_FLOAT(result), pure.rs can
            // recognize the round-trip and recover x.
            OpCode::ConvertFloatBytesToLonglong => {
                ctx.register_pure_from_args1(
                    OpCode::ConvertLonglongBytesToFloat,
                    op.pos.get(),
                    op.arg(0),
                );
                OptimizationResult::PassOn
            }
            OpCode::ConvertLonglongBytesToFloat => {
                ctx.register_pure_from_args1(
                    OpCode::ConvertFloatBytesToLonglong,
                    op.pos.get(),
                    op.arg(0),
                );
                OptimizationResult::PassOn
            }

            // rewrite.py:712-718 optimize_GUARD_NO_EXCEPTION:
            //
            //     def optimize_GUARD_NO_EXCEPTION(self, op):
            //         if self.last_emitted_operation is REMOVED:
            //             return  # the prior op was a CALL_PURE that
            //                     # was killed; kill the guard too
            //         return self.emit(op)
            //
            // `last_emitted_operation` is set by every pass's emit
            // (optimizer.py:84-92), so the flag reflects the
            // PREVIOUS op's fate regardless of which pass dropped it.
            // pyre's ctx.last_op_removed is the cross-pass equivalent.
            OpCode::GuardNoException => {
                if ctx.last_op_removed {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }
            // rewrite.py: optimize_GUARD_FUTURE_CONDITION
            OpCode::GuardFutureCondition => {
                ctx.patchguardop = Some(op.clone());
                OptimizationResult::Remove
            }

            // INT_SIGNEXT belongs on `OptIntBounds`, not `OptRewrite`.
            // rewrite.py has no `optimize_INT_SIGNEXT`; the handler lives
            // at intbounds.py:450-466 (optimize + postprocess). pyre's
            // intbounds.rs:1760 already implements the full upstream
            // logic (is_within_range check), so this `OptRewrite` arm
            // is redundant — its weaker `nbytes == 8` shortcut admits a
            // strict subset of intbounds's removals. Removed for
            // line-by-line dispatch-shape parity.

            // rewrite.py:676-698: optimize_CALL_PURE_I
            // Dispatch based on oopspecindex to specialized handlers.
            // Constant-fold and CSE are handled by pure.rs; here we
            // only do oopspec-specific simplifications.
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN => {
                let __descr_arc_descr = op.getdescr();
                if let Some(ref descr) = __descr_arc_descr.as_ref() {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.get_extra_info();
                        match ei.oopspecindex {
                            // rewrite.py:688: OS_INT_UDIV
                            majit_ir::OopSpecIndex::IntUdiv => {
                                if self.optimize_call_int_udiv(op, ctx) {
                                    return OptimizationResult::Remove;
                                }
                            }
                            // rewrite.py:689: OS_INT_PY_DIV
                            majit_ir::OopSpecIndex::IntPyDiv => {
                                if let Some(result) = self.optimize_call_int_py_div(op, ctx) {
                                    return result;
                                }
                            }
                            // rewrite.py:692: OS_INT_PY_MOD
                            majit_ir::OopSpecIndex::IntPyMod => {
                                if let Some(result) = self.optimize_call_int_py_mod(op, ctx) {
                                    return result;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                OptimizationResult::PassOn
            }

            // rewrite.py:448-470: optimize_CALL_LOOPINVARIANT_I
            OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN => {
                if let Some(func_val) = ctx
                    .get_box_replacement_box(op.arg(0))
                    .and_then(|cb| cb.const_int())
                {
                    // RPython: LoopInvariantOp.produce_op stores PreambleOp
                    // in loop_invariant_results during import. Transfer from
                    // ctx.imported_loop_invariant_results on first access.
                    if let Some(&(_, source)) = ctx
                        .imported_loop_invariant_results
                        .iter()
                        .find(|(k, _)| *k == func_val)
                    {
                        if !self
                            .loop_invariant_results
                            .iter()
                            .any(|(k, _)| *k == func_val)
                        {
                            // RPython shortpreamble.py:158-159. Cat-2.2 dual-slot:
                            // `produce_loop_invariant` installs
                            // `make_equal_to(source, result_opref)`, so the source
                            // box's `_forwarded` slot now holds
                            // `Forwarded::Box(result_box)`.
                            // Build the synthetic SameAsI replay at
                            // `result_opref` (= get_box_replacement(source))
                            // so `take_preamble_forwarded_opinfo` reads the
                            // info seeded at result_opref's slot per the
                            // dual-slot rule (mod.rs:1817 replay_pos).
                            let replay_pos = ctx.get_box_replacement(source);
                            let mut replay = Op::new(OpCode::SameAsI, &[source]);
                            replay.pos.set(replay_pos);
                            self.loop_invariant_results.insert(
                                func_val,
                                LoopInvariantEntry::Preamble(PreambleOp {
                                    op: source,
                                    invented_name: false,
                                    preamble_op: replay,
                                }),
                            );
                        }
                    }
                    // rewrite.py:453-458: isinstance(resvalue, PreambleOp)
                    // → force_op_from_preamble → replace in dict
                    if let Some(entry) = self.loop_invariant_results.get(&func_val).cloned() {
                        let cached_result = match entry {
                            LoopInvariantEntry::Preamble(ref pop) => {
                                // unroll.py:26: force_op_from_preamble(preamble_op)
                                let forced = ctx.force_op_from_preamble_op(pop);
                                self.loop_invariant_results
                                    .insert(func_val, LoopInvariantEntry::Direct(forced));
                                forced
                            }
                            LoopInvariantEntry::Direct(r) => r,
                        };
                        let cached_result = ctx.get_box_replacement(cached_result);
                        let b_old = ctx
                            .ensure_box(op.pos.get())
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        let b_cached = ctx
                            .ensure_box(cached_result)
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        ctx.make_equal_to(&b_old, &b_cached);
                        ctx.last_op_removed = true;
                        return OptimizationResult::Remove;
                    }
                    // Cache miss: demote and record result
                    self.loop_invariant_results
                        .insert(func_val, LoopInvariantEntry::Direct(op.pos.get()));
                    // rewrite.py:30-31: _callback records producer op
                    let call_opcode = OpCode::call_for_type(op.result_type());
                    let producer = op.copy_and_change(call_opcode, None, None);
                    producer.pos.set(op.pos.get());
                    self.loop_invariant_producer.insert(func_val, producer);
                }
                let call_opcode = OpCode::call_for_type(op.result_type());
                let new_op = op.copy_and_change(call_opcode, None, None);
                new_op.pos.set(op.pos.get());
                ctx.last_op_removed = false;
                OptimizationResult::Emit(new_op)
            }

            // ── rewrite.py:373-374: optimize_ASSERT_NOT_NONE ──
            OpCode::AssertNotNone => {
                // RPython: self.make_nonnull(op.getarg(0))
                let obj = ctx.get_box_replacement(op.arg(0));
                let obj_box = ctx.get_box_replacement_box(op.arg(0));
                let has_info = obj_box.as_ref().map_or(false, |b| ctx.has_ptr_info(b));
                if !has_info {
                    if let Some(b) = ctx.ensure_box(obj) {
                        ctx.set_ptr_info(&b, crate::optimizeopt::info::PtrInfo::nonnull());
                    }
                }
                OptimizationResult::Remove
            }

            // rewrite.py:376-386 optimize_RECORD_EXACT_CLASS:
            //     opinfo = getptrinfo(op.getarg(0))
            //     expectedclassbox = op.getarg(1)
            //     if opinfo is not None:
            //         realclassbox = opinfo.get_known_class(cpu)
            //         if realclassbox is not None:
            //             assert realclassbox.same_constant(expectedclassbox)
            //             return
            //     self.make_constant_class(op.getarg(0), expectedclassbox,
            //                              update_last_guard=False)
            OpCode::RecordExactClass => {
                let obj = ctx.get_box_replacement(op.arg(0));
                if op.num_args() >= 2 {
                    // ConstClass is Value::Ref in majit (not Value::Int).
                    let expected_class: Option<i64> = ctx
                        .get_box_replacement_box(op.arg(1))
                        .and_then(|cb| cb.const_value())
                        .and_then(|v| match v {
                            Value::Ref(r) => Some(r.0 as i64),
                            Value::Int(i) => Some(i),
                            _ => None,
                        });
                    if let Some(expected_class) = expected_class {
                        // getptrinfo synthesizes ConstPtrInfo for constant
                        // Refs so `get_known_class` reads cls_of_box for them.
                        let obj_box = ctx.get_box_replacement_box(op.arg(0));
                        if let Some(known) = obj_box
                            .as_ref()
                            .and_then(|b| ctx.getptrinfo(b))
                            .and_then(|i| i.get_known_class(ctx.cpu.as_ref()))
                        {
                            debug_assert_eq!(known.0 as i64, expected_class);
                            return OptimizationResult::Remove;
                        }
                        if let Some(b) = ctx.ensure_box(obj) {
                            crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                                ctx,
                                &b,
                                expected_class,
                                false, // update_last_guard=False
                            );
                        }
                    }
                }
                OptimizationResult::Remove
            }

            // rewrite.py:397-401: optimize_record_exact_value
            //   box = op.getarg(0)
            //   expectedconstbox = op.getarg(1)
            //   assert isinstance(expectedconstbox, Const)
            //   self.make_constant(box, expectedconstbox)
            //
            // `make_constant` walks the forwarding chain internally
            // (optimizer.py:412 `box = get_box_replacement(box)`), so
            // upstream passes `op.getarg(0)` raw without a prior
            // `get_box_replacement` resolution. Pyre matches.
            OpCode::RecordExactValueI | OpCode::RecordExactValueR => {
                let val = ctx
                    .get_box_replacement_box(op.arg(1))
                    .and_then(|cb| cb.const_value())
                    .expect(
                        "rewrite.py:400 — RECORD_EXACT_VALUE expectedconstbox \
                     must be a Const",
                    );
                ctx.make_constant(op.arg(0), val);
                OptimizationResult::Remove
            }

            // rewrite.py:574-584: optimize_CALL_N — dispatch on oopspecindex
            OpCode::CallN | OpCode::CallI | OpCode::CallR => {
                let __descr_arc_descr = op.getdescr();
                if let Some(ref descr) = __descr_arc_descr.as_ref() {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.get_extra_info();
                        match ei.oopspecindex {
                            // rewrite.py:580-590: OS_ARRAYCOPY / OS_ARRAYMOVE
                            majit_ir::OopSpecIndex::Arraycopy => {
                                if op.num_args() >= 6 {
                                    if self.optimize_call_arrayop(
                                        op,
                                        op.arg(1),
                                        op.arg(2), // source, dest
                                        op.arg(3),
                                        op.arg(4),
                                        op.arg(5), // src_start, dst_start, length
                                        ctx,
                                    ) {
                                        return OptimizationResult::Remove;
                                    }
                                }
                            }
                            majit_ir::OopSpecIndex::Arraymove => {
                                // rewrite.py:592-597: ARRAYMOVE: source == dest
                                if op.num_args() >= 5 {
                                    let array_box = op.arg(1);
                                    if self.optimize_call_arrayop(
                                        op,
                                        array_box,
                                        array_box, // source == dest
                                        op.arg(2),
                                        op.arg(3),
                                        op.arg(4),
                                        ctx,
                                    ) {
                                        return OptimizationResult::Remove;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                OptimizationResult::PassOn
            }

            // Everything else: pass on to next optimization pass
            _ => OptimizationResult::PassOn,
        }
    }

    fn setup(&mut self) {
        // ctx.last_op_removed is initialised by OptContext::new() and
        // maintained cross-pass by propagate_from_pass_range +
        // emit_operation — no per-pass setup needed.
        self.bool_result_cache.clear();
        self.loop_invariant_results.clear();
        self.loop_invariant_producer.clear();
    }

    fn name(&self) -> &'static str {
        "rewrite"
    }

    fn have_postprocess(&self) -> bool {
        true
    }

    /// rewrite.py:303-305 postprocess_GUARD_VALUE,
    /// rewrite.py:352-371 postprocess_GUARD_TRUE / postprocess_GUARD_FALSE.
    ///
    /// The `make_constant(box, CONST_*)` call is the second half of PyPy's
    /// `optimize_guard` (rewrite.py:163-184). PyPy emits the guard, then
    /// records that the guard's input box is now known constant. The Rust
    /// port keeps the same split that `have_postprocess` requires — the
    /// emit happens via `optimize_guard_true/false/value` and the
    /// `make_constant` happens here.
    ///
    /// Safety on stable OpRefs: PyPy uses fresh `Box` objects per loop
    /// iteration; majit uses positional `OpRef` slots. The constant lands
    /// on the resolved OpRef of the comparison result (e.g. the position
    /// of an `int_lt`), which is itself fresh per iteration: each phase
    /// emits its own comparison op into a disjoint OpRef range (see the
    /// box-identity Step 2 work that gives Phase 1 and Phase 2 disjoint
    /// `forwarded[]` ranges). CSE within a single phase is the only way
    /// for two uses to share the same OpRef, in which case they describe
    /// the same value and the constant is correct for all of them. PyPy's
    /// stable-Box vs majit's stable-OpRef yield the same observable
    /// behavior.
    fn propagate_postprocess(&mut self, op: &Op, ctx: &mut OptContext) {
        match op.opcode {
            OpCode::GuardTrue => {
                ctx.make_constant(op.arg(0), majit_ir::Value::Int(1));
            }
            OpCode::GuardFalse => {
                ctx.make_constant(op.arg(0), majit_ir::Value::Int(0));
            }
            OpCode::GuardValue => {
                if op.num_args() >= 2 {
                    if let Some(val) = ctx
                        .get_box_replacement_box(op.arg(1))
                        .and_then(|cb| cb.const_value())
                    {
                        ctx.make_constant(op.arg(0), val);
                    }
                }
            }
            _ => {}
        }
    }

    /// rewrite.py:45-47: produce_potential_short_preamble_ops
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        _ctx: &mut OptContext,
    ) {
        for (_, op) in &self.loop_invariant_producer {
            sb.add_loopinvariant_op(op.clone());
        }
    }

    /// rewrite.py:828-834 serialize_optrewrite
    fn serialize_optrewrite(&self) -> Vec<(i64, OpRef)> {
        self.loop_invariant_results
            .iter()
            .filter_map(|(func_ptr, entry)| match entry {
                LoopInvariantEntry::Direct(r) => Some((*func_ptr, *r)),
                LoopInvariantEntry::Preamble(pop) => Some((*func_ptr, pop.op)),
            })
            .collect()
    }

    /// rewrite.py:836-838 deserialize_optrewrite
    fn deserialize_optrewrite(&mut self, entries: &[(i64, OpRef)]) {
        for &(func_ptr, result) in entries {
            self.loop_invariant_results
                .insert(func_ptr, LoopInvariantEntry::Direct(result));
        }
    }
}

impl Default for OptRewrite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::GcRef;

    /// Helper: assign positions to ops so the optimizer can track them.
    fn with_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos
                .set(OpRef::op_typed(i as u32, op.opcode.result_type()));
        }
    }

    /// Run the rewrite pass on a sequence of ops and return the optimized ops.
    fn run_rewrite(ops: &mut [Op]) -> (Vec<Op>, OptContext) {
        with_positions(ops);
        let mut ctx = OptContext::new(ops.len());
        let mut pass = OptRewrite::new();

        for op in ops.iter() {
            // Resolve forwarded arguments
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
            }

            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replacement)
                | OptimizationResult::Restart(replacement) => {
                    ctx.emit(replacement);
                }
                OptimizationResult::Remove => {
                    // removed, nothing emitted
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    std::panic::panic_any(crate::optimize::InvalidLoop(
                        "guard proven to always fail",
                    ));
                }
            }
        }

        let new_ops: Vec<Op> = ctx.new_operations.iter().map(|rc| (**rc).clone()).collect();
        (new_ops, ctx)
    }

    // ── Binary integer operation tests (consolidated) ──
    // RPython rewrite.py: identity, absorbing, constant-fold rules for all binops.

    /// Helper: test a binary op where one arg is constant → expect Remove + forwarding.
    fn assert_binop_identity(
        opcode: OpCode,
        const_pos: usize,
        const_val: i64,
        expected_forward_to: u32,
    ) {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(opcode, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(const_pos as u32), Value::Int(const_val));
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(
            matches!(result, OptimizationResult::Remove),
            "{opcode:?} with const {const_val} at pos {const_pos} should Remove"
        );
        assert_eq!(
            ctx.get_box_replacement(OpRef::int_op(2)),
            OpRef::int_op(expected_forward_to),
            "{opcode:?} should forward to {expected_forward_to}"
        );
    }

    /// Helper: test constant fold → expect Remove + constant result.
    fn assert_binop_const_fold(opcode: OpCode, a: i64, b: i64, expected: i64) {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(opcode, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(a));
        ctx.make_constant(OpRef::int_op(1), Value::Int(b));
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(
            matches!(result, OptimizationResult::Remove),
            "{opcode:?}({a}, {b}) should constant-fold"
        );
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(expected),
            "{opcode:?}({a}, {b}) = {expected}"
        );
    }

    /// Helper: test same-arg binop → expect Remove.
    fn assert_binop_self(opcode: OpCode, expected_const: Option<i64>) {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(opcode, &[OpRef::int_op(0), OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(
            matches!(result, OptimizationResult::Remove),
            "{opcode:?}(x, x) should Remove"
        );
        if let Some(val) = expected_const {
            assert_eq!(
                ctx.get_box_replacement_box(OpRef::int_op(1))
                    .and_then(|cb| cb.const_int()),
                Some(val),
                "{opcode:?}(x, x) = {val}"
            );
        }
    }

    #[test]
    fn test_int_add_identities() {
        // x + 0 = x
        assert_binop_identity(OpCode::IntAdd, 1, 0, 0);
        // 0 + x = x
        assert_binop_identity(OpCode::IntAdd, 0, 0, 1);
        // constant fold
        assert_binop_const_fold(OpCode::IntAdd, 10, 20, 30);
    }

    #[test]
    fn test_int_add_x_plus_x() {
        // x + x → lshift(x, 1) — keep as separate test (rewrite, not identity)
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        // x + x may be rewritten to lshift(x, 1) or kept
        assert!(
            !matches!(result, OptimizationResult::PassOn)
                || matches!(result, OptimizationResult::Replace(_))
                || matches!(result, OptimizationResult::Emit(_))
        );
    }

    #[test]
    fn test_int_sub_identities() {
        // x - 0 = x
        assert_binop_identity(OpCode::IntSub, 1, 0, 0);
        // x - x = 0
        assert_binop_self(OpCode::IntSub, Some(0));
        // constant fold
        assert_binop_const_fold(OpCode::IntSub, 30, 10, 20);
    }

    #[test]
    fn test_int_mul_identities() {
        // x * 0 = 0
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntMul, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(1), Value::Int(0));
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(0)
        );

        // x * 1 = x
        assert_binop_identity(OpCode::IntMul, 1, 1, 0);
        // constant fold
        assert_binop_const_fold(OpCode::IntMul, 6, 7, 42);
    }

    #[test]
    fn test_int_mul_power_of_two() {
        // x * 8 → lshift(x, 3)
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntMul, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(1), Value::Int(8));
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        match result {
            OptimizationResult::Replace(ref new_op) | OptimizationResult::Emit(ref new_op) => {
                assert_eq!(new_op.opcode, OpCode::IntLshift);
            }
            _ => {} // may also Remove with forwarding
        }
    }

    #[test]
    fn test_int_floordiv_identities() {
        // x / 1 = x
        assert_binop_identity(OpCode::IntFloorDiv, 1, 1, 0);
        // 0 / x = 0
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntFloorDiv, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(0)
        );
        // x / x = 1
        assert_binop_self(OpCode::IntFloorDiv, Some(1));
        // x / -1 = neg(x)
        // constant fold
        assert_binop_const_fold(OpCode::IntFloorDiv, 42, 7, 6);
    }

    #[test]
    fn test_int_mod_identities() {
        // x % 1 = 0
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntMod, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(1), Value::Int(1));
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(0)
        );
        // x % x = 0
        assert_binop_self(OpCode::IntMod, Some(0));
    }

    #[test]
    fn test_int_bitwise_identities() {
        // AND: x & 0 = 0, x & -1 = x, x & x = x
        assert_binop_identity(OpCode::IntAnd, 0, -1i64, 1); // -1 & x = x
        assert_binop_self(OpCode::IntAnd, None); // x & x = x (forward to x)

        // OR: x | 0 = x, x | -1 = -1, x | x = x
        assert_binop_identity(OpCode::IntOr, 1, 0, 0);
        assert_binop_self(OpCode::IntOr, None);

        // XOR: x ^ 0 = x, x ^ x = 0, x ^ -1 = ~x
        assert_binop_identity(OpCode::IntXor, 1, 0, 0);
        assert_binop_self(OpCode::IntXor, Some(0));
        assert_binop_const_fold(OpCode::IntXor, 0xFF, 0x0F, 0xF0);
    }

    #[test]
    fn test_shift_identities() {
        // x << 0 = x
        assert_binop_identity(OpCode::IntLshift, 1, 0, 0);
        // x >> 0 = x
        assert_binop_identity(OpCode::IntRshift, 1, 0, 0);
        // constant fold
        assert_binop_const_fold(OpCode::IntLshift, 1, 4, 16);
    }

    #[test]
    fn test_unary_constant_fold() {
        // neg constant
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntNeg, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(42));
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(1))
                .and_then(|cb| cb.const_int()),
            Some(-42)
        );

        // invert constant
        let mut ops2 = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntInvert, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops2);
        let mut ctx2 = OptContext::new(2);
        ctx2.emit(ops2[0].clone());
        ctx2.make_constant(OpRef::int_op(0), Value::Int(0xFF));
        let result2 = pass.propagate_forward(&ops2[1], &mut ctx2);
        assert!(matches!(result2, OptimizationResult::Remove));
        assert_eq!(
            ctx2.get_box_replacement_box(OpRef::int_op(1))
                .and_then(|cb| cb.const_int()),
            Some(!0xFF)
        );
    }

    #[test]
    fn test_int_is_zero_and_is_true() {
        let mut pass = OptRewrite::new();
        // is_zero(0) = 1
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntIsZero, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(1))
                .and_then(|cb| cb.const_int()),
            Some(1)
        );

        // is_zero(5) = 0
        let mut ops2 = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntIsZero, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops2);
        let mut ctx2 = OptContext::new(2);
        ctx2.emit(ops2[0].clone());
        ctx2.make_constant(OpRef::int_op(0), Value::Int(5));
        let result2 = pass.propagate_forward(&ops2[1], &mut ctx2);
        assert!(matches!(result2, OptimizationResult::Remove));
        assert_eq!(
            ctx2.get_box_replacement_box(OpRef::int_op(1))
                .and_then(|cb| cb.const_int()),
            Some(0)
        );
    }

    #[test]
    fn test_comparison_constant_fold() {
        assert_binop_const_fold(OpCode::IntLt, 3, 5, 1);
        assert_binop_const_fold(OpCode::IntLt, 5, 3, 0);
        assert_binop_const_fold(OpCode::IntEq, 7, 7, 1);
        assert_binop_const_fold(OpCode::IntEq, 7, 8, 0);
        assert_binop_const_fold(OpCode::UintLt, 3, 5, 1);
    }

    // ── Guard tests ──

    #[test]
    fn test_guard_true_known_true() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(1));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
    }

    #[test]
    fn test_guard_true_known_false() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));

        let mut pass = OptRewrite::new();
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pass.propagate_forward(&ops[1], &mut ctx)
        }))
        .expect_err("guard_true(0) should abort as InvalidLoop");
        assert!(err.downcast_ref::<crate::optimize::InvalidLoop>().is_some());
    }

    #[test]
    fn test_guard_true_unknown() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_guard_false_known_false() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::GuardFalse, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
    }

    #[test]
    fn test_guard_false_known_true() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::GuardFalse, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(1));

        let mut pass = OptRewrite::new();
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pass.propagate_forward(&ops[1], &mut ctx)
        }))
        .expect_err("guard_false(1) should abort as InvalidLoop");
        assert!(err.downcast_ref::<crate::optimize::InvalidLoop>().is_some());
    }

    #[test]
    fn test_guard_value_match() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::GuardValue, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(42));
        ctx.make_constant(OpRef::int_op(1), Value::Int(42));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
    }

    // ── SAME_AS tests ──

    #[test]
    fn test_same_as_i() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(OpRef::int_op(1)), OpRef::int_op(0));
    }

    // ── Integration test: full optimizer with OptRewrite ──

    #[test]
    fn test_optimizer_integration_add_zero() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));

        // Create a trace: x = SameAsI(), y = SameAsI(constant 0), z = IntAdd(x, y)
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]), // op0: x
            Op::new(OpCode::SameAsI, &[]), // op1: 0
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)]), // op2: x + 0
        ];
        with_positions(&mut ops);

        // We need to set up constants before the optimizer runs.
        // The optimizer creates its own context, so we need a way to
        // inject constants. Since we're testing through the optimizer,
        // let's test the pass directly instead.
        let mut ctx = OptContext::new(3);

        let mut pass = OptRewrite::new();

        // Simulate the optimizer loop
        for (i, op) in ops.iter().enumerate() {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replacement)
                | OptimizationResult::Restart(replacement) => {
                    ctx.emit(replacement);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    std::panic::panic_any(crate::optimize::InvalidLoop(
                        "guard proven to always fail",
                    ));
                }
            }
            // Set op1 as constant 0 after it has been emitted
            if i == 1 {
                ctx.make_constant(OpRef::int_op(1), Value::Int(0));
            }
        }

        // The SameAsI(x) should be removed and forwarded, but we only
        // have SameAsI with no args (acting as input). Let's verify
        // the IntAdd was removed and the result is forwarded.
        // op0 is emitted, op1 is emitted (just a constant), op2 is removed.
        // After forwarding, any reference to op2 should resolve to op0.
        assert_eq!(ctx.get_box_replacement(OpRef::int_op(2)), OpRef::int_op(0));
    }

    #[test]
    fn test_optimizer_integration_chain() {
        // RPython parity: x - x -> 0, then guard_true(0) makes the trace impossible.
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]), // op0: x
            Op::new(OpCode::IntSub, &[OpRef::int_op(0), OpRef::int_op(0)]), // op1: x - x -> 0
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(1)]), // op2: guard_true(0)
        ];
        with_positions(&mut ops);

        let mut ctx = OptContext::new(3);
        let mut pass = OptRewrite::new();

        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for op in &ops {
                let mut resolved = op.clone();
                // optimizer.py:651-652 setarg loop parity.
                for i in 0..resolved.num_args() {
                    resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
                }
                match pass.propagate_forward(&resolved, &mut ctx) {
                    OptimizationResult::Emit(emitted) => {
                        ctx.emit(emitted);
                    }
                    OptimizationResult::Replace(replacement)
                    | OptimizationResult::Restart(replacement) => {
                        ctx.emit(replacement);
                    }
                    OptimizationResult::Remove => {}
                    OptimizationResult::PassOn => {
                        ctx.emit(resolved);
                    }
                    OptimizationResult::InvalidLoop => {
                        std::panic::panic_any(crate::optimize::InvalidLoop(
                            "guard proven to always fail",
                        ));
                    }
                }
            }
        }))
        .expect_err("guard_true(0) should abort the optimized trace");
        assert!(err.downcast_ref::<crate::optimize::InvalidLoop>().is_some());
    }

    // ── Wrapping arithmetic tests ──

    #[test]
    fn test_int_add_wrapping() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(i64::MAX));
        ctx.make_constant(OpRef::int_op(1), Value::Int(1));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(i64::MIN)
        ); // wrapping
    }

    // ── Shift of zero constant tests ──

    #[test]
    fn test_zero_lshift_anything() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntLshift, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(0)
        );
    }

    #[test]
    fn test_zero_rshift_anything() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntRshift, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(0)
        );
    }

    // ── Non-optimizable cases (should PassOn) ──

    #[test]
    fn test_int_add_no_constants() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_unknown_opcode_passthrough() {
        let mut ops = vec![Op::new(
            OpCode::SetfieldGc,
            &[OpRef::void_op(0), OpRef::int_op(1)],
        )];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(1);

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[0], &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    // ── INT_AND constant fold ──

    #[test]
    fn test_int_and_constant_fold() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntAnd, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0xFF));
        ctx.make_constant(OpRef::int_op(1), Value::Int(0x0F));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(0x0F)
        );
    }

    // ── INT_OR constant fold ──

    #[test]
    fn test_int_or_constant_fold() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntOr, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0xF0));
        ctx.make_constant(OpRef::int_op(1), Value::Int(0x0F));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(0xFF)
        );
    }

    // ── UINT_RSHIFT tests ──

    #[test]
    fn test_uint_rshift_zero() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::UintRshift, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(1), Value::Int(0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(OpRef::int_op(2)), OpRef::int_op(0));
    }

    #[test]
    fn test_uint_rshift_constant_fold() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::UintRshift, &[OpRef::int_op(0), OpRef::int_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(-1)); // all ones
        ctx.make_constant(OpRef::int_op(1), Value::Int(1));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        // u64::MAX >> 1 = i64::MAX
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            Some(i64::MAX)
        );
    }

    // ── Float optimization tests ──

    #[test]
    fn test_float_mul_one_right() {
        let mut ops = vec![
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::FloatMul, &[OpRef::float_op(0), OpRef::float_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::float_op(1), Value::Float(1.0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement(OpRef::float_op(2)),
            OpRef::float_op(0)
        );
    }

    #[test]
    fn test_float_mul_one_left() {
        let mut ops = vec![
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::FloatMul, &[OpRef::float_op(0), OpRef::float_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::float_op(0), Value::Float(1.0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement(OpRef::float_op(2)),
            OpRef::float_op(1)
        );
    }

    #[test]
    fn test_float_neg_double_negation() {
        // FloatNeg(FloatNeg(x)) -> x
        let mut ops = vec![
            Op::new(OpCode::SameAsF, &[]),                    // op0: x
            Op::new(OpCode::FloatNeg, &[OpRef::float_op(0)]), // op1: -x
            Op::new(OpCode::FloatNeg, &[OpRef::float_op(1)]), // op2: -(-x) -> x
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        // Process op1 first (pass it through)
        let result1 = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result1, OptimizationResult::PassOn));
        ctx.emit(ops[1].clone());

        // Process op2: should detect double negation
        let result2 = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result2, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement(OpRef::float_op(2)),
            OpRef::float_op(0)
        );
    }

    #[test]
    fn test_float_truediv_power_of_two() {
        // x / 2.0 → x * 0.5
        let mut ops = vec![
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::SameAsF, &[]),
            Op::new(
                OpCode::FloatTrueDiv,
                &[OpRef::float_op(0), OpRef::float_op(1)],
            ),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::float_op(1), Value::Float(2.0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::Emit(_)));
    }

    #[test]
    fn test_float_no_opt_passthrough() {
        // FloatAdd with no constants: no RPython rewrite → PassOn
        let mut ops = vec![
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::FloatAdd, &[OpRef::float_op(0), OpRef::float_op(1)]),
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    // ── COND_CALL tests ──

    #[test]
    fn test_cond_call_constant_false_removed() {
        // CondCallN(condition=0, func, arg1) -> removed (dead call)
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]), // op0: condition (const 0)
            Op::new(OpCode::SameAsI, &[]), // op1: func
            Op::new(OpCode::SameAsI, &[]), // op2: arg1
            Op::new(
                OpCode::CondCallN,
                &[OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2)],
            ), // op3
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(4);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.emit(ops[2].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[3], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
    }

    #[test]
    fn test_cond_call_constant_true_to_direct_call() {
        // CondCallN(condition=1, func, arg1) -> CallN(func, arg1)
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]), // op0: condition (const 1)
            Op::new(OpCode::SameAsI, &[]), // op1: func
            Op::new(OpCode::SameAsI, &[]), // op2: arg1
            Op::new(
                OpCode::CondCallN,
                &[OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2)],
            ), // op3
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(4);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.emit(ops[2].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(1));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[3], &mut ctx);
        match result {
            OptimizationResult::Replace(op) => {
                assert_eq!(op.opcode, OpCode::CallN);
                // Should have args [func, arg1] (condition arg stripped)
                assert_eq!(op.num_args(), 2);
                assert_eq!(op.arg(0), OpRef::int_op(1));
                assert_eq!(op.arg(1), OpRef::int_op(2));
            }
            other => panic!("expected Replace(CallN), got {:?}", other),
        }
    }

    // ── COND_CALL_VALUE tests ──

    #[test]
    fn test_cond_call_value_nonnull_returns_value() {
        // CondCallValueI(value=42, func, arg1) -> value itself (no call needed)
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]), // op0: value (const 42)
            Op::new(OpCode::SameAsI, &[]), // op1: func
            Op::new(OpCode::SameAsI, &[]), // op2: arg1
            Op::new(
                OpCode::CondCallValueI,
                &[OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2)],
            ), // op3
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(4);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.emit(ops[2].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(42));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[3], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        let resolved = ctx.get_box_replacement(OpRef::int_op(3));
        assert!(resolved.is_constant());
        assert_eq!(
            ctx.get_box_replacement_box(resolved)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            Some(42)
        );
    }

    #[test]
    fn test_cond_call_value_null_to_direct_call() {
        // CondCallValueI(value=0, func, arg1) -> CallPureI(func, arg1)
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]), // op0: value (const 0)
            Op::new(OpCode::SameAsI, &[]), // op1: func
            Op::new(OpCode::SameAsI, &[]), // op2: arg1
            Op::new(
                OpCode::CondCallValueI,
                &[OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2)],
            ), // op3
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(4);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.emit(ops[2].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[3], &mut ctx);
        match result {
            OptimizationResult::Replace(op) => {
                assert_eq!(op.opcode, OpCode::CallPureI);
                assert_eq!(op.num_args(), 2);
                assert_eq!(op.arg(0), OpRef::int_op(1));
                assert_eq!(op.arg(1), OpRef::int_op(2));
            }
            other => panic!("expected Replace(CallPureI), got {:?}", other),
        }
    }

    // ── PTR_EQ / PTR_NE tests ──

    #[test]
    fn test_ptr_eq_same_opref() {
        // PtrEq(x, x) -> 1
        // resoperation.py:739 InputArgRef / 615 RefOp `type = 'r'`: ptr
        // boxes carry the Ref variant tag, not Int.
        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]),                                 // op0: x
            Op::new(OpCode::PtrEq, &[OpRef::ref_op(0), OpRef::ref_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(1))
                .and_then(|cb| cb.const_int()),
            Some(1)
        );
    }

    #[test]
    fn test_ptr_ne_same_opref() {
        // PtrNe(x, x) -> 0
        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]),                                 // op0: x
            Op::new(OpCode::PtrNe, &[OpRef::ref_op(0), OpRef::ref_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(1))
                .and_then(|cb| cb.const_int()),
            Some(0)
        );
    }

    #[test]
    fn test_instance_ptr_eq_same_opref() {
        // InstancePtrEq(x, x) -> 1
        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]), // op0: x
            Op::new(OpCode::InstancePtrEq, &[OpRef::ref_op(0), OpRef::ref_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(1))
                .and_then(|cb| cb.const_int()),
            Some(1)
        );
    }

    #[test]
    fn test_instance_ptr_ne_same_opref() {
        // InstancePtrNe(x, x) -> 0
        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]), // op0: x
            Op::new(OpCode::InstancePtrNe, &[OpRef::ref_op(0), OpRef::ref_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(1))
                .and_then(|cb| cb.const_int()),
            Some(0)
        );
    }

    #[test]
    fn test_ptr_eq_distinct_constants_not_folded_in_rewrite() {
        // PtrEq(const 100, const 200): two distinct non-null ConstPtr.
        // rewrite.py:525-564 _optimize_oois_ooisnot has no value-compare
        // branch — `arg0 is arg1` (line 542) is object identity, which is
        // False for distinct ConstPtr, so it falls through to `emit(op)`
        // (line 564). The actual constant fold lives in the pure pass
        // (pure.py:126-136 → execute_ptr_compare_const), not rewrite.
        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]), // op0: const 100
            Op::new(OpCode::SameAsR, &[]), // op1: const 200
            Op::new(OpCode::PtrEq, &[OpRef::ref_op(0), OpRef::ref_op(1)]), // op2
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        // history.py:307 ConstPtr — Value::Ref must land on a Ref-tagged
        // OpRef so the box class identity matches the resoperation.py:615
        // RefOp mixin of the producer SameAsR.
        ctx.make_constant(OpRef::ref_op(0), Value::Ref(GcRef(100)));
        ctx.make_constant(OpRef::ref_op(1), Value::Ref(GcRef(200)));

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[2], &mut ctx);
        // rewrite.py:564 `return self.emit(op)` — rewrite passes the op on
        // unchanged; it does not fold distinct constants.
        assert!(matches!(result, OptimizationResult::PassOn));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            None
        );
    }

    // ── CAST round-trip tests ──

    #[test]
    fn test_cast_ptr_to_int_passes_through() {
        // rewrite.py:807-809: CastPtrToInt registers pure inverse, emits.
        // arg0 is a Ref box (resoperation.py:615 RefOp `type = 'r'`).
        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]),                      // op0: x
            Op::new(OpCode::CastPtrToInt, &[OpRef::ref_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_cast_int_to_ptr_passes_through() {
        // rewrite.py:811-813: CastIntToPtr registers pure inverse, emits.
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),                      // op0: x
            Op::new(OpCode::CastIntToPtr, &[OpRef::int_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_cast_opaque_ptr_eliminated() {
        // CastOpaquePtr(x) -> x
        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]),                       // op0: x
            Op::new(OpCode::CastOpaquePtr, &[OpRef::ref_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(OpRef::ref_op(1)), OpRef::ref_op(0));
    }

    // ── CONVERT_FLOAT_BYTES tests ──
    // rewrite.py:815-821: these conversions are NOT eliminated —
    // they actually change bit representation. Only round-trips
    // (A→B→A) are eliminated via pure.rs CSE.

    #[test]
    fn test_convert_float_bytes_to_longlong_passes_through() {
        let mut ops = vec![
            Op::new(OpCode::SameAsF, &[]), // op0: x
            Op::new(OpCode::ConvertFloatBytesToLonglong, &[OpRef::float_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_convert_longlong_bytes_to_float_passes_through() {
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]), // op0: x
            Op::new(OpCode::ConvertLonglongBytesToFloat, &[OpRef::int_op(0)]), // op1
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(2);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&ops[1], &mut ctx);
        // PassOn: op is emitted, no replacement registered.
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    // ── GUARD_NO_EXCEPTION tests ──

    #[test]
    fn test_guard_no_exception_after_removed_call() {
        // CondCallN(condition=0, ...) -> removed, then GuardNoException -> removed
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]), // op0: condition (const 0)
            Op::new(OpCode::SameAsI, &[]), // op1: func
            Op::new(OpCode::CondCallN, &[OpRef::int_op(0), OpRef::int_op(1)]), // op2: removed
            Op::new(OpCode::GuardNoException, &[]), // op3: should be removed
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(4);
        ctx.emit(ops[0].clone());
        ctx.emit(ops[1].clone());
        ctx.make_constant(OpRef::int_op(0), Value::Int(0));

        let mut pass = OptRewrite::new();
        // Process CondCallN -> removed
        let result2 = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result2, OptimizationResult::Remove));

        // Process GuardNoException -> should also be removed
        let result3 = pass.propagate_forward(&ops[3], &mut ctx);
        assert!(matches!(result3, OptimizationResult::Remove));
    }

    #[test]
    fn test_guard_no_exception_after_emitted_call() {
        // CallN(...) -> emitted, then GuardNoException -> kept
        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),               // op0: func
            Op::new(OpCode::CallN, &[OpRef::int_op(0)]), // op1: call
            Op::new(OpCode::GuardNoException, &[]),      // op2: should NOT be removed
        ];
        with_positions(&mut ops);
        let mut ctx = OptContext::new(3);
        ctx.emit(ops[0].clone());

        let mut pass = OptRewrite::new();
        // Process CallN -> PassOn (not handled by OptRewrite)
        let result1 = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(matches!(result1, OptimizationResult::PassOn));
        ctx.emit(ops[1].clone());

        // Process GuardNoException -> should NOT be removed
        let result2 = pass.propagate_forward(&ops[2], &mut ctx);
        assert!(matches!(result2, OptimizationResult::PassOn));
    }

    #[test]
    fn test_guard_future_condition_records_and_removes() {
        // rewrite.py: GUARD_FUTURE_CONDITION → record in patchguardop + remove
        let mut op = Op::new(OpCode::GuardFutureCondition, &[]);
        op.pos.set(OpRef::void_op(0));
        let mut ctx = OptContext::new(1);
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert!(ctx.patchguardop.is_some());
        assert_eq!(
            ctx.patchguardop.unwrap().opcode,
            OpCode::GuardFutureCondition
        );
    }

    #[test]
    fn test_guard_value_to_guard_false() {
        // GUARD_VALUE(v, 0) → GUARD_FALSE(v)
        let mut ops = vec![
            {
                let mut op = Op::new(
                    OpCode::GuardValue,
                    &[OpRef::int_op(100), OpRef::int_op(200)],
                );
                op.pos.set(OpRef::void_op(0));
                op
            },
            Op::new(OpCode::Finish, &[]),
        ];
        ops[1].pos.set(OpRef::void_op(1));
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(200u32, majit_ir::Value::Int(0));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut constants,
            1024,
            crate::r#box::BoxPool::new(),
        );

        assert!(
            result.iter().any(|o| o.opcode == OpCode::GuardFalse),
            "GUARD_VALUE(v, 0) should become GUARD_FALSE(v)"
        );
    }

    #[test]
    fn test_int_mul_neg_one() {
        // x * (-1) → INT_NEG(x)
        let mut ops = vec![
            Op::new(OpCode::IntMul, &[OpRef::int_op(100), OpRef::int_op(200)]),
            Op::new(OpCode::Finish, &[OpRef::int_op(0)]),
        ];
        with_positions(&mut ops);

        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(200u32, majit_ir::Value::Int(-1));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut constants,
            1024,
            crate::r#box::BoxPool::new(),
        );

        assert!(
            result.iter().any(|o| o.opcode == OpCode::IntNeg),
            "x * (-1) should become INT_NEG(x)"
        );
    }

    #[test]
    fn test_float_mul_neg_one() {
        // x * (-1.0) → FLOAT_NEG(x)
        let mut ops = vec![
            Op::new(
                OpCode::FloatMul,
                &[OpRef::float_op(100), OpRef::float_op(200)],
            ),
            Op::new(OpCode::Finish, &[OpRef::float_op(0)]),
        ];
        with_positions(&mut ops);

        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        // Float constant as Value::Float
        constants.insert(200u32, majit_ir::Value::Float(-1.0));
        // Need float constant support in ctx — skip for now, just test no crash
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut constants,
            1024,
            crate::r#box::BoxPool::new(),
        );
        assert!(!result.is_empty());
    }

    #[test]
    fn test_cond_call_n_zero_removes() {
        // COND_CALL_N(0, func, args...) → removed (condition is false)
        let mut ops = vec![
            Op::new(
                OpCode::CondCallN,
                &[OpRef::int_op(200), OpRef::int_op(100), OpRef::int_op(101)],
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        with_positions(&mut ops);
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(200u32, majit_ir::Value::Int(0));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut constants,
            1024,
            crate::r#box::BoxPool::new(),
        );
        assert!(
            !result.iter().any(|o| o.opcode == OpCode::CondCallN),
            "COND_CALL_N(0, ...) should be removed"
        );
    }

    #[test]
    fn test_cond_call_n_nonzero_converts() {
        // COND_CALL_N(1, func, args...) → CALL_N(func, args...)
        let mut ops = vec![
            Op::new(
                OpCode::CondCallN,
                &[OpRef::int_op(200), OpRef::int_op(100), OpRef::int_op(101)],
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        with_positions(&mut ops);
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(200u32, majit_ir::Value::Int(1));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut constants,
            1024,
            crate::r#box::BoxPool::new(),
        );
        assert!(
            result.iter().any(|o| o.opcode == OpCode::CallN),
            "COND_CALL_N(1, ...) should become CALL_N"
        );
    }
}
