/// Integer bounds optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/intbounds.py.
///
/// Propagates integer bounds information through the trace. When a guard tests
/// a condition that is already known true from integer bounds, the guard can be
/// removed. It also narrows bounds after guards and arithmetic operations.
use majit_ir::{Op, OpCode, OpRef, Value};

use crate::optimizeopt::intutils::IntBound;
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// autogenintrules.py:14-18 `_eq(box1, bound1, box2, bound2)` helper.
/// RPython's rule matcher tests identity OR constant-bound equality so
/// two different boxes with the same known constant value are treated
/// as `x is x`.
fn autogen_eq(box1: OpRef, bound1: &IntBound, box2: OpRef, bound2: &IntBound) -> bool {
    if box1 == box2 {
        return true;
    }
    if bound1.is_constant()
        && bound2.is_constant()
        && bound1.get_constant_int() == bound2.get_constant_int()
    {
        return true;
    }
    false
}

/// autogenintrules.py:54-55 rewrite helper:
/// `newop = self.replace_op_with(op, opcode, args=[...]);
///  self.optimizer.send_extra_operation(newop); return`.
/// Builds a fresh `Op` that reuses the original `op.pos` (preserving
/// Box identity) and returns `OptimizationResult::Restart` so the
/// dispatcher re-runs the new op from `first_optimization`, letting
/// chained OptIntBounds rules (add_zero, int_is_zero, further reassoc)
/// fire on the rewritten op. RPython's `send_extra_operation(opt=None)`
/// (optimizer.py:567-589) is what the dispatcher's `Restart` arm models.
fn replace_with(original: &Op, opcode: OpCode, args: &[OpRef]) -> OptimizationResult {
    let mut new_op = Op::new(opcode, args);
    new_op.pos = original.pos;
    OptimizationResult::Restart(new_op)
}

/// Integer bounds optimization pass.
///
/// Keeps track of the bounds placed on integers by guards and removes
/// redundant guards.
///
/// RPython parity: `OptIntBounds(Optimization)` has NO own bounds storage —
/// all bound state lives on `box._forwarded` and is accessed via the base
/// class `Optimization.getintbound`/`setintbound` (optimizer.py:99-125).
/// In majit the equivalent is `OptContext::forwarded[Forwarded::IntBound]`
/// accessed via `ctx.getintbound`/`ctx.setintbound`/`ctx.with_intbound_mut`.
pub struct OptIntBounds {
    /// intbounds.py: last_emitted_operation — opcode (for overflow guard handling).
    last_emitted_opcode: Option<OpCode>,
    /// intbounds.py: last_emitted_operation — args (for overflow guard handling).
    last_emitted_args: Vec<OpRef>,
    /// intbounds.py: last_emitted_operation — OpRef result.
    last_emitted_ref: OpRef,
}

impl OptIntBounds {
    pub fn new() -> Self {
        OptIntBounds {
            last_emitted_opcode: None,
            last_emitted_args: Vec::new(),
            last_emitted_ref: OpRef::NONE,
        }
    }

    /// optimizer.py:99-113 getintbound — thin wrapper over `ctx.getintbound`.
    /// optimizer.py:100 `assert op.type == 'i'` is enforced inside
    /// `ctx.getintbound`.
    fn getintbound(&self, opref: OpRef, ctx: &mut OptContext) -> IntBound {
        ctx.getintbound(opref)
    }

    /// Intersect a bound into the stored bound for opref. RPython:
    /// `self.getintbound(op).intersect(bound)` (mutates the IntBound stored
    /// on `op._forwarded` in place).
    fn intersect_bound(&mut self, opref: OpRef, bound: &IntBound, ctx: &mut OptContext) {
        ctx.setintbound(opref, bound);
    }

    /// optimizer.py:434: make_constant_int(box, intvalue) — RPython just
    /// forwards to `make_constant(box, ConstInt(intvalue))`. The bounds-range
    /// safety check + `make_eq_const` shrink (optimizer.py:415-426) live in
    /// `OptContext::make_constant`.
    fn make_constant_int(&mut self, op: &Op, value: i64, ctx: &mut OptContext) {
        ctx.make_constant(op.pos, Value::Int(value));
    }

    /// OpRef-keyed variant of make_constant_int. Used by
    /// `propagate_bounds_backward` and the IntIsTrue/IsZero arms which
    /// receive an OpRef rather than an &Op.
    fn make_constant_int_ref(&mut self, opref: OpRef, value: i64, ctx: &mut OptContext) {
        ctx.make_constant(opref, Value::Int(value));
    }

    /// intbounds.py:107 `inv_arg1 = ConstInt(-i1)` parity — synthesise a
    /// fresh `ConstInt` for the inverted constant used by INT_ADD / INT_SUB
    /// pure-cache rewriting.
    ///
    /// RPython mints a fresh `ConstInt(value)` here without consulting any
    /// existing Box; `same_box` later does the value-aware match on the
    /// pure cache side (history.py:204). The previous Rust port searched
    /// `ctx.constants` (op-namespace value cache, populated by
    /// `make_constant(box, ConstInt(...))` on arbitrary Boxes — not just
    /// op results) and returned `OpRef::int_op(idx)` on hit, which mints
    /// the wrong Box family when the original Box at slot `idx` was an
    /// `InputArgInt(idx)` constant-folded via `optimizer.py:410`. Under
    /// variant-aware OpRef Eq (resoperation.rs:290) `IntOp(idx) ==
    /// InputArgInt(idx)` is false, so cache keys silently key-mismatch.
    ///
    /// Mint via `make_constant_int` (constant namespace, CONST_BIT set,
    /// typed `OpRef::const_int(idx)` per history.py:220 ConstInt.type).
    /// Constants minted at this site land in a namespace disjoint from
    /// any op/inputarg Box family, so no aliasing hazard remains.
    fn get_or_make_const(&self, value: i64, ctx: &mut OptContext) -> OpRef {
        ctx.make_constant_int(value)
    }

    // ── Comparison optimizations ──

    fn optimize_int_lt(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_lt(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            OptimizationResult::PassOn
        }
    }

    fn optimize_int_gt(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_gt(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            OptimizationResult::PassOn
        }
    }

    fn optimize_int_le(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_gt(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            OptimizationResult::PassOn
        }
    }

    fn optimize_int_ge(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_lt(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            OptimizationResult::PassOn
        }
    }

    /// autogenintrules.py:1220-1320 optimize_INT_EQ — rules:
    /// eq_different_knownbits / eq_same / eq_one / eq_zero / eq_sub_eq.
    fn optimize_int_eq(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // eq_sub_eq: int_eq(int_sub(x, int_eq(x, a)), a) => 0 (4 forms)
        if let Some(arg0_sub) = self.as_operation(arg0, OpCode::IntSub, ctx) {
            let arg0_0 = ctx.get_box_replacement(arg0_sub.arg(0));
            let arg0_1 = ctx.get_box_replacement(arg0_sub.arg(1));
            let b_arg0_0 = self.getintbound(arg0_0, ctx);
            if let Some(inner_eq) = self.as_operation(arg0_1, OpCode::IntEq, ctx) {
                let inner_0 = ctx.get_box_replacement(inner_eq.arg(0));
                let inner_1 = ctx.get_box_replacement(inner_eq.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                // eq_sub_eq: int_eq(int_sub(x, int_eq(x, a)), a) => 0
                if autogen_eq(inner_0, &b_inner_0, arg0_0, &b_arg0_0)
                    && autogen_eq(arg1, &b1, inner_1, &b_inner_1)
                {
                    self.make_constant_int(op, 0, ctx);
                    return OptimizationResult::Remove;
                }
                // eq_sub_eq: int_eq(int_sub(x, int_eq(a, x)), a) => 0
                if autogen_eq(inner_1, &b_inner_1, arg0_0, &b_arg0_0)
                    && autogen_eq(arg1, &b1, inner_0, &b_inner_0)
                {
                    self.make_constant_int(op, 0, ctx);
                    return OptimizationResult::Remove;
                }
            }
        }
        if let Some(arg1_sub) = self.as_operation(arg1, OpCode::IntSub, ctx) {
            let arg1_0 = ctx.get_box_replacement(arg1_sub.arg(0));
            let arg1_1 = ctx.get_box_replacement(arg1_sub.arg(1));
            let b_arg1_0 = self.getintbound(arg1_0, ctx);
            if let Some(inner_eq) = self.as_operation(arg1_1, OpCode::IntEq, ctx) {
                let inner_0 = ctx.get_box_replacement(inner_eq.arg(0));
                let inner_1 = ctx.get_box_replacement(inner_eq.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                // eq_sub_eq: int_eq(a, int_sub(x, int_eq(x, a))) => 0
                if autogen_eq(inner_0, &b_inner_0, arg1_0, &b_arg1_0)
                    && autogen_eq(inner_1, &b_inner_1, arg0, &b0)
                {
                    self.make_constant_int(op, 0, ctx);
                    return OptimizationResult::Remove;
                }
                // eq_sub_eq: int_eq(a, int_sub(x, int_eq(a, x))) => 0
                if autogen_eq(inner_0, &b_inner_0, arg0, &b0)
                    && autogen_eq(inner_1, &b_inner_1, arg1_0, &b_arg1_0)
                {
                    self.make_constant_int(op, 0, ctx);
                    return OptimizationResult::Remove;
                }
            }
        }
        // eq_same: int_eq(x, x) => 1
        if autogen_eq(arg1, &b1, arg0, &b0) {
            self.make_constant_int(op, 1, ctx);
            return OptimizationResult::Remove;
        }
        // eq_different_knownbits: int_eq(x, y) => 0
        if b0.known_ne(&b1) {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // eq_different_knownbits: int_eq(y, x) => 0
        if b1.known_ne(&b0) {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // eq_one: int_eq(1, x) => x  (when x is bool)
        if b0.is_constant() && b0.get_constant_int() == 1 && b1.is_bool() {
            ctx.make_equal_to(op.pos, arg1);
            return OptimizationResult::Remove;
        }
        // eq_one: int_eq(x, 1) => x  (when x is bool)
        if b1.is_constant() && b1.get_constant_int() == 1 && b0.is_bool() {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        // eq_zero: int_eq(0, x) => int_is_zero(x)
        if b0.is_constant() && b0.get_constant_int() == 0 {
            return replace_with(op, OpCode::IntIsZero, &[arg1]);
        }
        // eq_zero: int_eq(x, 0) => int_is_zero(x)
        if b1.is_constant() && b1.get_constant_int() == 0 {
            return replace_with(op, OpCode::IntIsZero, &[arg0]);
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:1324-1360 optimize_INT_NE — rules:
    /// ne_different_knownbits / ne_same / ne_zero.
    fn optimize_int_ne(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // ne_same: int_ne(x, x) => 0
        if autogen_eq(arg1, &b1, arg0, &b0) {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // ne_different_knownbits: int_ne(x, y) => 1
        if b0.known_ne(&b1) {
            self.make_constant_int(op, 1, ctx);
            return OptimizationResult::Remove;
        }
        // ne_different_knownbits: int_ne(y, x) => 1
        if b1.known_ne(&b0) {
            self.make_constant_int(op, 1, ctx);
            return OptimizationResult::Remove;
        }
        // ne_zero: int_ne(0, x) => int_is_true(x)
        if b0.is_constant() && b0.get_constant_int() == 0 {
            return replace_with(op, OpCode::IntIsTrue, &[arg1]);
        }
        // ne_zero: int_ne(x, 0) => int_is_true(x)
        if b1.is_constant() && b1.get_constant_int() == 0 {
            return replace_with(op, OpCode::IntIsTrue, &[arg0]);
        }
        OptimizationResult::PassOn
    }

    // ── Unsigned comparison optimizations ──

    fn optimize_uint_lt(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_unsigned_lt(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_unsigned_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            OptimizationResult::PassOn
        }
    }

    fn optimize_uint_gt(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_unsigned_gt(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_unsigned_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            OptimizationResult::PassOn
        }
    }

    fn optimize_uint_le(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_unsigned_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_unsigned_gt(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            OptimizationResult::PassOn
        }
    }

    fn optimize_uint_ge(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_unsigned_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_unsigned_lt(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            OptimizationResult::PassOn
        }
    }

    // ── autogenintrules.py ports ──
    //
    // rpython/jit/metainterp/optimizeopt/autogenintrules.py is machine-
    // generated by ruleopt/generate.py and mixed into OptIntBounds via
    // `objectmodel.import_from_mixin(autogenintrules.OptIntAutoGenerated)`
    // at intbounds.py:806. Each optimize_* method below is a direct port
    // of the corresponding `def optimize_INT_*` block.

    /// autogenintrules.py:23-143 optimize_INT_ADD — rules:
    /// add_zero / add_reassoc_consts / add_sub_x_c_c / add_sub_c_x_c.
    fn optimize_int_add(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // add_zero: int_add(0, x) => x
        if b0.is_constant() && b0.get_constant_int() == 0 {
            ctx.make_equal_to(op.pos, arg1);
            return OptimizationResult::Remove;
        }
        // add_zero: int_add(x, 0) => x
        if b1.is_constant() && b1.get_constant_int() == 0 {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        // autogenintrules.py:42-88 — outer const on arg0, inner producer on arg1.
        if b0.is_constant() {
            let c_outer = b0.get_constant_int();
            if let Some(arg1_add) = self.as_operation(arg1, OpCode::IntAdd, ctx) {
                let inner_0 = ctx.get_box_replacement(arg1_add.arg(0));
                let inner_1 = ctx.get_box_replacement(arg1_add.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                // add_reassoc_consts: int_add(C2, int_add(C1, x)) => int_add(x, C1+C2)
                if b_inner_0.is_constant() {
                    let folded = c_outer.wrapping_add(b_inner_0.get_constant_int());
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntAdd, &[inner_1, const_ref]);
                }
                // add_reassoc_consts: int_add(C2, int_add(x, C1)) => int_add(x, C1+C2)
                if b_inner_1.is_constant() {
                    let folded = c_outer.wrapping_add(b_inner_1.get_constant_int());
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntAdd, &[inner_0, const_ref]);
                }
            } else if let Some(arg1_sub) = self.as_operation(arg1, OpCode::IntSub, ctx) {
                let inner_0 = ctx.get_box_replacement(arg1_sub.arg(0));
                let inner_1 = ctx.get_box_replacement(arg1_sub.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                // add_sub_c_x_c: int_add(C2, int_sub(C1, x)) => int_sub(C1+C2, x)
                if b_inner_0.is_constant() {
                    let folded = c_outer.wrapping_add(b_inner_0.get_constant_int());
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntSub, &[const_ref, inner_1]);
                }
                // add_sub_x_c_c: int_add(C2, int_sub(x, C1)) => int_add(x, C2-C1)
                if b_inner_1.is_constant() {
                    let folded = c_outer.wrapping_sub(b_inner_1.get_constant_int());
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntAdd, &[inner_0, const_ref]);
                }
            }
        } else {
            // autogenintrules.py:89-142 — inner producer on arg0, outer const on arg1.
            if let Some(arg0_add) = self.as_operation(arg0, OpCode::IntAdd, ctx) {
                let inner_0 = ctx.get_box_replacement(arg0_add.arg(0));
                let inner_1 = ctx.get_box_replacement(arg0_add.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                if b1.is_constant() {
                    let c_outer = b1.get_constant_int();
                    // add_reassoc_consts: int_add(int_add(C1, x), C2) => int_add(x, C1+C2)
                    if b_inner_0.is_constant() {
                        let folded = b_inner_0.get_constant_int().wrapping_add(c_outer);
                        let const_ref = ctx.make_constant_int(folded);
                        return replace_with(op, OpCode::IntAdd, &[inner_1, const_ref]);
                    }
                    // add_reassoc_consts: int_add(int_add(x, C1), C2) => int_add(x, C1+C2)
                    if b_inner_1.is_constant() {
                        let folded = b_inner_1.get_constant_int().wrapping_add(c_outer);
                        let const_ref = ctx.make_constant_int(folded);
                        return replace_with(op, OpCode::IntAdd, &[inner_0, const_ref]);
                    }
                }
            } else if let Some(arg0_sub) = self.as_operation(arg0, OpCode::IntSub, ctx) {
                let inner_0 = ctx.get_box_replacement(arg0_sub.arg(0));
                let inner_1 = ctx.get_box_replacement(arg0_sub.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                if b1.is_constant() {
                    let c_outer = b1.get_constant_int();
                    // add_sub_c_x_c: int_add(int_sub(C1, x), C2) => int_sub(C1+C2, x)
                    if b_inner_0.is_constant() {
                        let folded = b_inner_0.get_constant_int().wrapping_add(c_outer);
                        let const_ref = ctx.make_constant_int(folded);
                        return replace_with(op, OpCode::IntSub, &[const_ref, inner_1]);
                    }
                    // add_sub_x_c_c: int_add(int_sub(x, C1), C2) => int_add(x, C2-C1)
                    if b_inner_1.is_constant() {
                        let folded = c_outer.wrapping_sub(b_inner_1.get_constant_int());
                        let const_ref = ctx.make_constant_int(folded);
                        return replace_with(op, OpCode::IntAdd, &[inner_0, const_ref]);
                    }
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:147-311 optimize_INT_SUB — rules:
    /// sub_zero / sub_from_zero / sub_x_x / sub_add_consts / sub_add /
    /// sub_add_neg / sub_sub_left_x_c_c / sub_sub_left_c_x_c /
    /// sub_xor_x_y_y / sub_or_x_y_y / sub_invert_one.
    fn optimize_int_sub(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // sub_x_x: int_sub(x, x) => 0
        if autogen_eq(arg1, &b1, arg0, &b0) {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // sub_add: int_sub(int_add(x, y), y) => x / int_sub(int_add(y, x), y) => x
        if let Some(arg0_int_add) = self.as_operation(arg0, OpCode::IntAdd, ctx) {
            let arg0_0 = ctx.get_box_replacement(arg0_int_add.arg(0));
            let arg0_1 = ctx.get_box_replacement(arg0_int_add.arg(1));
            let b0_0 = self.getintbound(arg0_0, ctx);
            let b0_1 = self.getintbound(arg0_1, ctx);
            if autogen_eq(arg1, &b1, arg0_1, &b0_1) {
                ctx.make_equal_to(op.pos, arg0_0);
                return OptimizationResult::Remove;
            }
            if autogen_eq(arg1, &b1, arg0_0, &b0_0) {
                ctx.make_equal_to(op.pos, arg0_1);
                return OptimizationResult::Remove;
            }
        } else if let Some(arg0_int_or) = self.as_operation(arg0, OpCode::IntOr, ctx) {
            // sub_or_x_y_y: int_sub(int_or(x, y), y) => x  (when x & y == 0)
            let arg0_0 = ctx.get_box_replacement(arg0_int_or.arg(0));
            let arg0_1 = ctx.get_box_replacement(arg0_int_or.arg(1));
            let b0_0 = self.getintbound(arg0_0, ctx);
            let b0_1 = self.getintbound(arg0_1, ctx);
            if autogen_eq(arg1, &b1, arg0_1, &b0_1) && b0_0.and_bound(&b0_1).known_eq_const(0) {
                ctx.make_equal_to(op.pos, arg0_0);
                return OptimizationResult::Remove;
            }
            if autogen_eq(arg1, &b1, arg0_0, &b0_0) && b0_1.and_bound(&b0_0).known_eq_const(0) {
                ctx.make_equal_to(op.pos, arg0_1);
                return OptimizationResult::Remove;
            }
        } else if let Some(arg0_int_xor) = self.as_operation(arg0, OpCode::IntXor, ctx) {
            // sub_xor_x_y_y: int_sub(int_xor(x, y), y) => x  (when x & y == 0)
            let arg0_0 = ctx.get_box_replacement(arg0_int_xor.arg(0));
            let arg0_1 = ctx.get_box_replacement(arg0_int_xor.arg(1));
            let b0_0 = self.getintbound(arg0_0, ctx);
            let b0_1 = self.getintbound(arg0_1, ctx);
            if autogen_eq(arg1, &b1, arg0_1, &b0_1) && b0_0.and_bound(&b0_1).known_eq_const(0) {
                ctx.make_equal_to(op.pos, arg0_0);
                return OptimizationResult::Remove;
            }
            if autogen_eq(arg1, &b1, arg0_0, &b0_0) && b0_1.and_bound(&b0_0).known_eq_const(0) {
                ctx.make_equal_to(op.pos, arg0_1);
                return OptimizationResult::Remove;
            }
        }
        // sub_zero: int_sub(x, 0) => x
        if b1.is_constant() && b1.get_constant_int() == 0 {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        // sub_from_zero: int_sub(0, x) => int_neg(x)
        if b0.is_constant() && b0.get_constant_int() == 0 {
            return replace_with(op, OpCode::IntNeg, &[arg1]);
        }
        if let Some(arg0_int_invert) = self.as_operation(arg0, OpCode::IntInvert, ctx) {
            let arg0_0 = ctx.get_box_replacement(arg0_int_invert.arg(0));
            // sub_invert_one: int_sub(int_invert(x), -1) => int_neg(x)
            if b1.is_constant() && b1.get_constant_int() == -1 {
                return replace_with(op, OpCode::IntNeg, &[arg0_0]);
            }
        } else if let Some(arg0_int_add) = self.as_operation(arg0, OpCode::IntAdd, ctx) {
            let arg0_0 = ctx.get_box_replacement(arg0_int_add.arg(0));
            let arg0_1 = ctx.get_box_replacement(arg0_int_add.arg(1));
            let b0_0 = self.getintbound(arg0_0, ctx);
            let b0_1 = self.getintbound(arg0_1, ctx);
            if b1.is_constant() {
                let c_outer = b1.get_constant_int();
                // sub_add_consts: int_sub(int_add(C1, x), C2) => int_sub(x, C-C1)
                if b0_0.is_constant() {
                    let folded = c_outer.wrapping_sub(b0_0.get_constant_int());
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntSub, &[arg0_1, const_ref]);
                }
                // sub_add_consts: int_sub(int_add(x, C1), C2) => int_sub(x, C-C1)
                if b0_1.is_constant() {
                    let folded = c_outer.wrapping_sub(b0_1.get_constant_int());
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntSub, &[arg0_0, const_ref]);
                }
            }
        } else if let Some(arg0_int_sub) = self.as_operation(arg0, OpCode::IntSub, ctx) {
            let arg0_0 = ctx.get_box_replacement(arg0_int_sub.arg(0));
            let arg0_1 = ctx.get_box_replacement(arg0_int_sub.arg(1));
            let b0_0 = self.getintbound(arg0_0, ctx);
            let b0_1 = self.getintbound(arg0_1, ctx);
            if b1.is_constant() {
                let c_outer = b1.get_constant_int();
                // sub_sub_left_c_x_c: int_sub(int_sub(C1, x), C2) => int_sub(C1-C2, x)
                if b0_0.is_constant() {
                    let folded = b0_0.get_constant_int().wrapping_sub(c_outer);
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntSub, &[const_ref, arg0_1]);
                }
                // sub_sub_left_x_c_c: int_sub(int_sub(x, C1), C2) => int_sub(x, C1+C2)
                if b0_1.is_constant() {
                    let folded = b0_1.get_constant_int().wrapping_add(c_outer);
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntSub, &[arg0_0, const_ref]);
                }
            }
        }
        if let Some(arg1_int_add) = self.as_operation(arg1, OpCode::IntAdd, ctx) {
            let arg1_0 = ctx.get_box_replacement(arg1_int_add.arg(0));
            let arg1_1 = ctx.get_box_replacement(arg1_int_add.arg(1));
            let b1_0 = self.getintbound(arg1_0, ctx);
            let b1_1 = self.getintbound(arg1_1, ctx);
            // sub_add_neg: int_sub(y, int_add(x, y)) => int_neg(x)
            if autogen_eq(arg1_1, &b1_1, arg0, &b0) {
                return replace_with(op, OpCode::IntNeg, &[arg1_0]);
            }
            // sub_add_neg: int_sub(y, int_add(y, x)) => int_neg(x)
            if autogen_eq(arg1_0, &b1_0, arg0, &b0) {
                return replace_with(op, OpCode::IntNeg, &[arg1_1]);
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:315-410 optimize_INT_MUL — rules:
    /// mul_zero / mul_one / mul_minus_one / mul_pow2_const / mul_lshift.
    fn optimize_int_mul(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // mul_zero: int_mul(0, x) => 0
        if b0.is_constant() && b0.get_constant_int() == 0 {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // mul_zero: int_mul(x, 0) => 0
        if b1.is_constant() && b1.get_constant_int() == 0 {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // mul_one: int_mul(1, x) => x
        if b0.is_constant() && b0.get_constant_int() == 1 {
            ctx.make_equal_to(op.pos, arg1);
            return OptimizationResult::Remove;
        }
        // mul_one: int_mul(x, 1) => x
        if b1.is_constant() && b1.get_constant_int() == 1 {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        // Outer const on arg0: mul_minus_one / mul_pow2_const
        if b0.is_constant() {
            let c = b0.get_constant_int();
            // mul_minus_one: int_mul(-1, x) => int_neg(x)
            if c == -1 {
                return replace_with(op, OpCode::IntNeg, &[arg1]);
            }
            // mul_pow2_const: int_mul(C, x) where C > 0 && (C & (C-1)) == 0
            // => int_lshift(x, highest_bit(C))
            if c > 0 && (c & c.wrapping_sub(1)) == 0 {
                let shift = c.trailing_zeros() as i64;
                let shift_ref = ctx.make_constant_int(shift);
                return replace_with(op, OpCode::IntLshift, &[arg1, shift_ref]);
            }
        } else if let Some(arg0_lshift) = self.as_operation(arg0, OpCode::IntLshift, ctx) {
            // mul_lshift: int_mul(int_lshift(1, y), x) => int_lshift(x, y)
            let inner_0 = ctx.get_box_replacement(arg0_lshift.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_lshift.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_0.is_constant()
                && b_inner_0.get_constant_int() == 1
                && b_inner_1.known_ge_const(0)
                && b_inner_1.known_le_const(64)
            {
                return replace_with(op, OpCode::IntLshift, &[arg1, inner_1]);
            }
        }
        // Outer const on arg1: mul_minus_one / mul_pow2_const
        if b1.is_constant() {
            let c = b1.get_constant_int();
            // mul_minus_one: int_mul(x, -1) => int_neg(x)
            if c == -1 {
                return replace_with(op, OpCode::IntNeg, &[arg0]);
            }
            // mul_pow2_const: int_mul(x, C) where C > 0 && pow2
            if c > 0 && (c & c.wrapping_sub(1)) == 0 {
                let shift = c.trailing_zeros() as i64;
                let shift_ref = ctx.make_constant_int(shift);
                return replace_with(op, OpCode::IntLshift, &[arg0, shift_ref]);
            }
        } else if let Some(arg1_lshift) = self.as_operation(arg1, OpCode::IntLshift, ctx) {
            // mul_lshift: int_mul(x, int_lshift(1, y)) => int_lshift(x, y)
            let inner_0 = ctx.get_box_replacement(arg1_lshift.arg(0));
            let inner_1 = ctx.get_box_replacement(arg1_lshift.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_0.is_constant()
                && b_inner_0.get_constant_int() == 1
                && b_inner_1.known_ge_const(0)
                && b_inner_1.known_le_const(64)
            {
                return replace_with(op, OpCode::IntLshift, &[arg0, inner_1]);
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:414-581 optimize_INT_AND — rules:
    /// and_known_result / and_x_c_in_range / and_x_x / and_idempotent /
    /// and_reassoc_consts / and_absorb / and_or.
    fn optimize_int_and(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // and_known_result: int_and(a, b) bound is constant => ConstInt(C)
        let bound = b0.and_bound(&b1);
        if bound.is_constant() {
            self.make_constant_int(op, bound.get_constant_int(), ctx);
            return OptimizationResult::Remove;
        }
        let bound = b1.and_bound(&b0);
        if bound.is_constant() {
            self.make_constant_int(op, bound.get_constant_int(), ctx);
            return OptimizationResult::Remove;
        }
        // and_x_c_in_range: int_and(C, x) => x  (when 0 <= x.lower &&
        //                                      x.upper <= C & ~(C+1))
        if b0.is_constant() {
            let c = b0.get_constant_int();
            let mask = !((c as u64).wrapping_add(1)) as i64;
            if b1.lower >= 0 && b1.upper <= c & mask {
                ctx.make_equal_to(op.pos, arg1);
                return OptimizationResult::Remove;
            }
        }
        // and_x_c_in_range: int_and(x, C) => x
        if b1.is_constant() {
            let c = b1.get_constant_int();
            let mask = !((c as u64).wrapping_add(1)) as i64;
            if b0.lower >= 0 && b0.upper <= c & mask {
                ctx.make_equal_to(op.pos, arg0);
                return OptimizationResult::Remove;
            }
        }
        // and_x_x: int_and(a, a) => a
        if autogen_eq(arg1, &b1, arg0, &b0) {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        // and_idempotent: int_and(x, y) => x
        // when b1.tvalue | ~(b0.tvalue | b0.tmask) == all-ones
        if b1.tvalue | !(b0.tvalue | b0.tmask) == u64::MAX {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        // and_idempotent: int_and(y, x) => x
        if b0.tvalue | !(b1.tvalue | b1.tmask) == u64::MAX {
            ctx.make_equal_to(op.pos, arg1);
            return OptimizationResult::Remove;
        }
        if b0.is_constant() {
            let c_outer = b0.get_constant_int();
            if let Some(arg1_and) = self.as_operation(arg1, OpCode::IntAnd, ctx) {
                let inner_0 = ctx.get_box_replacement(arg1_and.arg(0));
                let inner_1 = ctx.get_box_replacement(arg1_and.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                // and_reassoc_consts: int_and(C2, int_and(C1, x)) => int_and(x, C1&C2)
                if b_inner_0.is_constant() {
                    let folded = b_inner_0.get_constant_int() & c_outer;
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntAnd, &[inner_1, const_ref]);
                }
                // and_reassoc_consts: int_and(C2, int_and(x, C1)) => int_and(x, C1&C2)
                if b_inner_1.is_constant() {
                    let folded = b_inner_1.get_constant_int() & c_outer;
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntAnd, &[inner_0, const_ref]);
                }
            }
        } else if let Some(arg0_and) = self.as_operation(arg0, OpCode::IntAnd, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_and.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_and.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_0.is_constant() {
                if b1.is_constant() {
                    // and_reassoc_consts: int_and(int_and(C1, x), C2) => int_and(x, C1&C2)
                    let folded = b_inner_0.get_constant_int() & b1.get_constant_int();
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntAnd, &[inner_1, const_ref]);
                }
            }
            if b_inner_1.is_constant() {
                if b1.is_constant() {
                    // and_reassoc_consts: int_and(int_and(x, C1), C2) => int_and(x, C1&C2)
                    let folded = b_inner_1.get_constant_int() & b1.get_constant_int();
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntAnd, &[inner_0, const_ref]);
                }
            }
            // and_absorb: int_and(int_and(a, b), a) => int_and(a, b)
            if autogen_eq(arg1, &b1, inner_0, &b_inner_0) {
                return replace_with(op, OpCode::IntAnd, &[inner_0, inner_1]);
            }
            // and_absorb: int_and(int_and(b, a), a) => int_and(a, b)
            if autogen_eq(arg1, &b1, inner_1, &b_inner_1) {
                return replace_with(op, OpCode::IntAnd, &[inner_1, inner_0]);
            }
        } else if let Some(arg0_or) = self.as_operation(arg0, OpCode::IntOr, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_or.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_or.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            // and_or: int_and(int_or(x, y), z) => int_and(x, z)
            if b_inner_1.and_bound(&b1).known_eq_const(0) {
                return replace_with(op, OpCode::IntAnd, &[inner_0, arg1]);
            }
            // and_or: int_and(int_or(y, x), z) => int_and(x, z)
            if b_inner_0.and_bound(&b1).known_eq_const(0) {
                return replace_with(op, OpCode::IntAnd, &[inner_1, arg1]);
            }
        }
        // Symmetric arm: producer on arg1.
        if let Some(arg1_and) = self.as_operation(arg1, OpCode::IntAnd, ctx) {
            let inner_0 = ctx.get_box_replacement(arg1_and.arg(0));
            let inner_1 = ctx.get_box_replacement(arg1_and.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            // and_absorb: int_and(a, int_and(a, b)) => int_and(a, b)
            if autogen_eq(inner_0, &b_inner_0, arg0, &b0) {
                return replace_with(op, OpCode::IntAnd, &[arg0, inner_1]);
            }
            // and_absorb: int_and(a, int_and(b, a)) => int_and(a, b)
            if autogen_eq(inner_1, &b_inner_1, arg0, &b0) {
                return replace_with(op, OpCode::IntAnd, &[arg0, inner_0]);
            }
        } else if let Some(arg1_or) = self.as_operation(arg1, OpCode::IntOr, ctx) {
            let inner_0 = ctx.get_box_replacement(arg1_or.arg(0));
            let inner_1 = ctx.get_box_replacement(arg1_or.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            // and_or: int_and(z, int_or(x, y)) => int_and(x, z)
            if b_inner_1.and_bound(&b0).known_eq_const(0) {
                return replace_with(op, OpCode::IntAnd, &[inner_0, arg0]);
            }
            // and_or: int_and(z, int_or(y, x)) => int_and(x, z)
            if b_inner_0.and_bound(&b0).known_eq_const(0) {
                return replace_with(op, OpCode::IntAnd, &[inner_1, arg0]);
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:585-787 optimize_INT_OR — rules:
    /// or_known_result / or_x_x / or_idempotent / or_reassoc_consts /
    /// or_and_two_parts / or_absorb.
    ///
    /// Note on RPython parity: the auto-generated source includes
    /// pairs of `or_and_two_parts` arms with identical `_eq` predicates
    /// (autogenintrules.py:661/668, 677/684, 701/708, 717/724); the
    /// second of each pair is dead because the first always returns
    /// when its predicate holds. Preserved here to mirror upstream's
    /// auto-generated output line for line.
    fn optimize_int_or(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // or_known_result: int_or(a, b) bound is constant => ConstInt(C)
        let bound = b0.or_bound(&b1);
        if bound.is_constant() {
            self.make_constant_int(op, bound.get_constant_int(), ctx);
            return OptimizationResult::Remove;
        }
        let bound = b1.or_bound(&b0);
        if bound.is_constant() {
            self.make_constant_int(op, bound.get_constant_int(), ctx);
            return OptimizationResult::Remove;
        }
        // or_x_x: int_or(a, a) => a
        if autogen_eq(arg1, &b1, arg0, &b0) {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        // or_idempotent: int_or(x, y) => x
        // when b0.tvalue | ~(b1.tvalue | b1.tmask) == all-ones
        if b0.tvalue | !(b1.tvalue | b1.tmask) == u64::MAX {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        // or_idempotent: int_or(y, x) => x
        if b1.tvalue | !(b0.tvalue | b0.tmask) == u64::MAX {
            ctx.make_equal_to(op.pos, arg1);
            return OptimizationResult::Remove;
        }
        if b0.is_constant() {
            let c_outer = b0.get_constant_int();
            if let Some(arg1_or) = self.as_operation(arg1, OpCode::IntOr, ctx) {
                let inner_0 = ctx.get_box_replacement(arg1_or.arg(0));
                let inner_1 = ctx.get_box_replacement(arg1_or.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                // or_reassoc_consts: int_or(C2, int_or(C1, x)) => int_or(x, C1|C2)
                if b_inner_0.is_constant() {
                    let folded = b_inner_0.get_constant_int() | c_outer;
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntOr, &[inner_1, const_ref]);
                }
                // or_reassoc_consts: int_or(C2, int_or(x, C1)) => int_or(x, C1|C2)
                if b_inner_1.is_constant() {
                    let folded = b_inner_1.get_constant_int() | c_outer;
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntOr, &[inner_0, const_ref]);
                }
            }
        } else if let Some(arg0_and) = self.as_operation(arg0, OpCode::IntAnd, ctx) {
            let arg0_0 = ctx.get_box_replacement(arg0_and.arg(0));
            let arg0_1 = ctx.get_box_replacement(arg0_and.arg(1));
            let b_arg0_0 = self.getintbound(arg0_0, ctx);
            let b_arg0_1 = self.getintbound(arg0_1, ctx);
            // or_and_two_parts shapes when outer = int_and(C, x):
            if b_arg0_0.is_constant() {
                let c_arg0_0 = b_arg0_0.get_constant_int();
                if let Some(arg1_and) = self.as_operation(arg1, OpCode::IntAnd, ctx) {
                    let arg1_0 = ctx.get_box_replacement(arg1_and.arg(0));
                    let arg1_1 = ctx.get_box_replacement(arg1_and.arg(1));
                    let b_arg1_0 = self.getintbound(arg1_0, ctx);
                    let b_arg1_1 = self.getintbound(arg1_1, ctx);
                    if b_arg1_0.is_constant() {
                        // int_or(int_and(C1, x), int_and(C2, x)) => int_and(x, C1|C2)
                        if autogen_eq(arg1_1, &b_arg1_1, arg0_1, &b_arg0_1) {
                            let folded = c_arg0_0 | b_arg1_0.get_constant_int();
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[arg0_1, const_ref]);
                        }
                        // dead twin per autogenintrules.py:668-673
                        if autogen_eq(arg1_1, &b_arg1_1, arg0_1, &b_arg0_1) {
                            let folded = b_arg1_0.get_constant_int() | c_arg0_0;
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[arg0_1, const_ref]);
                        }
                    }
                    if b_arg1_1.is_constant() {
                        // int_or(int_and(C, x), int_and(x, C1)) => int_and(x, C|C1)
                        if autogen_eq(arg1_0, &b_arg1_0, arg0_1, &b_arg0_1) {
                            let folded = b_arg1_1.get_constant_int() | c_arg0_0;
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[arg0_1, const_ref]);
                        }
                        // dead twin per autogenintrules.py:684-689
                        if autogen_eq(arg1_0, &b_arg1_0, arg0_1, &b_arg0_1) {
                            let folded = c_arg0_0 | b_arg1_1.get_constant_int();
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[arg0_1, const_ref]);
                        }
                    }
                }
            }
            if b_arg0_1.is_constant() {
                let c_arg0_1 = b_arg0_1.get_constant_int();
                if let Some(arg1_and) = self.as_operation(arg1, OpCode::IntAnd, ctx) {
                    let arg1_0 = ctx.get_box_replacement(arg1_and.arg(0));
                    let arg1_1 = ctx.get_box_replacement(arg1_and.arg(1));
                    let b_arg1_0 = self.getintbound(arg1_0, ctx);
                    let b_arg1_1 = self.getintbound(arg1_1, ctx);
                    if b_arg1_0.is_constant() {
                        // int_or(int_and(x, C1), int_and(C2, x)) => int_and(x, C)
                        if autogen_eq(arg1_1, &b_arg1_1, arg0_0, &b_arg0_0) {
                            let folded = c_arg0_1 | b_arg1_0.get_constant_int();
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[arg0_0, const_ref]);
                        }
                        // dead twin per autogenintrules.py:708-713
                        if autogen_eq(arg1_1, &b_arg1_1, arg0_0, &b_arg0_0) {
                            let folded = b_arg1_0.get_constant_int() | c_arg0_1;
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[arg0_0, const_ref]);
                        }
                    }
                    if b_arg1_1.is_constant() {
                        // int_or(int_and(x, C1), int_and(x, C2)) => int_and(x, C)
                        if autogen_eq(arg1_0, &b_arg1_0, arg0_0, &b_arg0_0) {
                            let folded = c_arg0_1 | b_arg1_1.get_constant_int();
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[arg0_0, const_ref]);
                        }
                        // dead twin per autogenintrules.py:724-729
                        if autogen_eq(arg1_0, &b_arg1_0, arg0_0, &b_arg0_0) {
                            let folded = b_arg1_1.get_constant_int() | c_arg0_1;
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[arg0_0, const_ref]);
                        }
                    }
                }
            }
        } else if let Some(arg0_or) = self.as_operation(arg0, OpCode::IntOr, ctx) {
            let arg0_0 = ctx.get_box_replacement(arg0_or.arg(0));
            let arg0_1 = ctx.get_box_replacement(arg0_or.arg(1));
            let b_arg0_0 = self.getintbound(arg0_0, ctx);
            let b_arg0_1 = self.getintbound(arg0_1, ctx);
            if b_arg0_0.is_constant() && b1.is_constant() {
                // or_reassoc_consts: int_or(int_or(C1, x), C2) => int_or(x, C1|C2)
                let folded = b_arg0_0.get_constant_int() | b1.get_constant_int();
                let const_ref = ctx.make_constant_int(folded);
                return replace_with(op, OpCode::IntOr, &[arg0_1, const_ref]);
            }
            if b_arg0_1.is_constant() && b1.is_constant() {
                // or_reassoc_consts: int_or(int_or(x, C1), C2) => int_or(x, C1|C2)
                let folded = b_arg0_1.get_constant_int() | b1.get_constant_int();
                let const_ref = ctx.make_constant_int(folded);
                return replace_with(op, OpCode::IntOr, &[arg0_0, const_ref]);
            }
            // or_absorb: int_or(int_or(a, b), a) => int_or(a, b)
            if autogen_eq(arg1, &b1, arg0_0, &b_arg0_0) {
                return replace_with(op, OpCode::IntOr, &[arg0_0, arg0_1]);
            }
            // or_absorb: int_or(int_or(b, a), a) => int_or(a, b)
            if autogen_eq(arg1, &b1, arg0_1, &b_arg0_1) {
                return replace_with(op, OpCode::IntOr, &[arg0_1, arg0_0]);
            }
        }
        // Symmetric arm: producer on arg1.
        if let Some(arg1_or) = self.as_operation(arg1, OpCode::IntOr, ctx) {
            let arg1_0 = ctx.get_box_replacement(arg1_or.arg(0));
            let arg1_1 = ctx.get_box_replacement(arg1_or.arg(1));
            let b_arg1_0 = self.getintbound(arg1_0, ctx);
            let b_arg1_1 = self.getintbound(arg1_1, ctx);
            // or_absorb: int_or(a, int_or(a, b)) => int_or(a, b)
            if autogen_eq(arg1_0, &b_arg1_0, arg0, &b0) {
                return replace_with(op, OpCode::IntOr, &[arg0, arg1_1]);
            }
            // or_absorb: int_or(a, int_or(b, a)) => int_or(a, b)
            if autogen_eq(arg1_1, &b_arg1_1, arg0, &b0) {
                return replace_with(op, OpCode::IntOr, &[arg0, arg1_0]);
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:791-942 optimize_INT_XOR — rules:
    /// xor_x_x / xor_reassoc_consts / xor_absorb / xor_zero /
    /// xor_minus_1 / xor_known_result / xor_is_not.
    fn optimize_int_xor(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // xor_known_result: int_xor(a, b) bound is constant => ConstInt(C)
        let bound = b0.xor_bound(&b1);
        if bound.is_constant() {
            self.make_constant_int(op, bound.get_constant_int(), ctx);
            return OptimizationResult::Remove;
        }
        let bound = b1.xor_bound(&b0);
        if bound.is_constant() {
            self.make_constant_int(op, bound.get_constant_int(), ctx);
            return OptimizationResult::Remove;
        }
        // xor_x_x: int_xor(a, a) => 0
        if autogen_eq(arg1, &b1, arg0, &b0) {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // xor_zero: int_xor(0, a) => a   (else: producer-xor absorbs)
        if b0.is_constant() {
            if b0.get_constant_int() == 0 {
                ctx.make_equal_to(op.pos, arg1);
                return OptimizationResult::Remove;
            }
        } else if let Some(arg0_xor) = self.as_operation(arg0, OpCode::IntXor, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_xor.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_xor.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            // xor_absorb: int_xor(int_xor(a, b), b) => a
            if autogen_eq(arg1, &b1, inner_1, &b_inner_1) {
                ctx.make_equal_to(op.pos, inner_0);
                return OptimizationResult::Remove;
            }
            // xor_absorb: int_xor(int_xor(b, a), b) => a
            if autogen_eq(arg1, &b1, inner_0, &b_inner_0) {
                ctx.make_equal_to(op.pos, inner_1);
                return OptimizationResult::Remove;
            }
        }
        // xor_zero: int_xor(a, 0) => a   (else: producer-xor absorbs)
        if b1.is_constant() {
            if b1.get_constant_int() == 0 {
                ctx.make_equal_to(op.pos, arg0);
                return OptimizationResult::Remove;
            }
        } else if let Some(arg1_xor) = self.as_operation(arg1, OpCode::IntXor, ctx) {
            let inner_0 = ctx.get_box_replacement(arg1_xor.arg(0));
            let inner_1 = ctx.get_box_replacement(arg1_xor.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            // xor_absorb: int_xor(b, int_xor(a, b)) => a
            if autogen_eq(inner_1, &b_inner_1, arg0, &b0) {
                ctx.make_equal_to(op.pos, inner_0);
                return OptimizationResult::Remove;
            }
            // xor_absorb: int_xor(b, int_xor(b, a)) => a
            if autogen_eq(inner_0, &b_inner_0, arg0, &b0) {
                ctx.make_equal_to(op.pos, inner_1);
                return OptimizationResult::Remove;
            }
        }
        // Constant-on-left arm (b_arg_0 constant): reassoc_consts/minus_1/is_not
        if b0.is_constant() {
            let c_outer = b0.get_constant_int();
            if let Some(arg1_xor) = self.as_operation(arg1, OpCode::IntXor, ctx) {
                let inner_0 = ctx.get_box_replacement(arg1_xor.arg(0));
                let inner_1 = ctx.get_box_replacement(arg1_xor.arg(1));
                let b_inner_0 = self.getintbound(inner_0, ctx);
                let b_inner_1 = self.getintbound(inner_1, ctx);
                // xor_reassoc_consts: int_xor(C2, int_xor(C1, x)) => int_xor(x, C1^C2)
                if b_inner_0.is_constant() {
                    let folded = b_inner_0.get_constant_int() ^ c_outer;
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntXor, &[inner_1, const_ref]);
                }
                // xor_reassoc_consts: int_xor(C2, int_xor(x, C1)) => int_xor(x, C1^C2)
                if b_inner_1.is_constant() {
                    let folded = b_inner_1.get_constant_int() ^ c_outer;
                    let const_ref = ctx.make_constant_int(folded);
                    return replace_with(op, OpCode::IntXor, &[inner_0, const_ref]);
                }
            }
            // xor_minus_1: int_xor(-1, x) => int_invert(x)
            if c_outer == -1 {
                return replace_with(op, OpCode::IntInvert, &[arg1]);
            }
            // xor_is_not: int_xor(1, x) => int_is_zero(x)  (when x is bool)
            if c_outer == 1 && b1.is_bool() {
                return replace_with(op, OpCode::IntIsZero, &[arg1]);
            }
        } else if let Some(arg0_xor) = self.as_operation(arg0, OpCode::IntXor, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_xor.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_xor.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_0.is_constant() && b1.is_constant() {
                // xor_reassoc_consts: int_xor(int_xor(C1, x), C2) => int_xor(x, C1^C2)
                let folded = b_inner_0.get_constant_int() ^ b1.get_constant_int();
                let const_ref = ctx.make_constant_int(folded);
                return replace_with(op, OpCode::IntXor, &[inner_1, const_ref]);
            }
            if b_inner_1.is_constant() && b1.is_constant() {
                // xor_reassoc_consts: int_xor(int_xor(x, C1), C2) => int_xor(x, C1^C2)
                let folded = b_inner_1.get_constant_int() ^ b1.get_constant_int();
                let const_ref = ctx.make_constant_int(folded);
                return replace_with(op, OpCode::IntXor, &[inner_0, const_ref]);
            }
        }
        // Constant-on-right arm (b_arg_1 constant): minus_1/is_not
        if b1.is_constant() {
            let c = b1.get_constant_int();
            // xor_minus_1: int_xor(x, -1) => int_invert(x)
            if c == -1 {
                return replace_with(op, OpCode::IntInvert, &[arg0]);
            }
            // xor_is_not: int_xor(x, 1) => int_is_zero(x)  (when x is bool)
            if c == 1 && b0.is_bool() {
                return replace_with(op, OpCode::IntIsZero, &[arg0]);
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:1432-1443 optimize_INT_INVERT — rules: invert_invert.
    fn optimize_int_invert(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        // invert_invert: int_invert(int_invert(x)) => x
        if let Some(inner) = self.as_operation(arg0, OpCode::IntInvert, ctx) {
            let inner_0 = ctx.get_box_replacement(inner.arg(0));
            ctx.make_equal_to(op.pos, inner_0);
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:1447-1458 optimize_INT_NEG — rules: neg_neg.
    fn optimize_int_neg(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        // neg_neg: int_neg(int_neg(x)) => x
        if let Some(inner) = self.as_operation(arg0, OpCode::IntNeg, ctx) {
            let inner_0 = ctx.get_box_replacement(inner.arg(0));
            ctx.make_equal_to(op.pos, inner_0);
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:946-1109 optimize_INT_LSHIFT — rules:
    /// lshift_zero_x / lshift_x_zero / lshift_rshift_c_c / lshift_and_rshift /
    /// lshift_urshift_c_c / lshift_and_urshift / lshift_lshift_c_c.
    /// LONG_BIT inlined as 64 (pyre is 64-bit only).
    fn optimize_int_lshift(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // lshift_zero_x: int_lshift(0, x) => 0
        if b0.is_constant() && b0.get_constant_int() == 0 {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // lshift_x_zero: int_lshift(x, 0) => x
        if b1.is_constant() && b1.get_constant_int() == 0 {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        if let Some(arg0_and) = self.as_operation(arg0, OpCode::IntAnd, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_and.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_and.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_0.is_constant() {
                let c_arg0_0 = b_inner_0.get_constant_int();
                if let Some(rshift) = self.as_operation(inner_1, OpCode::IntRshift, ctx) {
                    let r0 = ctx.get_box_replacement(rshift.arg(0));
                    let r1 = ctx.get_box_replacement(rshift.arg(1));
                    let b_r1 = self.getintbound(r1, ctx);
                    if b_r1.is_constant() && b1.is_constant() {
                        let c_r1 = b_r1.get_constant_int();
                        let c_arg1 = b1.get_constant_int();
                        // lshift_and_rshift: int_lshift(int_and(C2, int_rshift(x, C1)), C1) => int_and(x, C)
                        if c_arg1 == c_r1 && (0..64).contains(&c_r1) {
                            let folded = c_arg0_0.wrapping_shl(c_r1 as u32);
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[r0, const_ref]);
                        }
                    }
                } else if let Some(urshift) = self.as_operation(inner_1, OpCode::UintRshift, ctx) {
                    let r0 = ctx.get_box_replacement(urshift.arg(0));
                    let r1 = ctx.get_box_replacement(urshift.arg(1));
                    let b_r1 = self.getintbound(r1, ctx);
                    if b_r1.is_constant() && b1.is_constant() {
                        let c_r1 = b_r1.get_constant_int();
                        let c_arg1 = b1.get_constant_int();
                        // lshift_and_urshift: int_lshift(int_and(C2, uint_rshift(x, C1)), C1) => int_and(x, C)
                        if c_arg1 == c_r1 && (0..64).contains(&c_r1) {
                            let folded = c_arg0_0.wrapping_shl(c_r1 as u32);
                            let const_ref = ctx.make_constant_int(folded);
                            return replace_with(op, OpCode::IntAnd, &[r0, const_ref]);
                        }
                    }
                }
            } else if let Some(rshift) = self.as_operation(inner_0, OpCode::IntRshift, ctx) {
                let r0 = ctx.get_box_replacement(rshift.arg(0));
                let r1 = ctx.get_box_replacement(rshift.arg(1));
                let b_r1 = self.getintbound(r1, ctx);
                if b_r1.is_constant() && b_inner_1.is_constant() && b1.is_constant() {
                    let c_r1 = b_r1.get_constant_int();
                    let c_arg0_1 = b_inner_1.get_constant_int();
                    let c_arg1 = b1.get_constant_int();
                    // lshift_and_rshift: int_lshift(int_and(int_rshift(x, C1), C2), C1) => int_and(x, C)
                    if c_arg1 == c_r1 && (0..64).contains(&c_r1) {
                        let folded = c_arg0_1.wrapping_shl(c_r1 as u32);
                        let const_ref = ctx.make_constant_int(folded);
                        return replace_with(op, OpCode::IntAnd, &[r0, const_ref]);
                    }
                }
            } else if let Some(urshift) = self.as_operation(inner_0, OpCode::UintRshift, ctx) {
                let r0 = ctx.get_box_replacement(urshift.arg(0));
                let r1 = ctx.get_box_replacement(urshift.arg(1));
                let b_r1 = self.getintbound(r1, ctx);
                if b_r1.is_constant() && b_inner_1.is_constant() && b1.is_constant() {
                    let c_r1 = b_r1.get_constant_int();
                    let c_arg0_1 = b_inner_1.get_constant_int();
                    let c_arg1 = b1.get_constant_int();
                    // lshift_and_urshift: int_lshift(int_and(uint_rshift(x, C1), C2), C1) => int_and(x, C)
                    if c_arg1 == c_r1 && (0..64).contains(&c_r1) {
                        let folded = c_arg0_1.wrapping_shl(c_r1 as u32);
                        let const_ref = ctx.make_constant_int(folded);
                        return replace_with(op, OpCode::IntAnd, &[r0, const_ref]);
                    }
                }
            }
        } else if let Some(arg0_lshift) = self.as_operation(arg0, OpCode::IntLshift, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_lshift.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_lshift.arg(1));
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_1.is_constant() && b1.is_constant() {
                let c1 = b_inner_1.get_constant_int();
                let c2 = b1.get_constant_int();
                // lshift_lshift_c_c: int_lshift(int_lshift(x, C1), C2) => int_lshift(x, C)
                if (0..64).contains(&c1) && (0..64).contains(&c2) {
                    let folded = c1.wrapping_add(c2);
                    if folded < 64 {
                        let const_ref = ctx.make_constant_int(folded);
                        return replace_with(op, OpCode::IntLshift, &[inner_0, const_ref]);
                    }
                }
            }
        } else if let Some(arg0_rshift) = self.as_operation(arg0, OpCode::IntRshift, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_rshift.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_rshift.arg(1));
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_1.is_constant() && b1.is_constant() {
                let c1 = b_inner_1.get_constant_int();
                let c2 = b1.get_constant_int();
                // lshift_rshift_c_c: int_lshift(int_rshift(x, C1), C1) => int_and(x, -1 << C1)
                if c2 == c1 && (0..64).contains(&c1) {
                    let mask = (-1i64).wrapping_shl(c1 as u32);
                    let const_ref = ctx.make_constant_int(mask);
                    return replace_with(op, OpCode::IntAnd, &[inner_0, const_ref]);
                }
            }
        } else if let Some(arg0_urshift) = self.as_operation(arg0, OpCode::UintRshift, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_urshift.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_urshift.arg(1));
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_1.is_constant() && b1.is_constant() {
                let c1 = b_inner_1.get_constant_int();
                let c2 = b1.get_constant_int();
                // lshift_urshift_c_c: int_lshift(uint_rshift(x, C1), C1) => int_and(x, -1 << C1)
                if c2 == c1 && (0..64).contains(&c1) {
                    let mask = (-1i64).wrapping_shl(c1 as u32);
                    let const_ref = ctx.make_constant_int(mask);
                    return replace_with(op, OpCode::IntAnd, &[inner_0, const_ref]);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:1113-1169 optimize_INT_RSHIFT — rules:
    /// rshift_zero_x / rshift_x_zero / rshift_known_result / rshift_lshift /
    /// rshift_rshift_c_c. LONG_BIT inlined as 64.
    fn optimize_int_rshift(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // rshift_zero_x: int_rshift(0, x) => 0
        if b0.is_constant() && b0.get_constant_int() == 0 {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // rshift_known_result: int_rshift(a, b) bound is constant => ConstInt(C)
        let bound = b0.rshift_bound(&b1);
        if bound.is_constant() {
            self.make_constant_int(op, bound.get_constant_int(), ctx);
            return OptimizationResult::Remove;
        }
        if let Some(arg0_lshift) = self.as_operation(arg0, OpCode::IntLshift, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_lshift.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_lshift.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            // rshift_lshift: int_rshift(int_lshift(x, y), y) => x
            if autogen_eq(arg1, &b1, inner_1, &b_inner_1)
                && b_inner_0.lshift_bound_cannot_overflow(&b_inner_1)
            {
                ctx.make_equal_to(op.pos, inner_0);
                return OptimizationResult::Remove;
            }
        }
        // rshift_x_zero: int_rshift(x, 0) => x
        if b1.is_constant() && b1.get_constant_int() == 0 {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        if let Some(arg0_rshift) = self.as_operation(arg0, OpCode::IntRshift, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_rshift.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_rshift.arg(1));
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_1.is_constant() && b1.is_constant() {
                let c1 = b_inner_1.get_constant_int();
                let c2 = b1.get_constant_int();
                // rshift_rshift_c_c: int_rshift(int_rshift(x, C1), C2) => int_rshift(x, min(C1+C2, 63))
                if (0..64).contains(&c1) && (0..64).contains(&c2) {
                    let folded = c1.wrapping_add(c2).min(63);
                    if (0..64).contains(&folded) {
                        let const_ref = ctx.make_constant_int(folded);
                        return replace_with(op, OpCode::IntRshift, &[inner_0, const_ref]);
                    }
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:1364-1399 optimize_INT_IS_TRUE — rules:
    /// is_true_bool / is_true_true / is_true_and_minint.
    fn optimize_int_is_true(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let b0 = self.getintbound(arg0, ctx);
        // is_true_true: int_is_true(x) => 1
        // when bound excludes 0: lower > 0, or upper < 0, or any tvalue bit set.
        if b0.lower > 0 || b0.upper < 0 || b0.tvalue != 0 {
            self.make_constant_int(op, 1, ctx);
            return OptimizationResult::Remove;
        }
        // is_true_bool: int_is_true(x) => x  (when x is bool)
        if b0.is_bool() {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        if let Some(arg0_and) = self.as_operation(arg0, OpCode::IntAnd, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_and.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_and.arg(1));
            let b_inner_0 = self.getintbound(inner_0, ctx);
            let b_inner_1 = self.getintbound(inner_1, ctx);
            // is_true_and_minint: int_is_true(int_and(MININT, x)) => int_lt(x, 0)
            if b_inner_0.is_constant() && b_inner_0.get_constant_int() == i64::MIN {
                let zero_ref = ctx.make_constant_int(0);
                return replace_with(op, OpCode::IntLt, &[inner_1, zero_ref]);
            }
            // is_true_and_minint: int_is_true(int_and(x, MININT)) => int_lt(x, 0)
            if b_inner_1.is_constant() && b_inner_1.get_constant_int() == i64::MIN {
                let zero_ref = ctx.make_constant_int(0);
                return replace_with(op, OpCode::IntLt, &[inner_0, zero_ref]);
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:1403-1411 optimize_INT_IS_ZERO — rules: is_zero_true.
    fn optimize_int_is_zero(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let b0 = self.getintbound(arg0, ctx);
        // is_zero_true: int_is_zero(x) => 0
        // when bound excludes 0: lower > 0, or upper < 0, or any tvalue bit set.
        if b0.lower > 0 || b0.upper < 0 || b0.tvalue != 0 {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:1415-1428 optimize_INT_FORCE_GE_ZERO — rules:
    /// force_ge_zero_pos / force_ge_zero_neg.
    fn optimize_int_force_ge_zero(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let b0 = self.getintbound(arg0, ctx);
        // force_ge_zero_neg: int_force_ge_zero(x) => 0  (when x < 0)
        if b0.known_lt_const(0) {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // force_ge_zero_pos: int_force_ge_zero(x) => x  (when x >= 0)
        if b0.known_nonnegative() {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py:1173-1216 optimize_UINT_RSHIFT — rules:
    /// urshift_zero_x / urshift_x_zero / urshift_known_result /
    /// urshift_lshift_x_c_c.
    fn optimize_uint_rshift(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // urshift_zero_x: uint_rshift(0, x) => 0
        if b0.is_constant() && b0.get_constant_int() == 0 {
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        // urshift_known_result: uint_rshift(a, b) bound is constant => ConstInt(C)
        let bound = b0.urshift_bound(&b1);
        if bound.is_constant() {
            self.make_constant_int(op, bound.get_constant_int(), ctx);
            return OptimizationResult::Remove;
        }
        // urshift_x_zero: uint_rshift(x, 0) => x
        if b1.is_constant() && b1.get_constant_int() == 0 {
            ctx.make_equal_to(op.pos, arg0);
            return OptimizationResult::Remove;
        }
        if let Some(arg0_lshift) = self.as_operation(arg0, OpCode::IntLshift, ctx) {
            let inner_0 = ctx.get_box_replacement(arg0_lshift.arg(0));
            let inner_1 = ctx.get_box_replacement(arg0_lshift.arg(1));
            let b_inner_1 = self.getintbound(inner_1, ctx);
            if b_inner_1.is_constant() && b1.is_constant() {
                let c1 = b_inner_1.get_constant_int();
                let c2 = b1.get_constant_int();
                // urshift_lshift_x_c_c: uint_rshift(int_lshift(x, C), C) => int_and(x, mask)
                // mask = intmask(r_uint(-1 << C) >> r_uint(C))
                if c2 == c1 && (0..64).contains(&c1) {
                    let mask = ((((-1i64) as u64) << c1) >> c1) as i64;
                    let const_ref = ctx.make_constant_int(mask);
                    return replace_with(op, OpCode::IntAnd, &[inner_0, const_ref]);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// autogenintrules.py helper used for pattern matching:
    /// `as_operation(box, opnum)` returns the producing op iff its opcode
    /// matches, else None.
    fn as_operation(&self, opref: OpRef, opcode: OpCode, ctx: &OptContext) -> Option<Op> {
        let op = ctx.get_producing_op(opref)?;
        if op.opcode == opcode { Some(op) } else { None }
    }

    // ── Arithmetic postprocessing ──

    /// intbounds.py:114-146 postprocess_INT_ADD
    fn postprocess_int_add(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        // intbounds.py:119-123: if arg0 is arg1: b = b0.lshift_bound(1) (x+x is even)
        let b = if arg0 == arg1 {
            b0.lshift_bound(&IntBound::from_constant(1))
        } else {
            let b1 = self.getintbound(arg1, ctx);
            b0.add_bound(&b1)
        };
        self.intersect_bound(op.pos, &b, ctx);
        // intbounds.py:125-127:
        //   self.optimizer.pure_from_args2(rop.INT_SUB, op, arg1, arg0)
        //   self.optimizer.pure_from_args2(rop.INT_SUB, op, arg0, arg1)
        ctx.register_pure_from_args2(OpCode::IntSub, arg0, op.pos, arg1);
        ctx.register_pure_from_args2(OpCode::IntSub, arg1, op.pos, arg0);
        // intbounds.py:128-142: pick the constant arg, fall back to commutative
        // swap so `arg1` ends up holding the non-const operand.
        let (inv_const, other) = if let Some(c) = ctx.get_constant_int(arg0) {
            if c == i64::MIN {
                return;
            }
            (c, arg1)
        } else if let Some(c) = ctx.get_constant_int(arg1) {
            if c == i64::MIN {
                return;
            }
            (c, arg0)
        } else {
            return;
        };
        let neg_ref = self.get_or_make_const(-inv_const, ctx);
        // intbounds.py:143-146:
        //   self.optimizer.pure_from_args2(rop.INT_SUB, arg1, inv_arg0, op)
        //   self.optimizer.pure_from_args2(rop.INT_SUB, arg1, op, inv_arg0)
        //   self.optimizer.pure_from_args2(rop.INT_ADD, op, inv_arg0, arg1)
        //   self.optimizer.pure_from_args2(rop.INT_ADD, inv_arg0, op, arg1)
        ctx.register_pure_from_args2(OpCode::IntSub, op.pos, other, neg_ref);
        ctx.register_pure_from_args2(OpCode::IntSub, neg_ref, other, op.pos);
        ctx.register_pure_from_args2(OpCode::IntAdd, other, op.pos, neg_ref);
        ctx.register_pure_from_args2(OpCode::IntAdd, other, neg_ref, op.pos);
    }

    /// intbounds.py: INT_SUB postprocess with constant inversion synthesis.
    fn postprocess_int_sub(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        let b = b0.sub_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
        // Synthesis: INT_SUB(a,b)=res → INT_ADD(res,b)=a, INT_SUB(a,res)=b
        ctx.register_pure_from_args2(OpCode::IntAdd, arg0, op.pos, arg1);
        ctx.register_pure_from_args2(OpCode::IntSub, arg1, arg0, op.pos);
        // intbounds.py: constant inversion for INT_SUB
        if let Some(c1) = ctx.get_constant_int(arg1) {
            if c1 != i64::MIN {
                let neg_ref = self.get_or_make_const(-c1, ctx);
                ctx.register_pure_from_args2(OpCode::IntAdd, op.pos, arg0, neg_ref);
                ctx.register_pure_from_args2(OpCode::IntAdd, op.pos, neg_ref, arg0);
                ctx.register_pure_from_args2(OpCode::IntSub, arg0, op.pos, neg_ref);
                ctx.register_pure_from_args2(OpCode::IntSub, neg_ref, op.pos, arg0);
            }
        }
    }

    fn postprocess_int_mul(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.mul_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_int_and(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.and_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:60-71 postprocess_INT_OR
    fn postprocess_int_or(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // intbounds.py:65: if b0.and_bound(b1).known_eq_const(0):
        if b0.and_bound(&b1).known_eq_const(0) {
            // intbounds.py:66-69:
            //   pure_from_args2(rop.INT_ADD, arg0, arg1, op)
            //   pure_from_args2(rop.INT_XOR, arg0, arg1, op)
            ctx.register_pure_from_args2(OpCode::IntAdd, op.pos, arg0, arg1);
            ctx.register_pure_from_args2(OpCode::IntXor, op.pos, arg0, arg1);
        }
        let b = b0.or_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:73-84 postprocess_INT_XOR
    fn postprocess_int_xor(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // intbounds.py:78: if b0.and_bound(b1).known_eq_const(0):
        if b0.and_bound(&b1).known_eq_const(0) {
            // intbounds.py:79-82:
            //   pure_from_args2(rop.INT_ADD, arg0, arg1, op)
            //   pure_from_args2(rop.INT_OR, arg0, arg1, op)
            ctx.register_pure_from_args2(OpCode::IntAdd, op.pos, arg0, arg1);
            ctx.register_pure_from_args2(OpCode::IntOr, op.pos, arg0, arg1);
        }
        let b = b0.xor_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py: INT_LSHIFT pure_from_args synthesis.
    /// If res = INT_LSHIFT(a, b), then a = INT_RSHIFT(res, b).
    fn postprocess_int_lshift(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        let b = b0.lshift_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
        // intbounds.py:185: only synthesize reverse if lshift cannot overflow
        if b0.lshift_bound_cannot_overflow(&b1) {
            ctx.register_pure_from_args2(OpCode::IntRshift, arg0, op.pos, arg1);
        }
    }

    fn postprocess_int_rshift(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.rshift_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_uint_rshift(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.urshift_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_int_floordiv(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.py_div_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_int_mod(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.mod_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_int_neg(&mut self, op: &Op, ctx: &mut OptContext) {
        let b = self.getintbound(op.arg(0), ctx);
        let result = b.neg_bound();
        self.intersect_bound(op.pos, &result, ctx);
    }

    fn postprocess_int_invert(&mut self, op: &Op, ctx: &mut OptContext) {
        let b = self.getintbound(op.arg(0), ctx);
        let result = b.invert_bound();
        self.intersect_bound(op.pos, &result, ctx);
    }

    fn postprocess_int_force_ge_zero(&mut self, op: &Op, ctx: &mut OptContext) {
        let b_arg = self.getintbound(op.arg(0), ctx);
        let mut result = IntBound::nonnegative();
        if b_arg.upper >= 0 {
            let _ = result.make_le(&b_arg);
        }
        self.intersect_bound(op.pos, &result, ctx);
    }

    fn postprocess_arraylen_gc(&mut self, op: &Op, ctx: &mut OptContext) {
        // intbounds.py:503-505
        //     array = self.ensure_ptr_info_arg0(op)
        //     self.optimizer.setintbound(op, array.getlenbound(None))
        // `getlenbound` lazily fills `ArrayPtrInfo.lenbound` (`info.py:573`).
        // Wrapper triggers `mirror_forwarded_to_box` so the BoxRef snapshot
        // sees the populated lenbound — same rationale as STRLEN/UNICODELEN
        // postprocessors below.
        let bound = ctx.with_ensured_ptr_info_arg0(op, |mut array| array.getlenbound(None));
        if let Some(bound) = bound {
            ctx.setintbound(op.pos, &bound);
        }
    }

    fn postprocess_strlen(&mut self, op: &Op, ctx: &mut OptContext) {
        // intbounds.py:507-510
        //     self.make_nonnull_str(op.getarg(0), vstring.mode_string)
        //     array = getptrinfo(op.getarg(0))
        //     self.optimizer.setintbound(op, array.getlenbound(vstring.mode_string))
        //
        // Rust note: PyPy splits this into make_nonnull_str + getptrinfo
        // because make_nonnull_str installs the StrPtrInfo on box._forwarded
        // and getptrinfo reads it back. The Rust port reaches the same state
        // by calling ensure_ptr_info_arg0 (which constructs StrPtrInfo for
        // STRLEN per optimizer.py:490-491) and then invoking getlenbound on
        // the returned handle.
        ctx.make_nonnull_str(op.arg(0), 0);
        // `getlenbound` is a lazy-fill mutator on `StrPtrInfo.lenbound`
        // (`vstring.py:62`). Route through `with_ensured_ptr_info_arg0`
        // so `mirror_forwarded_to_box` re-syncs the BoxRef snapshot of
        // the StrPtrInfo after the lenbound slot is populated.
        let bound = ctx.with_ensured_ptr_info_arg0(op, |mut info| info.getlenbound(Some(0)));
        if let Some(bound) = bound {
            ctx.setintbound(op.pos, &bound);
        }
    }

    fn postprocess_unicodelen(&mut self, op: &Op, ctx: &mut OptContext) {
        // intbounds.py:512-515
        //     self.make_nonnull_str(op.getarg(0), vstring.mode_unicode)
        //     array = getptrinfo(op.getarg(0))
        //     self.optimizer.setintbound(op, array.getlenbound(vstring.mode_unicode))
        ctx.make_nonnull_str(op.arg(0), 1);
        // Same wrapper rationale as `postprocess_strlen` — lazy-fill
        // mutation on StrPtrInfo.lenbound needs BoxRef mirror.
        let bound = ctx.with_ensured_ptr_info_arg0(op, |mut info| info.getlenbound(Some(1)));
        if let Some(bound) = bound {
            ctx.setintbound(op.pos, &bound);
        }
    }

    // ── INT_SIGNEXT optimization ──

    fn optimize_int_signext(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let b = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        if b1.is_constant() {
            let byte_size = b1.get_constant_int();
            let numbits = byte_size * 8;
            let start = -(1i64 << (numbits - 1));
            let stop = 1i64 << (numbits - 1);
            if b.is_within_range(start, stop - 1) {
                // The value already fits; replace with the input.
                ctx.replace_op(op.pos, op.arg(0));
                return OptimizationResult::Remove;
            }
        }
        OptimizationResult::PassOn
    }

    fn postprocess_int_signext(&mut self, op: &Op, ctx: &mut OptContext) {
        let b1 = self.getintbound(op.arg(1), ctx);
        if b1.is_constant() {
            let byte_size = b1.get_constant_int();
            let numbits = byte_size * 8;
            let start = -(1i64 << (numbits - 1));
            let stop = 1i64 << (numbits - 1);
            let _ = ctx.with_intbound_mut(op.pos, |bm| bm.intersect_const(start, stop - 1));
        }
    }

    // ── Overflow operations ──

    /// intbounds.py:244-256 optimize_INT_ADD_OVF
    fn optimize_int_add_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        if b0.add_bound_cannot_overflow(&b1) {
            // replace_op_with(op, INT_ADD) + send_extra_operation
            let mut new_op = Op::new(OpCode::IntAdd, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            OptimizationResult::Emit(new_op)
        } else {
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_add_ovf(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b = if arg0 == arg1 {
            b0.mul2_bound_no_overflow()
        } else {
            let b1 = self.getintbound(arg1, ctx);
            b0.add_bound_no_overflow(&b1)
        };
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:275-287 optimize_INT_SUB_OVF
    fn optimize_int_sub_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        if arg0 == arg1 {
            // arg0.same_box(arg1) → x - x = 0
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.sub_bound_cannot_overflow(&b1) {
            // replace_op_with(op, INT_SUB) + send_extra_operation
            let mut new_op = Op::new(OpCode::IntSub, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            OptimizationResult::Emit(new_op)
        } else {
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_sub_ovf(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.sub_bound_no_overflow(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:298-305 optimize_INT_MUL_OVF
    fn optimize_int_mul_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        if b0.mul_bound_cannot_overflow(&b1) {
            // replace_op_with(op, INT_MUL) + send_extra_operation
            let mut new_op = Op::new(OpCode::IntMul, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            OptimizationResult::Emit(new_op)
        } else {
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_mul_ovf(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b = if arg0 == arg1 {
            b0.square_bound_no_overflow()
        } else {
            let b1 = self.getintbound(arg1, ctx);
            b0.mul_bound_no_overflow(&b1)
        };
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:209-229 optimize_GUARD_NO_OVERFLOW
    fn optimize_guard_no_overflow(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let _ = op;
        // intbounds.py:210-220:
        //   lastop = self.last_emitted_operation
        //   if lastop is not None:
        //     opnum = lastop.getopnum()
        //     if opnum not in (INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF):
        //       return   # guard killed
        //   else:
        //     return   # falls out of `if`, no emit, guard killed
        let Some(opcode) = self.last_emitted_opcode else {
            return OptimizationResult::Remove;
        };
        if !matches!(
            opcode,
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf
        ) {
            return OptimizationResult::Remove;
        }
        // intbounds.py:222-228: synthesize the non-overflowing inverse
        let result = self.last_emitted_ref;
        if self.last_emitted_args.len() >= 2 && !result.is_none() {
            let arg0 = self.last_emitted_args[0];
            let arg1 = self.last_emitted_args[1];
            match opcode {
                OpCode::IntAddOvf => {
                    ctx.register_pure_from_args2(OpCode::IntSub, arg0, result, arg1);
                    ctx.register_pure_from_args2(OpCode::IntSub, arg1, result, arg0);
                }
                OpCode::IntSubOvf => {
                    ctx.register_pure_from_args2(OpCode::IntAdd, arg0, result, arg1);
                    ctx.register_pure_from_args2(OpCode::IntSub, arg1, arg0, result);
                }
                _ => {}
            }
        }
        // intbounds.py:229: return self.emit(op)
        OptimizationResult::PassOn
    }

    /// intbounds.py:231-242 optimize_GUARD_OVERFLOW
    fn optimize_guard_overflow(&mut self, op: &Op) -> OptimizationResult {
        let _ = op;
        // intbounds.py:232-233: if lastop is None: return
        let Some(opcode) = self.last_emitted_opcode else {
            return OptimizationResult::Remove;
        };
        // intbounds.py:236-238: if opnum not in OVF_ops: raise InvalidLoop
        if !matches!(
            opcode,
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf
        ) {
            return OptimizationResult::InvalidLoop;
        }
        // intbounds.py:240: return self.emit(op)
        OptimizationResult::PassOn
    }

    // ── Guard optimizations ──

    /// RPython intbounds.py: no optimize_GUARD_TRUE handler.
    /// Guards are NOT handled in optimize (propagate_forward).
    /// The guard is emitted via the default emit path, and
    /// propagate_bounds_backward runs in postprocess_GUARD_TRUE
    /// (intbounds.py:56).
    ///
    /// The constant/bound checks for guard removal are handled by
    /// the default dispatch in RPython. In majit, we do them here
    /// because we don't have per-opcode default dispatch.
    fn optimize_guard_true(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let cond_ref = ctx.get_box_replacement(op.arg(0));

        // Constant check: if condition is known constant nonzero, remove guard.
        if let Some(val) = ctx.get_constant_int(cond_ref) {
            if val != 0 {
                return OptimizationResult::Remove;
            }
        }

        if !matches!(ctx.opref_type(cond_ref), Some(majit_ir::Type::Int)) {
            return OptimizationResult::PassOn;
        }

        // Bound check: if bound proves always nonzero, remove guard.
        let b = self.getintbound(cond_ref, ctx);
        if b.known_gt_const(0) {
            return OptimizationResult::Remove;
        }

        OptimizationResult::PassOn
    }

    fn optimize_guard_false(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let cond_ref = ctx.get_box_replacement(op.arg(0));

        if let Some(val) = ctx.get_constant_int(cond_ref) {
            if val == 0 {
                return OptimizationResult::Remove;
            }
        }

        if !matches!(ctx.opref_type(cond_ref), Some(majit_ir::Type::Int)) {
            return OptimizationResult::PassOn;
        }

        let b = self.getintbound(cond_ref, ctx);
        if b.known_eq_const(0) {
            return OptimizationResult::Remove;
        }

        OptimizationResult::PassOn
    }

    /// optimizer.py:366 as_operation parity:
    /// Find the operation that produced cond_ref by searching new_operations.
    /// RPython's `as_operation(box)` checks `_emittedoperations` directly;
    /// majit's flat OpRef model requires a positional lookup.
    fn find_producing_op<'a>(&self, cond_ref: OpRef, ctx: &'a OptContext) -> Option<&'a Op> {
        // First try direct index (when OpRef matches new_operations index)
        let idx = cond_ref.raw() as usize;
        if idx < ctx.new_operations.len() && ctx.new_operations[idx].pos == cond_ref {
            return Some(&ctx.new_operations[idx]);
        }
        // Otherwise search by pos field
        ctx.new_operations.iter().rfind(|op| op.pos == cond_ref)
    }

    // ── Bound narrowing helpers ──

    /// intbounds.py:564-570 make_int_lt
    ///
    /// ```python
    /// def make_int_lt(self, box1, box2):
    ///     b1 = self.getintbound(box1)
    ///     b2 = self.getintbound(box2)
    ///     if b1.make_lt(b2):
    ///         self.propagate_bounds_backward(box1)
    ///     if b2.make_gt(b1):
    ///         self.propagate_bounds_backward(box2)
    /// ```
    ///
    /// `IntBound::make_lt` returns `Ok(true)` exactly when RPython's
    /// `make_lt` returns truthy ("the bound was actually tightened"), so
    /// `matches!(.., Ok(true))` is the line-by-line equivalent of the
    /// RPython `if`.
    fn make_int_lt(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let changed1 = ctx.with_intbound_mut(box1, |b1| matches!(b1.make_lt(&b2), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(box1, ctx);
        }
        let b1 = self.getintbound(box1, ctx);
        let changed2 = ctx.with_intbound_mut(box2, |b2| matches!(b2.make_gt(&b1), Ok(true)));
        if changed2 {
            self.propagate_bounds_backward(box2, ctx);
        }
    }

    /// intbounds.py:572-578 make_int_le
    fn make_int_le(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let changed1 = ctx.with_intbound_mut(box1, |b1| matches!(b1.make_le(&b2), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(box1, ctx);
        }
        let b1 = self.getintbound(box1, ctx);
        let changed2 = ctx.with_intbound_mut(box2, |b2| matches!(b2.make_ge(&b1), Ok(true)));
        if changed2 {
            self.propagate_bounds_backward(box2, ctx);
        }
    }

    /// intbounds.py:580-581 make_int_gt
    fn make_int_gt(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        self.make_int_lt(box2, box1, ctx);
    }

    /// intbounds.py:583-584 make_int_ge
    fn make_int_ge(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        self.make_int_le(box2, box1, ctx);
    }

    /// intbounds.py:586-592 make_unsigned_lt
    fn make_unsigned_lt(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let changed1 =
            ctx.with_intbound_mut(box1, |b1| matches!(b1.make_unsigned_lt(&b2), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(box1, ctx);
        }
        let b1 = self.getintbound(box1, ctx);
        let changed2 =
            ctx.with_intbound_mut(box2, |b2| matches!(b2.make_unsigned_gt(&b1), Ok(true)));
        if changed2 {
            self.propagate_bounds_backward(box2, ctx);
        }
    }

    /// intbounds.py:594-600 make_unsigned_le
    fn make_unsigned_le(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let changed1 =
            ctx.with_intbound_mut(box1, |b1| matches!(b1.make_unsigned_le(&b2), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(box1, ctx);
        }
        let b1 = self.getintbound(box1, ctx);
        let changed2 =
            ctx.with_intbound_mut(box2, |b2| matches!(b2.make_unsigned_ge(&b1), Ok(true)));
        if changed2 {
            self.propagate_bounds_backward(box2, ctx);
        }
    }

    /// intbounds.py:602-603 make_unsigned_gt
    fn make_unsigned_gt(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        self.make_unsigned_lt(box2, box1, ctx);
    }

    /// intbounds.py:605-606 make_unsigned_ge
    fn make_unsigned_ge(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        self.make_unsigned_le(box2, box1, ctx);
    }

    /// intbounds.py:658-664 make_eq
    ///
    /// ```python
    /// def make_eq(self, arg0, arg1):
    ///     b0 = self.getintbound(arg0)
    ///     b1 = self.getintbound(arg1)
    ///     if b0.intersect(b1):
    ///         self.propagate_bounds_backward(arg0)
    ///     if b1.intersect(b0):
    ///         self.propagate_bounds_backward(arg1)
    /// ```
    fn make_eq(&mut self, arg0: OpRef, arg1: OpRef, ctx: &mut OptContext) {
        let b1 = self.getintbound(arg1, ctx);
        let changed0 = ctx.with_intbound_mut(arg0, |b0| matches!(b0.intersect(&b1), Ok(true)));
        if changed0 {
            self.propagate_bounds_backward(arg0, ctx);
        }
        let b0 = self.getintbound(arg0, ctx);
        let changed1 = ctx.with_intbound_mut(arg1, |b1| matches!(b1.intersect(&b0), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(arg1, ctx);
        }
    }

    /// intbounds.py:666-676 make_ne
    ///
    /// ```python
    /// def make_ne(self, arg0, arg1):
    ///     b0 = self.getintbound(arg0)
    ///     b1 = self.getintbound(arg1)
    ///     if b1.is_constant():
    ///         v1 = b1.get_constant_int()
    ///         if b0.make_ne_const(v1):
    ///             self.propagate_bounds_backward(arg0)
    ///     elif b0.is_constant():
    ///         v0 = b0.get_constant_int()
    ///         if b1.make_ne_const(v0):
    ///             self.propagate_bounds_backward(arg1)
    /// ```
    fn make_ne(&mut self, arg0: OpRef, arg1: OpRef, ctx: &mut OptContext) {
        let b1 = self.getintbound(arg1, ctx);
        if b1.is_constant() {
            let v1 = b1.get_constant_int();
            if ctx.with_intbound_mut(arg0, |b0| b0.make_ne_const(v1)) {
                self.propagate_bounds_backward(arg0, ctx);
            }
        } else {
            let b0 = self.getintbound(arg0, ctx);
            if b0.is_constant() {
                let v0 = b0.get_constant_int();
                if ctx.with_intbound_mut(arg1, |b1| b1.make_ne_const(v0)) {
                    self.propagate_bounds_backward(arg1, ctx);
                }
            }
        }
    }

    // ── Backward propagation after constant discovery ──

    /// intbounds.py:40-50 propagate_bounds_backward
    ///
    /// ```python
    /// def propagate_bounds_backward(self, box):
    ///     b = self.getintbound(box)
    ///     if b.is_constant():
    ///         self.make_constant_int(box, b.get_constant_int())
    ///     box1 = self.optimizer.as_operation(box)
    ///     if box1 is not None:
    ///         dispatch_bounds_ops(self, box1)
    /// ```
    ///
    /// Both branches run unconditionally (no early return after the
    /// constant fold) — RPython falls through into the `dispatch_bounds_ops`
    /// call so the producing op of a now-constant value also gets a chance
    /// to tighten its other arguments.
    fn propagate_bounds_backward(&mut self, opref: OpRef, ctx: &mut OptContext) {
        let b = self.getintbound(opref, ctx);
        if b.is_constant() {
            self.make_constant_int_ref(opref, b.get_constant_int(), ctx);
        }
        if let Some(producing_op) = self.find_producing_op(opref, ctx) {
            let producing_op = producing_op.clone();
            self.propagate_bounds_backward_op(&producing_op, ctx);
        }
    }

    /// intbounds.py:678-693 _propagate_int_is_true_or_zero (helper for
    /// propagate_bounds_INT_IS_TRUE / IS_ZERO).
    fn propagate_int_is_true_or_zero(
        &mut self,
        op: &Op,
        valnonzero: i64,
        valzero: i64,
        ctx: &mut OptContext,
    ) {
        if ctx.is_raw_ptr(op.arg(0)) {
            return;
        }
        let r = self.getintbound(op.pos, ctx);
        if !r.is_constant() {
            return;
        }
        let arg0 = op.arg(0);
        let r_const = r.get_constant_int();
        if r_const == valnonzero {
            let b1 = self.getintbound(arg0, ctx);
            if b1.known_nonnegative() {
                let _ = ctx.with_intbound_mut(arg0, |bm| bm.make_gt_const(0));
                self.propagate_bounds_backward(arg0, ctx);
            } else if b1.known_le_const(0) {
                let _ = ctx.with_intbound_mut(arg0, |bm| bm.make_lt_const(0));
                self.propagate_bounds_backward(arg0, ctx);
            }
        } else if r_const == valzero {
            self.make_constant_int_ref(arg0, 0, ctx);
            self.propagate_bounds_backward(arg0, ctx);
        }
    }

    fn propagate_bounds_backward_op(&mut self, op: &Op, ctx: &mut OptContext) {
        match op.opcode {
            // intbounds.py:701-712 propagate_bounds_INT_ADD
            //
            // ```python
            // def propagate_bounds_INT_ADD(self, op):
            //     if self.is_raw_ptr(op.getarg(0)) or self.is_raw_ptr(op.getarg(1)):
            //         return
            //     b1 = self.getintbound(op.getarg(0))
            //     b2 = self.getintbound(op.getarg(1))
            //     r = self.getintbound(op)
            //     b = r.sub_bound(b2)
            //     if b1.intersect(b):
            //         self.propagate_bounds_backward(op.getarg(0))
            //     b = r.sub_bound(b1)
            //     if b2.intersect(b):
            //         self.propagate_bounds_backward(op.getarg(1))
            // ```
            OpCode::IntAdd | OpCode::IntAddOvf => {
                if ctx.is_raw_ptr(op.arg(0)) || ctx.is_raw_ptr(op.arg(1)) {
                    return;
                }
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let b1 = self.getintbound(arg0, ctx);
                let b2 = self.getintbound(arg1, ctx);
                let r = self.getintbound(op.pos, ctx);
                let b = r.sub_bound(&b2);
                let changed0 =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed0 {
                    self.propagate_bounds_backward(arg0, ctx);
                }
                let b = r.sub_bound(&b1);
                let changed1 =
                    ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed1 {
                    self.propagate_bounds_backward(arg1, ctx);
                }
            }
            // intbounds.py:714-723 propagate_bounds_INT_SUB
            OpCode::IntSub | OpCode::IntSubOvf => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let b1 = self.getintbound(arg0, ctx);
                let b2 = self.getintbound(arg1, ctx);
                let r = self.getintbound(op.pos, ctx);
                let b = r.add_bound(&b2);
                let changed0 =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed0 {
                    self.propagate_bounds_backward(arg0, ctx);
                }
                let b = r.sub_bound(&b1).neg_bound();
                let changed1 =
                    ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed1 {
                    self.propagate_bounds_backward(arg1, ctx);
                }
            }
            // intbounds.py:725-737 propagate_bounds_INT_MUL
            OpCode::IntMul | OpCode::IntMulOvf => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let b1 = self.getintbound(arg0, ctx);
                let b2 = self.getintbound(arg1, ctx);
                if op.opcode != OpCode::IntMulOvf && !b1.mul_bound_cannot_overflow(&b2) {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.py_div_bound(&b2);
                let changed0 =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed0 {
                    self.propagate_bounds_backward(arg0, ctx);
                }
                let b = r.py_div_bound(&b1);
                let changed1 =
                    ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed1 {
                    self.propagate_bounds_backward(arg1, ctx);
                }
            }
            // intbounds.py:739-747 propagate_bounds_INT_LSHIFT
            OpCode::IntLshift => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let b1 = self.getintbound(arg0, ctx);
                let b2 = self.getintbound(arg1, ctx);
                if !b1.lshift_bound_cannot_overflow(&b2) {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                if let Ok(b) = r.lshift_bound_backwards(&b2) {
                    let changed =
                        ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg0, ctx);
                    }
                }
            }
            // intbounds.py:759-767 propagate_bounds_INT_RSHIFT
            OpCode::IntRshift => {
                let arg0 = op.arg(0);
                let b2 = self.getintbound(op.arg(1), ctx);
                if !b2.is_constant() {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.rshift_bound_backwards(&b2);
                let changed =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:749-757 propagate_bounds_UINT_RSHIFT
            OpCode::UintRshift => {
                let arg0 = op.arg(0);
                let b2 = self.getintbound(op.arg(1), ctx);
                if !b2.is_constant() {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.urshift_bound_backwards(&b2);
                let changed =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:773-782 propagate_bounds_INT_AND
            OpCode::IntAnd => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(arg0, ctx);
                let b1 = self.getintbound(arg1, ctx);
                if let Ok(b) = b0.and_bound_backwards(&r) {
                    let changed =
                        ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg1, ctx);
                    }
                }
                if let Ok(b) = b1.and_bound_backwards(&r) {
                    let changed =
                        ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg0, ctx);
                    }
                }
            }
            // intbounds.py:784-793 propagate_bounds_INT_OR
            OpCode::IntOr => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(arg0, ctx);
                let b1 = self.getintbound(arg1, ctx);
                if let Ok(b) = b0.or_bound_backwards(&r) {
                    let changed =
                        ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg1, ctx);
                    }
                }
                if let Ok(b) = b1.or_bound_backwards(&r) {
                    let changed =
                        ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg0, ctx);
                    }
                }
            }
            // intbounds.py:795-804 propagate_bounds_INT_XOR
            OpCode::IntXor => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(arg0, ctx);
                let b1 = self.getintbound(arg1, ctx);
                // xor is its own inverse
                let b = b0.xor_bound(&r);
                let changed1 =
                    ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed1 {
                    self.propagate_bounds_backward(arg1, ctx);
                }
                let b = b1.xor_bound(&r);
                let changed0 =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed0 {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:481-487 propagate_bounds_INT_INVERT
            OpCode::IntInvert => {
                let arg0 = op.arg(0);
                let bres = self.getintbound(op.pos, ctx);
                let bounds = bres.invert_bound();
                let changed =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&bounds), Ok(true)));
                if changed {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:489-495 propagate_bounds_INT_NEG
            OpCode::IntNeg => {
                let arg0 = op.arg(0);
                let bres = self.getintbound(op.pos, ctx);
                let bounds = bres.neg_bound();
                let changed =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&bounds), Ok(true)));
                if changed {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:608-615 propagate_bounds_INT_LT
            OpCode::IntLt => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_int_lt(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_int_ge(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:617-624 propagate_bounds_INT_GT
            OpCode::IntGt => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_int_gt(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_int_le(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:626-633 propagate_bounds_INT_LE
            OpCode::IntLe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_int_le(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_int_gt(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:635-642 propagate_bounds_INT_GE
            OpCode::IntGe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_int_ge(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_int_lt(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:644-651 propagate_bounds_INT_EQ
            OpCode::IntEq => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() && r.lower == 1 {
                    self.make_eq(op.arg(0), op.arg(1), ctx);
                } else if r.is_constant() && r.lower == 0 {
                    self.make_ne(op.arg(0), op.arg(1), ctx);
                }
            }
            // intbounds.py:653-656 propagate_bounds_INT_NE
            OpCode::IntNe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() && r.lower == 0 {
                    self.make_eq(op.arg(0), op.arg(1), ctx);
                } else if r.is_constant() && r.lower == 1 {
                    self.make_ne(op.arg(0), op.arg(1), ctx);
                }
            }
            // intbounds.py:379-386 propagate_bounds_UINT_LT
            OpCode::UintLt => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_unsigned_lt(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_unsigned_ge(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:400-407 propagate_bounds_UINT_GT
            OpCode::UintGt => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_unsigned_gt(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_unsigned_le(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:421-428 propagate_bounds_UINT_LE
            OpCode::UintLe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_unsigned_le(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_unsigned_gt(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:442-449 propagate_bounds_UINT_GE
            OpCode::UintGe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_unsigned_ge(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_unsigned_lt(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:678-693 _propagate_int_is_true_or_zero +
            // propagate_bounds_INT_IS_TRUE / IS_ZERO.
            //
            // ```python
            // def _propagate_int_is_true_or_zero(self, op, valnonzero, valzero):
            //     if self.is_raw_ptr(op.getarg(0)):
            //         return
            //     r = self.getintbound(op)
            //     if r.is_constant():
            //         if r.get_constant_int() == valnonzero:
            //             b1 = self.getintbound(op.getarg(0))
            //             if b1.known_nonnegative():
            //                 b1.make_gt_const(0)
            //                 self.propagate_bounds_backward(op.getarg(0))
            //             elif b1.known_le_const(0):
            //                 b1.make_lt_const(0)
            //                 self.propagate_bounds_backward(op.getarg(0))
            //         elif r.get_constant_int() == valzero:
            //             self.make_constant_int(op.getarg(0), 0)
            //             self.propagate_bounds_backward(op.getarg(0))
            //
            // def propagate_bounds_INT_IS_TRUE(self, op):
            //     self._propagate_int_is_true_or_zero(op, 1, 0)
            //
            // def propagate_bounds_INT_IS_ZERO(self, op):
            //     self._propagate_int_is_true_or_zero(op, 0, 1)
            // ```
            OpCode::IntIsTrue => {
                self.propagate_int_is_true_or_zero(op, 1, 0, ctx);
            }
            OpCode::IntIsZero => {
                self.propagate_int_is_true_or_zero(op, 0, 1, ctx);
            }
            _ => {}
        }
    }

    /// Record what we last emitted (for overflow guard removal).
    /// Mirrors RPython's self.last_emitted_operation = op in Optimization.emit().
    fn record_emitted(&mut self, op: &Op) {
        self.last_emitted_opcode = Some(op.opcode);
        self.last_emitted_args = op.args.to_vec();
        self.last_emitted_ref = op.pos;
    }
}

impl Default for OptIntBounds {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptIntBounds {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let result = match op.opcode {
            // ── Comparisons ──
            OpCode::IntLt => self.optimize_int_lt(op, ctx),
            OpCode::IntLe => self.optimize_int_le(op, ctx),
            OpCode::IntGt => self.optimize_int_gt(op, ctx),
            OpCode::IntGe => self.optimize_int_ge(op, ctx),
            OpCode::IntEq => self.optimize_int_eq(op, ctx),
            OpCode::IntNe => self.optimize_int_ne(op, ctx),
            OpCode::UintLt => self.optimize_uint_lt(op, ctx),
            OpCode::UintLe => self.optimize_uint_le(op, ctx),
            OpCode::UintGt => self.optimize_uint_gt(op, ctx),
            OpCode::UintGe => self.optimize_uint_ge(op, ctx),

            // ── Arithmetic folds (autogenintrules.py) ──
            OpCode::IntAdd => self.optimize_int_add(op, ctx),
            OpCode::IntSub => self.optimize_int_sub(op, ctx),
            OpCode::IntMul => self.optimize_int_mul(op, ctx),
            OpCode::IntAnd => self.optimize_int_and(op, ctx),
            OpCode::IntOr => self.optimize_int_or(op, ctx),
            OpCode::IntXor => self.optimize_int_xor(op, ctx),
            OpCode::IntNeg => self.optimize_int_neg(op, ctx),
            OpCode::IntInvert => self.optimize_int_invert(op, ctx),
            OpCode::IntLshift => self.optimize_int_lshift(op, ctx),
            OpCode::IntRshift => self.optimize_int_rshift(op, ctx),
            OpCode::UintRshift => self.optimize_uint_rshift(op, ctx),
            OpCode::IntIsTrue => self.optimize_int_is_true(op, ctx),
            OpCode::IntIsZero => self.optimize_int_is_zero(op, ctx),
            OpCode::IntForceGeZero => self.optimize_int_force_ge_zero(op, ctx),

            // ── Signext ──
            OpCode::IntSignext => self.optimize_int_signext(op, ctx),

            // ── Overflow arithmetic ──
            OpCode::IntAddOvf => self.optimize_int_add_ovf(op, ctx),
            OpCode::IntSubOvf => self.optimize_int_sub_ovf(op, ctx),
            OpCode::IntMulOvf => self.optimize_int_mul_ovf(op, ctx),

            // ── Overflow guards ──
            OpCode::GuardNoOverflow => self.optimize_guard_no_overflow(op, ctx),
            OpCode::GuardOverflow => self.optimize_guard_overflow(op),

            // ── Guards on conditions ──
            OpCode::GuardTrue => self.optimize_guard_true(op, ctx),
            OpCode::GuardFalse => self.optimize_guard_false(op, ctx),

            // ── All other ops: default emit (RPython's Optimization.emit) ──
            // Postprocess dispatch is in propagate_postprocess, matching
            // RPython's make_dispatcher_method(OptIntBounds, 'postprocess_').
            _ => OptimizationResult::PassOn,
        };

        // Track last emitted for overflow guard handling
        match &result {
            OptimizationResult::Emit(emitted_op) => self.record_emitted(emitted_op),
            OptimizationResult::PassOn => self.record_emitted(op),
            _ => {
                // Remove: don't update last_emitted
            }
        }

        result
    }

    fn setup(&mut self) {
        self.last_emitted_opcode = None;
        self.last_emitted_args.clear();
        self.last_emitted_ref = OpRef::NONE;
    }

    fn name(&self) -> &'static str {
        "intbounds"
    }

    /// RPython: have_dispatcher_method(OptIntBounds, 'postprocess_')
    ///
    /// Guard postprocess must be included in framework dispatch so
    /// intbounds can run its upstream `postprocess_GUARD_TRUE/FALSE` parity:
    /// after rewrite postprocess has fixed the guard result to 1/0,
    /// intbounds propagates that knowledge backward through the producing
    /// int expression before later ops (e.g. INT_ADD_OVF) are optimized.
    fn have_postprocess_op(&self, opcode: OpCode) -> bool {
        matches!(
            opcode,
            OpCode::GuardTrue
                | OpCode::GuardFalse
                | OpCode::GuardValue
                | OpCode::IntAdd
                | OpCode::IntSub
                | OpCode::IntMul
                | OpCode::IntAnd
                | OpCode::IntOr
                | OpCode::IntXor
                | OpCode::IntLshift
                | OpCode::IntRshift
                | OpCode::UintRshift
                | OpCode::IntFloorDiv
                | OpCode::IntMod
                | OpCode::IntNeg
                | OpCode::IntInvert
                | OpCode::IntForceGeZero
                | OpCode::IntSignext
                | OpCode::IntAddOvf
                | OpCode::IntSubOvf
                | OpCode::IntMulOvf
                | OpCode::ArraylenGc
                | OpCode::Strlen
                | OpCode::Unicodelen
                | OpCode::Strgetitem
                | OpCode::Unicodegetitem
                | OpCode::GetfieldRawI
                | OpCode::GetfieldGcI
                | OpCode::GetinteriorfieldGcI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldGcR
                | OpCode::GetinteriorfieldGcR
                | OpCode::GetfieldRawF
                | OpCode::GetfieldGcF
                | OpCode::GetinteriorfieldGcF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemGcI
                | OpCode::CallPureI
                | OpCode::CallI
        )
    }

    /// intbounds.py:52-58 _postprocess_guard_true_false_value parity.
    ///
    /// ```python
    /// def _postprocess_guard_true_false_value(self, op):
    ///     if op.getarg(0).type == 'i':
    ///         self.propagate_bounds_backward(op.getarg(0))
    /// ```
    ///
    /// Called AFTER the op has been emitted through ALL passes and added
    /// to new_operations. At this point, the heap pass has flushed any
    /// postponed comparison op, so find_producing_op succeeds.
    /// Matches RPython's make_dispatcher_method(OptIntBounds, 'postprocess_').
    /// Dispatches to the correct postprocess_* method based on opcode.
    fn propagate_postprocess(&mut self, op: &Op, ctx: &mut OptContext) {
        match op.opcode {
            // intbounds.py:52-58 _postprocess_guard_true_false_value
            //   if op.getarg(0).type == 'i':
            //       self.propagate_bounds_backward(op.getarg(0))
            OpCode::GuardTrue | OpCode::GuardFalse | OpCode::GuardValue => {
                let arg0 = ctx.get_box_replacement(op.arg(0));
                let is_int = ctx
                    .opref_type(arg0)
                    .map_or(true, |t| t == majit_ir::Type::Int);
                if !is_int {
                    return;
                }
                // intbounds.py:40-50 propagate_bounds_backward
                let b = self.getintbound(arg0, ctx);
                if b.is_constant() {
                    self.make_constant_int_ref(arg0, b.get_constant_int(), ctx);
                }
                if let Some(producing_op) = self.find_producing_op(arg0, ctx) {
                    let producing_op = producing_op.clone();
                    self.propagate_bounds_backward_op(&producing_op, ctx);
                }
            }

            // ── Arithmetic postprocess ──
            OpCode::IntAdd => self.postprocess_int_add(op, ctx),
            OpCode::IntSub => self.postprocess_int_sub(op, ctx),
            OpCode::IntMul => self.postprocess_int_mul(op, ctx),
            OpCode::IntAnd => self.postprocess_int_and(op, ctx),
            OpCode::IntOr => self.postprocess_int_or(op, ctx),
            OpCode::IntXor => self.postprocess_int_xor(op, ctx),
            OpCode::IntLshift => self.postprocess_int_lshift(op, ctx),
            OpCode::IntRshift => self.postprocess_int_rshift(op, ctx),
            OpCode::UintRshift => self.postprocess_uint_rshift(op, ctx),
            OpCode::IntFloorDiv => self.postprocess_int_floordiv(op, ctx),
            OpCode::IntMod => self.postprocess_int_mod(op, ctx),
            OpCode::IntNeg => self.postprocess_int_neg(op, ctx),
            OpCode::IntInvert => self.postprocess_int_invert(op, ctx),
            OpCode::IntForceGeZero => self.postprocess_int_force_ge_zero(op, ctx),
            OpCode::IntSignext => self.postprocess_int_signext(op, ctx),

            // ── Overflow arithmetic postprocess ──
            OpCode::IntAddOvf => self.postprocess_int_add_ovf(op, ctx),
            OpCode::IntSubOvf => self.postprocess_int_sub_ovf(op, ctx),
            OpCode::IntMulOvf => self.postprocess_int_mul_ovf(op, ctx),

            // ── Lengths ──
            OpCode::ArraylenGc => self.postprocess_arraylen_gc(op, ctx),
            OpCode::Strlen => self.postprocess_strlen(op, ctx),
            OpCode::Unicodelen => self.postprocess_unicodelen(op, ctx),

            // ── String/Unicode items ──
            OpCode::Strgetitem => {
                self.intersect_bound(op.pos, &IntBound::bounded(0, 255), ctx);
            }
            OpCode::Unicodegetitem => {
                self.intersect_bound(op.pos, &IntBound::nonnegative(), ctx);
            }

            // ── Field accesses ──
            OpCode::GetfieldRawI
            | OpCode::GetfieldGcI
            | OpCode::GetinteriorfieldGcI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldGcR
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetfieldRawF
            | OpCode::GetfieldGcF
            | OpCode::GetinteriorfieldGcF => {
                if let Some(ref d) = op.descr {
                    let (field_size, signed) = d.field_size_and_sign();
                    if field_size > 0 && field_size < 8 {
                        let (lo, hi) = if signed {
                            let half = 1i64 << (field_size * 8 - 1);
                            (-half, half - 1)
                        } else {
                            (0, (1i64 << (field_size * 8)) - 1)
                        };
                        self.intersect_bound(op.pos, &IntBound::bounded(lo, hi), ctx);
                    }
                }
            }

            // ── Array item accesses ──
            OpCode::GetarrayitemRawI | OpCode::GetarrayitemGcI => {
                if let Some(ref d) = op.descr {
                    if let Some(ad) = d.as_array_descr() {
                        let item_size = ad.item_size();
                        if item_size > 0 && item_size < 8 {
                            let signed = ad.is_item_signed();
                            let (lo, hi) = if signed {
                                let half = 1i64 << (item_size * 8 - 1);
                                (-half, half - 1)
                            } else {
                                (0, (1i64 << (item_size * 8)) - 1)
                            };
                            self.intersect_bound(op.pos, &IntBound::bounded(lo, hi), ctx);
                        }
                    }
                }
            }

            // ── Call postprocess ──
            OpCode::CallPureI | OpCode::CallI => {
                if let Some(ref d) = op.descr {
                    if let Some(cd) = d.as_call_descr() {
                        let ei = cd.get_extra_info();
                        match ei.oopspecindex {
                            majit_ir::OopSpecIndex::IntPyDiv => {
                                if op.num_args() >= 3 {
                                    // intbounds.py:165-169 post_call_INT_PY_DIV.
                                    let b1 = self.getintbound(op.arg(1), ctx);
                                    let b2 = self.getintbound(op.arg(2), ctx);
                                    let result_bound = b1.py_div_bound(&b2);
                                    self.intersect_bound(op.pos, &result_bound, ctx);
                                }
                            }
                            majit_ir::OopSpecIndex::IntPyMod => {
                                if op.num_args() >= 3 {
                                    // intbounds.py:171-175 post_call_INT_PY_MOD.
                                    let b1 = self.getintbound(op.arg(1), ctx);
                                    let b2 = self.getintbound(op.arg(2), ctx);
                                    let result_bound = b1.mod_bound(&b2);
                                    self.intersect_bound(op.pos, &result_bound, ctx);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            _ => {}
        }
    }

    /// intbounds.py: produce_potential_short_preamble_ops(sb)
    /// Contribute bounds guards to the short preamble.
    fn produce_potential_short_preamble_ops(
        &self,
        _sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        _ctx: &mut OptContext,
    ) {
        // In RPython, this adds INT_GE/INT_LE guards for known bounds
        // that the loop body depends on. The bounds are discovered during
        // preamble optimization and must be re-checked on bridge entry.
        // The actual bounds live on box._forwarded (Forwarded::IntBound) —
        // a real implementation would iterate non-trivial bounds via
        // ctx.getintbound and generate guard ops.
    }

    fn export_arg_int_bounds(
        &self,
        args: &[OpRef],
        ctx: &OptContext,
    ) -> std::collections::HashMap<OpRef, IntBound> {
        let mut exported = std::collections::HashMap::new();
        for &arg in args {
            let resolved = ctx.get_box_replacement(arg);
            if !matches!(ctx.opref_type(resolved), Some(majit_ir::Type::Int)) {
                continue;
            }
            if let Some(bound) = ctx.peek_intbound_via_box(resolved) {
                if bound.is_unbounded() {
                    continue;
                }
                exported.insert(resolved, bound);
            }
        }
        exported
    }
}

#[cfg(test)]
mod tests {
    //! Upstream parity anchor:
    //! `rpython/jit/metainterp/optimizeopt/test/test_optimizeopt.py`
    //! (`test_bound_*`, `test_strgetitem_bounds`, `test_arraylen_bound`, ...)
    //! and `rpython/jit/metainterp/optimizeopt/intbounds.py`.
    //!
    //! These Rust tests are mostly unit-level decompositions of those optimizer
    //! behaviors so the port can be checked below full trace-optimizer
    //! integration.

    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::{Descr, DescrRef};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr(u32);

    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
    }

    fn descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr(idx))
    }

    fn call_descr(idx: u32, oopspecindex: majit_ir::OopSpecIndex) -> DescrRef {
        Arc::new(majit_ir::SimpleCallDescr::new(
            idx,
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            true,
            std::mem::size_of::<i64>(),
            majit_ir::EffectInfo::new(majit_ir::ExtraEffect::ElidableCanRaise, oopspecindex),
        ))
    }

    fn run_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.propagate_all_forward(&ops)
    }

    /// Create an OptIntBounds with specific bounds pre-set and run it on ops.
    fn run_pass_with_bounds(
        ops: &[Op],
        initial_bounds: &[(OpRef, IntBound)],
    ) -> (Vec<Op>, OptContext) {
        let mut pass = OptIntBounds::new();
        // Compute num_inputs as the highest OpRef referenced anywhere (as
        // arg, result pos, or initial bound key) plus one. With bounds now
        // living on `ctx.forwarded`, reserve_pos must skip past every
        // pre-existing OpRef so freshly emitted ops (e.g. SameAsI from
        // rewrite's `x + x → x << 1`) cannot collide with an input arg.
        let max_arg = ops
            .iter()
            .flat_map(|op| op.args.iter().copied())
            .filter(|r| !r.is_constant() && !r.is_none())
            .map(|r| r.raw())
            .max()
            .unwrap_or(0);
        let max_pos = ops
            .iter()
            .map(|op| op.pos)
            .filter(|r| !r.is_constant() && !r.is_none())
            .map(|r| r.raw())
            .max()
            .unwrap_or(0);
        let max_initial = initial_bounds
            .iter()
            .map(|(r, _)| r.raw())
            .max()
            .unwrap_or(0);
        let num_inputs = (max_arg.max(max_pos).max(max_initial) as usize) + 1;
        let inputarg_types = vec![majit_ir::Type::Int; num_inputs];
        let mut ctx = OptContext::with_inputarg_types(ops.len(), &inputarg_types);

        // resume.py:397 parity: each guard must have a snapshot indexed
        // by `rd_resume_position`.  This helper drives the optimizer
        // without going through `Optimizer::optimize_with_constants_and_inputs`,
        // so seed the snapshots manually.
        let (seeded_ops, snapshots) = super::super::seed_empty_guard_snapshots(ops);
        ctx.snapshot_boxes = snapshots;
        let ops = &seeded_ops;

        pass.setup();
        for (opref, bound) in initial_bounds {
            ctx.setintbound(*opref, bound);
        }

        for op in ops.iter() {
            let mut resolved_op = op.clone();
            // Keep the op.pos as set by the test (not overriding with index)
            for arg in &mut resolved_op.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            let emitted_op = match pass.propagate_forward(&resolved_op, &mut ctx) {
                OptimizationResult::Emit(emit_op) => {
                    ctx.emit(emit_op.clone());
                    Some(emit_op)
                }
                OptimizationResult::Replace(rep_op) | OptimizationResult::Restart(rep_op) => {
                    ctx.emit(rep_op.clone());
                    Some(rep_op)
                }
                OptimizationResult::Remove => None,
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op.clone());
                    Some(resolved_op)
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            };
            if let Some(ref emitted) = emitted_op {
                // Simulate rewrite pass postprocess: make_constant on guard condition.
                // In the real optimizer, rewrite.rs postprocess_GUARD_TRUE sets
                // condition = CONST_1 before intbounds postprocess runs.
                match emitted.opcode {
                    OpCode::GuardTrue => {
                        let cond = ctx.get_box_replacement(emitted.arg(0));
                        ctx.make_constant(cond, majit_ir::Value::Int(1));
                    }
                    OpCode::GuardFalse => {
                        let cond = ctx.get_box_replacement(emitted.arg(0));
                        ctx.make_constant(cond, majit_ir::Value::Int(0));
                    }
                    _ => {}
                }
                pass.propagate_postprocess(emitted, &mut ctx);
            }
        }

        let result = ctx.new_operations.clone();
        let _ = pass;
        (result, ctx)
    }

    fn make_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef::op_typed(pos, opcode.result_type());
        op
    }

    /// Like `make_op` but stamps a fresh `ResumeGuardDescr` on the
    /// resulting op — required for guards that act as sharing donors
    /// (optimizer.py:691 `assert isinstance(last_descr, ResumeGuardDescr)`).
    /// Sharer guards (descrless) keep using `make_op`.
    #[allow(dead_code)]
    fn make_guard_with_descr(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = make_op(opcode, args, pos);
        assert!(
            opcode.is_guard(),
            "make_guard_with_descr requires a guard opcode"
        );
        op.descr = Some(crate::compile::make_resume_guard_descr_typed(Vec::new()));
        op
    }

    // ── Test: INT_ADD narrows bounds ──

    #[test]
    fn test_int_add_narrows_bounds() {
        // Set up two operands with known bounds and add them
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 10)),
            (OpRef::int_op(1), IntBound::bounded(5, 20)),
        ];
        let ops = vec![make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);

        // The result should have bounds [5, 30]
        let b = ctx.getintbound(OpRef::int_op(2));
        assert_eq!(b.lower, 5);
        assert_eq!(b.upper, 30);
    }

    // ── Test: Guard removal when bounds prove the condition ──

    #[test]
    fn test_guard_removal_known_true() {
        // i0 in [10, 20], i1 in [0, 5]
        // INT_LT(i1, i0) is always true (5 < 10)
        // GUARD_TRUE on it should be removable
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(10, 20)),
            (OpRef::int_op(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![
            // i2 = INT_LT(i1, i0)  -- i1 < i0 is always true
            make_op(OpCode::IntLt, &[OpRef::int_op(1), OpRef::int_op(0)], 2),
            // GUARD_TRUE(i2)
            make_op(OpCode::GuardTrue, &[OpRef::int_op(2)], 3),
        ];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        // INT_LT should be removed (replaced by constant 1)
        // GUARD_TRUE on a known constant 1 should also be removed
        assert!(
            result.is_empty()
                || result
                    .iter()
                    .all(|op| op.opcode != OpCode::GuardTrue && op.opcode != OpCode::IntLt),
            "Guard and comparison should both be removed, got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_guard_removal_known_false() {
        // i0 in [10, 20], i1 in [0, 5]
        // INT_GE(i1, i0) is always false (5 < 10, so not >=)
        // GUARD_FALSE on it should be removable
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(10, 20)),
            (OpRef::int_op(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![
            // i2 = INT_GE(i1, i0)  -- always false
            make_op(OpCode::IntGe, &[OpRef::int_op(1), OpRef::int_op(0)], 2),
            // GUARD_FALSE(i2)
            make_op(OpCode::GuardFalse, &[OpRef::int_op(2)], 3),
        ];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty()
                || result
                    .iter()
                    .all(|op| op.opcode != OpCode::GuardFalse && op.opcode != OpCode::IntGe),
            "Guard and comparison should both be removed, got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test: Guard narrowing (bounds updated after guard) ──

    #[test]
    fn test_guard_narrowing_int_lt() {
        // i0 unbounded, i1 in [10, 10] (constant 10)
        // After GUARD_TRUE on INT_LT(i0, i1), i0's upper should be < 10
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::unbounded()),
            (OpRef::int_op(1), IntBound::from_constant(10)),
        ];
        let ops = vec![
            // i2 = INT_LT(i0, i1)
            make_op(OpCode::IntLt, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            // GUARD_TRUE(i2)
            make_op(OpCode::GuardTrue, &[OpRef::int_op(2)], 3),
        ];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);

        // After the guard, i0 should be < 10, meaning upper <= 9
        let b0 = ctx.getintbound(OpRef::int_op(0));
        assert!(
            b0.upper <= 9,
            "After GUARD_TRUE(INT_LT(i0, 10)), i0.upper should be <= 9, got {}",
            b0.upper
        );
    }

    #[test]
    fn test_guard_narrowing_int_ge() {
        // i0 unbounded, i1 = 5
        // After GUARD_TRUE on INT_GE(i0, i1), i0.lower >= 5
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::unbounded()),
            (OpRef::int_op(1), IntBound::from_constant(5)),
        ];
        let ops = vec![
            make_op(OpCode::IntGe, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            make_op(OpCode::GuardTrue, &[OpRef::int_op(2)], 3),
        ];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);

        let b0 = ctx.getintbound(OpRef::int_op(0));
        assert!(
            b0.lower >= 5,
            "After GUARD_TRUE(INT_GE(i0, 5)), i0.lower should be >= 5, got {}",
            b0.lower
        );
    }

    // ── Test: Overflow guard removal ──

    #[test]
    fn test_add_ovf_cannot_overflow() {
        // i0 in [0, 100], i1 in [0, 100]
        // INT_ADD_OVF(i0, i1) cannot overflow, should be replaced by INT_ADD
        // GUARD_NO_OVERFLOW should be removed
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 100)),
            (OpRef::int_op(1), IntBound::bounded(0, 100)),
        ];
        let ops = vec![
            make_op(OpCode::IntAddOvf, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            make_op(OpCode::GuardNoOverflow, &[], 3),
        ];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        // The OVF should be replaced with IntAdd, and the guard removed
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntAdd),
            "INT_ADD_OVF should be transformed to INT_ADD"
        );
        assert!(
            !result.iter().any(|op| op.opcode == OpCode::IntAddOvf),
            "INT_ADD_OVF should not remain"
        );
        assert!(
            !result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should be removed"
        );
    }

    #[test]
    fn test_add_ovf_may_overflow() {
        // i0 in [0, i64::MAX - 1], i1 in [0, i64::MAX - 1]
        // INT_ADD_OVF(i0, i1) may overflow, should NOT be replaced
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, i64::MAX - 1)),
            (OpRef::int_op(1), IntBound::bounded(0, i64::MAX - 1)),
        ];
        let ops = vec![
            make_op(OpCode::IntAddOvf, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            make_op(OpCode::GuardNoOverflow, &[], 3),
        ];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntAddOvf),
            "INT_ADD_OVF should remain when overflow is possible"
        );
        assert!(
            result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should remain when overflow is possible"
        );
    }

    #[test]
    fn test_sub_ovf_cannot_overflow() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 100)),
            (OpRef::int_op(1), IntBound::bounded(0, 50)),
        ];
        let ops = vec![
            make_op(OpCode::IntSubOvf, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            make_op(OpCode::GuardNoOverflow, &[], 3),
        ];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntSub),
            "INT_SUB_OVF should be transformed to INT_SUB"
        );
        assert!(
            !result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should be removed"
        );
    }

    #[test]
    fn test_mul_ovf_cannot_overflow() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 10)),
            (OpRef::int_op(1), IntBound::bounded(0, 10)),
        ];
        let ops = vec![
            make_op(OpCode::IntMulOvf, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            make_op(OpCode::GuardNoOverflow, &[], 3),
        ];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntMul),
            "INT_MUL_OVF should be transformed to INT_MUL"
        );
    }

    #[test]
    fn test_second_overflow_guard_survives_after_first_guard() {
        // Use OpRef::int_op(3) for guard condition so it doesn't affect overflow operands.
        // GuardTrue(OpRef::int_op(3)) makes OpRef::int_op(3)=1 via rewrite postprocess,
        // but OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2) remain unbounded.
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::unbounded()),
            (OpRef::int_op(1), IntBound::unbounded()),
            (OpRef::int_op(2), IntBound::unbounded()),
            (OpRef::int_op(3), IntBound::unbounded()),
        ];
        let ops = vec![
            // The first guard becomes the donor for the two descrless
            // GuardNoOverflow sharers below — give it a real descr per
            // optimizer.py:691.
            make_guard_with_descr(OpCode::GuardTrue, &[OpRef::int_op(3)], 4),
            make_op(OpCode::IntSubOvf, &[OpRef::int_op(0), OpRef::int_op(1)], 5),
            make_op(OpCode::GuardNoOverflow, &[], 6),
            make_op(OpCode::IntMulOvf, &[OpRef::int_op(2), OpRef::int_op(1)], 7),
            make_op(OpCode::GuardNoOverflow, &[], 8),
            make_op(
                OpCode::Jump,
                &[OpRef::int_op(5), OpRef::int_op(5), OpRef::int_op(7)],
                9,
            ),
        ];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        let guard_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNoOverflow)
            .count();
        let opcodes: Vec<_> = result.iter().map(|op| op.opcode).collect();
        assert_eq!(
            guard_count, 2,
            "both overflow guards must remain, got {opcodes:?}"
        );
    }

    // ── Test: Comparison result when bounds determine outcome ──

    #[test]
    fn test_int_lt_known_true() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 5)),
            (OpRef::int_op(1), IntBound::bounded(10, 20)),
        ];
        let ops = vec![make_op(
            OpCode::IntLt,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // Should be removed (replaced by constant 1)
        assert!(
            result.is_empty(),
            "INT_LT should be removed when known true"
        );
        // The constant should be set
        let b = ctx.getintbound(OpRef::int_op(2));
        assert!(b.is_constant() && b.get_constant_int() == 1);
    }

    #[test]
    fn test_int_lt_known_false() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(10, 20)),
            (OpRef::int_op(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![make_op(
            OpCode::IntLt,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "INT_LT should be removed when known false"
        );
        let b = ctx.getintbound(OpRef::int_op(2));
        assert!(b.is_constant() && b.get_constant_int() == 0);
    }

    #[test]
    fn test_int_eq_same_arg() {
        let ops = vec![make_op(
            OpCode::IntEq,
            &[OpRef::int_op(0), OpRef::int_op(0)],
            1,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        assert!(
            result.is_empty(),
            "INT_EQ(x, x) should be removed (always 1)"
        );
        let b = ctx.getintbound(OpRef::int_op(1));
        assert!(b.is_constant() && b.get_constant_int() == 1);
    }

    #[test]
    fn test_int_ne_same_arg() {
        let ops = vec![make_op(
            OpCode::IntNe,
            &[OpRef::int_op(0), OpRef::int_op(0)],
            1,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        assert!(
            result.is_empty(),
            "INT_NE(x, x) should be removed (always 0)"
        );
        let b = ctx.getintbound(OpRef::int_op(1));
        assert!(b.is_constant() && b.get_constant_int() == 0);
    }

    #[test]
    fn test_int_le_known_true() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 5)),
            (OpRef::int_op(1), IntBound::bounded(5, 20)),
        ];
        let ops = vec![make_op(
            OpCode::IntLe,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "INT_LE should be removed when known true"
        );
    }

    #[test]
    fn test_int_ge_known_true() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(10, 20)),
            (OpRef::int_op(1), IntBound::bounded(0, 10)),
        ];
        let ops = vec![make_op(
            OpCode::IntGe,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "INT_GE should be removed when known true"
        );
    }

    // ── Test: Arithmetic bounds propagation ──

    #[test]
    fn test_int_sub_bounds() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(10, 20)),
            (OpRef::int_op(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![make_op(
            OpCode::IntSub,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = ctx.getintbound(OpRef::int_op(2));
        // [10, 20] - [0, 5] = [5, 20]
        assert_eq!(b.lower, 5);
        assert_eq!(b.upper, 20);
    }

    #[test]
    fn test_int_mul_bounds() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(2, 5)),
            (OpRef::int_op(1), IntBound::bounded(3, 7)),
        ];
        let ops = vec![make_op(
            OpCode::IntMul,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = ctx.getintbound(OpRef::int_op(2));
        // [2, 5] * [3, 7] = [6, 35]
        assert_eq!(b.lower, 6);
        assert_eq!(b.upper, 35);
    }

    #[test]
    fn test_int_and_bounds() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 255)),
            (OpRef::int_op(1), IntBound::bounded(0, 15)),
        ];
        let ops = vec![make_op(
            OpCode::IntAnd,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = ctx.getintbound(OpRef::int_op(2));
        // AND of [0, 255] and [0, 15] -> [0, 15]
        assert!(b.lower >= 0);
        assert!(b.upper <= 15);
    }

    /// autogenintrules.py rule and_known_result: int_and(a, b) where the
    /// AND of bounds is a single constant => fold to that constant.
    #[test]
    fn test_int_and_known_result_both_const() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::from_constant(0xff)),
            (OpRef::int_op(1), IntBound::from_constant(0x0f)),
        ];
        let ops = vec![make_op(
            OpCode::IntAnd,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];
        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().all(|op| op.opcode != OpCode::IntAnd),
            "INT_AND of constants should be folded out, got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        let b = ctx.getintbound(OpRef::int_op(2));
        assert!(b.is_constant(), "result should be constant");
        assert_eq!(b.get_constant_int(), 0x0f);
    }

    /// autogenintrules.py rule and_x_x: int_and(a, a) => a.
    #[test]
    fn test_int_and_x_x() {
        let ops = vec![make_op(
            OpCode::IntAnd,
            &[OpRef::int_op(0), OpRef::int_op(0)],
            1,
        )];
        let result = run_pass(&ops);
        assert!(
            result.iter().all(|op| op.opcode != OpCode::IntAnd),
            "INT_AND(a, a) should be removed via and_x_x, got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    /// autogenintrules.py rule and_x_c_in_range: int_and(C, x) => x when x
    /// is a non-negative bounded value already inside the C low-bit mask.
    #[test]
    fn test_int_and_x_c_in_range() {
        // Mask = 0xff (low-byte mask). x = bounded(0, 100). 100 < 256 and
        // 100 < 0xff & ~0x100 == 0xff so the rule fires.
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::from_constant(0xff)),
            (OpRef::int_op(1), IntBound::bounded(0, 100)),
        ];
        let ops = vec![make_op(
            OpCode::IntAnd,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];
        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().all(|op| op.opcode != OpCode::IntAnd),
            "INT_AND(0xff, [0,100]) should fold via and_x_c_in_range, got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    /// autogenintrules.py rule or_known_result: int_or(a, b) where the OR
    /// of bounds is a single constant => fold to that constant.
    #[test]
    fn test_int_or_known_result_both_const() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::from_constant(0xf0)),
            (OpRef::int_op(1), IntBound::from_constant(0x0f)),
        ];
        let ops = vec![make_op(
            OpCode::IntOr,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];
        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().all(|op| op.opcode != OpCode::IntOr),
            "INT_OR of constants should be folded out, got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        let b = ctx.getintbound(OpRef::int_op(2));
        assert!(b.is_constant(), "result should be constant");
        assert_eq!(b.get_constant_int(), 0xff);
    }

    /// autogenintrules.py rule or_x_x: int_or(a, a) => a.
    #[test]
    fn test_int_or_x_x() {
        let ops = vec![make_op(
            OpCode::IntOr,
            &[OpRef::int_op(0), OpRef::int_op(0)],
            1,
        )];
        let result = run_pass(&ops);
        assert!(
            result.iter().all(|op| op.opcode != OpCode::IntOr),
            "INT_OR(a, a) should be removed via or_x_x, got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    /// autogenintrules.py rule or_idempotent: int_or(x, y) => x when y's
    /// known bits are a subset of x's known bits (knownbits absorption).
    #[test]
    fn test_int_or_idempotent_subset() {
        // arg0 = constant 0xff (tvalue=0xff, tmask=0).
        // arg1 = constant 0x0f (tvalue=0x0f, tmask=0).
        // 0x0f's set bits are a subset of 0xff's set bits, so
        // int_or(0xff, 0x0f) collapses; this is also covered by
        // or_known_result as a fold to 0xff. Either rule satisfies the
        // assertion.
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::from_constant(0xff)),
            (OpRef::int_op(1), IntBound::from_constant(0x0f)),
        ];
        let ops = vec![make_op(
            OpCode::IntOr,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];
        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().all(|op| op.opcode != OpCode::IntOr),
            "INT_OR(0xff, 0x0f) should be folded, got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_int_force_ge_zero() {
        let initial_bounds = vec![(OpRef::int_op(0), IntBound::bounded(-10, 20))];
        let ops = vec![make_op(OpCode::IntForceGeZero, &[OpRef::int_op(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef::int_op(1));
        assert!(
            b.lower >= 0,
            "INT_FORCE_GE_ZERO result should be >= 0, got {}",
            b.lower
        );
        assert!(
            b.upper <= 20,
            "INT_FORCE_GE_ZERO result upper should be <= 20, got {}",
            b.upper
        );
    }

    #[test]
    fn test_arraylen_nonneg() {
        let mut op = make_op(OpCode::ArraylenGc, &[OpRef::int_op(0)], 1);
        op.descr = Some(descr(1));
        let ops = vec![op];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        let b = ctx.getintbound(OpRef::int_op(1));
        assert!(b.lower >= 0, "ARRAYLEN_GC result should be non-negative");
    }

    #[test]
    fn test_strlen_nonneg() {
        let ops = vec![make_op(OpCode::Strlen, &[OpRef::int_op(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        let b = ctx.getintbound(OpRef::int_op(1));
        assert!(b.lower >= 0, "STRLEN result should be non-negative");
    }

    #[test]
    fn test_int_neg_bounds() {
        let initial_bounds = vec![(OpRef::int_op(0), IntBound::bounded(3, 10))];
        let ops = vec![make_op(OpCode::IntNeg, &[OpRef::int_op(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef::int_op(1));
        // neg([3, 10]) = [-10, -3]
        assert_eq!(b.lower, -10);
        assert_eq!(b.upper, -3);
    }

    #[test]
    fn test_int_invert_bounds() {
        let initial_bounds = vec![(OpRef::int_op(0), IntBound::bounded(3, 10))];
        let ops = vec![make_op(OpCode::IntInvert, &[OpRef::int_op(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef::int_op(1));
        // invert([3, 10]) = [!10, !3] = [-11, -4]
        assert_eq!(b.lower, -11);
        assert_eq!(b.upper, -4);
    }

    #[test]
    fn test_sub_ovf_same_arg() {
        // INT_SUB_OVF(x, x) should be replaced by constant 0
        let initial_bounds = vec![(OpRef::int_op(0), IntBound::unbounded())];
        let ops = vec![make_op(
            OpCode::IntSubOvf,
            &[OpRef::int_op(0), OpRef::int_op(0)],
            1,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "INT_SUB_OVF(x, x) should be removed");
        let b = ctx.getintbound(OpRef::int_op(1));
        assert!(b.is_constant() && b.get_constant_int() == 0);
    }

    // ── Test: Unsigned comparison optimization ──

    #[test]
    fn test_uint_lt_known_true() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 5)),
            (OpRef::int_op(1), IntBound::bounded(10, 20)),
        ];
        let ops = vec![make_op(
            OpCode::UintLt,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "UINT_LT should be removed when known true"
        );
    }

    // ── Test: Lshift bounds ──

    #[test]
    fn test_int_lshift_bounds() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(1, 4)),
            (OpRef::int_op(1), IntBound::from_constant(2)),
        ];
        let ops = vec![make_op(
            OpCode::IntLshift,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef::int_op(2));
        // [1, 4] << 2 = [4, 16]
        assert_eq!(b.lower, 4);
        assert_eq!(b.upper, 16);
    }

    // ── Test: Rshift bounds ──

    #[test]
    fn test_int_rshift_bounds() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(8, 20)),
            (OpRef::int_op(1), IntBound::from_constant(2)),
        ];
        let ops = vec![make_op(
            OpCode::IntRshift,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef::int_op(2));
        // [8, 20] >> 2 = [2, 5]
        assert_eq!(b.lower, 2);
        assert_eq!(b.upper, 5);
    }

    // ── Test: INT_IS_TRUE and INT_IS_ZERO produce bool bounds ──

    #[test]
    fn test_int_is_true_passthrough() {
        // RPython: IntIsTrue has no postprocess — just passes through.
        let ops = vec![make_op(OpCode::IntIsTrue, &[OpRef::int_op(0)], 1)];
        let (result, _) = run_pass_with_bounds(&ops, &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntIsTrue);
    }

    #[test]
    fn test_int_is_zero_passthrough() {
        // RPython: IntIsZero has no postprocess — just passes through.
        let ops = vec![make_op(OpCode::IntIsZero, &[OpRef::int_op(0)], 1)];
        let (result, _) = run_pass_with_bounds(&ops, &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntIsZero);
    }

    // ── Test: Comparison with unknown bounds stays ──

    #[test]
    fn test_int_lt_unknown_not_removed() {
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::unbounded()),
            (OpRef::int_op(1), IntBound::unbounded()),
        ];
        let ops = vec![make_op(
            OpCode::IntLt,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(
            result.len(),
            1,
            "INT_LT should remain when bounds are unknown"
        );
        assert_eq!(result[0].opcode, OpCode::IntLt);
    }

    // ── Test: INT_SIGNEXT ──

    #[test]
    fn test_int_signext_eliminated() {
        // x in [-100, 100], signext to 2 bytes (-32768..32767) -> identity
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(-100, 100)),
            (OpRef::int_op(1), IntBound::from_constant(2)), // byte_size = 2
        ];
        let ops = vec![make_op(
            OpCode::IntSignext,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, _) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "signext should be eliminated when value fits"
        );
    }

    #[test]
    fn test_int_signext_kept() {
        // x in [-50000, 50000], signext to 1 byte (-128..127) -> can't eliminate
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(-50000, 50000)),
            (OpRef::int_op(1), IntBound::from_constant(1)), // byte_size = 1
        ];
        let ops = vec![make_op(
            OpCode::IntSignext,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntSignext);
        // Result should have bounds [-128, 127]
        let b = ctx.getintbound(OpRef::int_op(2));
        assert!(b.lower >= -128);
        assert!(b.upper <= 127);
    }

    // ── Test: Chain of operations with bound propagation ──

    #[test]
    fn test_chain_add_then_compare() {
        // x in [0, 5], y in [0, 5]
        // z = x + y   -> z in [0, 10]
        // z < 100     -> always true
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(0, 5)),
            (OpRef::int_op(1), IntBound::bounded(0, 5)),
            (OpRef::int_op(3), IntBound::from_constant(100)),
        ];
        let ops = vec![
            make_op(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            make_op(OpCode::IntLt, &[OpRef::int_op(2), OpRef::int_op(3)], 4),
        ];

        let (result, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // INT_ADD should remain, INT_LT should be eliminated as constant true
        assert_eq!(result.len(), 1, "only INT_ADD should remain");
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(ctx.get_constant_int(OpRef::int_op(4)), Some(1));
    }

    // ── Test: Guard narrowing with INT_IS_TRUE ──

    #[test]
    fn test_guard_true_on_int_is_true() {
        // i0 in [0, 100]
        // i1 = INT_IS_TRUE(i0)
        // GUARD_TRUE(i1)
        // After guard, i0 should have lower > 0
        let initial_bounds = vec![(OpRef::int_op(0), IntBound::bounded(0, 100))];
        let ops = vec![
            make_op(OpCode::IntIsTrue, &[OpRef::int_op(0)], 1),
            make_op(OpCode::GuardTrue, &[OpRef::int_op(1)], 2),
        ];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b0 = ctx.getintbound(OpRef::int_op(0));
        assert!(
            b0.lower >= 1,
            "After GUARD_TRUE(INT_IS_TRUE(i0)), i0.lower should be >= 1, got {}",
            b0.lower
        );
    }

    #[test]
    fn test_guard_false_on_int_is_true() {
        // i0 in [0, 100]
        // i1 = INT_IS_TRUE(i0)
        // GUARD_FALSE(i1)
        // After guard, i0 should be 0
        let initial_bounds = vec![(OpRef::int_op(0), IntBound::bounded(0, 100))];
        let ops = vec![
            make_op(OpCode::IntIsTrue, &[OpRef::int_op(0)], 1),
            make_op(OpCode::GuardFalse, &[OpRef::int_op(1)], 2),
        ];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b0 = ctx.getintbound(OpRef::int_op(0));
        assert!(
            b0.is_constant() && b0.get_constant_int() == 0,
            "After GUARD_FALSE(INT_IS_TRUE(i0)), i0 should be 0, got [{}, {}]",
            b0.lower,
            b0.upper
        );
    }

    // ── Test: x + x bounds ──

    #[test]
    fn test_int_add_x_plus_x() {
        // x in [3, 5], x + x -> should be [6, 10]
        let initial_bounds = vec![(OpRef::int_op(0), IntBound::bounded(3, 5))];
        let ops = vec![make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(0), OpRef::int_op(0)],
            1,
        )];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef::int_op(1));
        assert!(b.lower >= 6, "lower should be >= 6, got {}", b.lower);
        assert!(b.upper <= 10, "upper should be <= 10, got {}", b.upper);
    }

    // ── Test: Backward propagation through arithmetic ──

    #[test]
    fn test_backward_prop_int_neg() {
        // i0 unbounded
        // i1 = INT_NEG(i0)  -- i1 in [-5, -1] initially
        // After propagation, i0 should have bounds [1, 5]
        let initial_bounds = vec![(OpRef::int_op(0), IntBound::unbounded())];
        let ops = vec![make_op(OpCode::IntNeg, &[OpRef::int_op(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // Manually tighten the result and trigger backward prop. The
        // OptIntBounds pass is stateless except for `last_emitted_*`
        // (everything else lives on `ctx.forwarded`), so we can spin up a
        // fresh one to drive the backward propagation step.
        let mut pass = OptIntBounds::new();
        ctx.setintbound(OpRef::int_op(1), &IntBound::bounded(-5, -1));
        pass.propagate_bounds_backward(OpRef::int_op(1), &mut ctx);
        let b0 = ctx.getintbound(OpRef::int_op(0));
        assert!(
            b0.lower >= 1,
            "backward neg: lower should be >= 1, got {}",
            b0.lower
        );
        assert!(
            b0.upper <= 5,
            "backward neg: upper should be <= 5, got {}",
            b0.upper
        );
    }

    #[test]
    fn test_strgetitem_bounds() {
        // postprocess_STRGETITEM: result should be bounded to [0, 255].
        let ops = vec![make_op(
            OpCode::Strgetitem,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            2,
        )];
        let (_result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        let b = ctx.getintbound(OpRef::int_op(2));
        assert!(b.lower >= 0, "STRGETITEM lower should be >= 0");
        assert!(b.upper <= 255, "STRGETITEM upper should be <= 255");
    }

    #[test]
    fn test_call_int_py_div_uses_numerator_and_divisor_bounds() {
        // RPython intbounds.py:165-169 post_call_INT_PY_DIV:
        //   r.intersect(b1.py_div_bound(b2))
        // This must not use the modulo-style [0, divisor-1] bound.
        let mut call = make_op(
            OpCode::CallPureI,
            &[OpRef::int_op(10), OpRef::int_op(0), OpRef::int_op(1)],
            2,
        );
        call.descr = Some(call_descr(12, majit_ir::OopSpecIndex::IntPyDiv));
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(-100, -10)),
            (OpRef::int_op(1), IntBound::bounded(2, 4)),
        ];

        let (_result, mut ctx) = run_pass_with_bounds(&[call], &initial_bounds);
        let b = ctx.getintbound(OpRef::int_op(2));
        assert_eq!(b.lower, -50);
        assert_eq!(b.upper, -3);
    }

    #[test]
    fn test_call_int_py_mod_uses_mod_bound_for_negative_divisor() {
        // RPython intbounds.py:171-175 post_call_INT_PY_MOD:
        //   r.intersect(b1.mod_bound(b2))
        let mut call = make_op(
            OpCode::CallPureI,
            &[OpRef::int_op(10), OpRef::int_op(0), OpRef::int_op(1)],
            2,
        );
        call.descr = Some(call_descr(14, majit_ir::OopSpecIndex::IntPyMod));
        let initial_bounds = vec![
            (OpRef::int_op(0), IntBound::bounded(-100, 100)),
            (OpRef::int_op(1), IntBound::bounded(-4, -2)),
        ];

        let (_result, mut ctx) = run_pass_with_bounds(&[call], &initial_bounds);
        let b = ctx.getintbound(OpRef::int_op(2));
        assert_eq!(b.lower, -3);
        assert_eq!(b.upper, 0);
    }

    /// RPython postprocess_GUARD_TRUE parity test:
    /// GuardTrue(IntLt(i, len)) should tighten i's upper bound so that
    /// IntAddOvf(i, 1) is converted to IntAdd. This is the nbody inner
    /// loop pattern and relies on rewrite postprocess first fixing the
    /// comparison result to 1, then intbounds driving
    /// propagate_bounds_backward_op() / intbounds.py:608-651.
    #[test]
    fn test_guard_true_int_lt_enables_add_ovf_removal() {
        use crate::optimizeopt::optimizer::Optimizer;

        let ops = vec![
            make_op(OpCode::IntLt, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            make_op(OpCode::GuardTrue, &[OpRef::int_op(2)], 3),
            make_op(
                OpCode::IntAddOvf,
                &[OpRef::int_op(0), OpRef::int_op(200)],
                4,
            ),
            make_op(OpCode::GuardNoOverflow, &[], 5),
            make_op(OpCode::Jump, &[OpRef::int_op(4)], 6),
        ];

        let mut opt = Optimizer::default_pipeline();
        // IntLt/IntAddOvf operate on Int-typed inputs — override the
        // test default (Ref) used by `optimize_with_constants_and_inputs_at`.
        opt.trace_inputarg_types = vec![majit_ir::Type::Int; 1024];
        let mut constants = std::collections::HashMap::new();
        constants.insert(200, 1i64);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        let opcodes: Vec<_> = result.iter().map(|op| op.opcode).collect();
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntAdd),
            "INT_ADD_OVF should be transformed to INT_ADD after GuardTrue(IntLt); got {opcodes:?}"
        );
        assert!(
            !result.iter().any(|op| op.opcode == OpCode::IntAddOvf),
            "INT_ADD_OVF should not remain; got {opcodes:?}"
        );
    }
}
