/// Integer division by constant using magic number multiplication.
///
/// Translated from rpython/jit/metainterp/optimizeopt/intdiv.py.
///
/// Replaces signed integer division by a constant with a sequence of
/// UINT_MUL_HIGH + shift operations, avoiding the expensive `idiv` instruction.
use majit_ir::{Op, OpCode, OpRef};

use crate::optimizeopt::OptContext;

/// Compute magic numbers for division by constant `m`.
///
/// Returns `(k, i)` where `k` is the multiplier and `i` is the shift amount.
/// The relationship is: `k = 2^(64+i) / m + 1`.
///
/// Preconditions: `m >= 3`, `m` is not a power of two.
pub fn magic_numbers(m: i64) -> (u64, u32) {
    debug_assert!(m >= 3);
    debug_assert!(m & (m - 1) != 0, "m must not be a power of two");
    debug_assert!(m == m as i64, "m must fit in signed i64"); // always true

    let m_u = m as u64;

    // Find i such that 2^i < m < 2^(i+1)
    let mut i: u32 = 1;
    while (1u64 << (i + 1)) < m_u {
        i += 1;
    }

    // Compute quotient = 2^(64+i) // m using bit-by-bit long division.
    // We cannot represent 2^(64+i) directly, so we use the fact that
    // UINT_MUL_HIGH(t, m) gives us the high 64 bits of t*m.
    let high_word_dividend = 1u64 << i;
    let mut quotient: u64 = 0;
    for bit in (0..64).rev() {
        let t = quotient + (1u64 << bit);
        // Check: is t * m < 2^(64+i)?
        // Equivalently: UINT_MUL_HIGH(t, m) < high_word_dividend
        let (_, high) = full_mul_u64(t, m_u);
        if high < high_word_dividend {
            quotient = t;
        }
    }

    // k = 2^(64+i) // m + 1
    let k = quotient + 1;

    debug_assert!(k != 0);
    // k > 2^63 because m < 2^(i+1) implies 2^(64+i) // m >= 2^63
    debug_assert!(k > (1u64 << 63));

    (k, i)
}

/// Full 128-bit multiplication of two u64 values.
/// Returns (low, high) halves.
#[inline]
fn full_mul_u64(a: u64, b: u64) -> (u64, u64) {
    let result = (a as u128) * (b as u128);
    (result as u64, (result >> 64) as u64)
}

/// Emit a constant integer into the optimization context.
pub(crate) fn emit_constant_int(ctx: &mut OptContext, value: i64) -> OpRef {
    ctx.make_constant_int(value)
}

/// RPython intdiv.py: emit an op through the pass chain.
///
/// In RPython, intdiv returns a list of ops and the caller sends each
/// through `send_extra_operation()`. In majit, we use `emit_extra()`
/// to route through downstream passes, matching the upstream semantics.
fn emit_op(ctx: &mut OptContext, pass_idx: usize, op: Op) -> OpRef {
    ctx.emit_extra(pass_idx, op)
}

/// Generate division operations: `n // m` using multiply-and-shift.
///
/// `pass_idx`: caller's pass index, so synthesized ops route through
/// downstream passes via `emit_extra` (matching RPython's
/// `send_extra_operation`).
///
/// Algorithm:
/// ```text
///   t = n >> 63            (sign bits: 0 or -1)
///   nt = n ^ t             (absolute value - 1 if negative)
///   mul = UINT_MUL_HIGH(nt, k)
///   sh = UINT_RSHIFT(mul, i)
///   result = sh ^ t        (negate back if needed)
/// ```
///
/// When `known_nonneg` is true, skips sign correction (saves 3 ops):
/// ```text
///   mul = UINT_MUL_HIGH(n, k)
///   result = UINT_RSHIFT(mul, i)
/// ```
pub fn division_operations(
    n_ref: OpRef,
    m: i64,
    known_nonneg: bool,
    pass_idx: usize,
    ctx: &mut OptContext,
) -> OpRef {
    let (k, i) = magic_numbers(m);

    let k_ref = emit_constant_int(ctx, k as i64);
    let i_ref = emit_constant_int(ctx, i as i64);

    if !known_nonneg {
        // t = n >> 63
        let shift63_ref = emit_constant_int(ctx, 63);
        let t_ref = emit_op(
            ctx,
            pass_idx,
            Op::new(OpCode::IntRshift, &[n_ref, shift63_ref]),
        );

        // nt = n ^ t
        let nt_ref = emit_op(ctx, pass_idx, Op::new(OpCode::IntXor, &[n_ref, t_ref]));

        // mul = UINT_MUL_HIGH(nt, k)
        let mul_ref = emit_op(
            ctx,
            pass_idx,
            Op::new(OpCode::UintMulHigh, &[nt_ref, k_ref]),
        );

        // sh = UINT_RSHIFT(mul, i)
        let sh_ref = emit_op(
            ctx,
            pass_idx,
            Op::new(OpCode::UintRshift, &[mul_ref, i_ref]),
        );

        // result = sh ^ t
        emit_op(ctx, pass_idx, Op::new(OpCode::IntXor, &[sh_ref, t_ref]))
    } else {
        // mul = UINT_MUL_HIGH(n, k)
        let mul_ref = emit_op(ctx, pass_idx, Op::new(OpCode::UintMulHigh, &[n_ref, k_ref]));

        // result = UINT_RSHIFT(mul, i)
        emit_op(
            ctx,
            pass_idx,
            Op::new(OpCode::UintRshift, &[mul_ref, i_ref]),
        )
    }
}

/// Generate modulo operations: `n % m` using division + multiply + subtract.
///
/// Computes: `n - (n // m) * m`.
pub fn modulo_operations(
    n_ref: OpRef,
    m: i64,
    known_nonneg: bool,
    pass_idx: usize,
    ctx: &mut OptContext,
) -> OpRef {
    let div_ref = division_operations(n_ref, m, known_nonneg, pass_idx, ctx);

    // product = div_result * m
    let m_ref = emit_constant_int(ctx, m);
    let product_ref = emit_op(ctx, pass_idx, Op::new(OpCode::IntMul, &[div_ref, m_ref]));

    // remainder = n - product
    emit_op(
        ctx,
        pass_idx,
        Op::new(OpCode::IntSub, &[n_ref, product_ref]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::OptContext;

    fn drain_extra_ops(ctx: &mut OptContext) {
        while let Some((_, op)) = ctx.extra_operations_after.pop_front() {
            ctx.new_operations.push(op);
        }
    }

    // ── magic_numbers tests ──

    #[test]
    fn test_magic_numbers_3() {
        let (k, i) = magic_numbers(3);
        assert_eq!(i, 1);
        // k = 2^(64+1) // 3 + 1
        // 2^65 // 3 = 12297829382473034410
        // k = 12297829382473034411
        assert_eq!(k, 0xAAAAAAAAAAAAAAABu64);
    }

    #[test]
    fn test_magic_numbers_7() {
        let (k, i) = magic_numbers(7);
        assert_eq!(i, 2);
        // k should be > 2^63
        assert!(k > (1u64 << 63));
    }

    #[test]
    fn test_magic_numbers_10() {
        let (k, i) = magic_numbers(10);
        assert_eq!(i, 3);
        assert!(k > (1u64 << 63));
    }

    #[test]
    fn test_magic_numbers_various() {
        for m in [3, 5, 6, 7, 9, 10, 11, 12, 13, 100, 1000, 127, 255] {
            let (k, i) = magic_numbers(m);
            assert!(k > (1u64 << 63), "k too small for m={m}");
            assert!(i < 64, "i too large for m={m}");
        }
    }

    /// Verify the magic number multiplication gives correct division results.
    #[test]
    fn test_magic_numbers_correctness() {
        for m in [3i64, 5, 7, 10, 13, 100, 127, 1000] {
            let (k, i) = magic_numbers(m);
            // Test with positive dividends
            for n in [0i64, 1, 2, m - 1, m, m + 1, 100, 999, 10000, i64::MAX / 2] {
                let expected = n / m; // positive n: floor == trunc
                let actual = apply_magic_div(n, k, i);
                assert_eq!(
                    actual, expected,
                    "division failed: {n} / {m} = expected {expected}, got {actual}"
                );
            }
            // Test with negative dividends.
            // The algorithm produces floor division (towards -inf),
            // matching Python's // operator.
            for n in [-1i64, -2, -(m - 1), -m, -(m + 1), -100, -999, -10000] {
                let expected = floor_div(n, m);
                let actual = apply_magic_div(n, k, i);
                assert_eq!(
                    actual, expected,
                    "division failed: {n} // {m} = expected {expected}, got {actual}"
                );
            }
        }
    }

    /// Apply the magic number division algorithm manually.
    fn apply_magic_div(n: i64, k: u64, i: u32) -> i64 {
        let t = n >> 63; // 0 or -1
        let nt = n ^ t; // if negative: ~n; if positive: n
        let (_, high) = full_mul_u64(nt as u64, k);
        let sh = high >> i;
        (sh as i64) ^ t
    }

    /// Floor division (towards negative infinity), matching Python's // operator.
    fn floor_div(a: i64, b: i64) -> i64 {
        let d = a / b;
        let r = a % b;
        if (r != 0) && ((r ^ b) < 0) { d - 1 } else { d }
    }

    // ── division_operations tests ──

    #[test]
    fn test_division_ops_emits_correct_sequence() {
        let mut ctx = OptContext::new(16);
        // op0 = input variable n
        let n_op = Op::new(OpCode::SameAsI, &[]);
        let n_ref = ctx.emit(n_op);

        let result_ref = division_operations(n_ref, 7, false, 0, &mut ctx);
        drain_extra_ops(&mut ctx);

        // Constants live in the constant table, not new_operations.
        // The helper emits five queued operations after the input.
        assert_eq!(ctx.new_operations.len(), 6); // 1 input + 5 queued ops

        // Check the final op is IntXor (sign correction)
        let final_op = &ctx.new_operations[result_ref.raw() as usize];
        assert_eq!(final_op.opcode, OpCode::IntXor);

        // Verify UintMulHigh is present
        let has_mul_high = ctx
            .new_operations
            .iter()
            .any(|op| op.opcode == OpCode::UintMulHigh);
        assert!(has_mul_high, "should contain UintMulHigh");

        // Verify UintRshift is present
        let has_rshift = ctx
            .new_operations
            .iter()
            .any(|op| op.opcode == OpCode::UintRshift);
        assert!(has_rshift, "should contain UintRshift");
    }

    #[test]
    fn test_division_ops_known_nonneg() {
        let mut ctx = OptContext::new(8);
        let n_op = Op::new(OpCode::SameAsI, &[]);
        let n_ref = ctx.emit(n_op);

        let result_ref = division_operations(n_ref, 7, true, 0, &mut ctx);
        drain_extra_ops(&mut ctx);

        // known_nonneg emits two queued operations after the input.
        assert_eq!(ctx.new_operations.len(), 3); // 1 input + 2 queued ops

        let final_op = &ctx.new_operations[result_ref.raw() as usize];
        assert_eq!(final_op.opcode, OpCode::UintRshift);
    }

    // ── modulo_operations tests ──

    #[test]
    fn test_modulo_ops_emits_correct_sequence() {
        let mut ctx = OptContext::new(16);
        let n_op = Op::new(OpCode::SameAsI, &[]);
        let n_ref = ctx.emit(n_op);

        let result_ref = modulo_operations(n_ref, 7, false, 0, &mut ctx);
        drain_extra_ops(&mut ctx);

        // Five queued division ops + IntMul + IntSub after the input.
        assert_eq!(ctx.new_operations.len(), 8); // 1 input + 7 queued ops

        let final_op = &ctx.new_operations[result_ref.raw() as usize];
        assert_eq!(final_op.opcode, OpCode::IntSub);

        // Verify IntMul is present (div_result * m)
        let has_mul = ctx
            .new_operations
            .iter()
            .any(|op| op.opcode == OpCode::IntMul);
        assert!(has_mul, "should contain IntMul for div*m");
    }

    #[test]
    fn test_modulo_ops_known_nonneg() {
        let mut ctx = OptContext::new(12);
        let n_op = Op::new(OpCode::SameAsI, &[]);
        let n_ref = ctx.emit(n_op);

        let result_ref = modulo_operations(n_ref, 7, true, 0, &mut ctx);
        drain_extra_ops(&mut ctx);

        // Two queued division ops + IntMul + IntSub after the input.
        assert_eq!(ctx.new_operations.len(), 5); // 1 input + 4 queued ops

        let final_op = &ctx.new_operations[result_ref.raw() as usize];
        assert_eq!(final_op.opcode, OpCode::IntSub);
    }

    // ── Edge cases ──

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_magic_numbers_panics_for_power_of_two() {
        magic_numbers(4);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_magic_numbers_panics_for_two() {
        magic_numbers(2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_magic_numbers_panics_for_one() {
        magic_numbers(1);
    }

    #[test]
    fn test_magic_numbers_large_divisor() {
        // Large odd divisor
        let m = (1i64 << 50) + 1;
        let (k, i) = magic_numbers(m);
        assert!(k > (1u64 << 63));
        assert!(i < 64);
    }
}
