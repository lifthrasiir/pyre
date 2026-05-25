//! pypy/objspace/descroperation.py — binary/unary operation dispatch.
//!
//! The ObjSpace mediates all operations on Python objects. This module
//! contains the dispatch layer that routes `+`, `-`, `*`, `//`, `%`,
//! `**`, `<<`, `>>`, `&`, `|`, `^`, comparisons, and unary `+`/`-`/`~`
//! through type-specific fast paths and then the dunder protocol.
#![allow(non_camel_case_types, non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use malachite_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;

use pyre_object::strobject::is_str;
use pyre_object::*;

use crate::baseobjspace::{
    getattr, getitem, is_true, issubtype_w, lookup, lookup_in_type, lookup_in_type_where,
    unwrap_cell,
};
pub use crate::{PyError, PyErrorKind, PyResult};

// ── BigInt helpers ──────────────────────────────────────────────────

/// Extract a BigInt from an int or long object.

unsafe fn as_bigint(obj: PyObjectRef) -> BigInt {
    if is_int(obj) {
        BigInt::from(w_int_get_value(obj))
    } else if is_bool(obj) {
        BigInt::from(w_bool_get_value(obj) as i64)
    } else {
        w_long_get_value(obj).clone()
    }
}

/// Box a BigInt result, demoting to W_IntObject if it fits in i64.

fn bigint_result(value: BigInt) -> PyObjectRef {
    match value.to_i64() {
        Some(v) => w_int_new(v),
        None => w_long_new(value),
    }
}

#[majit_macros::elidable]
fn bigint_add(a: BigInt, b: BigInt) -> BigInt {
    a + b
}

#[majit_macros::elidable]
fn bigint_sub(a: BigInt, b: BigInt) -> BigInt {
    a - b
}

#[majit_macros::elidable]
fn bigint_mul(a: BigInt, b: BigInt) -> BigInt {
    a * b
}

#[majit_macros::elidable]
fn bigint_and(a: BigInt, b: BigInt) -> BigInt {
    a & b
}

#[majit_macros::elidable]
fn bigint_or(a: BigInt, b: BigInt) -> BigInt {
    a | b
}

#[majit_macros::elidable]
fn bigint_xor(a: BigInt, b: BigInt) -> BigInt {
    a ^ b
}

#[majit_macros::elidable]
fn bigint_lshift(a: BigInt, shift: usize) -> BigInt {
    a << shift
}

#[majit_macros::elidable]
fn bigint_rshift(a: BigInt, shift: usize) -> BigInt {
    a >> shift
}

#[majit_macros::elidable]
fn bigint_neg(a: BigInt) -> BigInt {
    -a
}

#[majit_macros::elidable]
fn bigint_invert(a: BigInt) -> BigInt {
    !a
}

#[majit_macros::elidable]
pub(crate) fn bigint_eq(a: BigInt, b: BigInt) -> bool {
    a == b
}

#[majit_macros::elidable]
fn bigint_lt(a: BigInt, b: BigInt) -> bool {
    a < b
}

#[majit_macros::elidable]
fn bigint_gt(a: BigInt, b: BigInt) -> bool {
    a > b
}

#[majit_macros::elidable]
fn bigint_mod(a: BigInt, b: BigInt) -> BigInt {
    a % b
}

// ── Arithmetic operations ─────────────────────────────────────────────

/// Integer addition fast path.
///
/// The JIT will specialize this via:
///   GuardClass(a, &INT_TYPE)
///   GuardClass(b, &INT_TYPE)
///   GetfieldGcI(a, intval_offset) → va
///   GetfieldGcI(b, intval_offset) → vb
///   IntAdd(va, vb) → result
///   New(W_IntObject) + SetfieldGcI(result)

unsafe fn int_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_add(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_add(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_sub(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_sub(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_mul(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_mul(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    // Python floor division: rounds toward negative infinity.
    // i64::MIN / -1 overflows → fall back to BigInt.
    let q = match va.checked_div(vb) {
        Some(q) => q,
        None => return Ok(bigint_result(BigInt::from(va).div_floor(&BigInt::from(vb)))),
    };
    let r = va % vb;
    // Adjust: if remainder is nonzero and signs of operands differ, subtract 1.
    let q = if r != 0 && (r ^ vb) < 0 { q - 1 } else { q };
    Ok(w_int_new(q))
}

unsafe fn int_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    // Python modulo: result has the same sign as the divisor.
    let r = va % vb;
    let r = if r != 0 && (r ^ vb) < 0 { r + vb } else { r };
    Ok(w_int_new(r))
}

// ── Long (BigInt) arithmetic operations ─────────────────────────────

unsafe fn long_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_add(as_bigint(a), as_bigint(b))))
}

unsafe fn long_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_sub(as_bigint(a), as_bigint(b))))
}

unsafe fn long_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_mul(as_bigint(a), as_bigint(b))))
}

unsafe fn long_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_eq(vb.clone(), BigInt::from(0)) {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(bigint_result(as_bigint(a).div_floor(&vb)))
}

unsafe fn long_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_eq(vb.clone(), BigInt::from(0)) {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(bigint_result(as_bigint(a).mod_floor(&vb)))
}

// ── Float arithmetic operations ──────────────────────────────────────

/// Coerce an operand to f64. Works for int, long, and float objects.
unsafe fn as_float(obj: PyObjectRef) -> f64 {
    if is_float(obj) {
        w_float_get_value(obj)
    } else if is_int(obj) {
        w_int_get_value(obj) as f64
    } else {
        // long → f64 (may lose precision for very large values)
        w_long_get_value(obj).to_f64().unwrap_or(f64::INFINITY)
    }
}

/// True if both operands are numeric and at least one is float.

unsafe fn is_float_pair(a: PyObjectRef, b: PyObjectRef) -> bool {
    let a_num = is_int(a) || is_float(a) || is_long(a);
    let b_num = is_int(b) || is_float(b) || is_long(b);
    a_num && b_num && (is_float(a) || is_float(b))
}

unsafe fn float_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) + as_float(b)))
}

unsafe fn float_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) - as_float(b)))
}

unsafe fn float_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) * as_float(b)))
}

unsafe fn float_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_float(b);
    if vb == 0.0 {
        return Err(PyError::zero_division("float division by zero"));
    }
    Ok(w_float_new(as_float(a) / vb))
}

/// floatobject.py:508-512: descr_floordiv → _divmod_w()[0].
unsafe fn float_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (floordiv, _mod) = float_divmod_w(as_float(a), as_float(b))?;
    Ok(w_float_new(floordiv))
}

/// floatobject.py:520-540: descr_mod with math_fmod + sign correction.
unsafe fn float_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let x = as_float(a);
    let y = as_float(b);
    if y == 0.0 {
        // floatobject.py:526
        return Err(PyError::zero_division("float modulo"));
    }
    let mut m = x % y; // fmod
    if m != 0.0 {
        // floatobject.py:529-531: ensure remainder has same sign as denominator
        if (y < 0.0) != (m < 0.0) {
            m += y;
        }
    } else {
        // floatobject.py:536-538: signed zero — copysign(0.0, y)
        m = f64::copysign(0.0, y);
    }
    Ok(w_float_new(m))
}

/// floatobject.py:758-793: _divmod_w.
fn float_divmod_w(x: f64, y: f64) -> Result<(f64, f64), PyError> {
    if y == 0.0 {
        // floatobject.py:761
        return Err(PyError::zero_division("float modulo"));
    }
    let mut m = x % y; // fmod
    // floatobject.py:767: div = (x - mod) / y
    let mut div = (x - m) / y;
    if m != 0.0 {
        // floatobject.py:769-771: sign correction
        if (y < 0.0) != (m < 0.0) {
            m += y;
            div -= 1.0;
        }
    } else {
        // floatobject.py:776-778: signed zero
        // "mod *= mod" hides "+0" from optimizer, then negate if y < 0
        m = m * m; // hide from optimizer
        if y < 0.0 {
            m = -m;
        }
    }
    // floatobject.py:784-790: snap quotient to nearest integral value
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        // floatobject.py:789-790: zero with sign of true quotient
        let d = div * div; // hide from optimizer
        d * x / y
    };
    Ok((floordiv, m))
}

// ── Power ────────────────────────────────────────────────────────────

unsafe fn int_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Ok(w_float_new((va as f64).powf(vb as f64)));
    }
    // intobject.py:415 / longobject.py:229: x ** 0 == 1 for any x.
    if vb == 0 {
        return Ok(w_int_new(1));
    }
    // longobject.py:224-231: rbigint.pow handles arbitrary exponents.
    // Rust BigInt::pow takes u32; short-circuit trivial bases so that
    // e.g. `1 ** huge` returns 1 instead of MemoryError.
    match va {
        0 => return Ok(w_int_new(0)),
        1 => return Ok(w_int_new(1)),
        -1 => return Ok(w_int_new(if vb % 2 == 0 { 1 } else { -1 })),
        _ => {}
    }
    let vb = match u32::try_from(vb) {
        Ok(v) => v,
        Err(_) => return Err(PyError::memory_error("exponent too large")),
    };
    match va.checked_pow(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va).pow(vb))),
    }
}

unsafe fn long_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_lt(vb.clone(), BigInt::from(0)) {
        let fa = as_float(a);
        let fb = as_float(b);
        return Ok(w_float_new(fa.powf(fb)));
    }
    // longobject.py:229: `if not exp_bigint: return int_pow(0)` → 1.
    if vb.sign() == malachite_bigint::Sign::NoSign {
        return Ok(w_int_new(1));
    }
    // longobject.py:224-231: rbigint.pow handles arbitrary exponents.
    // Short-circuit trivial bases before the u32 narrowing so that
    // 1 ** huge, (-1) ** huge, 0 ** huge succeed.
    let va = as_bigint(a);
    if va.sign() == malachite_bigint::Sign::NoSign {
        return Ok(w_int_new(0));
    }
    if va == BigInt::from(1) {
        return Ok(w_int_new(1));
    }
    if va == BigInt::from(-1) {
        let even = vb.clone() % BigInt::from(2) == BigInt::from(0);
        return Ok(w_int_new(if even { 1 } else { -1 }));
    }
    let exp = match vb.to_u32() {
        Some(v) => v,
        None => return Err(PyError::memory_error("exponent too large")),
    };
    Ok(bigint_result(va.pow(exp)))
}

// ── Shift operations ─────────────────────────────────────────────────

unsafe fn int_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    // `i64::checked_shl` only fails when the shift amount is >= 64, so it
    // happily returns a wrapped result when the VALUE overflows (e.g.
    // `(10**18) << 4`). Detect real value overflow by computing the shift
    // in BigInt and demoting to i64 only when the result fits.
    let big = bigint_lshift(BigInt::from(va), vb as usize);
    Ok(bigint_result(big))
}

unsafe fn int_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    let vb = vb as u32;
    Ok(w_int_new(va >> vb.min(63)))
}

unsafe fn long_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_lt(vb.clone(), BigInt::from(0)) {
        return Err(PyError::value_error("negative shift count"));
    }
    // longobject.py:375-380: shift overflows → 0 if base is zero,
    // OverflowError otherwise.
    let shift = match vb.to_usize() {
        Some(v) => v,
        None => {
            let va = as_bigint(a);
            if va.sign() == malachite_bigint::Sign::NoSign {
                return Ok(w_int_new(0));
            }
            return Err(PyError::overflow_error("shift count too large"));
        }
    };
    Ok(bigint_result(bigint_lshift(as_bigint(a), shift)))
}

unsafe fn long_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_lt(vb.clone(), BigInt::from(0)) {
        return Err(PyError::value_error("negative shift count"));
    }
    // longobject.py:393-397: shift overflows → positive yields 0,
    // negative yields -1 (all bits shifted out).
    let shift = match vb.to_usize() {
        Some(v) => v,
        None => {
            let va = as_bigint(a);
            let va = as_bigint(a);
            return Ok(w_int_new(if va.sign() == malachite_bigint::Sign::Minus {
                -1
            } else {
                0
            }));
        }
    };
    Ok(bigint_result(bigint_rshift(as_bigint(a), shift)))
}

// ── bool-as-int helpers ──────────────────────────────────────────────

/// True when obj is int or bool (bool is a subclass of int in Python).
#[inline]
pub(crate) unsafe fn is_int_like(obj: PyObjectRef) -> bool {
    is_int(obj) || is_bool(obj)
}

/// Extract i64 from an int or bool object.
#[inline]
pub(crate) unsafe fn int_value(obj: PyObjectRef) -> i64 {
    if is_bool(obj) {
        w_bool_get_value(obj) as i64
    } else {
        w_int_get_value(obj)
    }
}

// ── Bitwise operations ───────────────────────────────────────────────

unsafe fn int_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let r = int_value(a) & int_value(b);
    // bool & bool → bool
    if is_bool(a) && is_bool(b) {
        return Ok(w_bool_from(r != 0));
    }
    Ok(w_int_new(r))
}

unsafe fn int_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let r = int_value(a) | int_value(b);
    if is_bool(a) && is_bool(b) {
        return Ok(w_bool_from(r != 0));
    }
    Ok(w_int_new(r))
}

unsafe fn int_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let r = int_value(a) ^ int_value(b);
    if is_bool(a) && is_bool(b) {
        return Ok(w_bool_from(r != 0));
    }
    Ok(w_int_new(r))
}

unsafe fn long_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_and(as_bigint(a), as_bigint(b))))
}

unsafe fn long_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_or(as_bigint(a), as_bigint(b))))
}

unsafe fn long_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_xor(as_bigint(a), as_bigint(b))))
}

// ── String operations ────────────────────────────────────────────────

/// Concatenate two str objects.

unsafe fn str_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let sa = w_str_get_value(a);
    let sb = w_str_get_value(b);
    let mut result = String::with_capacity(sa.len() + sb.len());
    result.push_str(sa);
    result.push_str(sb);
    Ok(w_str_new(&result))
}

/// Repeat a str object `n` times.

unsafe fn str_repeat(s: PyObjectRef, n: PyObjectRef) -> PyResult {
    let sv = w_str_get_value(s);
    let nv = w_int_get_value(n);
    let count = if nv < 0 { 0 } else { nv as usize };
    Ok(w_str_new(&sv.repeat(count)))
}

unsafe fn list_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_list_len(a);
    let len_b = w_list_len(b);
    let mut items = Vec::with_capacity(len_a + len_b);
    for i in 0..len_a {
        if let Some(item) = w_list_getitem(a, i as i64) {
            items.push(item);
        }
    }
    for i in 0..len_b {
        if let Some(item) = w_list_getitem(b, i as i64) {
            items.push(item);
        }
    }
    Ok(w_list_new(items))
}

unsafe fn tuple_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_tuple_len(a);
    let len_b = w_tuple_len(b);
    let mut items = Vec::with_capacity(len_a + len_b);
    for i in 0..len_a {
        if let Some(item) = w_tuple_getitem(a, i as i64) {
            items.push(item);
        }
    }
    for i in 0..len_b {
        if let Some(item) = w_tuple_getitem(b, i as i64) {
            items.push(item);
        }
    }
    Ok(w_tuple_new(items))
}

unsafe fn list_repeat(list: PyObjectRef, n: PyObjectRef) -> PyResult {
    let nv = w_int_get_value(n);
    let count = if nv < 0 { 0 } else { nv as usize };
    let len = w_list_len(list);
    let mut items = Vec::with_capacity(len * count);
    for _ in 0..count {
        for i in 0..len {
            if let Some(item) = w_list_getitem(list, i as i64) {
                items.push(item);
            }
        }
    }
    Ok(w_list_new(items))
}

// ── Comparison operations ─────────────────────────────────────────────

unsafe fn int_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) < int_value(b)))
}

unsafe fn int_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) <= int_value(b)))
}

unsafe fn int_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) > int_value(b)))
}

unsafe fn int_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) >= int_value(b)))
}

unsafe fn int_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) == int_value(b)))
}

unsafe fn int_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) != int_value(b)))
}

unsafe fn float_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) < as_float(b)))
}

unsafe fn float_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) <= as_float(b)))
}

unsafe fn float_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) > as_float(b)))
}

unsafe fn float_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) >= as_float(b)))
}

unsafe fn float_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) == as_float(b)))
}

unsafe fn float_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) != as_float(b)))
}

// ── Public dispatch API ───────────────────────────────────────────────

/// Try to call a dunder method on an instance for binary ops.
///
/// PyPy: descroperation.py `_binop_impl` →
///   1. Try `a.__op__(b)` (forward)
///   2. If not found or returns NotImplemented, try `b.__rop__(a)` (reverse)
///
/// descroperation.py:432-437 parity: `space.get_and_call_function` raises
/// OperationError; the NotImplemented return value alone triggers the
/// fallback. We mirror that by propagating PyError immediately — the
/// pyre `call_function` shim stashes exceptions in PENDING_CALL_ERROR
/// so we must consume them via `call_function_impl_result` to match.
unsafe fn try_instance_binop(a: PyObjectRef, b: PyObjectRef, dunder: &str) -> Option<PyResult> {
    let a_is_inst = is_instance(a);
    let b_is_inst = is_instance(b);

    // PyPy: descroperation.py _binop_impl
    // If b's type is a proper subtype of a's type, try reverse first.
    // This matches Python's "subclass reflected op takes priority" rule.
    let try_reverse_first = if a_is_inst && b_is_inst {
        if let Some(rdunder) = reverse_dunder(dunder) {
            let a_type = w_instance_get_type(a);
            let b_type = w_instance_get_type(b);
            !std::ptr::eq(a_type, b_type)
                && issubtype_cached(b_type, a_type)
                && lookup_in_type_where(b_type, rdunder).is_some()
        } else {
            false
        }
    } else {
        false
    };

    if try_reverse_first {
        let rdunder = reverse_dunder(dunder).unwrap();
        let w_type = w_instance_get_type(b);
        if let Some(method) = lookup_in_type_where(w_type, rdunder) {
            match crate::call::call_function_impl_result(method, &[b, a]) {
                Ok(result) => {
                    if !is_not_implemented(result) {
                        return Some(Ok(result));
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }

    // Forward: a.__op__(b)
    if a_is_inst {
        let w_type = w_instance_get_type(a);
        if let Some(method) = lookup_in_type_where(w_type, dunder) {
            match crate::call::call_function_impl_result(method, &[a, b]) {
                Ok(result) => {
                    if !is_not_implemented(result) {
                        return Some(Ok(result));
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }

    // Reverse: b.__rop__(a) — only if not already tried above
    if !try_reverse_first && b_is_inst {
        if let Some(rdunder) = reverse_dunder(dunder) {
            let w_type = w_instance_get_type(b);
            if let Some(method) = lookup_in_type_where(w_type, rdunder) {
                match crate::call::call_function_impl_result(method, &[b, a]) {
                    Ok(result) => {
                        if !is_not_implemented(result) {
                            return Some(Ok(result));
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        }
    }

    None
}

/// `descroperation.py _binop_impl` typedef-driven fallback for
/// non-instance LHS / RHS — pyre's `try_instance_binop` only fires
/// when at least one operand is `is_instance`, but built-in W_Root
/// types (dict_view, exception, generator, …) also expose dunder
/// methods through their typedef.  This helper does the same
/// forward + reverse MRO lookup but routes through
/// `crate::typedef::r#type` instead of `w_instance_get_type`, so
/// `dict_keys() | set()` etc. find the typedef-installed `__or__`
/// and friends.  Returns `None` when neither side defines the
/// method (caller falls through to the existing TypeError path).
unsafe fn try_typedef_binop(a: PyObjectRef, b: PyObjectRef, dunder: &str) -> Option<PyResult> {
    if let Some(a_type) = crate::typedef::r#type(a) {
        if let Some(method) = lookup_in_type_where(a_type, dunder) {
            match crate::call::call_function_impl_result(method, &[a, b]) {
                Ok(result) => {
                    if !is_not_implemented(result) {
                        return Some(Ok(result));
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
    if let Some(rdunder) = reverse_dunder(dunder) {
        if let Some(b_type) = crate::typedef::r#type(b) {
            if let Some(method) = lookup_in_type_where(b_type, rdunder) {
                match crate::call::call_function_impl_result(method, &[b, a]) {
                    Ok(result) => {
                        if !is_not_implemented(result) {
                            return Some(Ok(result));
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        }
    }
    None
}

/// Check if w_type is a subtype of cls using cached MRO.
unsafe fn issubtype_cached(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    let mro_ptr = w_type_get_mro(w_type);
    if !mro_ptr.is_null() {
        return (*mro_ptr).iter().any(|&t| std::ptr::eq(t, cls));
    }
    false
}

/// Map forward dunder to reverse dunder.
/// PyPy: descroperation.py `_make_binop_impl` generates both directions.
fn reverse_dunder(dunder: &str) -> Option<&'static str> {
    Some(match dunder {
        // Arithmetic — PyPy: descroperation.py _make_binop_impl
        "__add__" => "__radd__",
        "__sub__" => "__rsub__",
        "__mul__" => "__rmul__",
        "__truediv__" => "__rtruediv__",
        "__floordiv__" => "__rfloordiv__",
        "__mod__" => "__rmod__",
        "__pow__" => "__rpow__",
        "__lshift__" => "__rlshift__",
        "__rshift__" => "__rrshift__",
        "__and__" => "__rand__",
        "__or__" => "__ror__",
        "__xor__" => "__rxor__",
        // Comparison reflected — PyPy: descroperation.py _cmp_dispatch
        "__lt__" => "__gt__",
        "__le__" => "__ge__",
        "__gt__" => "__lt__",
        "__ge__" => "__le__",
        "__eq__" => "__eq__",
        "__ne__" => "__ne__",
        _ => return None,
    })
}

/// Try to call a unary dunder on an instance.
///
/// PyPy: `ObjSpace.call_function(space.lookup(w_obj, dunder), w_obj)`
/// The Python-level OperationError must propagate to the caller; use the
/// Result-returning call path so PENDING_CALL_ERROR is consumed.
unsafe fn try_instance_unaryop(a: PyObjectRef, dunder: &str) -> Option<PyResult> {
    if is_instance(a) {
        if let Some(method) = lookup(a, dunder) {
            return Some(crate::call::call_function_impl_result(method, &[a]));
        }
    }
    None
}

pub fn add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_add(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_add(a, b);
        }
        if is_float_pair(a, b) {
            return float_add(a, b);
        }
        if is_str(a) && is_str(b) {
            return str_concat(a, b);
        }
        if is_list(a) && is_list(b) {
            return list_concat(a, b);
        }
        if is_tuple(a) && is_tuple(b) {
            return tuple_concat(a, b);
        }
        if pyre_object::bytesobject::is_bytes_like(a) && pyre_object::bytesobject::is_bytes_like(b)
        {
            let a_data = pyre_object::bytesobject::bytes_like_data(a);
            let b_data = pyre_object::bytesobject::bytes_like_data(b);
            let mut result = a_data.to_vec();
            result.extend_from_slice(b_data);
            // bytes + bytes → bytes; anything with bytearray → bytearray
            return Ok(
                if pyre_object::bytesobject::is_bytes(a) && pyre_object::bytesobject::is_bytes(b) {
                    pyre_object::bytesobject::w_bytes_from_bytes(&result)
                } else {
                    pyre_object::bytearrayobject::w_bytearray_from_bytes(&result)
                },
            );
        }
        // Instance dunder dispatch: __add__
        if let Some(result) = try_instance_binop(a, b, "__add__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for +: '{}' and '{}'",
            a_name, b_name,
        )))
    }
}

pub fn sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_sub(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_sub(a, b);
        }
        if is_float_pair(a, b) {
            return float_sub(a, b);
        }
        // set / frozenset difference — PyPy: setobject.py W_BaseSetObject.descr_sub
        if pyre_object::is_set_or_frozenset(a) {
            let other_items = crate::builtins::collect_iterable(b)?;
            let probe = pyre_object::w_set_from_items(&other_items);
            let result: Vec<PyObjectRef> = pyre_object::w_set_items(a)
                .into_iter()
                .filter(|&item| !pyre_object::w_set_contains(probe, item))
                .collect();
            return Ok(if pyre_object::is_frozenset(a) {
                pyre_object::w_frozenset_from_items(&result)
            } else {
                pyre_object::w_set_from_items(&result)
            });
        }
        if let Some(result) = try_instance_binop(a, b, "__sub__") {
            return result;
        }
        // Built-in W_Root types (dict_view, …) expose `__sub__` /
        // `__rsub__` through their typedef.
        if let Some(result) = try_typedef_binop(a, b, "__sub__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for -: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_mul(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mul(a, b);
        }
        if is_float_pair(a, b) {
            return float_mul(a, b);
        }
        if is_str(a) && is_int(b) {
            return str_repeat(a, b);
        }
        if is_int(a) && is_str(b) {
            return str_repeat(b, a);
        }
        // list * int
        if is_list(a) && is_int(b) {
            return list_repeat(a, b);
        }
        if is_int(a) && is_list(b) {
            return list_repeat(b, a);
        }
        // tuple * int
        if is_tuple(a) && is_int(b) {
            let n = w_int_get_value(b).max(0) as usize;
            let len = w_tuple_len(a);
            let mut items = Vec::with_capacity(n * len);
            for _ in 0..n {
                for i in 0..len {
                    if let Some(item) = w_tuple_getitem(a, i as i64) {
                        items.push(item);
                    }
                }
            }
            return Ok(w_tuple_new(items));
        }
        if is_int(a) && is_tuple(b) {
            return mul(b, a);
        }
        // bytes/bytearray * int — bytesobject.py descr_mul / bytearrayobject.py descr_mul
        if pyre_object::bytesobject::is_bytes_like(a) && is_int(b) {
            let data = pyre_object::bytesobject::bytes_like_data(a);
            let n = w_int_get_value(b).max(0) as usize;
            let mut buf = Vec::with_capacity(data.len() * n);
            for _ in 0..n {
                buf.extend_from_slice(data);
            }
            return Ok(if pyre_object::bytesobject::is_bytes(a) {
                pyre_object::bytesobject::w_bytes_from_bytes(&buf)
            } else {
                pyre_object::bytearrayobject::w_bytearray_from_bytes(&buf)
            });
        }
        if is_int(a) && pyre_object::bytesobject::is_bytes_like(b) {
            return mul(b, a);
        }
        if let Some(result) = try_instance_binop(a, b, "__mul__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for *: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_floordiv(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_floordiv(a, b);
        }
        if is_float_pair(a, b) {
            return float_floordiv(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__floordiv__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for //: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn mod_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_mod(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mod(a, b);
        }
        if is_float_pair(a, b) {
            return float_mod(a, b);
        }
        // str % args — PyPy: unicodeobject.py mod__String_ANY
        if is_str(a) {
            return crate::objspace::std::formatting::str_format_percent(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__mod__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for %: '{a_name}' and '{b_name}'"
        )))
    }
}

/// True division (`/` operator) — always produces a float result.

pub fn truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if a_num && b_num {
            return float_truediv(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__truediv__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for /: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Power operation dispatch (`**` operator).

pub fn pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_pow(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_pow(a, b);
        }
        if is_float_pair(a, b) {
            return float_pow_impl(as_float(a), as_float(b));
        }
        if let Some(result) = try_instance_binop(a, b, "__pow__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for **: '{a_name}' and '{b_name}'"
        )))
    }
}

// ── descroperation helpers — pypy/objspace/descroperation.py ──────────
//
// These helpers implement the standard "forward + reverse with
// NotImplemented fallback" dispatch that PyPy generates from
// `_make_binop_impl` / `_make_descr_unaryop`. They were originally in
// `baseobjspace` (not in `builtins`) because they are space-level
// semantics shared between the builtin module, the weakproxy wrappers,
// and any future opcode dispatch — every caller needs the same rule
// or NotImplemented from the forward path silently swallows the
// reflected operand.

/// `space.lookup(w_obj, dunder)` — descroperation.py.
pub(crate) unsafe fn lookup_type_special(obj: PyObjectRef, dunder: &str) -> Option<PyObjectRef> {
    crate::typedef::r#type(obj).and_then(|tp| lookup_in_type(tp, dunder))
}

/// descroperation.py `_binop_impl` — when `type(rhs)` is a proper
/// subtype of `type(lhs)` and defines the reflected dunder, the
/// reflected operand is tried first.
pub(crate) unsafe fn should_try_reverse_first(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    rdunder: &str,
) -> bool {
    let Some(lhs_type) = crate::typedef::r#type(lhs) else {
        return false;
    };
    let Some(rhs_type) = crate::typedef::r#type(rhs) else {
        return false;
    };
    !std::ptr::eq(lhs_type, rhs_type)
        && issubtype_w(rhs_type, lhs_type)
        && lookup_in_type(rhs_type, rdunder).is_some()
}

/// Call a special method and treat NotImplemented as "no result", per
/// descroperation.py `_check_notimplemented`.
pub(crate) fn try_call_special(
    method: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<Option<PyObjectRef>, PyError> {
    let result = crate::call::call_function_impl_result(method, args)?;
    if unsafe { is_not_implemented(result) } {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// descroperation.py `_make_binop_impl` — forward `__op__` then
/// reflected `__rop__`, with the reflected-first reordering rule when
/// `type(rhs)` subtypes `type(lhs)`.
pub(crate) fn try_dispatch_binary_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    dunder: &str,
    rdunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    let try_reverse_first = unsafe { should_try_reverse_first(lhs, rhs, rdunder) };
    if try_reverse_first {
        if let Some(method) = unsafe { lookup_type_special(rhs, rdunder) } {
            if let Some(result) = try_call_special(method, &[rhs, lhs])? {
                return Ok(Some(result));
            }
        }
    }
    if let Some(method) = unsafe { lookup_type_special(lhs, dunder) } {
        if let Some(result) = try_call_special(method, &[lhs, rhs])? {
            return Ok(Some(result));
        }
    }
    if !try_reverse_first {
        if let Some(method) = unsafe { lookup_type_special(rhs, rdunder) } {
            if let Some(result) = try_call_special(method, &[rhs, lhs])? {
                return Ok(Some(result));
            }
        }
    }
    Ok(None)
}

/// descroperation.py:399 `def pow(space, w_obj1, w_obj2, w_obj3)` —
/// the same forward/reverse dance as the binary version but threading
/// the third (modulo) operand through to both arms.
pub(crate) fn try_dispatch_ternary_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    third: PyObjectRef,
    dunder: &str,
    rdunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    let try_reverse_first = unsafe { should_try_reverse_first(lhs, rhs, rdunder) };
    if try_reverse_first {
        if let Some(method) = unsafe { lookup_type_special(rhs, rdunder) } {
            if let Some(result) = try_call_special(method, &[rhs, lhs, third])? {
                return Ok(Some(result));
            }
        }
    }
    if let Some(method) = unsafe { lookup_type_special(lhs, dunder) } {
        if let Some(result) = try_call_special(method, &[lhs, rhs, third])? {
            return Ok(Some(result));
        }
    }
    if !try_reverse_first {
        if let Some(method) = unsafe { lookup_type_special(rhs, rdunder) } {
            if let Some(result) = try_call_special(method, &[rhs, lhs, third])? {
                return Ok(Some(result));
            }
        }
    }
    Ok(None)
}

/// `(int|long) ** (int|long) % (int|long)` fast path used by `space.pow`
/// when a modulus is supplied — longobject.py `int_pow`.
pub(crate) fn try_int_long_pow_with_modulo(
    base: PyObjectRef,
    exp: PyObjectRef,
    modulus: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        if !is_int_or_long(base) || !is_int_or_long(exp) || !is_int_or_long(modulus) {
            return Ok(None);
        }

        let base = crate::builtins::obj_to_bigint(base);
        let exp = crate::builtins::obj_to_bigint(exp);
        let modulus = crate::builtins::obj_to_bigint(modulus);

        if bigint_eq(modulus.clone(), BigInt::from(0)) {
            return Err(PyError::value_error("pow() 3rd argument cannot be 0"));
        }
        if bigint_lt(exp.clone(), BigInt::from(0)) {
            return Err(PyError::type_error(
                "pow() 2nd argument cannot be negative when 3rd argument specified",
            ));
        }
        if bigint_eq(exp.clone(), BigInt::from(0)) {
            return Ok(Some(box_bigint_result(bigint_mod(
                BigInt::from(1),
                modulus,
            ))));
        }

        let negative_modulus = bigint_lt(modulus.clone(), BigInt::from(0));
        let abs_modulus = if negative_modulus {
            bigint_neg(modulus.clone())
        } else {
            modulus.clone()
        };
        let mut result = base.modpow(&exp, &abs_modulus);
        if negative_modulus && bigint_gt(result.clone(), BigInt::from(0)) {
            result = bigint_sub(result, abs_modulus);
        }
        Ok(Some(box_bigint_result(result)))
    }
}

pub(crate) fn box_bigint_result(value: BigInt) -> PyObjectRef {
    use num_traits::ToPrimitive;
    if let Some(small) = value.to_i64() {
        w_int_new(small)
    } else {
        w_long_new(value)
    }
}

pub(crate) fn binary_builtin_type_error(
    opname: &str,
    lhs: PyObjectRef,
    rhs: PyObjectRef,
) -> PyError {
    let lhs_name = unsafe {
        match crate::typedef::r#type(lhs) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*lhs).ob_type).name.to_string(),
        }
    };
    let rhs_name = unsafe {
        match crate::typedef::r#type(rhs) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*rhs).ob_type).name.to_string(),
        }
    };
    PyError::type_error(format!(
        "unsupported operand type(s) for {opname}: '{lhs_name}' and '{rhs_name}'"
    ))
}

/// 3-arg `pow(a, b, c)` dispatch — pypy/objspace/descroperation.py:399
/// `def pow(space, w_obj1, w_obj2, w_obj3)`. Tries the int/long modulus
/// fast path, then forward `__pow__` and reverse `__rpow__` with the
/// usual NotImplemented fallback. PyPy's MethodTable lists `pow` with
/// arity=3 (`('pow', '**', 3, ['__pow__', '__rpow__'])`) so a 3-arg
/// space op exists for the proxy wrapper to call.
pub fn pow3(base: PyObjectRef, exp: PyObjectRef, modulus: PyObjectRef) -> PyResult {
    let base = unwrap_cell(base);
    let exp = unwrap_cell(exp);
    let modulus = unwrap_cell(modulus);
    if unsafe { is_none(modulus) } {
        return pow(base, exp);
    }
    if let Some(result) = try_int_long_pow_with_modulo(base, exp, modulus)? {
        return Ok(result);
    }
    if let Some(result) = try_dispatch_ternary_special(base, exp, modulus, "__pow__", "__rpow__")? {
        return Ok(result);
    }
    Err(binary_builtin_type_error("pow()", base, exp))
}

/// `divmod(a, b)` dispatch — pypy/interpreter/baseobjspace.py:2159
/// `('divmod', 'divmod', 2, ['__divmod__', '__rdivmod__'])`. Numeric
/// fast path then forward + reverse special-method dispatch with the
/// standard NotImplemented fallback.
pub fn divmod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        let lhs_num = is_int(a) || is_long(a) || is_float(a);
        let rhs_num = is_int(b) || is_long(b) || is_float(b);
        if lhs_num && rhs_num {
            let q = floordiv(a, b)?;
            let r = mod_(a, b)?;
            return Ok(w_tuple_new(vec![q, r]));
        }
    }
    if let Some(result) = try_dispatch_binary_special(a, b, "__divmod__", "__rdivmod__")? {
        return Ok(result);
    }
    Err(binary_builtin_type_error("divmod()", a, b))
}

pub fn float_pow_raw(x: f64, y: f64) -> Result<f64, PyError> {
    // floatobject.py:800-801
    if y == 2.0 {
        return Ok(x * x);
    }
    // floatobject.py:803-804
    if y == 0.0 {
        return Ok(1.0);
    }
    // floatobject.py:806-807
    if x.is_nan() {
        return Ok(x);
    }
    // floatobject.py:809-814
    if y.is_nan() {
        return Ok(if x == 1.0 { 1.0 } else { y });
    }
    // floatobject.py:815-827
    if y.is_infinite() {
        let ax = x.abs();
        if ax == 1.0 {
            return Ok(1.0);
        }
        return Ok(if (y > 0.0) == (ax > 1.0) {
            f64::INFINITY
        } else {
            0.0
        });
    }
    // floatobject.py:828-842
    if x.is_infinite() {
        let y_is_odd = y.abs() % 2.0 == 1.0;
        return Ok(if y > 0.0 {
            if y_is_odd { x } else { x.abs() }
        } else if y_is_odd {
            f64::copysign(0.0, x)
        } else {
            0.0
        });
    }
    // floatobject.py:844-847
    if x == 0.0 && y < 0.0 {
        return Err(PyError::zero_division(
            "0.0 cannot be raised to a negative power",
        ));
    }
    // floatobject.py:849-862
    let mut negate_result = false;
    let mut bx = x;
    if bx < 0.0 {
        if y.floor() != y {
            return Err(PyError::value_error(
                "negative number cannot be raised to a fractional power",
            ));
        }
        bx = -bx;
        negate_result = y.abs() % 2.0 == 1.0;
    }
    // floatobject.py:864-869
    if bx == 1.0 {
        return Ok(if negate_result { -1.0 } else { 1.0 });
    }
    // floatobject.py:871-877
    let z = bx.powf(y);
    if z.is_infinite() && !bx.is_infinite() {
        return Err(PyError::overflow_error("float power"));
    }
    // floatobject.py:879-881
    Ok(if negate_result { -z } else { z })
}

/// floatobject.py:562 `W_FloatObject.descr_pow` boxing wrapper over `_pow`.
/// Calls `float_pow_raw` and boxes the raw result into W_FloatObject.
fn float_pow_impl(x: f64, y: f64) -> PyResult {
    use pyre_object::w_float_new;
    Ok(w_float_new(float_pow_raw(x, y)?))
}

/// Left shift dispatch (`<<` operator).

pub fn lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_lshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_lshift(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__lshift__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for <<: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Right shift dispatch (`>>` operator).

pub fn rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_rshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_rshift(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__rshift__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for >>: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Bitwise AND dispatch (`&` operator).

pub fn and_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        // boolobject.py:74 W_BoolObject.descr_and — both operands bool
        // → space.newbool(op(a, b)). MRO ensures this runs before the
        // W_IntObject.descr_and fallback in int_bitand.
        if is_bool(a) && is_bool(b) {
            return Ok(pyre_object::bool_descr_and(a, b));
        }
        if is_int(a) && is_int(b) {
            return int_bitand(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitand(a, b);
        }
        // set / frozenset intersection — PyPy: setobject.py W_BaseSetObject.descr_and
        if pyre_object::is_set_or_frozenset(a) {
            let other_items = crate::builtins::collect_iterable(b)?;
            let probe = pyre_object::w_set_from_items(&other_items);
            let result: Vec<PyObjectRef> = pyre_object::w_set_items(a)
                .into_iter()
                .filter(|&item| pyre_object::w_set_contains(probe, item))
                .collect();
            return Ok(if pyre_object::is_frozenset(a) {
                pyre_object::w_frozenset_from_items(&result)
            } else {
                pyre_object::w_set_from_items(&result)
            });
        }
        if let Some(result) = try_instance_binop(a, b, "__and__") {
            return result;
        }
        // Built-in W_Root types (dict_view, …) expose `__and__` /
        // `__rand__` through their typedef but are not is_instance
        // — fall back to typedef-driven MRO dispatch before TypeError.
        if let Some(result) = try_typedef_binop(a, b, "__and__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for &: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Check if an object can participate in `X | Y` union syntax.
///
/// PyPy equivalent: _unionable() in _pypy_generic_alias.py
#[inline]
fn unionable(obj: PyObjectRef) -> bool {
    unsafe { is_none(obj) || is_type(obj) || pyre_object::is_union(obj) }
}

/// Bitwise OR dispatch (`|` operator).

pub fn or_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    // `pypy/objspace/std/dictproxyobject.py:51 descr_or` /
    // `pypy/objspace/std/dictproxyobject.py:60 descr_ror` —
    // mappingproxy `|` dispatches by copying the proxy's wrapped
    // mapping then `update`-ing with the other operand.  Pre-unwrap
    // each side so the dict-arm below sees plain dicts and produces
    // the same merge result.  The proxy-on-rhs case mirrors
    // `descr_ror` (proxy wraps the rhs operand inside `__or__`).
    let a = unsafe {
        if pyre_object::is_dict_proxy(a) {
            pyre_object::w_dict_proxy_get_mapping(a)
        } else {
            a
        }
    };
    let b = unsafe {
        if pyre_object::is_dict_proxy(b) {
            pyre_object::w_dict_proxy_get_mapping(b)
        } else {
            b
        }
    };
    unsafe {
        // boolobject.py:75 W_BoolObject.descr_or — both operands bool
        // → space.newbool(op(a, b)).
        if is_bool(a) && is_bool(b) {
            return Ok(pyre_object::bool_descr_or(a, b));
        }
        if is_int(a) && is_int(b) {
            return int_bitor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitor(a, b);
        }
        // set / frozenset union — PyPy: setobject.py W_BaseSetObject.descr_or
        if pyre_object::is_set_or_frozenset(a) {
            let mut items = pyre_object::w_set_items(a);
            for item in crate::builtins::collect_iterable(b)? {
                items.push(item);
            }
            return Ok(if pyre_object::is_frozenset(a) {
                pyre_object::w_frozenset_from_items(&items)
            } else {
                pyre_object::w_set_from_items(&items)
            });
        }
        // dict | dict — PEP 584 merge. PyPy: dictmultiobject.py descr_or.
        // Returns a new dict built from `a`'s items, then updated with `b`'s.
        if pyre_object::is_dict(a) && pyre_object::is_dict(b) {
            let new_dict = pyre_object::w_dict_new();
            for (k, v) in pyre_object::w_dict_items(a) {
                pyre_object::w_dict_store(new_dict, k, v);
            }
            for (k, v) in pyre_object::w_dict_items(b) {
                pyre_object::w_dict_store(new_dict, k, v);
            }
            return Ok(new_dict);
        }
        if let Some(result) = try_instance_binop(a, b, "__or__") {
            return result;
        }
        // type | type — PEP 604 union types (Python 3.10+)
        // PyPy: typeobject.py descr_or → _pypy_generic_alias._create_union
        if unionable(a) && unionable(b) {
            return Ok(pyre_object::w_union_new(a, b));
        }
        if let Some(result) = try_instance_binop(a, b, "__ror__") {
            return result;
        }
        // Built-in W_Root types (dict_view, …) expose `__or__` /
        // `__ror__` through their typedef.
        if let Some(result) = try_typedef_binop(a, b, "__or__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for |: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Bitwise XOR dispatch (`^` operator).

pub fn xor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_bool(a) && is_bool(b) {
            return Ok(pyre_object::bool_descr_xor(a, b));
        }
        if is_int(a) && is_int(b) {
            return int_bitxor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitxor(a, b);
        }
        // set / frozenset symmetric difference — `pypy/objspace/std/
        // setobject.py W_BaseSetObject.descr_xor`.  Mirrors `and_`'s
        // intersection arm: walk both sides, keep elements present in
        // exactly one set.  Result type follows the left operand
        // (frozenset stays frozenset).
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            let mut out: Vec<PyObjectRef> = pyre_object::w_set_items(a)
                .into_iter()
                .filter(|&item| !pyre_object::w_set_contains(b, item))
                .collect();
            for item in pyre_object::w_set_items(b) {
                if !pyre_object::w_set_contains(a, item) {
                    out.push(item);
                }
            }
            return Ok(if pyre_object::is_frozenset(a) {
                pyre_object::w_frozenset_from_items(&out)
            } else {
                pyre_object::w_set_from_items(&out)
            });
        }
        if let Some(result) = try_instance_binop(a, b, "__xor__") {
            return result;
        }
        // Built-in W_Root types (dict_view, …) expose `__xor__` /
        // `__rxor__` through their typedef.
        if let Some(result) = try_typedef_binop(a, b, "__xor__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for ^: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Comparison operation dispatch.

pub fn compare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return match op {
                CompareOp::Lt => int_lt(a, b),
                CompareOp::Le => int_le(a, b),
                CompareOp::Gt => int_gt(a, b),
                CompareOp::Ge => int_ge(a, b),
                CompareOp::Eq => int_eq(a, b),
                CompareOp::Ne => int_ne(a, b),
            };
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            let va = as_bigint(a);
            let vb = as_bigint(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => va < vb,
                CompareOp::Le => va <= vb,
                CompareOp::Gt => va > vb,
                CompareOp::Ge => va >= vb,
                CompareOp::Eq => va == vb,
                CompareOp::Ne => va != vb,
            }));
        }
        if is_float_pair(a, b) {
            return match op {
                CompareOp::Lt => float_lt(a, b),
                CompareOp::Le => float_le(a, b),
                CompareOp::Gt => float_gt(a, b),
                CompareOp::Ge => float_ge(a, b),
                CompareOp::Eq => float_eq(a, b),
                CompareOp::Ne => float_ne(a, b),
            };
        }
        if is_str(a) && is_str(b) {
            let sa = w_str_get_value(a);
            let sb = w_str_get_value(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => sa < sb,
                CompareOp::Le => sa <= sb,
                CompareOp::Gt => sa > sb,
                CompareOp::Ge => sa >= sb,
                CompareOp::Eq => sa == sb,
                CompareOp::Ne => sa != sb,
            }));
        }
        // Tuple lexicographic comparison — PyPy: tupleobject.py descr_lt / _eq / etc.
        if is_tuple(a) && is_tuple(b) {
            let la = w_tuple_len(a);
            let lb = w_tuple_len(b);
            let min_len = la.min(lb);
            for i in 0..min_len {
                let ea = w_tuple_getitem(a, i as i64).unwrap_or(PY_NULL);
                let eb = w_tuple_getitem(b, i as i64).unwrap_or(PY_NULL);
                let eq = match compare(ea, eb, CompareOp::Eq) {
                    Ok(r) => is_true(r),
                    Err(_) => false,
                };
                if !eq {
                    return compare(ea, eb, op);
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Lt => la < lb,
                CompareOp::Le => la <= lb,
                CompareOp::Gt => la > lb,
                CompareOp::Ge => la >= lb,
                CompareOp::Eq => la == lb,
                CompareOp::Ne => la != lb,
            }));
        }
        // dict equality — `pypy/objspace/std/dictobject.py
        // W_DictMultiObject.descr_eq` is order-independent: same length
        // AND each key-value pair in `a` exists with equal value in `b`.
        // CPython only defines == / != for dicts (no ordering), so we
        // restrict to those ops; other ops fall through to the dunder
        // dispatch which currently raises TypeError, matching the
        // unimplemented `__lt__` etc. on plain dict.
        if is_dict(a) && is_dict(b) && matches!(op, CompareOp::Eq | CompareOp::Ne) {
            let la = pyre_object::w_dict_len(a);
            let lb = pyre_object::w_dict_len(b);
            let mut equal = la == lb;
            if equal {
                for (k, v) in pyre_object::w_dict_items(a) {
                    match pyre_object::w_dict_lookup(b, k) {
                        Some(other_v) => {
                            let same = compare(v, other_v, CompareOp::Eq)
                                .map(|r| is_true(r))
                                .unwrap_or(false);
                            if !same {
                                equal = false;
                                break;
                            }
                        }
                        None => {
                            equal = false;
                            break;
                        }
                    }
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => !equal,
                _ => unreachable!(),
            }));
        }
        // `dictmultiobject.py:1619-1623 _is_set_like` parity — when
        // one side is a set/frozenset and the other is a set-like
        // dict_view (Keys / Items), the comparison reduces to the
        // set-set arm with the dict_view materialised through its
        // snapshot.  Without this arm, `set == d.keys()` would fall
        // through to `object.__eq__`'s identity check and return
        // False even when the contents match.
        if (pyre_object::is_set_or_frozenset(a) || pyre_object::is_set_or_frozenset(b))
            && (pyre_object::dictviewobject::is_dict_view(a)
                || pyre_object::dictviewobject::is_dict_view(b))
        {
            let view_set_like = |obj: PyObjectRef| -> bool {
                if pyre_object::is_set_or_frozenset(obj) {
                    return true;
                }
                if pyre_object::dictviewobject::is_dict_view(obj) {
                    let kind = pyre_object::dictviewobject::w_dict_view_get_kind(obj);
                    return matches!(
                        kind,
                        pyre_object::dictviewobject::DictViewKind::Keys
                            | pyre_object::dictviewobject::DictViewKind::Items
                    );
                }
                false
            };
            if view_set_like(a) && view_set_like(b) {
                let a_items = if pyre_object::is_set_or_frozenset(a) {
                    pyre_object::w_set_items(a)
                } else {
                    crate::type_methods::dict_view_snapshot(a)
                };
                let b_items = if pyre_object::is_set_or_frozenset(b) {
                    pyre_object::w_set_items(b)
                } else {
                    crate::type_methods::dict_view_snapshot(b)
                };
                let a_set = pyre_object::w_set_from_items(&a_items);
                let b_set = pyre_object::w_set_from_items(&b_items);
                let la = pyre_object::w_set_len(a_set);
                let lb = pyre_object::w_set_len(b_set);
                let a_subset_b = || {
                    pyre_object::w_set_items(a_set)
                        .into_iter()
                        .all(|item| pyre_object::w_set_contains(b_set, item))
                };
                let b_subset_a = || {
                    pyre_object::w_set_items(b_set)
                        .into_iter()
                        .all(|item| pyre_object::w_set_contains(a_set, item))
                };
                return Ok(w_bool_from(match op {
                    CompareOp::Eq => la == lb && a_subset_b(),
                    CompareOp::Ne => la != lb || !a_subset_b(),
                    CompareOp::Le => la <= lb && a_subset_b(),
                    CompareOp::Lt => la < lb && a_subset_b(),
                    CompareOp::Ge => la >= lb && b_subset_a(),
                    CompareOp::Gt => la > lb && b_subset_a(),
                }));
            }
        }
        // set / frozenset comparison — subset / superset / equality.
        // PyPy: setobject.py W_BaseSetObject.descr_eq, descr_le, descr_lt
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            let la = pyre_object::w_set_len(a);
            let lb = pyre_object::w_set_len(b);
            let a_subset_b = || {
                pyre_object::w_set_items(a)
                    .into_iter()
                    .all(|item| pyre_object::w_set_contains(b, item))
            };
            let b_subset_a = || {
                pyre_object::w_set_items(b)
                    .into_iter()
                    .all(|item| pyre_object::w_set_contains(a, item))
            };
            return Ok(w_bool_from(match op {
                CompareOp::Eq => la == lb && a_subset_b(),
                CompareOp::Ne => la != lb || !a_subset_b(),
                CompareOp::Le => la <= lb && a_subset_b(),
                CompareOp::Lt => la < lb && a_subset_b(),
                CompareOp::Ge => la >= lb && b_subset_a(),
                CompareOp::Gt => la > lb && b_subset_a(),
            }));
        }
        // List lexicographic comparison — same logic as tuple.
        if is_list(a) && is_list(b) {
            let la = pyre_object::w_list_len(a);
            let lb = pyre_object::w_list_len(b);
            let min_len = la.min(lb);
            for i in 0..min_len {
                let ea = pyre_object::w_list_getitem(a, i as i64).unwrap_or(PY_NULL);
                let eb = pyre_object::w_list_getitem(b, i as i64).unwrap_or(PY_NULL);
                let eq = match compare(ea, eb, CompareOp::Eq) {
                    Ok(r) => is_true(r),
                    Err(_) => false,
                };
                if !eq {
                    return compare(ea, eb, op);
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Lt => la < lb,
                CompareOp::Le => la <= lb,
                CompareOp::Gt => la > lb,
                CompareOp::Ge => la >= lb,
                CompareOp::Eq => la == lb,
                CompareOp::Ne => la != lb,
            }));
        }
        // Instance dunder dispatch for comparison
        let dunder = match op {
            CompareOp::Lt => "__lt__",
            CompareOp::Le => "__le__",
            CompareOp::Gt => "__gt__",
            CompareOp::Ge => "__ge__",
            CompareOp::Eq => "__eq__",
            CompareOp::Ne => "__ne__",
        };
        if let Some(result) = try_instance_binop(a, b, dunder) {
            return result;
        }
        // `dictmultiobject.py:1628-1656 SetLikeDictView` — dict_keys
        // / dict_items expose `__eq__` / `__ne__` / `__lt__` / etc.
        // through the typedef.  `try_instance_binop` only fires for
        // is_instance-shaped objects, so dict views (a separate
        // W_Root type) need an explicit MRO-driven dispatch here.
        // Reflected: if RHS is a dict view, try `b.dunder(a)` —
        // PyPy's `_is_set_like(other)` short-circuits the LHS-side
        // descr_eq when the other is set-like, so the reflected call
        // path is the one that succeeds for `set == d.keys()`.
        if let Some(a_type) = crate::typedef::r#type(a) {
            if let Some(method) = lookup_in_type_where(a_type, dunder) {
                if let Ok(result) = crate::call::call_function_impl_result(method, &[a, b]) {
                    if !is_not_implemented(result) {
                        return Ok(result);
                    }
                }
            }
        }
        if let Some(rdunder) = reverse_dunder(dunder) {
            if let Some(b_type) = crate::typedef::r#type(b) {
                if let Some(method) = lookup_in_type_where(b_type, rdunder) {
                    if let Ok(result) = crate::call::call_function_impl_result(method, &[b, a]) {
                        if !is_not_implemented(result) {
                            return Ok(result);
                        }
                    }
                }
            }
        }
        // Identity comparison fallback for == and !=
        if matches!(op, CompareOp::Eq) {
            return Ok(w_bool_from(std::ptr::eq(a, b)));
        }
        if matches!(op, CompareOp::Ne) {
            return Ok(w_bool_from(!std::ptr::eq(a, b)));
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "'{op:?}' not supported between instances of '{a_name}' and '{b_name}'"
        )))
    }
}

/// Comparison operator enum (mirrors RustPython's ComparisonOperator).
#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Unary positive (`+a`).

pub fn pos(a: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    unsafe {
        if is_int(a) || is_bool(a) {
            return Ok(w_int_new(int_value(a)));
        }
        if is_long(a) {
            return Ok(bigint_result(w_long_get_value(a).clone()));
        }
        if is_float(a) {
            return Ok(w_float_new(w_float_get_value(a)));
        }
        if let Some(result) = try_instance_unaryop(a, "__pos__") {
            return result;
        }
        if a.is_null() {
            return Err(PyError::type_error(
                "bad operand type for unary +: 'NoneType'",
            ));
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary +: '{}'",
            (*(*a).ob_type).name,
        )))
    }
}

/// Unary negation.

pub fn neg(a: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    unsafe {
        if is_int(a) || is_bool(a) {
            let v = int_value(a);
            return match v.checked_neg() {
                Some(r) => Ok(w_int_new(r)),
                None => Ok(w_long_new(bigint_neg(BigInt::from(v)))),
            };
        }
        if is_long(a) {
            return Ok(bigint_result(bigint_neg(w_long_get_value(a).clone())));
        }
        if is_float(a) {
            return Ok(w_float_new(-w_float_get_value(a)));
        }
        // Instance __neg__
        if let Some(result) = try_instance_unaryop(a, "__neg__") {
            return result;
        }
        if a.is_null() {
            return Err(PyError::type_error(
                "bad operand type for unary -: 'NoneType'",
            ));
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary -: '{}'",
            (*(*a).ob_type).name,
        )))
    }
}

/// Unary bitwise inversion.

pub fn invert(a: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    unsafe {
        if is_int(a) || is_bool(a) {
            return Ok(w_int_new(!int_value(a)));
        }
        if is_long(a) {
            return Ok(bigint_result(bigint_invert(w_long_get_value(a).clone())));
        }
        if let Some(result) = try_instance_unaryop(a, "__invert__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary ~: '{}'",
            (*(*a).ob_type).name,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_add() {
        let a = w_int_new(3);
        let b = w_int_new(4);
        let result = add(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_int_compare() {
        let a = w_int_new(5);
        let b = w_int_new(10);
        let result = compare(a, b, CompareOp::Lt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_zero_division() {
        let a = w_int_new(5);
        let b = w_int_new(0);
        assert!(floordiv(a, b).is_err());
    }

    #[test]
    fn test_truthiness() {
        assert!(is_true(w_int_new(1)));
        assert!(!is_true(w_int_new(0)));
        assert!(!is_true(w_none()));
        assert!(is_true(w_bool_from(true)));
        assert!(!is_true(w_bool_from(false)));
    }

    #[test]
    fn test_int_add_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(1);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_sub_overflow() {
        let a = w_int_new(i64::MIN);
        let b = w_int_new(1);
        let result = sub(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MIN) - BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_mul_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(2);
        let result = mul(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) * BigInt::from(2)
            );
        }
    }

    #[test]
    fn test_long_add() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(100);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(101)
            );
        }
    }

    #[test]
    fn test_long_demote_to_int() {
        // long + long that fits back in i64 → W_IntObject
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(-1);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), i64::MAX);
        }
    }

    #[test]
    fn test_negate_min_int() {
        let a = w_int_new(i64::MIN);
        let result = neg(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), -BigInt::from(i64::MIN));
        }
    }

    #[test]
    fn test_invert_int() {
        let result = invert(w_int_new(6)).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), !6);
        }
    }

    #[test]
    fn test_long_compare() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(i64::MAX);
        let result = compare(a, b, CompareOp::Gt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_long_truthiness() {
        assert!(is_true(w_long_new(
            BigInt::from(i64::MAX) + BigInt::from(1)
        )));
        assert!(!is_true(w_long_new(BigInt::from(0))));
    }

    #[test]
    fn test_int_pow() {
        let result = pow(w_int_new(2), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_pow_overflow() {
        let result = pow(w_int_new(2), w_int_new(63)).unwrap();
        unsafe {
            // 2^63 overflows i64, should be long
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(2).pow(63));
        }
    }

    #[test]
    fn test_int_pow_negative_exponent() {
        let result = pow(w_int_new(2), w_int_new(-1)).unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 0.5);
        }
    }

    #[test]
    fn test_int_lshift() {
        let result = lshift(w_int_new(1), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_lshift_overflow() {
        let result = lshift(w_int_new(1), w_int_new(64)).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(1) << 64);
        }
    }

    #[test]
    fn test_int_rshift() {
        let result = rshift(w_int_new(1024), w_int_new(3)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 128) };
    }

    #[test]
    fn test_negative_shift_count() {
        assert!(lshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(rshift(w_int_new(1), w_int_new(-1)).is_err());
    }

    #[test]
    fn test_int_bitand() {
        let result = and_(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0x0F) };
    }

    #[test]
    fn test_int_bitor() {
        let result = or_(w_int_new(0xF0), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xFF) };
    }

    #[test]
    fn test_int_bitxor() {
        let result = xor(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xF0) };
    }

    #[test]
    fn test_long_bitand() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(0xFF);
        let result = and_(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0) };
    }

    #[test]
    fn test_invert_long() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let result = invert(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                !(BigInt::from(i64::MAX) + BigInt::from(1))
            );
        }
    }
}
