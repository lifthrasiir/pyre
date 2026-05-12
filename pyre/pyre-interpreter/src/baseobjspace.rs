//! ObjSpace — Python object operation dispatch.
#![allow(non_camel_case_types, non_snake_case)]
//!
//! The ObjSpace mediates all operations on Python objects. This is the layer
//! where type-specific fast paths live, and where the JIT inserts `GuardClass`
//! to specialize operations.

// Suppress unsafe-in-unsafe-fn warnings; our unsafe fns are inherently
// working with raw pointers throughout and wrapping every call in an
// additional unsafe block adds noise without safety benefit.
#![allow(unsafe_op_in_unsafe_fn)]

use malachite_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;

use std::cell::RefCell;
use std::collections::HashMap;

use crate::function::is_function;
pub use crate::{PyError, PyErrorKind, PyResult};
use pyre_object::strobject::is_str;
use pyre_object::*;

/// Compatibility alias for PyPy's base-object type.
/// PyPy frequently models interpreter values as subclasses of `W_Root`.
pub type W_Root = PyObjectRef;

/// Compatibility marker for a type mismatch in descriptor lookup.
#[derive(Debug, Clone)]
pub struct DescrMismatch;

/// Compatibility marker for lock-sensitive APIs that are disabled under
/// this no-GIL runtime.
#[derive(Debug, Clone)]
pub struct CannotHaveLock;

/// Minimal compatibility placeholder for PyPy-style cache objects.
#[derive(Debug, Default)]
pub struct SpaceCache {
    space: PyObjectRef,
    _entries: RefCell<HashMap<usize, PyObjectRef>>,
}

impl SpaceCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            space,
            _entries: RefCell::new(HashMap::new()),
        }
    }

    #[inline]
    pub fn getorbuild(&self, _key: PyObjectRef) -> PyObjectRef {
        std::ptr::null_mut()
    }

    #[inline]
    pub fn ready(&self, _result: PyObjectRef) {}
}

/// Compatibility cache variant with `callable(self)` construction path.
#[derive(Debug, Default)]
pub struct InternalSpaceCache {
    base: SpaceCache,
}

impl InternalSpaceCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            base: SpaceCache::new(space),
        }
    }

    #[inline]
    pub fn getorbuild<F>(&self, f: F) -> PyObjectRef
    where
        F: FnOnce(PyObjectRef) -> PyObjectRef,
    {
        let _ = self.base.space;
        f(std::ptr::null_mut())
    }
}

/// Compatibility helper used by `ObjSpace` bootstrap in PyPy.
#[derive(Debug, Default)]
pub struct AppExecCache {
    base: SpaceCache,
}

impl AppExecCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            base: SpaceCache::new(space),
        }
    }

    pub fn build(&self, _source: PyObjectRef) -> PyObjectRef {
        let _ = self.base.space;
        std::ptr::null_mut()
    }
}

/// Very small compatibility object for PyPy's `ObjSpace` interface.
/// The full object-space API is implemented as free functions in this module.
#[derive(Debug, Default)]
pub struct ObjSpace {
    fromcache: Option<PyObjectRef>,
}

impl ObjSpace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fromcache<T, F>(&self, mut build: F, cache: &SpaceCache) -> T
    where
        T: Default,
        F: FnMut(&SpaceCache) -> T,
    {
        let _ = cache.getorbuild(std::ptr::null_mut());
        build(cache)
    }
}

// ── Cell unwrap ──────────────────────────────────────────────────────
// CPython 3.13 unified locals+cells means LoadFast can return cell
// objects. All operations must transparently unwrap cells.
// PyPy: each opcode implementation calls space.unwrap_cell() implicitly.

/// Unwrap a cell object to its contents. Non-cells pass through.
#[inline(always)]
pub fn unwrap_cell(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return obj;
    }
    if unsafe { is_cell(obj) } {
        let inner = unsafe { w_cell_get(obj) };
        if !inner.is_null() {
            return inner;
        }
        // Cell with null content — return cell itself (caller will handle)
        return obj;
    }
    obj
}

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
fn bigint_eq(a: BigInt, b: BigInt) -> bool {
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
        // Negative exponent → float result
        return Ok(w_float_new((va as f64).powf(vb as f64)));
    }
    let vb = vb as u32;
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
    let exp = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(as_bigint(a).pow(exp)))
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
    let shift = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(bigint_lshift(as_bigint(a), shift as usize)))
}

unsafe fn long_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_lt(vb.clone(), BigInt::from(0)) {
        return Err(PyError::value_error("negative shift count"));
    }
    let shift = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(bigint_rshift(as_bigint(a), shift as usize)))
}

// ── bool-as-int helpers ──────────────────────────────────────────────

/// True when obj is int or bool (bool is a subclass of int in Python).
#[inline]
unsafe fn is_int_like(obj: PyObjectRef) -> bool {
    is_int(obj) || is_bool(obj)
}

/// Extract i64 from an int or bool object.
#[inline]
unsafe fn int_value(obj: PyObjectRef) -> i64 {
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
        // Also check per-instance attributes (ATTR_TABLE)
        if let Ok(method) = getattr(a, dunder) {
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

/// Binary operation dispatch.
///
/// Checks types and dispatches to the appropriate fast path.
/// The JIT traces through this function, recording `GuardClass` on operand types.

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
            return str_format_percent(a, b);
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

/// `str % args` — printf-style string formatting.
/// PyPy: unicodeobject.py mod__String_ANY → formatting.py
unsafe fn str_format_percent(fmt: PyObjectRef, args: PyObjectRef) -> PyResult {
    let fmt_str = w_str_get_value(fmt);
    // `formatting.py:39-46 StringFormatter.__init__` — when `args`
    // is a tuple, `values_w` is the unpacked positional list; for
    // any other shape (mapping, single value), `values_w` stays
    // None and `checkconsumed` is skipped.  Pyre tracks the same
    // distinction via `args_is_tuple` so the trailing surplus-args
    // check at the end fires only for tuple input — without it,
    // `"%s"`-only formats against a mapping that exposes
    // `__getitem__` would always trip "not all arguments converted"
    // even when the mapping was the (single) intended value.
    let args_is_tuple = is_tuple(args);
    let arg_list: Vec<PyObjectRef> = if args_is_tuple {
        let n = w_tuple_len(args);
        (0..n)
            .filter_map(|i| w_tuple_getitem(args, i as i64))
            .collect()
    } else {
        vec![args]
    };

    let mut result = String::new();
    let mut arg_idx = 0;
    let bytes = fmt_str.as_bytes();
    let mut i = 0;
    let take_next_arg =
        |arg_idx: &mut usize, arg_list: &[PyObjectRef]| -> Result<PyObjectRef, PyError> {
            // `formatting.py:574-582 nextinputvalue` — surfaces the
            // canonical "not enough arguments" message when the
            // positional pool runs dry.
            if *arg_idx >= arg_list.len() {
                return Err(PyError::type_error(
                    "not enough arguments for format string",
                ));
            }
            let a = arg_list[*arg_idx];
            *arg_idx += 1;
            Ok(a)
        };
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 1 < bytes.len() {
            i += 1;
            // Named format: %(name)s — `formatting.py:174-195
            // getmappingkey` scans up to the *balanced* closing
            // `)` (paren-depth counting so `%(a(b)c)s` parses key
            // `a(b)c`), then `space.getitem(args, w_key)` looks up
            // the mapping value (mapping-like object, not just
            // exact dict).
            let named_arg = if i < bytes.len() && bytes[i] == b'(' {
                i += 1; // skip the opening '('
                let key_start = i;
                let mut pcount: usize = 1;
                while i < bytes.len() {
                    let c = bytes[i];
                    if c == b')' {
                        pcount -= 1;
                        if pcount == 0 {
                            break;
                        }
                    } else if c == b'(' {
                        pcount += 1;
                    }
                    i += 1;
                }
                if i >= bytes.len() {
                    // `formatting.py:184-186 incomplete format key` —
                    // ran off the end of the format string without
                    // the matching `)`.
                    return Err(PyError::new(
                        PyErrorKind::ValueError,
                        "incomplete format key".to_string(),
                    ));
                }
                let key = String::from_utf8_lossy(&bytes[key_start..i]).into_owned();
                i += 1; // skip the closing ')'
                // Fast path for exact dict: avoid building a W_StrObject
                // when we can probe the dict storage directly.
                if is_dict(args) {
                    w_dict_getitem_str(args, &key)
                } else {
                    let w_key = pyre_object::w_str_new(&key);
                    Some(getitem(args, w_key)?)
                }
            } else {
                None
            };
            // `pypy/objspace/std/formatting.py StringFormatter._parse_spec`
            // — flags (`-`, `+`, ` `, `0`, `#`), width digits, optional
            // `.precision` digits, then conversion type.  pyre handles
            // the common subset; the asterisk (`*`) form is left to a
            // future round.
            let mut left_align = false;
            let mut zero_pad = false;
            let mut explicit_sign = false;
            let mut blank_sign = false;
            let mut alt_form = false;
            while i < bytes.len() {
                match bytes[i] {
                    b'-' => left_align = true,
                    b'0' => zero_pad = true,
                    b'+' => explicit_sign = true,
                    b' ' => blank_sign = true,
                    // `formatting.py:240-242` — `#` selects the
                    // alternate form (0x/0o/0b prefix on integers).
                    b'#' => alt_form = true,
                    _ => break,
                }
                i += 1;
            }
            // `formatting.py:266-274 peel_num` — `*` reads width /
            // precision from the next positional input.
            let mut width = 0usize;
            if i < bytes.len() && bytes[i] == b'*' {
                i += 1;
                let star_arg = take_next_arg(&mut arg_idx, &arg_list)?;
                if !is_int_like(star_arg) {
                    return Err(PyError::type_error("* wants int"));
                }
                let val = int_value(star_arg);
                if val < 0 {
                    // Negative star width acts like `-` flag with abs(width).
                    left_align = true;
                    width = (-val) as usize;
                } else {
                    width = val as usize;
                }
            } else {
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    width = width * 10 + (bytes[i] - b'0') as usize;
                    i += 1;
                }
            }
            let mut precision: Option<usize> = None;
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                if i < bytes.len() && bytes[i] == b'*' {
                    i += 1;
                    let star_arg = take_next_arg(&mut arg_idx, &arg_list)?;
                    if !is_int_like(star_arg) {
                        return Err(PyError::type_error("* wants int"));
                    }
                    let v = int_value(star_arg);
                    precision = Some(if v < 0 { 0 } else { v as usize });
                } else {
                    let mut p = 0usize;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        p = p * 10 + (bytes[i] - b'0') as usize;
                        i += 1;
                    }
                    precision = Some(p);
                }
            }
            if i >= bytes.len() {
                break;
            }
            let spec = bytes[i] as char;
            i += 1;
            if spec == '%' {
                result.push('%');
                continue;
            }
            let arg = if let Some(na) = named_arg {
                na
            } else {
                take_next_arg(&mut arg_idx, &arg_list)?
            };
            // Helper closure: pad-and-emit body string respecting
            // width/left-align (zero-pad only matters for numeric paths
            // that build their own body with sign awareness).
            let pad = |body: String| -> String {
                if body.chars().count() >= width {
                    return body;
                }
                let need = width - body.chars().count();
                let mut s = String::with_capacity(width);
                if left_align {
                    s.push_str(&body);
                    for _ in 0..need {
                        s.push(' ');
                    }
                } else {
                    for _ in 0..need {
                        s.push(' ');
                    }
                    s.push_str(&body);
                }
                s
            };
            let sign_prefix = |val: i64| -> &'static str {
                if val < 0 {
                    "-"
                } else if explicit_sign {
                    "+"
                } else if blank_sign {
                    " "
                } else {
                    ""
                }
            };
            match spec {
                's' => {
                    let mut body = crate::py_str(arg);
                    if let Some(p) = precision {
                        body = body.chars().take(p).collect();
                    }
                    result.push_str(&pad(body));
                }
                'r' => {
                    let mut body = crate::py_repr(arg);
                    if let Some(p) = precision {
                        body = body.chars().take(p).collect();
                    }
                    result.push_str(&pad(body));
                }
                // `formatting.py fmt_a` (CPython `%a`) — repr() force-
                // ASCII-encoded.  Pyre's `py_repr` already produces
                // ASCII-clean output for the types it covers, so the
                // result matches PyPy `fmt_a` for the supported subset.
                'a' => {
                    let mut body = crate::py_repr(arg);
                    if let Some(p) = precision {
                        body = body.chars().take(p).collect();
                    }
                    result.push_str(&pad(body));
                }
                // `formatting.py:549-552 fmt_b` — `%b` is a *bytes-only*
                // conversion (formats a `bytes`/`bytearray`/`__bytes__`
                // object); for the unicode formatter PyPy calls
                // `self.unknown_fmtchar()` which raises ValueError
                // "unsupported format character 'b' (0x62) at index N".
                'b' => {
                    return Err(PyError::new(
                        PyErrorKind::ValueError,
                        format!("unsupported format character 'b' (0x62) at index {}", i - 1),
                    ));
                }
                // `formatting.py fmt_u` is documented as a deprecated
                // alias for `%d` and dispatches to the same body.
                'd' | 'i' | 'u' | 'x' | 'X' | 'o' => {
                    // `formatting.py:296-302 fmt_d / fmt_x ...` raises
                    // TypeError when the argument is not int-like
                    // (`%X format: an integer is required, not <type>`),
                    // not silently coerce via str().
                    if !is_int_like(arg) {
                        return Err(PyError::type_error(format!(
                            "%{spec} format: an integer is required, not {}",
                            object_functionstr_type_name(arg),
                        )));
                    }
                    let val = int_value(arg);
                    let abs = val.unsigned_abs();
                    let digits = match spec {
                        'x' => format!("{abs:x}"),
                        'X' => format!("{abs:X}"),
                        'o' => format!("{abs:o}"),
                        _ => format!("{abs}"),
                    };
                    // `formatting.py:240-242` — alt-form adds the
                    // base prefix (`0x`, `0X`, `0o`) for the matching
                    // radix; `%d/%i/%u` ignore it.  `%b` is not a
                    // unicode-formatter conversion (rejected above
                    // per `:549-552 fmt_b`).
                    let prefix = if alt_form {
                        match spec {
                            'x' => "0x",
                            'X' => "0X",
                            'o' => "0o",
                            _ => "",
                        }
                    } else {
                        ""
                    };
                    let sign = sign_prefix(val);
                    // Sign-aware zero pad (`%05d`, `%-5d` parity):
                    // `formatting.py:_pad_string` reserves width for
                    // sign + alt-prefix before padding zeros.
                    // Left-align disables zero pad.
                    if zero_pad && !left_align {
                        let need = width.saturating_sub(sign.len() + prefix.len() + digits.len());
                        let mut s = String::with_capacity(width);
                        s.push_str(sign);
                        s.push_str(prefix);
                        for _ in 0..need {
                            s.push('0');
                        }
                        s.push_str(&digits);
                        result.push_str(&s);
                    } else {
                        result.push_str(&pad(format!("{sign}{prefix}{digits}")));
                    }
                }
                // `formatting.py fmt_e / fmt_E / fmt_g / fmt_G` —
                // scientific / general float formatting.  Default
                // precision is 6 (CPython %e/%g), `%g` strips
                // trailing zeros and chooses scientific vs fixed
                // based on exponent magnitude (PyPy delegates to
                // `rfloat.formatd` with the matching format char).
                'e' | 'E' | 'g' | 'G' => {
                    // `formatting.py:303-308 fmt_e / fmt_g ...` raises
                    // TypeError when the argument is not numeric;
                    // silently substituting 0.0 for non-numeric input
                    // hid type bugs at the format site.
                    let val = if is_float(arg) {
                        pyre_object::floatobject::w_float_get_value(arg)
                    } else if is_int_like(arg) {
                        int_value(arg) as f64
                    } else {
                        return Err(PyError::type_error(format!(
                            "must be real number, not {}",
                            object_functionstr_type_name(arg),
                        )));
                    };
                    let prec = precision.unwrap_or(6);
                    let abs = val.abs();
                    let body = match spec {
                        'e' => normalise_exponent(&format!("{:.*e}", prec, abs), false),
                        'E' => normalise_exponent(&format!("{:.*E}", prec, abs), true),
                        // `%g` precision counts significant digits;
                        // Rust has no built-in formatter that exactly
                        // matches CPython's `%g` rules, so emit
                        // scientific when |v| >= 10^prec or < 1e-4
                        // (matching the CPython threshold), else
                        // fixed with (prec - 1 - exponent) decimals
                        // and trailing zero stripping.
                        'g' | 'G' => format_g_like(abs, prec, spec == 'G', alt_form),
                        _ => unreachable!(),
                    };
                    let sign = if val.is_sign_negative() && !val.is_nan() {
                        "-"
                    } else if explicit_sign {
                        "+"
                    } else if blank_sign {
                        " "
                    } else {
                        ""
                    };
                    if zero_pad && !left_align {
                        let need = width.saturating_sub(sign.len() + body.len());
                        let mut s = String::with_capacity(width);
                        s.push_str(sign);
                        for _ in 0..need {
                            s.push('0');
                        }
                        s.push_str(&body);
                        result.push_str(&s);
                    } else {
                        result.push_str(&pad(format!("{sign}{body}")));
                    }
                }
                'f' | 'F' => {
                    // `formatting.py:303-308 fmt_f` — same TypeError as
                    // `%e/%g` for non-numeric arguments.
                    let val = if is_float(arg) {
                        pyre_object::floatobject::w_float_get_value(arg)
                    } else if is_int_like(arg) {
                        int_value(arg) as f64
                    } else {
                        return Err(PyError::type_error(format!(
                            "must be real number, not {}",
                            object_functionstr_type_name(arg),
                        )));
                    };
                    let prec = precision.unwrap_or(6);
                    let abs_body = format!("{:.*}", prec, val.abs());
                    let sign = if val.is_sign_negative() && !val.is_nan() {
                        "-"
                    } else if explicit_sign {
                        "+"
                    } else if blank_sign {
                        " "
                    } else {
                        ""
                    };
                    if zero_pad && !left_align {
                        let need = width.saturating_sub(sign.len() + abs_body.len());
                        let mut s = String::with_capacity(width);
                        s.push_str(sign);
                        for _ in 0..need {
                            s.push('0');
                        }
                        s.push_str(&abs_body);
                        result.push_str(&s);
                    } else {
                        result.push_str(&pad(format!("{sign}{abs_body}")));
                    }
                }
                'c' => {
                    // `formatting.py:283-294 fmt_c` parity:
                    //   - int branch: `if value < 0 or value > 0x10FFFF:
                    //     OverflowError("%c arg not in range(0x110000)")`
                    //   - str branch: `if len(s) != 1: TypeError(
                    //     "%c requires int or single character")`
                    //   - other types: same TypeError.
                    if is_int_like(arg) {
                        let v = int_value(arg);
                        if v < 0 || v > 0x10FFFF {
                            return Err(PyError::new(
                                PyErrorKind::OverflowError,
                                "%c arg not in range(0x110000)".to_string(),
                            ));
                        }
                        let c = char::from_u32(v as u32).ok_or_else(|| {
                            PyError::new(
                                PyErrorKind::OverflowError,
                                "%c arg not in range(0x110000)".to_string(),
                            )
                        })?;
                        result.push_str(&pad(c.to_string()));
                    } else if is_str(arg) {
                        let s = w_str_get_value(arg);
                        if s.chars().count() != 1 {
                            return Err(PyError::type_error("%c requires int or single character"));
                        }
                        result.push_str(&pad(s.to_string()));
                    } else {
                        return Err(PyError::type_error(format!(
                            "%c requires int or char, not {}",
                            object_functionstr_type_name(arg),
                        )));
                    }
                }
                // `formatting.py:328-329 unknown_fmtchar` raises
                // ValueError on any spec char outside FORMATTER_CHARS.
                _ => {
                    return Err(PyError::value_error(format!(
                        "unsupported format character '{spec}' (0x{:x}) at index {}",
                        spec as u32,
                        i.saturating_sub(1)
                    )));
                }
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    // `formatting.py:572-580 checkconsumed` — surplus positional
    // arguments are an error only when the input came as a tuple
    // (PyPy's `values_w` invariant).  Mapping inputs (dict and any
    // other shape) skip this check because every replacement field
    // pulls from `args` by name, not by sequential index.
    if args_is_tuple && arg_idx < arg_list.len() {
        return Err(PyError::type_error(
            "not all arguments converted during string formatting",
        ));
    }
    Ok(w_str_new(&result))
}

/// `formatting.py fmt_g / fmt_G` parity helper — emits the `%g` body
/// for an *absolute* float value.  CPython's `%g` rules:
///   * precision ≤ 0 is treated as 1 (PyPy: `prec = max(prec, 1)`)
///   * choose scientific when `exp < -4` or `exp >= prec`
///     (`pypy/objspace/std/formatting.py:120 format_e_g_complex`)
///   * scientific body uses (`prec - 1`) decimals
///   * fixed body uses `(prec - 1 - exp)` decimals
///   * trailing zeros (and the trailing '.') are stripped unless
///     `#` (alt-form) was set; with alt-form the trailing zeros AND
///     the dangling '.' survive so the output advertises the full
///     requested precision.
pub(crate) fn format_g_like(abs: f64, prec: usize, upper: bool, alt_form: bool) -> String {
    if abs == 0.0 {
        return if alt_form {
            // `formatting.py:120-128` — alt-form `%#g` keeps `prec`
            // significant digits even for 0.0, so `"%#.4g" % 0`
            // renders as `"0.000"` (one digit before the dot, then
            // `prec - 1` zeros after).
            let after = prec.saturating_sub(1);
            if after == 0 {
                "0".to_string()
            } else {
                format!("0.{}", "0".repeat(after))
            }
        } else {
            "0".to_string()
        };
    }
    if !abs.is_finite() {
        return format!("{abs}");
    }
    let prec = prec.max(1);
    let exp = abs.log10().floor() as i32;
    let use_sci = exp < -4 || exp >= prec as i32;
    let raw = if use_sci {
        let rust_exp = format!("{:.*e}", prec - 1, abs);
        normalise_exponent(&rust_exp, upper)
    } else {
        let dec = (prec as i32 - 1 - exp).max(0) as usize;
        format!("{abs:.dec$}")
    };
    if alt_form {
        // Alt-form preserves trailing zeros and the dangling '.';
        // ensure a '.' is present even when the natural body has
        // none (e.g. `"%#g" % 1` → `"1.00000"`).
        if use_sci {
            return raw;
        }
        return if raw.contains('.') {
            raw
        } else {
            // `prec - 1 - exp` was 0, so the body has no decimals;
            // re-render with the alt-form '.' suffix + trailing
            // zeros to advertise the full precision.
            let after = (prec as i32 - 1 - exp).max(0) as usize;
            if after == 0 {
                format!("{raw}.")
            } else {
                format!("{raw}.{}", "0".repeat(after))
            }
        };
    }
    // Strip trailing zeros from the mantissa (and a dangling '.').
    if use_sci {
        if let Some(epos) = raw.find(|c: char| c == 'e' || c == 'E') {
            let (mantissa, exp_part) = raw.split_at(epos);
            let trimmed = trim_trailing_zeros(mantissa);
            format!("{trimmed}{exp_part}")
        } else {
            raw
        }
    } else if raw.contains('.') {
        trim_trailing_zeros(&raw)
    } else {
        raw
    }
}

/// Convert Rust's `{:e}` / `{:E}` exponent encoding (no sign, minimal
/// width — `1.5e3` / `1.5e-3`) to PyPy / CPython `%e`-style
/// (`1.5e+03` / `1.5e-03` — explicit sign, exponent zero-padded to at
/// least two digits).  Mirrors `pypy/objspace/std/formatting.py
/// fmt_e` which delegates to `rfloat.formatd` with the C printf-style
/// padding rules.
pub(crate) fn normalise_exponent(raw: &str, upper: bool) -> String {
    let marker = if upper { 'E' } else { 'e' };
    let lower_marker = marker.to_ascii_lowercase();
    let upper_marker = marker.to_ascii_uppercase();
    let pos = match raw.find([lower_marker, upper_marker].as_ref()) {
        Some(p) => p,
        None => return raw.to_string(),
    };
    let (mantissa, exp_part) = raw.split_at(pos);
    let exp_str = &exp_part[1..]; // skip the marker
    let (sign_char, digits) = if let Some(rest) = exp_str.strip_prefix('-') {
        ('-', rest)
    } else if let Some(rest) = exp_str.strip_prefix('+') {
        ('+', rest)
    } else {
        ('+', exp_str)
    };
    let padded = if digits.len() < 2 {
        format!("0{digits}")
    } else {
        digits.to_string()
    };
    format!("{mantissa}{marker}{sign_char}{padded}")
}

fn trim_trailing_zeros(body: &str) -> String {
    if !body.contains('.') {
        return body.to_string();
    }
    let trimmed = body.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
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
// `_make_binop_impl` / `_make_descr_unaryop`. They live here in
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

/// pypy/interpreter/baseobjspace.py `issubtype_w` — `cls` is in
/// `w_type.mro_w`. Uses the cached MRO when present, otherwise
/// recomputes via `compute_default_mro`.
pub(crate) unsafe fn issubtype_w(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    if w_type.is_null() {
        return false;
    }
    // PyPy's issubtype_w is only valid for type objects.  Use the same
    // object-space test as abstractinst.py (`space.isinstance_w(...,
    // space.w_type)`) instead of peeking at Rust layout internals.
    if !is_type_like_w(w_type) {
        return false;
    }
    let mro_ptr = w_type_get_mro(w_type);
    if !mro_ptr.is_null() {
        return (*mro_ptr).iter().any(|&t| std::ptr::eq(t, cls));
    }
    compute_default_mro(w_type)
        .iter()
        .any(|&t| std::ptr::eq(t, cls))
}

/// pypy/interpreter/baseobjspace.py:1359 `exception_is_valid_obj_as_class_w`.
///
///   def exception_is_valid_obj_as_class_w(self, w_obj):
///       if not self.isinstance_w(w_obj, self.w_type):
///           return False
///       return self.issubtype_w(w_obj, self.w_BaseException)
///
/// Canonical `BaseException` comes from the EXC_CLASS_REGISTRY populated at
/// `make_exc_type` time — not from the mutable builtins dict — so a user
/// rebinding `builtins.BaseException` cannot redirect the gate.
pub unsafe fn exception_is_valid_obj_as_class_w(w_obj: PyObjectRef) -> bool {
    if !is_type_like_w(w_obj) {
        return false;
    }
    let Some(base_exc) = crate::builtins::lookup_exc_class("BaseException") else {
        return false;
    };
    issubtype_w(w_obj, base_exc)
}

/// pypy/interpreter/baseobjspace.py:1364-1365 `exception_is_valid_class_w`.
///
///   def exception_is_valid_class_w(self, w_cls):
///       return self.issubtype_w(w_cls, self.w_BaseException)
///
/// Like `exception_is_valid_obj_as_class_w` but skips the
/// `isinstance_w(w_cls, w_type)` precheck — the caller already knows
/// `w_cls` is a class object.
pub unsafe fn exception_is_valid_class_w(w_cls: PyObjectRef) -> bool {
    let Some(base_exc) = crate::builtins::lookup_exc_class("BaseException") else {
        return false;
    };
    issubtype_w(w_cls, base_exc)
}

/// pypy/interpreter/baseobjspace.py:1367-1368 `exception_getclass`.
///
///   def exception_getclass(self, w_obj):
///       return self.type(w_obj)
pub fn exception_getclass(w_obj: PyObjectRef) -> PyObjectRef {
    crate::typedef::r#type(w_obj).unwrap_or(pyre_object::PY_NULL)
}

/// pypy/interpreter/baseobjspace.py:1370-1371 `exception_issubclass_w`.
///
///   def exception_issubclass_w(self, w_cls1, w_cls2):
///       return self.issubtype_w(w_cls1, w_cls2)
pub unsafe fn exception_issubclass_w(w_cls1: PyObjectRef, w_cls2: PyObjectRef) -> bool {
    unsafe { issubtype_w(w_cls1, w_cls2) }
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

/// abstractinst.py:18-31 `_get_bases(space, w_cls)`.
/// Returns `Some(bases_tuple)` when `getattr(w_cls, "__bases__")` exists
/// and is a tuple, `None` when the attribute is missing or not a tuple.
/// AttributeError is swallowed; other errors propagate.
fn _get_bases(w_cls: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    let w_bases = match getattr(w_cls, "__bases__") {
        Ok(b) => b,
        Err(e) if e.kind == PyErrorKind::AttributeError => return Ok(None),
        Err(e) => return Err(e),
    };
    if w_bases.is_null() {
        return Ok(None);
    }
    if unsafe { is_tuple(w_bases) } {
        Ok(Some(w_bases))
    } else {
        Ok(None)
    }
}

/// abstractinst.py:33-34 `abstract_isclass_w(space, w_obj)`.
fn abstract_isclass_w(w_obj: PyObjectRef) -> Result<bool, PyError> {
    Ok(_get_bases(w_obj)?.is_some())
}

/// abstractinst.py:36-38 `check_class(space, w_obj, msg)`. Raises
/// `TypeError(msg)` when `w_obj` lacks a tuple-valued `__bases__`.
fn check_class(w_obj: PyObjectRef, msg: &str) -> Result<(), PyError> {
    if !abstract_isclass_w(w_obj)? {
        return Err(PyError::type_error(msg.to_string()));
    }
    Ok(())
}

/// abstractinst.py:74-88 `p_recursive_isinstance_type_w`. Assumes
/// `w_type` is a real type object: tries the MRO walk via `isinstance_w`
/// first, then consults `w_inst.__class__` to honour any custom class
/// override.
unsafe fn p_recursive_isinstance_type_w(
    w_inst: PyObjectRef,
    w_type: PyObjectRef,
) -> Result<bool, PyError> {
    if isinstance_w(w_inst, w_type) {
        return Ok(true);
    }
    let w_abstractclass = match getattr(w_inst, "__class__") {
        Ok(cls) => cls,
        Err(e) if e.kind == PyErrorKind::AttributeError => return Ok(false),
        Err(e) => return Err(e),
    };
    let w_inst_type = crate::typedef::r#type(w_inst).unwrap_or(pyre_object::PY_NULL);
    if !std::ptr::eq(w_abstractclass, w_inst_type) && is_type_like_w(w_abstractclass) {
        return Ok(issubtype_w(w_abstractclass, w_type));
    }
    Ok(false)
}

/// abstractinst.py:53-72 `p_recursive_isinstance_w`. The Py3 port drops
/// the `W_ClassObject`/`W_InstanceObject` Py2 fast path. Validates
/// `w_cls` via `check_class()` before falling back to the abstract
/// `__class__` / `__bases__` walk.
unsafe fn p_recursive_isinstance_w(
    w_inst: PyObjectRef,
    w_cls: PyObjectRef,
) -> Result<bool, PyError> {
    if is_type_like_w(w_cls) {
        return p_recursive_isinstance_type_w(w_inst, w_cls);
    }
    check_class(
        w_cls,
        "isinstance() arg 2 must be a type, a tuple of types, or a union",
    )?;
    let w_abstractclass = match getattr(w_inst, "__class__") {
        Ok(cls) => cls,
        Err(e) if e.kind == PyErrorKind::AttributeError => return Ok(false),
        Err(e) => return Err(e),
    };
    p_abstract_issubclass_w(w_abstractclass, w_cls)
}

/// abstractinst.py:53-56 / 154-156:
/// `space.isinstance_w(obj, space.w_type)`.
///
/// This deliberately goes through pyre's object-space `isinstance_w`,
/// which consults the Python-level class (`w_class` / W_TypeObject MRO).
/// Do not replace it with `pyre_object::is_type_or_subtype()`: that helper
/// inspects the static Rust `PyType` tag and is not the RPython data path.
unsafe fn is_type_like_w(obj: PyObjectRef) -> bool {
    let w_type = crate::typedef::w_type();
    !w_type.is_null() && isinstance_w(obj, w_type)
}

/// abstractinst.py:127-147 `p_abstract_issubclass_w`. Walks
/// `w_derived.__bases__` looking for an identity match with `w_cls`.
/// Recursion is bounded by avoiding the last entry of each `__bases__`
/// tuple — that one is followed by re-entering the loop.
fn p_abstract_issubclass_w(w_derived: PyObjectRef, w_cls: PyObjectRef) -> Result<bool, PyError> {
    let mut w_derived = w_derived;
    loop {
        if is_w(w_derived, w_cls) {
            return Ok(true);
        }
        let w_bases = match _get_bases(w_derived)? {
            Some(b) => b,
            None => return Ok(false),
        };
        let n = unsafe { w_tuple_len(w_bases) };
        if n == 0 {
            return Ok(false);
        }
        let last_index = n - 1;
        for i in 0..last_index {
            let base = match unsafe { w_tuple_getitem(w_bases, i as i64) } {
                Some(b) => b,
                None => return Ok(false),
            };
            if p_abstract_issubclass_w(base, w_cls)? {
                return Ok(true);
            }
        }
        w_derived = match unsafe { w_tuple_getitem(w_bases, last_index as i64) } {
            Some(b) => b,
            None => return Ok(false),
        };
    }
}

/// abstractinst.py:150-169 `p_recursive_issubclass_w`. The both-types
/// fast path is the common case; otherwise both arguments are validated
/// via `check_class()` before entering the abstract walk.
unsafe fn p_recursive_issubclass_w(
    w_derived: PyObjectRef,
    w_cls: PyObjectRef,
) -> Result<bool, PyError> {
    if is_type_like_w(w_cls) && is_type_like_w(w_derived) {
        return Ok(issubtype_w(w_derived, w_cls));
    }
    check_class(w_derived, "issubclass() arg 1 must be a class")?;
    check_class(
        w_cls,
        "issubclass() arg 2 must be a class or tuple of classes",
    )?;
    p_abstract_issubclass_w(w_derived, w_cls)
}

/// pypy/module/__builtin__/abstractinst.py:91-122
/// `abstract_isinstance_w(space, w_obj, w_klass_or_tuple, allow_override=True)`.
/// Handles tuple/union recursion, the `__instancecheck__` override
/// looked up via `space.lookup(w_klass_or_tuple, "__instancecheck__")`,
/// then the abstract `__class__`/`__bases__` walk.
pub fn isinstance(obj: PyObjectRef, classinfo: PyObjectRef) -> Result<bool, PyError> {
    let obj = unwrap_cell(obj);
    let classinfo = unwrap_cell(classinfo);
    unsafe {
        // abstractinst.py:104-106 — quick exact-type test.
        if let Some(t) = crate::typedef::r#type(obj) {
            if std::ptr::eq(t, classinfo) {
                return Ok(true);
            }
        }
        // abstractinst.py:108-114 — tuple recursion.
        if is_tuple(classinfo) {
            let n = w_tuple_len(classinfo);
            for i in 0..n {
                if let Some(c) = w_tuple_getitem(classinfo, i as i64) {
                    if isinstance(obj, c)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        // PEP 604 `X | Y` union recursion — pypy/objspace/std/union.py.
        if pyre_object::is_union(classinfo) {
            let union_args = pyre_object::w_union_get_args(classinfo);
            let n = w_tuple_len(union_args);
            for i in 0..n {
                if let Some(c) = w_tuple_getitem(union_args, i as i64) {
                    if isinstance(obj, c)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        // abstractinst.py:117-124 — `__instancecheck__` override
        // (`allow_override=True`). PyPy uses
        // `space.lookup(w_klass_or_tuple, "__instancecheck__")`, which
        // is a metaclass-MRO lookup (`lookup_in_type(type(cls), …)`),
        // not `getattr(cls, …)`. The distinction matters for the
        // weakproxy proxy_typedef_dict row: pyre's `getattr` runs the
        // `force()` fast path at entry and would dereference the proxy
        // before the typedef row ever gets a chance to fire. Going
        // through `lookup_in_type` on `type(classinfo)` keeps the
        // proxy's typedef wrapper installed via `proxy_typedef_dict`
        // visible. For real type objects pyre's `type` does not yet
        // install an `__instancecheck__` slot, so this falls through
        // to `p_recursive_isinstance_w` below — semantics-equivalent to
        // PyPy's `type.__instancecheck__` slot calling back into
        // `p_recursive_isinstance_type_w`.
        if let Some(cls_type) = crate::typedef::r#type(classinfo) {
            if let Some(check) = lookup_in_type(cls_type, "__instancecheck__") {
                let result = crate::call::call_function_impl_result(check, &[classinfo, obj])?;
                return Ok(is_true(result));
            }
        }
        p_recursive_isinstance_w(obj, classinfo)
    }
}

/// pypy/module/__builtin__/abstractinst.py:169-198
/// `abstract_issubclass_w(space, w_derived, w_klass_or_tuple, allow_override=True)`.
/// Tuple/union recursion, `__subclasscheck__` override looked up on
/// `type(classinfo)`, then the abstract `__bases__` walk.
pub fn issubclass(derived: PyObjectRef, classinfo: PyObjectRef) -> Result<bool, PyError> {
    let derived = unwrap_cell(derived);
    let classinfo = unwrap_cell(classinfo);
    unsafe {
        // abstractinst.py:181-187 — tuple recursion.
        if is_tuple(classinfo) {
            let n = w_tuple_len(classinfo);
            for i in 0..n {
                if let Some(c) = w_tuple_getitem(classinfo, i as i64) {
                    if issubclass(derived, c)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if pyre_object::is_union(classinfo) {
            let union_args = pyre_object::w_union_get_args(classinfo);
            let n = w_tuple_len(union_args);
            for i in 0..n {
                if let Some(c) = w_tuple_getitem(union_args, i as i64) {
                    if issubclass(derived, c)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        // abstractinst.py:190-196 — `__subclasscheck__` override.
        // Same `lookup_in_type(type(classinfo), …)` rationale as
        // `isinstance` above.
        if let Some(cls_type) = crate::typedef::r#type(classinfo) {
            if let Some(check) = lookup_in_type(cls_type, "__subclasscheck__") {
                let result = crate::call::call_function_impl_result(check, &[classinfo, derived])?;
                return Ok(is_true(result));
            }
        }
        p_recursive_issubclass_w(derived, classinfo)
    }
}

/// floatobject.py:799-881: `_pow(space, x, y)` parity — raw float power
/// with Python-correct semantics. Handles NaN/Inf edge cases and raises
/// ZeroDivisionError / ValueError / OverflowError for domain errors.
///
/// Returns raw `f64` (matching RPython which returns `r_float`) so the
/// JIT fast path can call this directly with raw-float arguments without
/// any intermediate W_FloatObject allocation. The interpreter wrapper
/// `float_pow_impl` boxes the result into a W_FloatObject.
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
                if !crate::is_function(method) {
                    if let Ok(result) = crate::call::call_function_impl_result(method, &[a, b]) {
                        if !is_not_implemented(result) {
                            return Ok(result);
                        }
                    }
                }
                if crate::is_function(method) {
                    if let Ok(result) = crate::call::call_function_impl_result(method, &[a, b]) {
                        if !is_not_implemented(result) {
                            return Ok(result);
                        }
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

/// Test if an object is truthy (for branch conditions).
///
/// Python truthiness rules:
/// - None → false
/// - bool → its value
/// - int → nonzero

pub fn is_true(obj: PyObjectRef) -> bool {
    let obj = unwrap_cell(obj);
    unsafe {
        if is_bool(obj) {
            return w_bool_get_value(obj);
        }
        if is_int(obj) {
            return w_int_get_value(obj) != 0;
        }
        if is_long(obj) {
            return !bigint_eq(w_long_get_value(obj).clone(), BigInt::from(0));
        }
        if is_float(obj) {
            return w_float_get_value(obj) != 0.0;
        }
        if is_str(obj) {
            return w_str_len(obj) != 0;
        }
        if is_list(obj) {
            return w_list_len(obj) > 0;
        }
        if is_tuple(obj) {
            return w_tuple_len(obj) > 0;
        }
        if is_dict(obj) {
            return w_dict_len(obj) > 0;
        }
        if pyre_object::is_set_or_frozenset(obj) {
            return pyre_object::w_set_len(obj) > 0;
        }
        if is_none(obj) {
            return false;
        }
        // Instance __bool__ / __len__ — PyPy: descroperation.py is_true
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            // Try __bool__ first (type MRO)
            if let Some(method) = lookup_in_type_where(w_type, "__bool__") {
                let result = crate::call_function(method, &[obj]);
                if !result.is_null() {
                    if is_bool(result) {
                        return w_bool_get_value(result);
                    }
                    if is_int(result) {
                        return w_int_get_value(result) != 0;
                    }
                    return true; // non-null → truthy fallback
                }
            }
            // Then __len__ (type MRO) — nonzero length = truthy
            if let Some(method) = lookup_in_type_where(w_type, "__len__") {
                let result = crate::call_function(method, &[obj]);
                if !result.is_null() && is_int(result) {
                    return w_int_get_value(result) != 0;
                }
            }
            // Also check per-instance __len__ (ATTR_TABLE)
            if let Ok(method) = getattr(obj, "__len__") {
                let result = crate::call_function(method, &[obj]);
                if !result.is_null() && is_int(result) {
                    return w_int_get_value(result) != 0;
                }
            }
        }
        true // default: objects are truthy
    }
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

// ── Subscript operations ─────────────────────────────────────────────

/// Normalize a slice to (start, stop, step) for a sequence of `length`.
///
/// PyPy: sliceobject.py descr_indices, mirroring CPython
/// `PySlice_Unpack` + `PySlice_AdjustIndices`. Handles negative `step`
/// (which CPython adjusts the start/stop bounds for separately from
/// positive `step`).
pub(crate) unsafe fn normalize_slice(
    index: PyObjectRef,
    length: i64,
) -> Result<(i64, i64, i64), PyError> {
    let start_obj = w_slice_get_start(index);
    let stop_obj = w_slice_get_stop(index);
    let step_obj = w_slice_get_step(index);
    let step = if is_none(step_obj) {
        1
    } else {
        w_int_get_value(step_obj)
    };
    if step == 0 {
        return Err(PyError::new(
            PyErrorKind::ValueError,
            "slice step cannot be zero",
        ));
    }
    let (lower, upper) = if step > 0 {
        (0, length)
    } else {
        (-1, length - 1)
    };
    let start = if is_none(start_obj) {
        if step > 0 { 0 } else { length - 1 }
    } else {
        let v = w_int_get_value(start_obj);
        let v = if v < 0 { v + length } else { v };
        v.max(lower).min(upper)
    };
    let stop = if is_none(stop_obj) {
        if step > 0 { length } else { -1 }
    } else {
        let v = w_int_get_value(stop_obj);
        let v = if v < 0 { v + length } else { v };
        v.max(lower).min(upper)
    };
    Ok((start, stop, step))
}

/// Get item by index: `obj[index]`.
///
/// Dispatches based on the type of `obj`.

pub fn getitem(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    // `pypy/objspace/std/dictproxyobject.py:35 descr_getitem` →
    // `space.getitem(self.w_mapping, w_key)` — forward through the
    // proxy to its wrapped mapping.  The unwrap happens at the
    // entrance so all downstream dict arms (and dict-subclass
    // overrides via the wrapped W_DictObject) see the underlying
    // mapping unchanged.
    let obj = unsafe {
        if pyre_object::is_dict_proxy(obj) {
            pyre_object::w_dict_proxy_get_mapping(obj)
        } else {
            obj
        }
    };
    unsafe {
        if is_list(obj) {
            if is_slice(index) {
                let len = w_list_len(obj) as i64;
                let (start, stop, step) = normalize_slice(index, len)?;
                let mut items = Vec::new();
                let mut i = start;
                while (step > 0 && i < stop) || (step < 0 && i > stop) {
                    if let Some(v) = w_list_getitem(obj, i) {
                        items.push(v);
                    }
                    i += step;
                }
                return Ok(w_list_new(items));
            }
            if !is_int(index) {
                return Err(PyError::type_error(
                    "list indices must be integers or slices",
                ));
            }
            let idx = w_int_get_value(index);
            match w_list_getitem(obj, idx) {
                Some(val) => Ok(val),
                None => Err(PyError::new(
                    PyErrorKind::IndexError,
                    "list index out of range",
                )),
            }
        } else if is_tuple(obj) {
            if is_slice(index) {
                // PyPy: tupleobject.py descr_getslice → slice.indices.
                let len = w_tuple_len(obj) as i64;
                let (start, stop, step) = normalize_slice(index, len)?;
                let mut items = Vec::new();
                let mut i = start;
                while (step > 0 && i < stop) || (step < 0 && i > stop) {
                    if let Some(v) = w_tuple_getitem(obj, i) {
                        items.push(v);
                    }
                    i += step;
                }
                return Ok(w_tuple_new(items));
            }
            if !is_int(index) {
                return Err(PyError::type_error("tuple indices must be integers"));
            }
            let idx = w_int_get_value(index);
            match w_tuple_getitem(obj, idx) {
                Some(val) => Ok(val),
                None => Err(PyError::new(
                    PyErrorKind::IndexError,
                    "tuple index out of range",
                )),
            }
        } else if is_dict(obj) {
            match w_dict_lookup(obj, index) {
                Some(val) => Ok(val),
                None => {
                    let key_repr = if is_str(index) {
                        format!("'{}'", w_str_get_value(index))
                    } else if is_int(index) {
                        format!("{}", w_int_get_value(index))
                    } else if !index.is_null() {
                        crate::py_repr(index)
                    } else {
                        "<null>".to_string()
                    };
                    Err(PyError::new(PyErrorKind::KeyError, key_repr))
                }
            }
        } else if is_str(obj) {
            let s = w_str_get_value(obj);
            if is_slice(index) {
                // `pypy/objspace/std/unicodeobject.py W_UnicodeObject._getitem_slice`
                // → `slice.indices(len)` (`pypy/objspace/std/sliceobject.py`).
                // Reuse the shared `normalize_slice` helper so negative-step
                // defaults (`s[::-1]`, `s[5::-1]`) match list/tuple semantics.
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len() as i64;
                let (start, stop, step) = normalize_slice(index, len)?;
                let mut result = String::new();
                let mut i = start;
                while (step > 0 && i < stop) || (step < 0 && i > stop) {
                    if i >= 0 && (i as usize) < chars.len() {
                        result.push(chars[i as usize]);
                    }
                    i += step;
                }
                Ok(w_str_new(&result))
            } else if is_int(index) {
                let idx = w_int_get_value(index);
                let chars: Vec<char> = s.chars().collect();
                let actual_idx = if idx < 0 {
                    chars.len() as i64 + idx
                } else {
                    idx
                } as usize;
                if actual_idx < chars.len() {
                    Ok(w_str_new(&chars[actual_idx].to_string()))
                } else {
                    Err(PyError::new(
                        PyErrorKind::IndexError,
                        "string index out of range",
                    ))
                }
            } else {
                Err(PyError::type_error("string indices must be integers"))
            }
        } else if pyre_object::bytesobject::is_bytes_like(obj) {
            let is_bytes = pyre_object::bytesobject::is_bytes(obj);
            if is_int(index) {
                let idx = w_int_get_value(index);
                let len = pyre_object::bytesobject::bytes_like_len(obj) as i64;
                let actual = if idx < 0 { len + idx } else { idx };
                if actual >= 0 && actual < len {
                    return Ok(w_int_new(pyre_object::bytesobject::bytes_like_getitem(
                        obj,
                        actual as usize,
                    ) as i64));
                }
                let name = if is_bytes { "bytes" } else { "bytearray" };
                return Err(PyError::new(
                    PyErrorKind::IndexError,
                    format!("{name} index out of range"),
                ));
            }
            if is_slice(index) {
                let len = pyre_object::bytesobject::bytes_like_len(obj) as i64;
                let start = w_slice_get_start(index);
                let stop = w_slice_get_stop(index);
                let step = w_slice_get_step(index);
                let step_val = if is_none(step) {
                    1
                } else {
                    w_int_get_value(step)
                };
                let s_val = if is_none(start) {
                    if step_val < 0 { len - 1 } else { 0 }
                } else {
                    let v = w_int_get_value(start);
                    if v < 0 { (len + v).max(0) } else { v.min(len) }
                };
                let e_val = if is_none(stop) {
                    if step_val < 0 { -1 } else { len }
                } else {
                    let v = w_int_get_value(stop);
                    if v < 0 { (len + v).max(-1) } else { v.min(len) }
                };
                let mut result = Vec::new();
                let mut i = s_val;
                if step_val > 0 {
                    while i < e_val {
                        if i >= 0 && i < len {
                            result.push(pyre_object::bytesobject::bytes_like_getitem(
                                obj, i as usize,
                            ));
                        }
                        i += step_val;
                    }
                } else if step_val < 0 {
                    while i > e_val {
                        if i >= 0 && i < len {
                            result.push(pyre_object::bytesobject::bytes_like_getitem(
                                obj, i as usize,
                            ));
                        }
                        i += step_val;
                    }
                }
                return Ok(if is_bytes {
                    pyre_object::bytesobject::w_bytes_from_bytes(&result)
                } else {
                    pyre_object::bytearrayobject::w_bytearray_from_bytes(&result)
                });
            }
            let name = if is_bytes { "bytes" } else { "bytearray" };
            return Err(PyError::type_error(format!(
                "{name} indices must be integers"
            )));
        } else if is_type(obj) {
            // Python 3.9+ generic subscript: type[X] → __class_getitem__(X)
            // PyPy: typeobject.py type.__class_getitem__
            if let Some(method) = lookup_in_type_where(obj, "__class_getitem__") {
                let result = crate::call_function(method, &[obj, index]);
                // Fallback if the user-defined __class_getitem__ raised
                // a stub error or returned NULL — return the type so
                // `class Foo(Generic[T]): pass` keeps working.
                if !result.is_null() {
                    return Ok(result);
                }
            }
            // Default: return the type itself (stub for GenericAlias)
            Ok(obj)
        } else if is_instance(obj) {
            // PyPy: descroperation.py __getitem__
            if let Some(method) = lookup_in_type_where(w_instance_get_type(obj), "__getitem__") {
                return crate::call::call_function_impl_result(method, &[obj, index]);
            }
            Err(PyError::type_error(format!(
                "'{}' object is not subscriptable",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else if is_range_iter(obj) {
            let r = &*(obj as *const pyre_object::rangeobject::W_RangeIterator);
            let len = if r.step > 0 {
                (r.stop - r.current + r.step - 1) / r.step
            } else if r.step < 0 {
                (r.current - r.stop - r.step - 1) / (-r.step)
            } else {
                0
            };
            if is_int(index) {
                // range[i]
                let i = w_int_get_value(index);
                let idx = if i < 0 { len + i } else { i };
                if idx < 0 || idx >= len {
                    Err(PyError::new(
                        PyErrorKind::IndexError,
                        "range object index out of range",
                    ))
                } else {
                    Ok(w_int_new(r.current + idx * r.step))
                }
            } else if is_slice(index) {
                // range[start:stop:step] → returns a list
                let s_raw = w_slice_get_start(index);
                let e_raw = w_slice_get_stop(index);
                let step_raw = w_slice_get_step(index);
                let s = if is_none(s_raw) {
                    0
                } else {
                    w_int_get_value(s_raw)
                };
                let e = if is_none(e_raw) {
                    len
                } else {
                    w_int_get_value(e_raw)
                };
                let sl_step = if is_none(step_raw) {
                    1
                } else {
                    w_int_get_value(step_raw)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) };
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) };
                let mut items = Vec::new();
                let mut i = s;
                while (sl_step > 0 && i < e) || (sl_step < 0 && i > e) {
                    items.push(w_int_new(r.current + i * r.step));
                    i += sl_step;
                }
                Ok(w_list_new(items))
            } else {
                Err(PyError::type_error(
                    "range indices must be integers or slices",
                ))
            }
        } else {
            Err(PyError::type_error(format!(
                "'{}' object is not subscriptable",
                (*(*obj).ob_type).name,
            )))
        }
    }
}

/// `pypy/interpreter/baseobjspace.py:870 finditem` — return the value
/// for `key` in `obj`, or `None` if absent.  PyPy catches only the
/// `KeyError` arm and re-raises any other `OperationError`; in Rust
/// the re-raise surfaces as `Result::Err`, the absent case as
/// `Ok(None)`, and a hit as `Ok(Some(value))`.
pub fn finditem(obj: PyObjectRef, index: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    match getitem(obj, index) {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.kind == crate::PyErrorKind::KeyError => Ok(None),
        Err(err) => Err(err),
    }
}

/// Set item by index: `obj[index] = value`.

pub fn setitem(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    let value = unwrap_cell(value);
    unsafe {
        // `pypy/objspace/std/dictproxyobject.py` exposes neither
        // `__setitem__` nor `__delitem__`, so `space.setitem` on a
        // mappingproxy raises `TypeError: 'mappingproxy' object does
        // not support item assignment`.  Detect proxy before any
        // dict-like assignment fallthrough.
        if pyre_object::is_dict_proxy(obj) {
            return Err(PyError::type_error(
                "'mappingproxy' object does not support item assignment",
            ));
        }
        if is_list(obj) {
            if is_slice(index) {
                let len = w_list_len(obj) as i64;
                let (start, stop, step) = normalize_slice(index, len)?;
                // listobject.py:709-714 wraps non-list iterables into a
                // temporary W_ListObject so the strategy-aware setslice
                // (`listobject.py:1746-1758`) and extended-slice
                // (`listobject.py:descr_setitem` step != 1 branch) paths
                // see a list operand.
                let w_other = if pyre_object::is_list(value) {
                    value
                } else {
                    let items = crate::builtins::collect_iterable(value)?;
                    pyre_object::listobject::w_list_new(items)
                };
                if step == 1 {
                    let s_lo = start.max(0) as usize;
                    let s_hi = stop.max(0) as usize;
                    pyre_object::listobject::w_list_setslice(obj, s_lo, s_hi, w_other)
                        .expect("w_other is always a valid list");
                    return Ok(w_none());
                }
                // Extended slice: `pypy/objspace/std/listobject.py
                // W_ListObject.descr_setitem` enforces equal length
                // ("attempt to assign sequence of size %d to extended
                // slice of size %d") and writes positions in order.
                let mut indices = Vec::new();
                let mut i = start;
                while (step > 0 && i < stop) || (step < 0 && i > stop) {
                    if i >= 0 && i < len {
                        indices.push(i);
                    }
                    i += step;
                }
                let other_len = pyre_object::w_list_len(w_other);
                if other_len != indices.len() {
                    return Err(PyError::new(
                        PyErrorKind::ValueError,
                        format!(
                            "attempt to assign sequence of size {} to extended slice of size {}",
                            other_len,
                            indices.len()
                        ),
                    ));
                }
                for (k, &idx) in indices.iter().enumerate() {
                    let item = pyre_object::w_list_getitem(w_other, k as i64)
                        .expect("k < other_len by construction");
                    if !pyre_object::w_list_setitem(obj, idx, item) {
                        return Err(PyError::new(
                            PyErrorKind::IndexError,
                            "list assignment index out of range",
                        ));
                    }
                }
                return Ok(w_none());
            }
            if !is_int(index) {
                let tp = if index.is_null() {
                    "NULL"
                } else {
                    (*(*index).ob_type).name
                };
                return Err(PyError::type_error(format!(
                    "list indices must be integers, not {tp}"
                )));
            }
            let idx = w_int_get_value(index);
            if w_list_setitem(obj, idx, value) {
                Ok(w_none())
            } else {
                Err(PyError::new(
                    PyErrorKind::IndexError,
                    "list assignment index out of range",
                ))
            }
        } else if is_dict(obj) {
            w_dict_store(obj, index, value);
            Ok(w_none())
        } else if pyre_object::bytearrayobject::is_bytearray(obj) {
            if is_int(index) {
                let idx = w_int_get_value(index);
                let len = pyre_object::bytearrayobject::w_bytearray_len(obj) as i64;
                let actual = if idx < 0 { len + idx } else { idx };
                if actual >= 0 && actual < len {
                    let val = w_int_get_value(value) as u8;
                    pyre_object::bytearrayobject::w_bytearray_setitem(obj, actual as usize, val);
                    return Ok(w_none());
                }
                return Err(PyError::new(
                    PyErrorKind::IndexError,
                    "bytearray index out of range",
                ));
            }
            Err(PyError::type_error("bytearray indices must be integers"))
        } else if is_instance(obj) {
            // PyPy: descroperation.py __setitem__ — `space.get_and_call_function`
            // raises on instance error.  pyre `call_function` stashes errors
            // as PY_NULL; `call_and_check` recovers them.
            if let Some(method) = lookup_in_type_where(w_instance_get_type(obj), "__setitem__") {
                crate::builtins::call_and_check(method, &[obj, index, value])?;
                return Ok(w_none());
            }
            Err(PyError::type_error(format!(
                "'{}' object does not support item assignment",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else {
            Err(PyError::type_error(format!(
                "'{}' object does not support item assignment",
                (*(*obj).ob_type).name,
            )))
        }
    }
}

/// String-keyed `finditem` shorthand: `space.finditem_str(w_obj, key)`.
pub fn finditem_str(obj: PyObjectRef, key: &str) -> Result<Option<PyObjectRef>, PyError> {
    finditem(obj, w_str_new(key))
}

/// PyPy-compatible identity check returning a raw boolean value.
pub fn is_w(w_one: PyObjectRef, w_two: PyObjectRef) -> bool {
    std::ptr::eq(w_one, w_two)
}

/// PyPy-compatible identity check returning a Python bool object.
pub fn is_(w_one: PyObjectRef, w_two: PyObjectRef) -> PyObjectRef {
    w_bool_from(is_w(w_one, w_two))
}

/// Python-level `not` operation.
pub fn not_(obj: PyObjectRef) -> PyObjectRef {
    w_bool_from(!is_true(obj))
}

/// PyPy-compatible attribute lookup returning `None` when not found.
pub fn findattr(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    if unsafe { is_none(obj) } {
        return None;
    }
    match getattr(obj, name) {
        Ok(value) => Some(value),
        Err(err) => {
            if err.kind == crate::PyErrorKind::AttributeError
                || err.kind == crate::PyErrorKind::NameError
            {
                None
            } else {
                panic!("space.findattr: unexpected {err:?}");
            }
        }
    }
}

/// Check whether `exc_type` matches `check_class`, including tuple/list class inputs.
pub fn exception_match(exc_type: PyObjectRef, check_class: PyObjectRef) -> bool {
    let (exc_type, check_class) = (exc_type, check_class);
    if unsafe { is_none(check_class) || is_none(exc_type) } {
        return false;
    }

    let is_tuple_check = unsafe { is_tuple(check_class) };
    if is_tuple_check {
        let len = unsafe { w_tuple_len(check_class) };
        for i in 0..len {
            let candidate = unsafe { w_tuple_getitem(check_class, i as i64) };
            if let Some(candidate) = candidate {
                if exception_match(exc_type, candidate) {
                    return true;
                }
            }
        }
        return false;
    }

    // Python 3: except clause only accepts tuple, not list.
    if !unsafe { is_type(check_class) } {
        return false;
    }

    if is_w(exc_type, check_class) {
        return true;
    }

    let mro_ptr = unsafe { w_type_get_mro(exc_type) };
    if mro_ptr.is_null() {
        return false;
    }

    let mro = unsafe { &*mro_ptr };
    mro.iter().any(|&klass| is_w(klass, check_class))
}

/// Get the length of a container: `len(obj)`.
pub fn len(obj: PyObjectRef) -> PyResult {
    // `pypy/objspace/std/dictproxyobject.py:32 descr_len` →
    // `space.len(self.w_mapping)`.
    let obj = unsafe {
        if pyre_object::is_dict_proxy(obj) {
            pyre_object::w_dict_proxy_get_mapping(obj)
        } else {
            obj
        }
    };
    // `pypy/objspace/std/dictmultiobject.py W_DictMultiViewKeysObject
    // .descr_len` returns `space.len(self.w_dict)` for all three view
    // kinds.  Forward to the source dict so the view's len reflects
    // live mutations on the dict, matching PyPy's view semantics.
    unsafe {
        if pyre_object::dictviewobject::is_dict_view(obj) {
            let dict = pyre_object::dictviewobject::w_dict_view_get_dict(obj);
            if dict.is_null() {
                return Ok(w_int_new(0));
            }
            return Ok(w_int_new(pyre_object::w_dict_len(dict) as i64));
        }
    }
    unsafe {
        if is_list(obj) {
            Ok(w_int_new(w_list_len(obj) as i64))
        } else if is_tuple(obj) {
            Ok(w_int_new(w_tuple_len(obj) as i64))
        } else if is_dict(obj) {
            Ok(w_int_new(w_dict_len(obj) as i64))
        } else if pyre_object::is_set_or_frozenset(obj) {
            Ok(w_int_new(pyre_object::w_set_len(obj) as i64))
        } else if is_str(obj) {
            Ok(w_int_new(w_str_len(obj) as i64))
        } else if pyre_object::bytesobject::is_bytes_like(obj) {
            Ok(w_int_new(
                pyre_object::bytesobject::bytes_like_len(obj) as i64
            ))
        } else if is_range_iter(obj) {
            // PRE-EXISTING-ADAPTATION: pyre conflates `range` and
            // `range_iterator` into a single `W_RangeIterator` (see
            // `builtin_range` in `builtins.rs`). PyPy keeps them
            // separate: `pypy/module/__builtin__/functional.py:444
            // W_Range` carries `w_length` and exposes
            // `descr_len → self.w_length` (line 485-486), while the
            // iterator (`pypy/objspace/std/iterobject.py
            // W_AbstractSeqIterObject:47 descr_length_hint`) exposes
            // only `__length_hint__`.  The convergence path is to
            // split pyre's `W_RangeIterator` into `W_Range` +
            // `W_RangeIterator`. Until then, derive the remaining
            // count from `(stop - current) / step` so `len(range(N))`
            // matches CPython's `range.__len__` semantics.
            let r = &*(obj as *const pyre_object::rangeobject::W_RangeIterator);
            let count = if r.step > 0 {
                ((r.stop - r.current).max(0) + r.step - 1) / r.step
            } else if r.step < 0 {
                ((r.current - r.stop).max(0) + (-r.step) - 1) / (-r.step)
            } else {
                0
            };
            Ok(w_int_new(count.max(0)))
        } else if is_instance(obj) {
            // descroperation.py:294-298 `_len` — `space.lookup(w_obj,
            // '__len__')` then `space.get_and_call_function(w_descr,
            // w_obj)`.  PyPy `get_and_call_function` raises on user
            // exception; pyre's `call_function` stashes errors as PY_NULL.
            // Use `call_and_check` so user-raised exceptions propagate.
            if let Some(method) = lookup_in_type_where(w_instance_get_type(obj), "__len__") {
                return crate::builtins::call_and_check(method, &[obj]);
            }
            // Per-instance __len__ via the unified getattr path (live dict + ATTR_TABLE).
            if let Ok(method) = getattr(obj, "__len__") {
                return crate::builtins::call_and_check(method, &[obj]);
            }
            Err(PyError::type_error(format!(
                "object of type '{}' has no len()",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else {
            Err(PyError::type_error(format!(
                "object of type '{}' has no len()",
                (*(*obj).ob_type).name,
            )))
        }
    }
}

// ── Attribute operations ──────────────────────────────────────────────

thread_local! {
    /// Side table mapping object addresses to their instance __dict__.
    ///
    /// Every object can have attributes stored here. This avoids modifying
    /// the repr(C) layout of existing object types.
    pub static ATTR_TABLE: RefCell<HashMap<usize, HashMap<String, PyObjectRef>>> =
        RefCell::new(HashMap::new());
}

// `INSTANCE_DICT` and `WEAKREF_TABLE` live in `objspace/std/mapdict.rs`,
// mirroring PyPy's `MapdictDictSupport` and `MapdictWeakrefSupport`.

/// interpreter/baseobjspace.py:43-44 W_Root.getdict(space).
///
/// ```python
/// def getdict(self, space):
///     return None
/// ```
///
/// objspace/std/mapdict.py:817-818 MapdictDictSupport.getdict overrides
/// it to call `_obj_getdict`. pyre dispatches at runtime via the type's
/// hasdict flag because Rust has no per-class virtual table.
pub fn getdict(obj: PyObjectRef) -> PyObjectRef {
    let w_type = match crate::typedef::r#type(obj) {
        Some(tp) => tp,
        None => return pyre_object::PY_NULL,
    };
    if unsafe { pyre_object::w_type_get_hasdict(w_type) } {
        crate::objspace::std::mapdict::_obj_getdict(obj)
    } else {
        // W_Root.getdict default — return None
        pyre_object::PY_NULL
    }
}

/// interpreter/baseobjspace.py:70-73 W_Root.setdict(space, w_dict).
///
/// ```python
/// def setdict(self, space, w_dict):
///     raise oefmt(space.w_TypeError,
///                  "attribute '__dict__' of %T objects is not writable",
///                  self)
/// ```
///
/// objspace/std/mapdict.py:820-821 MapdictDictSupport.setdict overrides
/// it to call `_obj_setdict`.
pub fn setdict(obj: PyObjectRef, w_dict: PyObjectRef) -> Result<(), PyError> {
    let w_type = match crate::typedef::r#type(obj) {
        Some(tp) => tp,
        None => {
            return Err(PyError::type_error(
                "attribute '__dict__' of object is not writable".to_string(),
            ));
        }
    };
    if unsafe { pyre_object::w_type_get_hasdict(w_type) } {
        crate::objspace::std::mapdict::_obj_setdict(obj, w_dict)
    } else {
        let tp_name = unsafe { pyre_object::w_type_get_name(w_type) };
        Err(PyError::type_error(format!(
            "attribute '__dict__' of '{}' objects is not writable",
            tp_name,
        )))
    }
}

/// interpreter/baseobjspace.py:142-143 W_Root.getweakref().
///
/// ```python
/// def getweakref(self):
///     return None
/// ```
///
/// MapdictWeakrefSupport.getweakref overrides it.
pub fn getweakref(obj: PyObjectRef) -> Option<PyObjectRef> {
    let w_type = crate::typedef::r#type(obj)?;
    if unsafe { pyre_object::w_type_get_weakrefable(w_type) } {
        crate::objspace::std::mapdict::getweakref(obj)
    } else {
        None
    }
}

/// interpreter/baseobjspace.py:145-147 W_Root.setweakref(space, weakreflifeline).
///
/// ```python
/// def setweakref(self, space, weakreflifeline):
///     raise oefmt(space.w_TypeError,
///                  "cannot create weak reference to '%T' object", self)
/// ```
///
/// MapdictWeakrefSupport.setweakref overrides it.
pub fn setweakref(obj: PyObjectRef, weakreflifeline: PyObjectRef) -> Result<(), PyError> {
    let w_type = match crate::typedef::r#type(obj) {
        Some(tp) => tp,
        None => {
            return Err(PyError::type_error(
                "cannot create weak reference to object".to_string(),
            ));
        }
    };
    if unsafe { pyre_object::w_type_get_weakrefable(w_type) } {
        crate::objspace::std::mapdict::setweakref(obj, weakreflifeline);
        Ok(())
    } else {
        let tp_name = unsafe { pyre_object::w_type_get_name(w_type) };
        Err(PyError::type_error(format!(
            "cannot create weak reference to '{}' object",
            tp_name,
        )))
    }
}

/// interpreter/baseobjspace.py:149-150 W_Root.delweakref().
///
/// ```python
/// def delweakref(self):
///     pass
/// ```
pub fn delweakref(obj: PyObjectRef) {
    let w_type = match crate::typedef::r#type(obj) {
        Some(tp) => tp,
        None => return,
    };
    if unsafe { pyre_object::w_type_get_weakrefable(w_type) } {
        crate::objspace::std::mapdict::delweakref(obj);
    }
}

/// `pypy/interpreter/module.py:77 Module.getdict()` parity: return
/// the **canonical** `W_DictObject` already paired with this storage,
/// not a fresh snapshot.  When the storage was first allocated
/// (`w_module_new`, exec/eval anonymous path, etc.) it was bound to
/// a sibling `W_DictObject` via `set_mirror_target` so that
/// storage-side writes back-mirror into that one dict's entries Vec.
/// This lookup retrieves that canonical dict so
/// `function.__globals__`, `globals()`, and the module's own
/// `__dict__` all share **one** identity (`f.__globals__ is
/// m.__dict__` invariant) and the iterating surfaces (`keys`,
/// `values`, `items`, `update`, `copy`, `iter`, `repr`) line up with
/// `lookup` / `len` on the same logical state.
///
/// `type.__dict__` is **not** routed through this helper: PyPy
/// `pypy/objspace/std/typeobject.py:1277 descr_get_dict` returns
/// `W_DictProxyObject(w_dict)` (a read-only live view), not the
/// type's underlying `W_DictObject`.  The dictproxy keeps its own
/// identity per call and forwards reads/iterations to the type's
/// `w_dict`; pyre's type.__dict__ readers stay on that path.
///
/// Lazy-canonical fallback: a storage that has not yet been paired
/// (the `set_mirror_target` call has not happened) gets one allocated
/// here and registered as the `mirror_target`, so subsequent calls
/// return the same object.
pub fn dict_storage_to_dict(ns_ptr: *const crate::DictStorage) -> PyObjectRef {
    if ns_ptr.is_null() {
        return pyre_object::w_dict_new();
    }
    let storage = unsafe { &mut *(ns_ptr as *mut crate::DictStorage) };
    let target = storage.mirror_target();
    if !target.is_null() {
        return target;
    }
    // Lazy canonical: snapshot-populate a fresh W_DictObject and
    // register it as the storage's permanent back-mirror target so
    // every subsequent `dict_storage_to_dict` call on this storage
    // returns this same dict.  The snapshot pre-populates the entries
    // Vec for storage entries that pre-date this call (the only path
    // that adds bindings to the storage without going through the
    // back-mirror).
    let dict = pyre_object::dictobject::w_dict_new_with_dict_storage(ns_ptr as *mut u8);
    unsafe {
        for (key, &value) in storage.entries() {
            if !value.is_null() {
                pyre_object::dictobject::w_dict_setitem_str_no_proxy(dict, key, value);
            }
        }
    }
    storage.set_mirror_target(dict);
    dict
}

/// Get an attribute from an object: `obj.name`.
///
/// For module objects, looks up the name in the module's namespace dict
/// (PyPy: Module.getdict → w_dict lookup).
/// For other objects, looks up the attribute in the per-object side table.

pub fn getattr(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = unwrap_cell(obj);

    // pypy/module/_weakref/interp__weakref.py:356-394 — proxy_typedef_dict
    // wraps every space op in `force(space, w_obj)`. PyPy then dispatches
    // through the type's `__getattribute__` slot at the C level, so the
    // proxy's wrapper runs before any inline path. pyre's `getattr` does
    // not consult the type's `__getattribute__`, so we apply the same
    // effect by forcing the receiver here. `force()` is a no-op for any
    // non-proxy operand, costing only one ptr-equality check on the hot
    // path.
    let obj = crate::module::_weakref::interp_weakref::force(obj)?;

    // super proxy — PyPy: superobject.py super_getattro
    // Looks up `name` in cls's MRO starting AFTER super_type.
    unsafe {
        if pyre_object::superobject::is_super(obj) {
            let super_type = pyre_object::superobject::w_super_get_type(obj);
            let bound_obj = pyre_object::superobject::w_super_get_obj(obj);

            // Walk obj's type MRO, skip until we pass super_type.
            // Fall back to `crate::typedef::r#type(obj)` so non-INSTANCE
            // built-in subclasses (W_ExceptionObject, etc.) resolve their
            // class through the same path that powers `type(obj)` —
            // `pypy/objspace/std/typeobject.py:1083 type_get_mro`.
            let w_obj_type = if is_instance(bound_obj) {
                w_instance_get_type(bound_obj)
            } else if is_type(bound_obj) {
                bound_obj
            } else if let Some(cls) = crate::typedef::r#type(bound_obj) {
                cls
            } else {
                return Err(PyError::type_error("super: bad obj type"));
            };
            let mro_ptr = w_type_get_mro(w_obj_type);
            if !mro_ptr.is_null() {
                let mro = &*mro_ptr;
                let mut past_super = false;
                for &t in mro {
                    if std::ptr::eq(t, super_type) {
                        past_super = true;
                        continue;
                    }
                    if !past_super {
                        continue;
                    }
                    if is_type(t) {
                        // Look in this class's own dict only (not its MRO),
                        // since we are already iterating the full MRO ourselves.
                        let ns_ptr = w_type_get_dict_ptr(t) as *mut crate::DictStorage;
                        let found = if !ns_ptr.is_null() {
                            (*ns_ptr).get(name).copied()
                        } else {
                            None
                        };
                        if let Some(attr) = found {
                            // superobject.py super_getattro:
                            // Invoke descriptor __get__ protocol.
                            // function.__get__(obj, type) → bound method
                            // __new__ is implicitly static — never bind.
                            if name != "__new__"
                                && crate::is_function(attr)
                                && !pyre_object::is_staticmethod(attr)
                                && !pyre_object::is_classmethod(attr)
                            {
                                return Ok(pyre_object::w_method_new(attr, bound_obj, w_obj_type));
                            }
                            return Ok(attr);
                        }
                    }
                }
            }
            return Err(PyError::new(
                PyErrorKind::AttributeError,
                format!("'super' object has no attribute '{name}'"),
            ));
        }
    }

    // Generator/coroutine methods — PyPy: generator.py GeneratorIterator
    //
    // Return W_Method(func, gen) so the generator is passed as args[0].
    unsafe {
        if pyre_object::generatorobject::is_generator(obj) {
            let (sname, func, arity): (&str, fn(&[PyObjectRef]) -> PyResult, Option<u16>) =
                match name {
                    "send" => ("send", generator_send_method, Some(2)),
                    "throw" => ("throw", generator_throw_method, None),
                    "close" => ("close", generator_close_method, Some(1)),
                    "__next__" => ("__next__", generator_next_method, Some(1)),
                    "__iter__" => return Ok(obj),
                    _ => ("", generator_next_method, None), // sentinel — won't match
                };
            if !sname.is_empty() {
                let func_obj = if let Some(a) = arity {
                    crate::make_builtin_function_with_arity(sname, func, a)
                } else {
                    crate::make_builtin_function(sname, func)
                };
                return Ok(pyre_object::w_method_new(
                    func_obj,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
        }
    }

    // itertools.count / itertools.repeat methods — PyPy interp_itertools.py
    // Expose `__next__` and `__iter__` so `_count(1).__next__` and
    // `iter(counter)` work.
    unsafe {
        if pyre_object::itertoolsmodule::is_count(obj)
            || pyre_object::itertoolsmodule::is_repeat(obj)
        {
            match name {
                "__iter__" => return Ok(obj),
                "__next__" => {
                    let func_obj =
                        crate::make_builtin_function_with_arity("__next__", iter_next_method, 1);
                    return Ok(pyre_object::w_method_new(
                        func_obj,
                        obj,
                        pyre_object::PY_NULL,
                    ));
                }
                _ => {}
            }
        }
    }

    // Property descriptor methods — PyPy: descriptor.py W_Property.setter / getter / deleter
    // Returns a bound method (W_Method) that captures the property via w_self,
    // so the static handler can extract the property from args[0].
    unsafe {
        if is_property(obj) {
            let static_name: Option<(
                &'static str,
                fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
            )> = match name {
                "setter" => Some(("setter", property_setter_impl)),
                "getter" => Some(("getter", property_getter_impl)),
                "deleter" => Some(("deleter", property_deleter_impl)),
                _ => None,
            };
            if let Some((sname, func)) = static_name {
                let builtin = crate::make_builtin_function_with_arity(sname, func, 2);
                return Ok(pyre_object::methodobject::w_method_new(
                    builtin,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
            match name {
                "fget" => return Ok(w_property_get_fget(obj)),
                "fset" => return Ok(w_property_get_fset(obj)),
                "fdel" => return Ok(w_property_get_fdel(obj)),
                "__doc__" => {
                    // Try INSTANCE_DICT first (set via property.__doc__ = "...")
                    let w_dict = crate::objspace::std::mapdict::_obj_getdict(obj);
                    if let Some(v) =
                        pyre_object::w_dict_lookup(w_dict, pyre_object::w_str_new("__doc__"))
                    {
                        return Ok(v);
                    }
                    return Ok(w_none());
                }
                _ => {}
            }
        }
    }

    // Member descriptor attributes — typedef.py:443 Member.__name__, __objclass__
    unsafe {
        if pyre_object::memberobject::is_member(obj) {
            match name {
                "__name__" => {
                    return Ok(pyre_object::w_str_new(pyre_object::w_member_get_name(obj)));
                }
                "__objclass__" => return Ok(pyre_object::w_member_get_cls(obj)),
                _ => {}
            }
        }
    }

    // Module objects: look up in module namespace.
    // PyPy `space.getattr(w_module, w_name) → Module.getdictvalue(space,
    // name)` (`pypy/interpreter/module.py:Module.getdictvalue`
    // inherited from `baseobjspace.py:45-48 W_Root.getdictvalue`):
    //
    //     w_dict = self.getdict(space)        # module.py:77 → self.w_dict
    //     if w_dict is not None:
    //         return space.finditem_str(w_dict, attr)
    //     return None
    //
    // Routing through `space.finditem_str` (rather than reading the
    // backing storage directly) gives dict subclass `__getitem__`
    // overrides their PyPy chance to fire on the user-supplied
    // `__builtins__` aliasing case (`moduledef.py:102-103
    // Module(space, None, w_builtin)`), and routes through the
    // storage-authoritative read path so transient W_DictObject
    // snapshots can't shadow the live storage state.  The Result-
    // bearing variant propagates non-KeyError errors from subclass
    // overrides (`baseobjspace.py:870 finditem` re-raise).
    unsafe {
        if is_module(obj) {
            if name == "__dict__" {
                // module.py:20 — `Module.getdict(space)` returns
                // `self.w_dict`.  Always non-null after construction.
                return Ok(pyre_object::w_module_get_w_dict(obj));
            }
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                if let Some(value) = finditem_str(w_dict, name)? {
                    if !value.is_null() {
                        return Ok(value);
                    }
                }
            }
        }
    }

    // Instance objects — PyPy: descroperation.py descr__getattribute__
    //
    // Full descriptor protocol (PEP 252):
    //   1. Look up name in type MRO → w_descr
    //   2. If w_descr is a data descriptor (__get__ + __set__/__delete__):
    //      → call w_descr.__get__(obj, type)
    //   3. Check instance dict
    //   4. If w_descr is a non-data descriptor (__get__ only):
    //      → call w_descr.__get__(obj, type)
    //   5. Return w_descr as-is
    unsafe {
        // `pypy/interpreter/typedef.py:825-826 Method.typedef` exposes
        // `__func__` / `__self__` as `interp_attrproperty_w` getset
        // descriptors that resolve to the wrapped function / instance
        // directly on attribute access.  Pyre's method typedef
        // registers them as regular `make_builtin_function` entries
        // which the descriptor protocol below would surface as bound
        // methods (binding the `__func__` helper to the method
        // instance), breaking `m.__func__ is C.m` and `m.__self__ is
        // c` identity.  Short-circuit before the `is_instance` branch
        // so the type dispatch path matches PyPy's getset semantics.
        // PyPy3 exposes only the dunder names — `im_func` / `im_self`
        // were dropped in 3.x, so do not surface them here.
        if pyre_object::methodobject::is_method(obj) {
            match name {
                "__func__" => {
                    return Ok(pyre_object::methodobject::w_method_get_func(obj));
                }
                "__self__" => {
                    return Ok(pyre_object::methodobject::w_method_get_self(obj));
                }
                _ => {}
            }
        }
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);

            // `pypy/objspace/descroperation.py descr__getattribute__`
            // dispatches through the receiver type's `__getattribute__`
            // slot before running the default descriptor protocol.
            // Users routinely override this to customise *all* attribute
            // access (e.g. lazy proxies, validating wrappers).  Detect a
            // non-default override by comparing against the canonical
            // `object.__getattribute__` registered at type init.
            if name != "__getattribute__" {
                if let Some(slot) = lookup_in_type(w_type, "__getattribute__") {
                    let default_slot =
                        lookup_in_type(crate::typedef::w_object(), "__getattribute__");
                    let is_default = match default_slot {
                        Some(d) => std::ptr::eq(slot, d),
                        None => false,
                    };
                    if !is_default {
                        let name_obj = w_str_new(name);
                        return crate::call::call_function_impl_result(slot, &[obj, name_obj]);
                    }
                }
            }

            // Step 1: look up in type MRO
            let w_descr = lookup_in_type_where(w_type, name);

            // Step 2: data descriptor takes priority over instance dict
            if let Some(descr) = w_descr {
                if is_data_descr(descr) {
                    if let Some(result) = get(descr, obj, w_type)? {
                        return Ok(result);
                    }
                }
            }

            // Step 3: instance dict
            // First check the Python dict (INSTANCE_DICT) for live-view semantics,
            // then ATTR_TABLE for slot values and legacy attributes.
            let w_dict = getdict(obj);
            if !w_dict.is_null() {
                if let Some(value) = pyre_object::w_dict_getitem_str(w_dict, name) {
                    return Ok(value);
                }
            }
            // Fallback: ATTR_TABLE (slot values via Member descriptor side-store)
            let found = ATTR_TABLE.with(|table| {
                let table = table.borrow();
                table
                    .get(&(obj as usize))
                    .and_then(|d| d.get(name).copied())
            });
            if let Some(value) = found {
                return Ok(value);
            }

            // Step 4: non-data descriptor
            // PyPy: descroperation.py — invoke __get__ to bind methods
            if let Some(descr) = w_descr {
                if let Some(result) = get(descr, obj, w_type)? {
                    return Ok(result);
                }
                // Step 5: builtin methods found in base type MRO need binding
                // CPython: PyFunction_GET_CODE slot → bound method
                if crate::is_function(descr)
                    && !crate::is_builtin_code(
                        crate::function_get_code(descr) as pyre_object::PyObjectRef
                    )
                {
                    return Ok(pyre_object::w_method_new(descr, obj, w_type));
                }
                return Ok(descr);
            }

            // Special attributes — PyPy: descroperation.py
            if name == "__class__" {
                return Ok(w_type);
            }

            // descroperation.py descr__getattribute__: on AttributeError,
            // check the type for `__getattr__` and call it.  Used by every
            // wrapper class that delegates attribute lookup to a backing
            // stream/buffer (unittest._WritelnDecorator, pathlib, etc.).
            if let Some(getattr_fn) = lookup_in_type_where(w_type, "__getattr__") {
                let name_obj = w_str_new(name);
                let result = crate::call_function(getattr_fn, &[obj, name_obj]);
                if !result.is_null() {
                    return Ok(result);
                }
            }

            return Err(PyError::new(
                PyErrorKind::AttributeError,
                format!(
                    "'{}' object has no attribute '{name}'",
                    w_type_get_name(w_type),
                ),
            ));
        }
    }

    // Type objects: look up in type's own dict → base dicts
    // PyPy: typeobject.py lookup_where → MRO search + descriptor unwrap
    unsafe {
        if is_type(obj) {
            // Special type attributes — PyPy: typeobject.py
            if name == "__class__" {
                // `pypy/objspace/std/typeobject.py:198 type___class__getter`
                // returns `self.w_metaclass` (the metaclass).  pyre stamps
                // each registered builtin type's `w_class` to the
                // `type` typeobject in `init_typeobjects`'s post-loop
                // (typedef.rs:489-499).  Return that directly; falling
                // through to `lookup_in_type_where` would hit the
                // `__class__` getset descriptor on the metatype and
                // recurse.  When `w_class` is null (bootstrap or a
                // type built before `init_typeobjects`), fall back to
                // the `type` typeobject so `int.__class__ is type`
                // still holds.
                let mc = (*obj).w_class;
                if !mc.is_null() {
                    return Ok(mc);
                }
                let w_type_type = crate::typedef::w_type();
                if !w_type_type.is_null() {
                    return Ok(w_type_type);
                }
            }
            if name == "__name__" {
                return Ok(w_str_new(w_type_get_name(obj)));
            }
            if name == "__qualname__" {
                // Check if __qualname__ was explicitly set in class body
                if let Some(qn) = lookup_in_type_where(obj, "__qualname__") {
                    return Ok(qn);
                }
                return Ok(w_str_new(w_type_get_name(obj)));
            }
            if name == "__mro__" {
                let mro_ptr = w_type_get_mro(obj);
                if !mro_ptr.is_null() {
                    return Ok(w_tuple_new((*mro_ptr).clone()));
                }
            }
            if name == "__dict__" {
                // `pypy/objspace/std/typeobject.py:1277 descr_get_dict`
                // returns `W_DictProxyObject(w_dict)` — a read-only
                // **live** view of the type's namespace.  The proxy's
                // identity is fresh per call (a new wrapper) but its
                // `w_mapping` is the type's canonical W_DictObject, so
                // a subsequent `cls.x = 1; d['x']` resolves through the
                // dict_storage_proxy and surfaces the live binding.
                let dict_ptr = w_type_get_dict_ptr(obj) as *const crate::DictStorage;
                if dict_ptr.is_null() {
                    return Ok(pyre_object::w_dict_proxy_new(pyre_object::w_dict_new()));
                }
                let canonical = dict_storage_to_dict(dict_ptr);
                return Ok(pyre_object::w_dict_proxy_new(canonical));
            }
            if name == "__bases__" {
                return Ok(w_type_get_bases(obj));
            }
            // PEP 649 lazy annotations: when `cls.__annotations__` is
            // requested and only `__annotate_func__` (or `__annotate__`)
            // is set, call the annotate function with format=1 to
            // produce the actual dict.  CPython 3.14+ stops emitting
            // `__annotations__` directly in class bodies in favour of
            // this lazy form.
            if name == "__annotations__" {
                if let Some(v) = lookup_in_type_where(obj, "__annotations__") {
                    return Ok(v);
                }
                if let Some(annotate_fn) = lookup_in_type_where(obj, "__annotate_func__")
                    .or_else(|| lookup_in_type_where(obj, "__annotate__"))
                {
                    if !annotate_fn.is_null() && !is_none(annotate_fn) {
                        // format=1 (VALUE) — return runtime values.
                        return Ok(crate::call_function(annotate_fn, &[w_int_new(1)]));
                    }
                }
                return Ok(pyre_object::w_dict_new());
            }
            // PEP 649: `__annotate__` and `__annotate_func__` are the
            // same slot. Bytecode stores it as `__annotate_func__` in the
            // class dict; user code reads it as `__annotate__`. Forward
            // either name to the other, matching CPython's mapping in
            // typeobject.c type_get___annotate__.
            if name == "__annotate__" || name == "__annotate_func__" {
                if let Some(v) = lookup_in_type_where(obj, name) {
                    return Ok(v);
                }
                let alt = if name == "__annotate__" {
                    "__annotate_func__"
                } else {
                    "__annotate__"
                };
                if let Some(v) = lookup_in_type_where(obj, alt) {
                    return Ok(v);
                }
                return Ok(w_none());
            }
            // `__abstractmethods__` is a descriptor on `type` that raises
            // AttributeError when the slot is not populated, NOT a getter
            // that returns None. abc.update_abstractmethods relies on
            // hasattr() returning False to short-circuit non-ABCs.
            if name == "__abstractmethods__" {
                if let Some(v) = lookup_in_type_where(obj, name) {
                    return Ok(v);
                }
                return Err(PyError::new(
                    PyErrorKind::AttributeError,
                    format!(
                        "type object '{}' has no attribute '__abstractmethods__'",
                        w_type_get_name(obj),
                    ),
                ));
            }
            if name == "__doc__"
                || name == "__flags__"
                || name == "__code__"
                || name == "__func__"
                || name == "__self__"
                || name == "__wrapped__"
                || name == "__globals__"
                || name == "__closure__"
                || name == "__defaults__"
                || name == "__kwdefaults__"
            {
                // Check class dict first, then return None
                if let Some(v) = lookup_in_type_where(obj, name) {
                    return Ok(v);
                }
                return Ok(w_none());
            }
            // `__module__` is NOT in the short-circuit list — it falls
            // through to the normal descriptor protocol so type's
            // `__module__` GetSetProperty (`typedef.rs init_type_type`)
            // can resolve via PyPy's `typeobject.py:614-624 get_module`
            // (heaptype reads class dict, builtin types use the dot-
            // split of the class name with `"builtins"` fallback).

            if let Some(value) = lookup_in_type_where(obj, name) {
                if let Some(result) = get(value, PY_NULL, obj)? {
                    return Ok(result);
                }
                return Ok(value);
            }
            // Metaclass attribute lookup — PyPy: type.__getattribute__
            // baseobjspace.py:76 — the metaclass is type(C), read from w_class.
            let w_metaclass = {
                let w_class = (*obj).w_class;
                let w_type_type = crate::typedef::w_type();
                if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                    Some(w_class)
                } else {
                    None
                }
            };
            // PyPy: type.__getattribute__ → metatype descriptor protocol.
            // Search metaclass MRO. Binding is handled by load_method.
            let w_metaclasses: [Option<PyObjectRef>; 2] =
                [w_metaclass, crate::typedef::gettypefor((*obj).ob_type)];
            for w_metaclass in w_metaclasses.iter().flatten() {
                let w_metaclass = *w_metaclass;
                if is_type(w_metaclass) {
                    if let Some(value) = lookup_in_type_where(w_metaclass, name) {
                        if let Some(result) = get(value, obj, w_metaclass)? {
                            return Ok(result);
                        }
                        return Ok(value);
                    }
                }
            }
            return Err(PyError::new(
                PyErrorKind::AttributeError,
                format!(
                    "type object '{}' has no attribute '{name}'",
                    w_type_get_name(obj),
                ),
            ));
        }
    }

    // Builtin type method lookup via TypeDef registry.
    //
    // PyPy: space.type(w_obj) → W_TypeObject → MRO lookup in type dict.
    // Each builtin type (list, str, dict, etc.) has a W_TypeObject with
    // methods pre-installed, matching PyPy's TypeDef interpleveldefs.
    if let Some(w_type) = crate::typedef::r#type(obj) {
        if let Some(method) = unsafe { lookup_in_type_where(w_type, name) } {
            if unsafe { crate::is_function(method) } {
                return Ok(pyre_object::w_method_new(method, obj, w_type));
            }
            if let Some(result) = unsafe { get(method, obj, w_type)? } {
                return Ok(result);
            }
            return Ok(method);
        }
    }

    // Function object attributes — PyPy: funcobject.py W_Function
    // Check the live W_DictObject (functions are hasdict per typedef.py:735
    // __dict__ = getset_func_dict) before falling through to legacy ATTR_TABLE.
    if unsafe { crate::is_function(obj) } {
        let w_dict = getdict(obj);
        if !w_dict.is_null() {
            if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) } {
                return Ok(v);
            }
        }
        let found = ATTR_TABLE.with(|table| {
            table
                .borrow()
                .get(&(obj as usize))
                .and_then(|d| d.get(name).copied())
        });
        if let Some(v) = found {
            return Ok(v);
        }
    }
    unsafe {
        if crate::is_function(obj) {
            match name {
                "__code__" => {
                    // function_get_code returns Code-level pointer (W_CodeObject or BuiltinCode)
                    let code = crate::function_get_code(obj) as PyObjectRef;
                    if code.is_null() {
                        return Ok(w_none());
                    }
                    return Ok(code);
                }
                "__name__" => {
                    return Ok(w_str_new(crate::function_get_name(obj)));
                }
                "__closure__" => {
                    let closure = crate::function_get_closure(obj);
                    return Ok(if closure.is_null() { w_none() } else { closure });
                }
                "__globals__" => {
                    // `funcobject.py:325 fget_func_globals` returns
                    // `self.w_func_globals` directly.  Pyre's cached
                    // `function_get_globals_obj` returns the same
                    // canonical W_DictObject as
                    // `dict_storage_to_dict(function_get_globals(obj))`
                    // (mirror_target invariant) but skips the
                    // HashMap lookup on subsequent reads — every
                    // `f.__globals__` access on the same function
                    // re-uses the slot stamped on first call.
                    return Ok(unsafe { crate::function_get_globals_obj(obj) });
                }
                "__defaults__" => {
                    let defaults = crate::function_get_defaults(obj);
                    return Ok(if defaults.is_null() {
                        w_none()
                    } else {
                        defaults
                    });
                }
                "__kwdefaults__" => {
                    let kwdefaults = crate::function_get_kwdefaults(obj);
                    return Ok(if kwdefaults.is_null() {
                        w_none()
                    } else {
                        kwdefaults
                    });
                }
                "__qualname__" => {
                    // function.py:470-471 fget_func_qualname returns
                    // space.newtext(self.qualname); the typed
                    // `function_get_qualname` mirrors PyPy's `qualname or
                    // self.name` short-circuit (w_qualname slot →
                    // code.co_qualname → name).
                    let s = crate::function::function_get_qualname(obj);
                    return Ok(w_str_new(&s));
                }
                "__doc__" => {
                    // `pypy/interpreter/function.py:395-398 fget_func_doc`
                    // — instance dict override first, then lazy
                    // `code.getdocstring(space)`.  Pyre's
                    // `function_get_doc` mirrors that shape (instance
                    // dict → BuiltinCode `docstring` → user
                    // CodeObject `HAS_DOCSTRING` first const).  The
                    // generic `__doc__` fallback would otherwise
                    // return None for every user-defined function
                    // because no caller routes to `function_get_doc`.
                    return Ok(crate::function::function_get_doc(obj));
                }
                "__module__" => {
                    // `pypy/interpreter/function.py:507 fget___module__`
                    // lazy-resolves from `w_func_globals['__name__']` on
                    // first read and caches into `self.w_module`.  Pyre's
                    // `crate::function::fget___module__` mirrors that
                    // shape — `(*func).w_module` stamps on first call so
                    // subsequent reads (and explicit `setattr(f,
                    // '__module__', x)`) take the cache path.  The
                    // generic `__module__` fallback at the end of
                    // `getattr` would otherwise return `None` for every
                    // function (function.rs:48 init `w_module = PY_NULL`).
                    return Ok(unsafe { crate::function::fget___module__(obj) });
                }
                "__annotations__" => {
                    // `pypy/interpreter/function.py:548-551
                    // fget_func_annotations` returns
                    // `self.w_ann`, allocating an empty dict on first
                    // access if none was set, and stamping it back so
                    // identity holds.
                    //
                    // Pyre stores the eager dict on the typed
                    // `Function.w_ann` slot via
                    // `function_set_annotations` at MAKE_FUNCTION
                    // ANNOTATIONS time (eval.rs).  PEP 649 lazy
                    // annotations (`MakeFunctionFlag::Annotate`,
                    // default in the RustPython compiler) keep a
                    // `__annotate_func__` callable in ATTR_TABLE that
                    // we invoke with `format=1` to materialise the
                    // dict; the result is stamped onto `w_ann` to
                    // freeze identity for subsequent reads.
                    let raw = unsafe { (*(obj as *mut crate::function::Function)).w_ann };
                    if !raw.is_null() {
                        return Ok(raw);
                    }
                    let annotate_fn = ATTR_TABLE.with(|table| {
                        let table = table.borrow();
                        let entry = table.get(&(obj as usize))?;
                        entry
                            .get("__annotate_func__")
                            .copied()
                            .or_else(|| entry.get("__annotate__").copied())
                    });
                    if let Some(annotate_fn) = annotate_fn {
                        if !annotate_fn.is_null() && !is_none(annotate_fn) {
                            let dict = crate::call_function(annotate_fn, &[w_int_new(1)]);
                            unsafe {
                                crate::function::function_set_annotations(obj, dict);
                            }
                            return Ok(dict);
                        }
                    }
                    // Lazy-fill via the helper so the slot is stamped
                    // and `f.__annotations__ is f.__annotations__`
                    // identity holds across reads.
                    return Ok(unsafe { crate::function::function_get_annotations(obj) });
                }
                _ => {}
            }
        }
        // staticmethod/classmethod descriptor attributes
        // PyPy: function.py StaticMethod.__func__, ClassMethod.__func__
        if pyre_object::is_staticmethod(obj) {
            if name == "__func__" || name == "__wrapped__" {
                return Ok(pyre_object::w_staticmethod_get_func(obj));
            }
        }
        if pyre_object::is_classmethod(obj) {
            if name == "__func__" || name == "__wrapped__" {
                return Ok(pyre_object::w_classmethod_get_func(obj));
            }
        }
        if crate::pycode::is_code(obj) {
            let code_ptr = crate::pycode::w_code_get_ptr(obj) as *const crate::CodeObject;
            if code_ptr.is_null() {
                return Ok(w_none());
            }
            let code = &*code_ptr;
            match name {
                "co_varnames" => {
                    let items = code
                        .varnames
                        .iter()
                        .map(|item| w_str_new(item.as_ref()))
                        .collect();
                    return Ok(w_tuple_new(items));
                }
                "co_argcount" => return Ok(w_int_new(code.arg_count as i64)),
                "co_kwonlyargcount" => return Ok(w_int_new(code.kwonlyarg_count as i64)),
                "co_name" => return Ok(w_str_new(code.obj_name.as_ref())),
                "co_filename" => return Ok(w_str_new(code.source_path.as_ref())),
                "co_flags" => return Ok(w_int_new(code.flags.bits() as i64)),
                _ => {}
            }
        }
    }

    // Common special attributes — return defaults for any object type
    if name == "__doc__"
        || name == "__module__"
        || name == "__wrapped__"
        || name == "__annotations__"
    {
        // Check ATTR_TABLE first, then return None as default
        let found = ATTR_TABLE.with(|table| {
            let table = table.borrow();
            table
                .get(&(obj as usize))
                .and_then(|d| d.get(name).copied())
        });
        return Ok(found.unwrap_or(w_none()));
    }
    // Exception attributes — PyPy: W_BaseException attributes
    if unsafe { pyre_object::is_exception(obj) } {
        match name {
            "__traceback__" => {
                // `pypy/module/exceptions/interp_exceptions.py:196-201
                // W_BaseException.descr_gettraceback` returns the
                // `w_traceback` slot stamped by `descr_settraceback`
                // and the `raise` machinery's
                // `OperationError.normalize_exception` path.  Defaults
                // to `None` in CPython when none has been set.  Pyre's
                // stdlib bundle (`lib-python/3/types.py:53-57`) probes
                // `type(exc.__traceback__.tb_frame)` even before any
                // `raise` reaches except, so returning `None` here
                // explodes module-level type initialisation for
                // `types`, `functools`, `enum`, ...
                //
                // PRE-EXISTING-ADAPTATION — until pyre grows real
                // traceback objects (`pypy/interpreter/pytraceback.py
                // PyTraceback`), surface a stub `W_InstanceObject`
                // carrying `tb_frame`/`tb_lineno`/`tb_next` so the
                // type-derivation pattern survives.  Explicit
                // `e.__traceback__ = tb` writes already land in the
                // typed `w_traceback` slot and take precedence.
                let stored = unsafe { pyre_object::excobject::w_exception_get_traceback(obj) };
                if !stored.is_null() {
                    return Ok(stored);
                }
                let tb = pyre_object::w_instance_new(crate::typedef::w_object());
                let frame_obj = pyre_object::w_instance_new(crate::typedef::w_object());
                ATTR_TABLE.with(|t| {
                    let mut t = t.borrow_mut();
                    let fd = t.entry(frame_obj as usize).or_default();
                    fd.insert("f_locals".into(), w_dict_new());
                    fd.insert("f_globals".into(), w_dict_new());
                    fd.insert("f_code".into(), w_none());
                    fd.insert("f_lineno".into(), w_int_new(0));
                    let td = t.entry(tb as usize).or_default();
                    td.insert("tb_frame".into(), frame_obj);
                    td.insert("tb_lineno".into(), w_int_new(0));
                    td.insert("tb_next".into(), w_none());
                });
                return Ok(tb);
            }
            "__cause__" => {
                // `interp_exceptions.py:163-164 descr_getcause`.
                let stored = unsafe { pyre_object::excobject::w_exception_get_cause(obj) };
                return Ok(if stored.is_null() { w_none() } else { stored });
            }
            "__context__" => {
                // `interp_exceptions.py:180-181 descr_getcontext`.
                let stored = unsafe { pyre_object::excobject::w_exception_get_context(obj) };
                return Ok(if stored.is_null() { w_none() } else { stored });
            }
            "__suppress_context__" => {
                // `interp_exceptions.py:212-213 descr_getsuppresscontext`
                // returns `space.newbool(self.suppress_context)`.
                // Defaults to False per `:117 W_BaseException` class
                // default; `descr_setcause` flips to True.
                let b = unsafe { pyre_object::excobject::w_exception_get_suppress_context(obj) };
                return Ok(pyre_object::w_bool_from(b));
            }
            "args" => {
                // `pypy/module/exceptions/interp_exceptions.py:153
                // W_BaseException.descr_getargs` returns
                // `space.newtuple(self.args_w)` — a freshly-built
                // tuple per call.  `w_exception_get_args` does the
                // same: it walks the internal list slot and rebuilds
                // a `W_TupleObject`, returning the empty tuple when
                // the slot was never stamped.
                return Ok(unsafe { pyre_object::excobject::w_exception_get_args(obj) });
            }
            "value" => {
                // `pypy/module/exceptions/interp_exceptions.py
                // W_StopIteration.descr_init` stores `value = w_args[0]`,
                // exposed as `fget_value`.  `generator_send_ex` stamps
                // the generator's return value into the exception's
                // `args` tuple; mirror PyPy by returning `args[0]` and
                // defaulting to `None`.  Only StopIteration uses this
                // attribute — other exception kinds keep the regular
                // attribute lookup fall-through.
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if kind == pyre_object::excobject::ExcKind::StopIteration {
                    let args_tuple = unsafe { pyre_object::excobject::w_exception_get_args(obj) };
                    // `w_exception_get_args` always returns a real
                    // tuple — empty tuple when `args_w` was never
                    // stamped — so the null-check above is unneeded.
                    let len = unsafe { pyre_object::w_tuple_len(args_tuple) };
                    if len > 0 {
                        if let Some(v) = unsafe { pyre_object::w_tuple_getitem(args_tuple, 0) } {
                            return Ok(v);
                        }
                    }
                    return Ok(w_none());
                }
            }
            _ => {}
        }
    }
    // __dict__: use getdict() — only returns a dict for hasdict objects,
    // matching PyPy's descriptor-based __dict__ control.
    if name == "__dict__" {
        let w_dict = getdict(obj);
        if !w_dict.is_null() {
            return Ok(w_dict);
        }
    }
    // __class__: read directly from w_class field (the single source of truth).
    // objectobject.py:133-134 descr_get___class__ → space.type(w_obj)
    if name == "__class__" {
        if let Some(tp) = crate::typedef::r#type(obj) {
            return Ok(tp);
        }
    }

    // objspace/std/mapdict.py:826-840 `MapdictDictSupport.getdict` parity.
    //
    // User subclasses of builtin types (`class MyInt(int): ...`) have
    // `hasdict=True` on the subclass type and their instances are still
    // laid out as the builtin (W_IntObject etc.), so `is_instance(obj)`
    // is False and the early descriptor-protocol block at :2858 skipped
    // the instance dict. `setattr` however stores into
    // `INSTANCE_DICT[obj as usize]` via `setdictvalue` → `_obj_setdict`,
    // so the dict is populated but would never be read back.
    //
    // Check the per-instance W_DictObject here (same API PyPy's
    // `descr__getattribute__` uses at descroperation.py:50). This is the
    // second half of the "hasdict instance dict" protocol and must fire
    // before the legacy `ATTR_TABLE` fallback.
    let w_dict = getdict(obj);
    if !w_dict.is_null() {
        if let Some(value) = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) } {
            return Ok(value);
        }
    }

    // Instance attributes from side table (excludes __class__ which lives
    // in the w_class header field, not ATTR_TABLE).
    let found = ATTR_TABLE.with(|table| {
        let table = table.borrow();
        let key = obj as usize;
        table.get(&key).and_then(|dict| dict.get(name).copied())
    });
    if let Some(value) = found {
        return Ok(value);
    }

    // MRO lookup on the object's Python class (w_class) for method resolution.
    let w_class = unsafe { (*obj).w_class };
    if !w_class.is_null() && unsafe { is_type(w_class) } {
        if let Some(method) = unsafe { lookup_in_type_where(w_class, name) } {
            if unsafe {
                crate::is_function(method)
                    && !crate::is_builtin_code(
                        crate::function_get_code(method) as pyre_object::PyObjectRef
                    )
            } {
                return Ok(pyre_object::w_method_new(method, obj, w_class));
            }
            if let Some(result) = unsafe { get(method, obj, w_class)? } {
                return Ok(result);
            }
            return Ok(method);
        }
    }

    unsafe {
        let tp_name = if obj.is_null() {
            "NULL"
        } else {
            (*(*obj).ob_type).name
        };
        Err(PyError::new(
            PyErrorKind::AttributeError,
            format!("'{tp_name}' object has no attribute '{name}'"),
        ))
    }
}

// Builtin type method implementations moved to type_methods.rs
// (PyPy: listobject.py, unicodeobject.py, dictobject.py, tupleobject.py)

/// baseobjspace.py:317-339 `W_Root.int(space)` — the number protocol
/// portion of `space.int(w_obj)`. Look up `__int__`; if absent, fall
/// back to `__index__`. Validate the result is a `W_AbstractIntObject`.
///
/// Note: `__trunc__` is NOT consulted here. `__trunc__` belongs to the
/// `int(...)` builtin path (`intobject.py:989 _new_baseint`), not to
/// `space.int()` / `space.int_w()`.
fn space_int(obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    // baseobjspace.py:319 `w_impl = space.lookup(self, '__int__')`
    let w_impl = unsafe { lookup(obj, "__int__") }
        // baseobjspace.py:321-323 `w_impl = space.lookup(self, '__index__')`
        .or_else(|| unsafe { lookup(obj, "__index__") });
    let Some(method) = w_impl else {
        // baseobjspace.py:323 `self._typed_unwrap_error(space, "integer")`
        return Err(PyError::type_error("expected integer"));
    };
    // baseobjspace.py:324 `w_result = space.get_and_call_function(w_impl, self)`
    let w_result = crate::builtins::call_and_check(method, &[obj])?;
    // baseobjspace.py:326-337 validate that w_result is a W_AbstractIntObject.
    if unsafe { pyre_object::pyobject::is_int_or_long(w_result) } {
        return Ok(w_result);
    }
    // baseobjspace.py:338-339 non-int result → TypeError.
    Err(PyError::type_error("__int__ returned non-int"))
}

/// baseobjspace.py:1811-1824 `ObjSpace.int_w(w_obj,
/// allow_conversion=True)` composed with `baseobjspace.py:279-285
/// W_Root.int_w`:
///
/// ```python
/// # ObjSpace.int_w
/// return w_obj.int_w(self, allow_conversion)
/// # W_Root.int_w
/// w_obj = self
/// if allow_conversion:
///     w_obj = space.int(self)
/// return w_obj._int_w(space)
/// ```
///
/// Fast paths for `W_IntObject` / `W_LongObject` match
/// `intobject.py:558` / `longobject.py` `_int_w`. For non-int/long
/// objects, delegate to `space_int` (the `space.int(self)` protocol)
/// and then re-apply `_int_w`. `allow_conversion=True` is implicit —
/// the `unwrap_spec` call sites that pyre supports all opt in.
///
/// Floats are explicitly rejected by `floatobject.py:177`.
pub fn int_w(obj: PyObjectRef) -> Result<i64, PyError> {
    if obj.is_null() {
        return Err(PyError::type_error("int_w: null object"));
    }
    // floatobject.py:177 `int_w` — floats are explicitly rejected.
    if unsafe { pyre_object::pyobject::is_float(obj) } {
        return Err(PyError::type_error(
            "an integer is required (got type float)",
        ));
    }
    // intobject.py:558 `W_IntObject._int_w` — self.intval. Fast path.
    if unsafe { pyre_object::pyobject::is_int(obj) } {
        return Ok(unsafe { pyre_object::intobject::w_int_get_value(obj) });
    }
    // longobject.py:157 `W_LongObject._int_w` — self.num.toint(), raises
    // OverflowError if the bigint does not fit in a machine word. Fast path.
    if unsafe { pyre_object::pyobject::is_long(obj) } {
        let big = unsafe { pyre_object::longobject::w_long_get_value(obj) };
        return i64::try_from(big)
            .map_err(|_| PyError::overflow_error("int too large to convert to int"));
    }
    // baseobjspace.py:284 `w_obj = space.int(self)` — __int__ or __index__.
    let w_obj = space_int(obj)?;
    // baseobjspace.py:285 `return w_obj._int_w(space)` — re-apply the
    // typed unwrap on the (int/long) result space.int returned.
    if unsafe { pyre_object::pyobject::is_int(w_obj) } {
        return Ok(unsafe { pyre_object::intobject::w_int_get_value(w_obj) });
    }
    if unsafe { pyre_object::pyobject::is_long(w_obj) } {
        let big = unsafe { pyre_object::longobject::w_long_get_value(w_obj) };
        return i64::try_from(big)
            .map_err(|_| PyError::overflow_error("int too large to convert to int"));
    }
    // Unreachable: space_int returns W_AbstractIntObject or errors.
    Err(PyError::type_error("__int__ returned non-int"))
}

/// pypy/interpreter/baseobjspace.py:1957 `gateway_int_w = int_w`.
/// The gateway entry point used by `@unwrap_spec` coercion.
#[inline]
pub fn gateway_int_w(obj: PyObjectRef) -> Result<i64, PyError> {
    int_w(obj)
}

/// pypy/interpreter/baseobjspace.py:1976-1982 `c_int_w(w_obj)`.
///
/// ```python
/// def c_int_w(self, w_obj):
///     value = self.gateway_int_w(w_obj)
///     if value < INT_MIN or value > INT_MAX:
///         raise oefmt(self.w_OverflowError, "expected a 32-bit integer")
///     return value
/// ```
///
/// Used by `@unwrap_spec(name="c_int")` (gateway.py). The only caller
/// today is `sys.setrecursionlimit` (pypy/module/sys/vm.py:63), whose
/// argument is typed as `c_int`; values outside the 32-bit signed
/// range surface as `OverflowError` rather than a silent clamp.
pub fn c_int_w(obj: PyObjectRef) -> Result<i32, PyError> {
    let value = gateway_int_w(obj)?;
    if !(i32::MIN as i64..=i32::MAX as i64).contains(&value) {
        return Err(PyError::overflow_error("expected a 32-bit integer"));
    }
    Ok(value as i32)
}

/// Look up a descriptor on an object's type.
///
/// PyPy equivalent: `space.lookup(w_obj, name)`.
pub unsafe fn lookup(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let w_type = crate::typedef::r#type(obj)?;
    lookup_in_type(w_type, name)
}

/// Look up a name on a type by walking the C3 MRO.
///
/// PyPy equivalent: `space.lookup_in_type(w_type, name)`.
pub unsafe fn lookup_in_type(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    lookup_in_type_where(w_type, name)
}

/// typeobject.py `_lookup_where(self, key)` — linear search through `self.mro_w`.
/// NOTE: PyPy's elidable wrapper (_pure_lookup_where_with_method_cache) takes
/// a version_tag argument to invalidate on type mutation. Until pyre has
/// version tags, this raw lookup must NOT be marked elidable.
unsafe fn lookup_in_type_where(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    if w_type.is_null() || !is_type(w_type) {
        return None;
    }
    // Use cached MRO if available (PyPy: W_TypeObject.mro_w)
    let cached = w_type_get_mro(w_type);
    let mro_owned;
    let mro: &[PyObjectRef] = if !cached.is_null() {
        &*cached
    } else {
        mro_owned = compute_mro(w_type);
        &mro_owned
    };
    for cls in mro {
        if (*cls).is_null() || !is_type(*cls) {
            continue;
        }
        let ns_ptr = w_type_get_dict_ptr(*cls) as *mut crate::DictStorage;
        if !ns_ptr.is_null() {
            let ns = &*ns_ptr;
            if let Some(&value) = ns.get(name) {
                if !value.is_null() {
                    return Some(value);
                }
            }
        }
    }
    None
}

/// Determine what `self` value to bind for a super-resolved attribute.
///
/// Walks the MRO of `self_obj` starting after `super_type`, finds the
/// raw descriptor for `name`, and returns:
///   - PY_NULL       if it is a staticmethod (no binding)
///   - the class obj if it is a classmethod  (bind class)
///   - `self_obj`    otherwise                (bind instance)
pub unsafe fn super_lookup_binding(
    super_type: PyObjectRef,
    self_obj: PyObjectRef,
    name: &str,
) -> PyObjectRef {
    use pyre_object::*;
    let w_obj_type = if is_instance(self_obj) {
        w_instance_get_type(self_obj)
    } else if is_type(self_obj) {
        self_obj
    } else {
        return self_obj;
    };
    let mro_ptr = w_type_get_mro(w_obj_type);
    if !mro_ptr.is_null() {
        let mro = &*mro_ptr;
        let mut past_super = false;
        for &t in mro {
            if std::ptr::eq(t, super_type) {
                past_super = true;
                continue;
            }
            if !past_super {
                continue;
            }
            if is_type(t) {
                if let Some(raw) = lookup_in_type_where(t, name) {
                    if is_staticmethod(raw) {
                        return PY_NULL;
                    }
                    if is_classmethod(raw) {
                        return w_obj_type;
                    }
                    // `__new__` is implicitly static (type.__new__ is a
                    // builtin_function_or_method, not a Python function)
                    if name == "__new__" {
                        return PY_NULL;
                    }
                    return self_obj;
                }
            }
        }
    }
    self_obj
}

/// C3 linearization — PyPy: typeobject.py `compute_default_mro`.
///
/// Computes the Method Resolution Order for a type following the C3
/// algorithm (Python 2.3+). Handles diamond inheritance correctly.
///
/// Public wrapper for use by isinstance and other external callers.
pub unsafe fn compute_default_mro(w_type: PyObjectRef) -> Vec<PyObjectRef> {
    compute_mro(w_type)
}

unsafe fn compute_mro(w_type: PyObjectRef) -> Vec<PyObjectRef> {
    let mut result = vec![w_type];
    let bases = w_type_get_bases(w_type);
    if bases.is_null() || !is_tuple(bases) {
        return result;
    }
    let n = w_tuple_len(bases);
    if n == 0 {
        return result;
    }

    // Build candidate lists: [base.mro() for base in bases] + [list(bases)]
    // Accept metaclass-created classes too, not just `is_type` ones —
    // ABCMeta's `class Rational(Real): pass` still produces a proper
    // W_TypeObject layout, just with a non-default `ob_type`.
    let mut lists: Vec<Vec<PyObjectRef>> = Vec::with_capacity(n + 1);
    for i in 0..n {
        if let Some(base) = w_tuple_getitem(bases, i as i64) {
            if is_type_like_w(base) {
                lists.push(compute_mro(base));
            }
        }
    }
    let mut bases_list = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(base) = w_tuple_getitem(bases, i as i64) {
            bases_list.push(base);
        }
    }
    lists.push(bases_list);

    // C3 merge
    loop {
        // Remove empty lists
        lists.retain(|l| !l.is_empty());
        if lists.is_empty() {
            break;
        }
        // Find a candidate: head of some list that doesn't appear in
        // the tail of any other list.
        let mut found = None;
        for list in &lists {
            let candidate = list[0];
            let in_tail = lists.iter().any(|other| {
                other.len() > 1 && other[1..].iter().any(|&x| std::ptr::eq(x, candidate))
            });
            if !in_tail {
                found = Some(candidate);
                break;
            }
        }
        let Some(next) = found else {
            // C3 inconsistency — fall back to first available
            break;
        };
        result.push(next);
        // Remove next from the head of all lists
        for list in &mut lists {
            if !list.is_empty() && std::ptr::eq(list[0], next) {
                list.remove(0);
            }
        }
    }
    result
}

// ── Descriptor protocol ──────────────────────────────────────────────
// PyPy equivalent: descroperation.py is_data_descr / space.get

/// Check if a descriptor is a data descriptor (has __set__ or __delete__).
///
/// PyPy: descroperation.py `space.is_data_descr(w_descr)`
///
/// In Python, a data descriptor is any object whose type defines __set__
/// or __delete__. For pyre's current object model, we check the ATTR_TABLE
/// and type dict for these names.
/// baseobjspace.py isinstance_w: check if w_obj is instance of w_cls
/// by walking the MRO of type(w_obj) and comparing with w_cls.
pub unsafe fn isinstance_w(w_obj: PyObjectRef, w_cls: PyObjectRef) -> bool {
    let w_obj_type = if is_instance(w_obj) {
        w_instance_get_type(w_obj)
    } else {
        crate::typedef::r#type(w_obj).unwrap_or(pyre_object::PY_NULL)
    };
    if w_obj_type.is_null() {
        return false;
    }
    if std::ptr::eq(w_obj_type, w_cls) {
        return true;
    }
    // Walk MRO
    let mro_ptr = w_type_get_mro(w_obj_type);
    if !mro_ptr.is_null() {
        for &t in &*mro_ptr {
            if std::ptr::eq(t, w_cls) {
                return true;
            }
        }
    }
    false
}

/// pypy/interpreter/baseobjspace.py:419-420 DescrMismatch.
///
/// Construct a DescrMismatch error. Used internally by
/// `descr_self_interp_w`; caught by GetSetProperty.descr_property_get/set/del
/// which then call `descr_call_mismatch` to raise the user-visible TypeError.
#[inline]
pub fn descr_mismatch_error() -> PyError {
    PyError::new(PyErrorKind::DescrMismatch, String::new())
}

/// pypy/interpreter/baseobjspace.py:929-933 ObjSpace.descr_self_interp_w.
///
/// ```python
/// @specialize.arg(1)
/// def descr_self_interp_w(self, RequiredClass, w_obj):
///     if not isinstance(w_obj, RequiredClass):
///         raise DescrMismatch()
///     return w_obj
/// ```
pub fn descr_self_interp_w(
    required_class: PyObjectRef,
    w_obj: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    if required_class.is_null() {
        return Ok(w_obj);
    }
    if w_obj.is_null() {
        return Err(descr_mismatch_error());
    }
    if !unsafe { isinstance_w(w_obj, required_class) } {
        return Err(descr_mismatch_error());
    }
    Ok(w_obj)
}

/// pypy/interpreter/baseobjspace.py:132-138 W_Root.descr_call_mismatch.
///
/// ```python
/// def descr_call_mismatch(self, space, opname, RequiredClass, args):
///     if RequiredClass is None:
///         classname = '?'
///     else:
///         classname = wrappable_class_name(RequiredClass)
///     raise oefmt(space.w_TypeError,
///                 "'%s' object expected, got '%T' instead", classname, self)
/// ```
///
/// `_opname` is preserved for parity with PyPy's signature even though the
/// error message ignores it (PyPy raises the same TypeError regardless of
/// whether the mismatch came through __getattribute__/__setattr__/__delattr__).
pub fn descr_call_mismatch(
    w_obj: PyObjectRef,
    _opname: &str,
    required_class: PyObjectRef,
) -> PyError {
    let classname: String = if required_class.is_null() {
        "?".to_string()
    } else {
        unsafe { pyre_object::w_type_get_name(required_class).to_string() }
    };
    // PyPy `'%T' % obj` formats space.type(obj).getname(space) — the
    // user-visible class name from `w_obj.w_class`, not the underlying
    // ob_type tag. Pyre's `crate::typedef::r#type` walks the same chain.
    let obj_typename: String = if w_obj.is_null() {
        "NoneType".to_string()
    } else {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => unsafe { pyre_object::w_type_get_name(tp).to_string() },
            None => unsafe { (*(*w_obj).ob_type).name.to_string() },
        }
    };
    PyError::type_error(format!(
        "'{}' object expected, got '{}' instead",
        classname, obj_typename
    ))
}

unsafe fn is_data_descr(descr: PyObjectRef) -> bool {
    if descr.is_null() {
        return false;
    }
    // property objects are always data descriptors
    if is_property(descr) {
        return true;
    }
    // typedef.py:492-496 Member is a data descriptor (__get__, __set__, __delete__)
    if pyre_object::is_member(descr) {
        return true;
    }
    // `typedef.py:312-320 GetSetProperty` is a data descriptor by
    // virtue of always exposing `__set__`/`__delete__` slots in its
    // typedef (regardless of whether `fset`/`fdel` are non-null —
    // `descr_property_set` raises `readonly_attribute` for the
    // null-fset case).  Pyre's W_GetSetProperty no longer rides on
    // INSTANCE_TYPE so the generic `is_instance + lookup_in_type`
    // branch below would miss it; short-circuit here.
    if pyre_object::getsetproperty::is_getset_property(descr) {
        return true;
    }
    // Check if the descriptor's class has __set__ or __delete__
    if is_instance(descr) {
        let w_type = w_instance_get_type(descr);
        if !w_type.is_null() && is_type(w_type) {
            return lookup_in_type_where(w_type, "__set__").is_some()
                || lookup_in_type_where(w_type, "__delete__").is_some();
        }
    }
    false
}

/// Call a descriptor's __get__ method.
///
/// PyPy: descroperation.py `space.get(w_descr, w_obj)` →
/// `w_descr.__get__(w_obj, w_type)`
///
/// Returns Some(result) if __get__ was found and called, None otherwise.
/// Call a descriptor's __get__ method.
///
/// PyPy: descroperation.py `space.get(w_descr, w_obj)` →
/// dispatch on descriptor type, then fallback to __get__ MRO lookup.
unsafe fn get(
    descr: PyObjectRef,
    obj: PyObjectRef,
    w_type: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if descr.is_null() {
        return Ok(None);
    }

    // PyPy splits BuiltinFunction from FunctionWithFixedCode at the typedef
    // layer: BuiltinFunction omits __get__, while FunctionWithFixedCode keeps
    // Function.__get__ and binds like a normal method descriptor.
    if crate::is_function(descr) {
        let ob_type = unsafe { (*descr).ob_type };
        if std::ptr::eq(ob_type, &crate::BUILTIN_FUNCTION_TYPE as *const _) {
            return Ok(Some(descr));
        }
        if std::ptr::eq(ob_type, &crate::FUNCTION_TYPE as *const _)
            && crate::is_builtin_code(crate::function_get_code(descr) as pyre_object::PyObjectRef)
        {
            if obj.is_null() || is_none(obj) {
                return Ok(Some(descr));
            }
            return Ok(Some(pyre_object::w_method_new(descr, obj, w_type)));
        }
    }

    // property: PyPy W_Property.get → call fget(obj)
    if is_property(descr) {
        if obj.is_null() {
            return Ok(Some(descr));
        }
        let fget = w_property_get_fget(descr);
        if fget.is_null() || is_none(fget) {
            return Ok(None);
        }
        return Ok(Some(crate::call_function(fget, &[obj])));
    }

    // typedef.py:464-475 Member.descr_member_get:
    //   if space.is_w(w_obj, space.w_None): return self
    //   self.typecheck(space, w_obj)
    //   w_result = w_obj.getslotvalue(self.index)
    //   if w_result is None: raise AttributeError(self.name)
    //   return w_result
    if pyre_object::is_member(descr) {
        // typedef.py:467
        if obj.is_null() || is_none(obj) {
            return Ok(Some(descr));
        }
        // typedef.py:470: self.typecheck(space, w_obj) → TypeError
        let w_cls = pyre_object::w_member_get_cls(descr);
        if !w_cls.is_null() && is_type(w_cls) && !isinstance_w(obj, w_cls) {
            let slot_name = pyre_object::w_member_get_name(descr);
            return Err(crate::PyError::type_error(format!(
                "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                slot_name,
                pyre_object::w_type_get_name(w_cls),
                (*(*obj).ob_type).name,
            )));
        }
        let slot_name = pyre_object::w_member_get_name(descr);
        let found = ATTR_TABLE.with(|table| {
            let table = table.borrow();
            table
                .get(&(obj as usize))
                .and_then(|d| d.get(slot_name).copied())
        });
        // typedef.py:472-474: if w_result is None: raise AttributeError(self.name)
        if found.is_none() {
            return Err(crate::PyError::new(
                crate::PyErrorKind::AttributeError,
                slot_name.to_string(),
            ));
        }
        return Ok(found);
    }

    // staticmethod: PyPy StaticMethod.descr_staticmethod_get → return w_function
    if is_staticmethod(descr) {
        return Ok(Some(w_staticmethod_get_func(descr)));
    }

    // classmethod: PyPy ClassMethod.descr_classmethod_get → func bound to class
    if is_classmethod(descr) {
        let func = w_classmethod_get_func(descr);
        let receiver = if !w_type.is_null() { w_type } else { obj };
        if receiver.is_null() || is_none(receiver) {
            return Ok(Some(func));
        }
        let owner = crate::typedef::r#type(receiver).unwrap_or(PY_NULL);
        return Ok(Some(pyre_object::w_method_new(func, receiver, owner)));
    }

    // General __get__: look up __get__ on the descriptor's own type MRO
    if let Some(descr_type) = crate::typedef::r#type(descr) {
        if let Some(get_fn) = lookup_in_type_where(descr_type, "__get__") {
            if !get_fn.is_null() {
                let result = crate::call::call_function_impl_result(get_fn, &[descr, obj, w_type])?;
                return Ok(Some(result));
            }
        }
    }
    Ok(None)
}

/// Call a descriptor's __set__ method.
///
/// PyPy: descroperation.py `descr__setattr__` →
/// `space.get_and_call_function(w_set, w_descr, w_obj, w_value)`
unsafe fn set(
    descr: PyObjectRef,
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<bool, crate::PyError> {
    if descr.is_null() {
        return Ok(false);
    }

    // property: PyPy W_Property.set → call_function(fset, obj, value).
    // Read-only properties (no `fset` / `@x.setter` never registered)
    // raise AttributeError ("can't set attribute") rather than falling
    // through to the instance dict (`descrobject.c property_descr_set`,
    // mirrored at `pypy/objspace/std/typeobject.py W_Property.descr_set`).
    if is_property(descr) {
        let fset = w_property_get_fset(descr);
        if fset.is_null() || is_none(fset) {
            return Err(crate::PyError::new(
                crate::PyErrorKind::AttributeError,
                "property has no setter".to_string(),
            ));
        }
        crate::call_function(fset, &[obj, value]);
        return Ok(true);
    }

    // typedef.py:477-481 Member.descr_member_set:
    //   self.typecheck(space, w_obj)
    //   w_obj.setslotvalue(self.index, w_value)
    if pyre_object::is_member(descr) {
        // typedef.py:480: self.typecheck(space, w_obj) → TypeError
        let w_cls = pyre_object::w_member_get_cls(descr);
        if !w_cls.is_null() && is_type(w_cls) && !isinstance_w(obj, w_cls) {
            let slot_name = pyre_object::w_member_get_name(descr);
            return Err(crate::PyError::type_error(format!(
                "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                slot_name,
                pyre_object::w_type_get_name(w_cls),
                (*(*obj).ob_type).name,
            )));
        }
        let slot_name = pyre_object::w_member_get_name(descr);
        ATTR_TABLE.with(|table| {
            let mut table = table.borrow_mut();
            table
                .entry(obj as usize)
                .or_default()
                .insert(slot_name.to_string(), value);
        });
        return Ok(true);
    }

    // General __set__: look up on descriptor's type MRO.  GetSetProperty
    // is no longer INSTANCE_TYPE-shaped (it carries `GETSET_DESCRIPTOR
    // _TYPE` so its W_GetSetProperty payload is GC-traced), so resolve
    // the type through `crate::typedef::r#type` rather than the
    // `is_instance` branch.
    let descr_type = if pyre_object::getsetproperty::is_getset_property(descr) {
        crate::typedef::r#type(descr).unwrap_or(std::ptr::null_mut())
    } else if is_instance(descr) {
        w_instance_get_type(descr)
    } else {
        std::ptr::null_mut()
    };
    if !descr_type.is_null() {
        if let Some(set_fn) = lookup_in_type_where(descr_type, "__set__") {
            if !set_fn.is_null() {
                crate::call::call_function_impl_result(set_fn, &[descr, obj, value])?;
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Call a descriptor's __delete__ method.
///
/// descroperation.py `space.delete(w_descr, w_obj)`
unsafe fn delete(descr: PyObjectRef, obj: PyObjectRef) -> Result<(), crate::PyError> {
    // property: call fdel(obj)
    if is_property(descr) {
        let fdel = w_property_get_fdel(descr);
        if fdel.is_null() || is_none(fdel) {
            return Err(crate::PyError::new(
                crate::PyErrorKind::AttributeError,
                "cannot delete attribute".to_string(),
            ));
        }
        crate::call::call_function_impl_result(fdel, &[obj])?;
        return Ok(());
    }
    // typedef.py:483-490 Member.descr_member_del
    if pyre_object::is_member(descr) {
        let w_cls = pyre_object::w_member_get_cls(descr);
        if !w_cls.is_null() && is_type(w_cls) && !isinstance_w(obj, w_cls) {
            let slot_name = pyre_object::w_member_get_name(descr);
            return Err(crate::PyError::type_error(format!(
                "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                slot_name,
                pyre_object::w_type_get_name(w_cls),
                (*(*obj).ob_type).name,
            )));
        }
        let slot_name = pyre_object::w_member_get_name(descr);
        let removed = ATTR_TABLE.with(|table| {
            let mut table = table.borrow_mut();
            table
                .get_mut(&(obj as usize))
                .and_then(|d| d.remove(slot_name))
                .is_some()
        });
        if !removed {
            return Err(crate::PyError::new(
                crate::PyErrorKind::AttributeError,
                slot_name.to_string(),
            ));
        }
        return Ok(());
    }
    // General __delete__: look up on descriptor's type MRO — same
    // shape as `set` above (resolve type through `r#type` so non-
    // INSTANCE_TYPE descriptors like `W_GetSetProperty` are reached).
    let descr_type = if pyre_object::getsetproperty::is_getset_property(descr) {
        crate::typedef::r#type(descr).unwrap_or(std::ptr::null_mut())
    } else if is_instance(descr) {
        w_instance_get_type(descr)
    } else {
        std::ptr::null_mut()
    };
    if !descr_type.is_null() {
        if let Some(del_fn) = lookup_in_type_where(descr_type, "__delete__") {
            if !del_fn.is_null() {
                crate::call::call_function_impl_result(del_fn, &[descr, obj])?;
                return Ok(());
            }
        }
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::AttributeError,
        "cannot delete attribute".to_string(),
    ))
}

/// Set an attribute on an object: `obj.name = value`.
///
/// Stores the attribute in the per-object side table.
/// PyPy: descroperation.py descr__setattr__

/// objectobject.py:137-154 `descr_set___class__(space, w_obj, w_newcls)`.
///
/// Validates and performs `obj.__class__ = newcls`.
fn descr_set___class__(w_obj: PyObjectRef, w_newcls: PyObjectRef) -> PyResult {
    unsafe {
        // objectobject.py:139-142 — w_newcls must be a W_TypeObject
        if !is_type(w_newcls) {
            return Err(crate::PyError::type_error(format!(
                "__class__ must be set to new-style class, not '{}' object",
                (*(*w_newcls).ob_type).name,
            )));
        }
        // objectobject.py:143-145 — w_newcls must be a heap type.
        if !w_type_is_heaptype(w_newcls) {
            return Err(crate::PyError::type_error(
                "__class__ assignment: only for heap types".to_string(),
            ));
        }
        // objectobject.py:146-147 — get the old class
        let w_oldcls = match crate::typedef::r#type(w_obj) {
            Some(c) => c,
            None => {
                return Err(crate::PyError::type_error(
                    "__class__ assignment: cannot determine current class".to_string(),
                ));
            }
        };
        // objectobject.py:148-154 — get_full_instance_layout() must match.
        // typeobject.py:125-129 Layout.expand() compares 5-tuple:
        //   (typedef, newslotnames, base_layout, hasdict, weakrefable)
        let layouts_compatible = pyre_object::typeobject::Layout::expands_equal(
            pyre_object::w_type_get_layout_ptr(w_oldcls),
            pyre_object::w_type_get_hasdict(w_oldcls),
            pyre_object::w_type_get_weakrefable(w_oldcls),
            pyre_object::w_type_get_layout_ptr(w_newcls),
            pyre_object::w_type_get_hasdict(w_newcls),
            pyre_object::w_type_get_weakrefable(w_newcls),
        );
        if !layouts_compatible {
            return Err(crate::PyError::type_error(format!(
                "__class__ assignment: '{}' object layout differs from '{}'",
                pyre_object::w_type_get_name(w_oldcls),
                pyre_object::w_type_get_name(w_newcls),
            )));
        }
        // objectobject.py:150 — w_obj.setclass(space, w_newcls)
        (*w_obj).w_class = w_newcls;
    }
    Ok(w_none())
}

pub fn setattr(obj: PyObjectRef, name: &str, value: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let value = unwrap_cell(value);
    // pypy/module/_weakref/interp__weakref.py:356-394 — proxy delegation.
    // Mirrors the `__setattr__` entry that `register_proxy_typedef_dict`
    // installs on the proxy types: force the receiver, then delegate.
    let obj = crate::module::_weakref::interp_weakref::force(obj)?;
    // Module objects: PyPy `module.py:Module` does not override
    // `descr__setattr__`, so the call falls through to W_Root's
    // `setdictvalue` (`baseobjspace.py:51-56`):
    //
    //     w_dict = self.getdict(space)
    //     if w_dict is not None:
    //         space.setitem_str(w_dict, attr, w_value)
    //
    // `space.setitem_str` is the generic dispatch: for an exact
    // `W_DictMultiObject` it goes direct, but for a dict subclass
    // (`moduledef.py:102-103` user-supplied `__builtins__`) it
    // dispatches through the subclass's `__setitem__`.  pyre's
    // `setitem` mirrors that — `is_dict(obj)` writes
    // entries+storage in lock-step, `is_instance(obj)` looks up
    // `__setitem__` in the MRO and calls it.
    unsafe {
        if is_module(obj) {
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                setitem(w_dict, w_str_new(name), value)?;
                return Ok(w_none());
            }
        }
    }
    // Data descriptor __set__ takes priority (PyPy: descroperation.py
    // descr__setattr__ step 1). PyPy walks `space.type(obj)` regardless of
    // whether `obj` is a Python-level instance, so the lookup must run for
    // every object whose type pyre can resolve — not just W_InstanceObject.
    unsafe {
        let w_type = if is_instance(obj) {
            w_instance_get_type(obj)
        } else if is_type(obj) {
            // For type objects pyre stores attributes in the type's own
            // dict below; the descriptor walk uses the metaclass MRO so
            // metatype-installed setters (e.g. on `type`) still fire.
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        } else {
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        };
        if !w_type.is_null() {
            if let Some(descr) = lookup_in_type_where(w_type, name) {
                if set(descr, obj, value)? {
                    return Ok(w_none());
                }
            }
        }
    }
    // Type objects: store in the type's own namespace (class dict).
    // PyPy: typeobject.py type.__setattr__ → w_type.dict_w[name] = w_value
    unsafe {
        if is_type(obj) {
            let dict_ptr = w_type_get_dict_ptr(obj) as *mut crate::DictStorage;
            if !dict_ptr.is_null() {
                crate::dict_storage_store(&mut *dict_ptr, name, value);
                return Ok(w_none());
            }
        }
    }
    // objectobject.py:137-154 descr_set___class__
    if name == "__class__" {
        return descr_set___class__(obj, value);
    }
    // descroperation.py:108-123 Object.descr__setattr__:
    //
    //     def descr__setattr__(space, w_obj, w_name, w_value):
    //         name = space.text_w(w_name)
    //         w_descr = space.lookup(w_obj, name)
    //         if w_descr is not None:
    //             w_set = space.lookup(w_descr, '__set__')
    //             if w_set is not None:
    //                 return space.get_and_call_function(w_set, w_descr, w_obj, w_value)
    //             if space.lookup(w_descr, '__delete__') is not None:
    //                 raise oefmt(space.w_AttributeError,
    //                             "'%T' object is not a descriptor with set", w_descr)
    //         if w_obj.setdictvalue(space, name, w_value):
    //             return
    //         raiseattrerror(space, w_obj, name, w_descr)
    //
    // The descriptor + type/module short-circuits above already handle the
    // first half of this. What remains is `setdictvalue` + raiseattrerror.
    if setdictvalue(obj, name, value) {
        return Ok(w_none());
    }
    // Property and similar non-instance objects: store via INSTANCE_DICT.
    // property.__doc__ = "..." is common in stdlib (dis.py, etc.)
    unsafe {
        if is_property(obj) || pyre_object::memberobject::is_member(obj) {
            let w_dict = crate::objspace::std::mapdict::_obj_getdict(obj);
            pyre_object::w_dict_setitem_str(w_dict, name, value);
            return Ok(w_none());
        }
    }
    // Exception instances accept arbitrary attribute writes —
    // `pypy/module/exceptions/interp_exceptions.py` declares
    // W_BaseException.typedef with `__dict__ = GetSetProperty(descr_get_dict)`,
    // so user code routinely does `e.foo = bar` (e.g.
    // `argparse.ArgumentTypeError`'s `e.message = ...` pattern).
    // pyre's W_ExceptionObject has no per-instance W_DictObject yet;
    // fall back to ATTR_TABLE which the matching getattr branch reads.
    if unsafe { pyre_object::is_exception(obj) } {
        // `pypy/module/exceptions/interp_exceptions.py:156-157
        // W_BaseException.descr_setargs` →
        //   self.args_w = space.fixedview(w_newargs)
        // `space.fixedview` materialises any iterable into a list of
        // wrapped objects; pyre stores `args_w` as a tuple `PyObjectRef`,
        // so coerce the incoming value into a tuple shape (tuple stays
        // as-is, list wraps into tuple, anything else iterates).
        if name == "args" {
            let coerced = unsafe { coerce_to_list_for_args(value)? };
            unsafe { pyre_object::excobject::w_exception_set_args(obj, coerced) };
            return Ok(w_none());
        }
        // `interp_exceptions.py:165-219` — the four special exception
        // attributes (`__cause__`, `__context__`, `__traceback__`,
        // `__suppress_context__`) are registered as `GetSetProperty`
        // setters on `W_BaseException.typedef` and each validates its
        // input before storing into the matching typed slot
        // (`w_cause`/`w_context`/`w_traceback`/`suppress_context`,
        // line 113-117).  Storage lives on `W_ExceptionObject`
        // directly — no ATTR_TABLE side store for these four names.
        match name {
            "__cause__" => {
                // `interp_exceptions.py:166-174 descr_setcause` — None
                // OR an instance whose type derives from `BaseException`,
                // and always flips `suppress_context` to True.
                if !unsafe { pyre_object::is_none(value) } {
                    let value_type = crate::typedef::r#type(value).unwrap_or(pyre_object::PY_NULL);
                    if value_type.is_null() || !unsafe { exception_is_valid_class_w(value_type) } {
                        return Err(PyError::type_error(
                            "exception cause must be None or derive from BaseException",
                        ));
                    }
                }
                unsafe {
                    pyre_object::excobject::w_exception_set_cause(obj, value);
                    pyre_object::excobject::w_exception_set_suppress_context(obj, true);
                };
                return Ok(w_none());
            }
            "__context__" => {
                // `interp_exceptions.py:183-190 descr_setcontext` — None
                // OR an instance whose type derives from `BaseException`.
                if !unsafe { pyre_object::is_none(value) } {
                    let value_type = crate::typedef::r#type(value).unwrap_or(pyre_object::PY_NULL);
                    if value_type.is_null() || !unsafe { exception_is_valid_class_w(value_type) } {
                        return Err(PyError::type_error(
                            "exception context must be None or derive from BaseException",
                        ));
                    }
                }
                unsafe { pyre_object::excobject::w_exception_set_context(obj, value) };
                return Ok(w_none());
            }
            "__traceback__" => {
                // `interp_exceptions.py:202-206 descr_settraceback` —
                // accept None or PyTraceback only.  Now that real
                // W_PyTraceback exists, narrow the type check to the
                // exact pair PyPy accepts; reject everything else as
                // TypeError per PyPy.
                let accept = unsafe {
                    pyre_object::is_none(value) || crate::pytraceback::is_pytraceback(value)
                };
                if !accept {
                    return Err(PyError::type_error(
                        "__traceback__ must be a traceback or None",
                    ));
                }
                let stored = if unsafe { pyre_object::is_none(value) } {
                    pyre_object::PY_NULL
                } else {
                    value
                };
                unsafe { pyre_object::excobject::w_exception_set_traceback(obj, stored) };
                return Ok(w_none());
            }
            "__suppress_context__" => {
                // `interp_exceptions.py:215-216 descr_setsuppresscontext`
                // — `space.bool_w(w_value)` coerces via `__bool__`.
                let b = is_true(value);
                unsafe { pyre_object::excobject::w_exception_set_suppress_context(obj, b) };
                return Ok(w_none());
            }
            _ => {}
        }
        ATTR_TABLE.with(|table| {
            table
                .borrow_mut()
                .entry(obj as usize)
                .or_default()
                .insert(name.to_string(), value);
        });
        return Ok(w_none());
    }
    Err(raiseattrerror(obj, name))
}

/// `pypy/module/exceptions/interp_exceptions.py:156-157
/// W_BaseException.descr_setargs` parity helper:
///
/// ```python
/// def descr_setargs(self, space, w_newargs):
///     self.args_w = space.fixedview(w_newargs)
/// ```
///
/// `space.fixedview` materialises any iterable into a RPython list
/// of `W_Root`; pyre stores `args_w` as a `W_ListObject` so the
/// getter (`w_exception_get_args`) can build a fresh tuple per read
/// (matching `descr_getargs: return space.newtuple(self.args_w)`).
unsafe fn coerce_to_list_for_args(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if value.is_null() {
        return Ok(w_list_new(vec![]));
    }
    let items = fixedview(value, -1)?;
    Ok(w_list_new(items))
}

/// baseobjspace.py:52-57 W_Root.setdictvalue (default).
///
/// ```python
/// def setdictvalue(self, space, attr, w_value):
///     w_dict = self.getdict(space)
///     if w_dict is not None:
///         space.setitem_str(w_dict, attr, w_value)
///         return True
///     return False
/// ```
fn setdictvalue(obj: PyObjectRef, name: &str, value: PyObjectRef) -> bool {
    let w_dict = getdict(obj);
    if w_dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_setitem_str(w_dict, name, value) };
    true
}

/// descroperation.py:63-69 raiseattrerror.
///
/// ```python
/// def raiseattrerror(space, w_obj, name, w_descr=None):
///     if w_descr is None:
///         raise oefmt(space.w_AttributeError,
///                     "'%T' object has no attribute '%s'", w_obj, name)
///     else:
///         raise oefmt(space.w_AttributeError,
///                     "'%T' object attribute '%s' is read-only", w_obj, name)
/// ```
fn raiseattrerror(obj: PyObjectRef, name: &str) -> PyError {
    let tp_name = unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*obj).ob_type).name.to_string(),
        }
    };
    PyError::new(
        PyErrorKind::AttributeError,
        format!("'{}' object has no attribute '{}'", tp_name, name),
    )
}

/// Delete an attribute: `del obj.name`.
///
/// PyPy: descroperation.py descr__delattr__
pub fn delattr(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = unwrap_cell(obj);
    // pypy/module/_weakref/interp__weakref.py:356-394 — proxy delegation
    // (matches the `__delattr__` entry installed by
    // `register_proxy_typedef_dict`).
    let obj = crate::module::_weakref::interp_weakref::force(obj)?;
    // Module objects: PyPy `module.py:Module` does not override
    // `descr__delattr__`, so the call falls through to W_Root's
    // `deldictvalue` (`baseobjspace.py:58-67`):
    //
    //     w_dict = self.getdict(space)
    //     if w_dict is not None:
    //         try: space.delitem(w_dict, space.newtext(attr))
    //         except KeyError: ...
    //
    // `space.delitem` is the generic dispatch: exact W_DictObject
    // goes direct, dict subclass (moduledef.py:102-103
    // user-supplied `__builtins__`) routes through the subclass's
    // `__delitem__`.  KeyError is swallowed (returning False from
    // `deldictvalue`); pyre falls through to `raiseattrerror` at
    // the end of the function for the same observable behaviour.
    //
    //     def deldictvalue(self, space, attr):
    //         w_dict = self.getdict(space)
    //         if w_dict is not None:
    //             try:
    //                 space.delitem(w_dict, space.newtext(attr))
    //                 return True
    //             except OperationError as ex:
    //                 if not ex.match(space, space.w_KeyError):
    //                     raise
    //         return False
    unsafe {
        if is_module(obj) {
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                match delitem(w_dict, w_str_new(name)) {
                    Ok(()) => return Ok(w_none()),
                    Err(err) if err.kind == crate::PyErrorKind::KeyError => {
                        // descroperation.py descr__delattr__: deldictvalue
                        // returning False raises AttributeError immediately.
                        return Err(raiseattrerror(obj, name));
                    }
                    Err(err) => return Err(err),
                }
            }
        }
    }
    // Type objects: set to PY_NULL in class dict
    // (DictStorage doesn't support removal, null slot acts as deleted)
    unsafe {
        if is_type(obj) {
            let dict_ptr = w_type_get_dict_ptr(obj) as *mut crate::DictStorage;
            if !dict_ptr.is_null() {
                crate::dict_storage_store(&mut *dict_ptr, name, PY_NULL);
                return Ok(w_none());
            }
        }
    }
    // descroperation.py descr__delattr__: data descriptor __delete__ takes
    // priority. PyPy walks `space.type(obj)`, so the lookup must run for
    // any object whose type pyre can resolve — not just W_InstanceObject.
    unsafe {
        let w_type = if is_instance(obj) {
            w_instance_get_type(obj)
        } else if is_type(obj) {
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        } else {
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        };
        if !w_type.is_null() {
            if let Some(descr) = lookup_in_type_where(w_type, name) {
                if is_data_descr(descr) {
                    delete(descr, obj)?;
                    return Ok(w_none());
                }
            }
        }
    }
    // `pypy/module/exceptions/interp_exceptions.py:159-161
    // W_BaseException.descr_delargs` → unconditional TypeError
    // ("args may not be deleted").  Reject `del e.args` before the
    // generic dict/ATTR_TABLE removal path, which would otherwise
    // succeed silently when an entry existed there.
    if unsafe { pyre_object::is_exception(obj) } && name == "args" {
        return Err(PyError::type_error("args may not be deleted"));
    }
    // Instance/general: remove from instance dict, then ATTR_TABLE
    let w_dict = getdict(obj);
    if !w_dict.is_null() {
        let removed = unsafe { pyre_object::w_dict_delitem_str(w_dict, name) };
        if removed {
            return Ok(w_none());
        }
    }
    let removed = ATTR_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        table
            .get_mut(&(obj as usize))
            .and_then(|d| d.remove(name))
            .is_some()
    });
    if removed {
        Ok(w_none())
    } else {
        let tp_name = unsafe { (*(*obj).ob_type).name };
        Err(PyError::new(
            PyErrorKind::AttributeError,
            format!("'{tp_name}' object has no attribute '{name}'"),
        ))
    }
}

/// PyPy: baseobjspace.py `call`.
///
/// Call a Python callable with packed positional arguments and optional kwargs.
pub fn call(
    callable: PyObjectRef,
    w_args: PyObjectRef,
    w_kwds: Option<PyObjectRef>,
) -> PyObjectRef {
    if let Some(w_kwargs) = w_kwds {
        if !w_kwargs.is_null() && !unsafe { is_none(w_kwargs) } {
            panic!("call with kwargs is not yet implemented in pyre");
        }
    }

    let mut args = Vec::new();
    unsafe {
        if is_tuple(w_args) {
            let len = w_tuple_len(w_args);
            args.reserve(len);
            for i in 0..len {
                if let Some(arg) = w_tuple_getitem(w_args, i as i64) {
                    args.push(arg);
                }
            }
        } else if is_list(w_args) {
            let len = w_list_len(w_args);
            args.reserve(len);
            for i in 0..len {
                if let Some(arg) = w_list_getitem(w_args, i as i64) {
                    args.push(arg);
                }
            }
        } else if !w_args.is_null() {
            panic!("call() expects tuple or list positional arguments");
        }
    }
    call_function(callable, &args)
}

/// PyPy: baseobjspace.py `call_obj_args` — add a leading object before args.
pub fn call_obj_args(callable: PyObjectRef, obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    if obj.is_null() {
        return call_function(callable, args);
    }
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(obj);
    call_args.extend_from_slice(args);
    call_function(callable, &call_args)
}

/// PyPy: baseobjspace.py `call_valuestack`.
pub fn call_valuestack(
    callable: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
    methodcall: bool,
) -> PyObjectRef {
    let mut args = Vec::with_capacity(nargs);
    for _ in 0..nargs {
        args.push(frame.pop());
    }
    args.reverse();

    let mut remaining_to_drop = dropvalues.saturating_sub(nargs);

    let null_or_self = if methodcall {
        let value = if remaining_to_drop > 0 {
            remaining_to_drop -= 1;
            Some(frame.pop())
        } else {
            None
        };
        if remaining_to_drop > 0 {
            frame.pop();
            remaining_to_drop -= 1;
        }
        value
    } else {
        if remaining_to_drop > 0 {
            frame.pop();
            remaining_to_drop -= 1;
        }
        None
    };

    for _ in 0..remaining_to_drop {
        frame.pop();
    }

    if let Some(null_or_self) = null_or_self {
        if !null_or_self.is_null() && !unsafe { is_none(null_or_self) } {
            args.insert(0, null_or_self);
        }
    }
    call_function(callable, &args)
}

/// PyPy: baseobjspace.py:1269-1277 `call_args_and_c_profile`.
///
/// ```python
/// def call_args_and_c_profile(self, frame, w_func, args):
///     ec = self.getexecutioncontext()
///     ec.c_call_trace(frame, w_func, args)
///     try:
///         w_res = self.call_args(w_func, args)
///     except OperationError:
///         ec.c_exception_trace(frame, w_func)
///         raise
///     ec.c_return_trace(frame, w_func, args)
///     return w_res
/// ```
///
/// Pyre's `call_function` returns `PyObjectRef` and stashes any error
/// via `set_call_error`; we recover it through `take_call_error` to
/// run the c_exception_trace branch.  Trace-callback errors raised by
/// the c_call/c_return/c_exception events propagate via the same TLS
/// stash so the JIT-side and interpreter-side error paths see them.
///
/// This wrapper is for call sites that already have a positional-only
/// slice.  Call sites that know keyword_names_w / keywords_w must call
/// `call_args_and_c_profile_args` with `Arguments::with_kw`, mirroring
/// pyopcode.py's `CALL_FUNCTION_KW` / `CALL_FUNCTION_EX` construction of
/// a single `Arguments` object before the profiled-builtin branch.
pub fn call_args_and_c_profile(
    frame: &mut crate::pyframe::PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyObjectRef {
    let arguments = crate::argument::Arguments::positional_only(args);
    call_args_and_c_profile_args(frame, callable, &arguments, args)
}

/// `baseobjspace.py:1269-1278 call_args_and_c_profile` with a
/// pre-built `Arguments` instance.
///
/// Step 2 of the Arguments port (continuation of `argument.rs`):
/// callers that have positional and kwargs separated (currently
/// `call::call_with_kwargs` for the builtin path) construct
/// `Arguments::with_kw(pos_args, keyword_names_w, keywords_w)` and
/// route through this helper, instead of wrapping the merged slice
/// as positional-only.  This way `firstarg()` reads `pos_args[0]`
/// rather than surfacing the trailing kwargs dict that pyre's flat
/// call surface otherwise appends.
///
/// `flat_args` is the legacy flat slice (positional + trailing kwargs
/// dict) that `call_function` still expects until the call surface
/// itself learns about Arguments.
pub fn call_args_and_c_profile_args(
    frame: &mut crate::pyframe::PyFrame,
    callable: PyObjectRef,
    arguments: &crate::argument::Arguments,
    flat_args: &[PyObjectRef],
) -> PyObjectRef {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if !ec.is_null() {
        if let Err(err) = unsafe {
            (*ec).c_call_trace(
                frame as *mut crate::pyframe::PyFrame,
                callable,
                Some(arguments),
            )
        } {
            crate::call::set_call_error(err);
            return pyre_object::PY_NULL;
        }
    }
    let w_res = call_function(callable, flat_args);
    if w_res == pyre_object::PY_NULL {
        if !ec.is_null() {
            // baseobjspace.py:1274-1276 — `except OperationError:
            // ec.c_exception_trace(frame, w_func); raise`. The bare
            // `raise` re-raises the active exception, but Python
            // semantics are that an exception raised from inside an
            // `except` block replaces the in-flight one. Pyre's call
            // stash already holds the original OperationError; if
            // c_exception_trace raises, overwrite the stash so the
            // tracer error is what propagates.
            if let Err(trace_err) =
                unsafe { (*ec).c_exception_trace(frame as *mut crate::pyframe::PyFrame, callable) }
            {
                crate::call::set_call_error(trace_err);
            }
        }
        return pyre_object::PY_NULL;
    }
    if !ec.is_null() {
        if let Err(err) = unsafe {
            (*ec).c_return_trace(
                frame as *mut crate::pyframe::PyFrame,
                callable,
                Some(arguments),
            )
        } {
            crate::call::set_call_error(err);
            return pyre_object::PY_NULL;
        }
    }
    w_res
}

/// PyPy: baseobjspace.py `call_method`.
pub fn call_method(obj: PyObjectRef, methname: &str, args: &[PyObjectRef]) -> PyObjectRef {
    let method =
        getattr(obj, methname).unwrap_or_else(|e| panic!("call_method({methname}) failed: {e}"));
    call_function(method, args)
}

/// PyPy: baseobjspace.py `call_function`.
///
/// Dispatches to builtins, user functions, and type objects.
pub fn call_function(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::call::call_function_impl(callable, args)
}

/// PyPy: baseobjspace.py `callable_w`.
pub fn callable_w(obj: PyObjectRef) -> bool {
    unsafe {
        is_function(obj)
            || is_type(obj)
            || (is_instance(obj) && lookup_in_type(w_instance_get_type(obj), "__call__").is_some())
    }
}

/// PyPy: baseobjspace.py `callable`.
pub fn callable(obj: PyObjectRef) -> PyObjectRef {
    if callable_w(obj) {
        w_bool_from(true)
    } else {
        w_bool_from(false)
    }
}

/// PyPy `ObjSpace.call_function_or_identity`.
pub fn call_function_or_identity(obj: PyObjectRef, dunder: &str) -> PyObjectRef {
    unsafe {
        if is_instance(obj) {
            if let Some(method) = lookup(obj, dunder) {
                return call_function(method, &[obj]);
            }
        }
    }
    obj
}

/// PyPy baseobjspace.py equivalent.
pub fn get_printable_location(greenkey: PyObjectRef) -> String {
    format!("unpackiterable [{:?}]", greenkey)
}

/// PyPy baseobjspace.py equivalent.
pub fn wrappable_class_name(class: PyObjectRef) -> String {
    if class.is_null() {
        return "internal subclass".to_string();
    }
    unsafe {
        let type_name = (*(*class).ob_type).name;
        if is_type(class) {
            type_name.to_string()
        } else {
            format!("internal subclass of {type_name}")
        }
    }
}

/// pypy/interpreter/baseobjspace.py:983-998 `unpackiterable`.
///
/// ```python
/// def unpackiterable(self, w_iterable, expected_length=-1):
///     """Unpack an iterable into a real (interpreter-level) list.
///     Raise an OperationError(w_ValueError) if the length is wrong."""
///     w_iterator = self.iter(w_iterable)
///     if expected_length == -1:
///         if self.is_generator(w_iterator):
///             # special hack for speed
///             lst_w = []
///             w_iterator.unpack_into(lst_w)
///             return lst_w
///         return self._unpackiterable_unknown_length(w_iterator, w_iterable)
///     else:
///         lst_w = self._unpackiterable_known_length(w_iterator,
///                                                   expected_length)
///         return lst_w[:]     # make the resulting list resizable
/// ```
///
/// `expected_length = -1` is PyPy's sentinel for "any length".  When
/// the caller supplies a positive expected_length, the length-validation
/// arm at `baseobjspace.py:1031-1053
/// `_unpackiterable_known_length_jitlook` runs and raises ValueError
/// on mismatch (`too many values to unpack` /
/// `not enough values to unpack`).
pub fn unpackiterable(
    w_iterable: PyObjectRef,
    expected_length: isize,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let w_iterator = iter(w_iterable)?;
    if expected_length == -1 {
        // baseobjspace.py:989-993 — generator fast path.  PyPy comments
        // (`generator.py:322 "This is a hack for performance"`) flag this
        // as an optimization, but the structural difference from the
        // generic next-loop is observable: `unpack_into` runs each yield
        // through the same suspended frame without the per-iteration
        // PyTypeObject/__next__ slot lookup, and uses a private
        // `_invoke_execute_frame(space.w_None)` instead of `space.next`.
        // Port both branches.
        if unsafe { pyre_object::generatorobject::is_generator(w_iterator) } {
            let mut lst_w: Vec<PyObjectRef> = Vec::new();
            generator_unpack_into(w_iterator, &mut lst_w)?;
            return Ok(lst_w);
        }
        _unpackiterable_unknown_length(w_iterator, w_iterable)
    } else {
        // baseobjspace.py:996-998 — known-length path with shape validation.
        _unpackiterable_known_length_jitlook(w_iterator, expected_length as usize)
    }
}

/// pypy/interpreter/baseobjspace.py:368-372 `iterator_greenkey`.
///
/// ```python
/// def iterator_greenkey(self, space):
///     """ Return something that can be used as a green key in jit
///     drivers that iterate over self. by default, it's just the type
///     of self, but custom iterators should override it. """
///     return space.type(self)
/// ```
///
/// Default implementation returning `space.type(w_iterable)`.  Pyre's
/// W_Root subclasses don't carry per-type overrides yet, so every
/// caller hits this default — matching PyPy's
/// `baseobjspace.py:2099-2103 ObjSpace.iterator_greenkey` after the
/// trivial `w_iterable.iterator_greenkey(self)` indirection.
pub fn iterator_greenkey(w_iterable: PyObjectRef) -> PyObjectRef {
    if w_iterable.is_null() {
        return pyre_object::PY_NULL;
    }
    crate::typedef::r#type(w_iterable).unwrap_or(pyre_object::PY_NULL)
}

/// pypy/interpreter/baseobjspace.py:1010 `unpackiterable_driver`
/// JitDriver merge-point hint.
///
/// PyPy declares `unpackiterable_driver = JitDriver(greens=['greenkey'],
/// reds='auto', name='unpackiterable')` and calls
/// `unpackiterable_driver.jit_merge_point(greenkey=greenkey)` once per
/// loop turn so the JIT specialises the loop trace per
/// `iterator_greenkey(w_iterator)` value.
///
/// Pyre's metainterp drives compilation from bytecode-level
/// `BC_JIT_MERGE_POINT` opcodes; an in-Rust `_unpackiterable_unknown_length`
/// is residual-call'd from the JIT'd interpreter loop, so the merge-point
/// inside this body is not visible to the live tracer.  The structural
/// port keeps the greenkey computation + the call so the per-greenkey
/// dispatch contract is documented at the call site; the runtime hook
/// is a no-op until the metainterp grows a Rust-callee merge-point
/// observer.
#[inline]
fn unpackiterable_driver_jit_merge_point(_greenkey: PyObjectRef) {
    // No-op: see doc comment above.
}

/// pypy/interpreter/generator.py:317-343 `_create_unpack_into` body.
///
/// ```python
/// def unpack_into(self, results):
///     """This is a hack for performance: runs the generator and
///     collects all produced items in a list."""
///     frame = self.frame
///     if frame is None:    # already finished
///         return
///     pycode = self.pycode
///     while True:
///         jitdriver.jit_merge_point(pycode=pycode)
///         space = self.space
///         try:
///             w_result = self._invoke_execute_frame(space.w_None)
///         except OperationError as e:
///             if not e.match(space, space.w_StopIteration):
///                 raise
///             break
///         if frame.frame_finished_execution:
///             self.frame_is_finished()
///             break
///         results.append(w_result)     # YIELDed
/// ```
///
/// Pyre stores the suspended PyFrame on the W_GeneratorObject as
/// `frame_ptr`; an exhausted generator has either `exhausted=true` or a
/// null frame_ptr.  `_invoke_execute_frame(space.w_None)` corresponds to
/// the frame's own `execute_frame(None, None)` resume — same routing as
/// `generator_send_ex` for the `already_started=true, w_arg=None` path.
fn generator_unpack_into(
    gen_obj: PyObjectRef,
    results: &mut Vec<PyObjectRef>,
) -> Result<(), crate::PyError> {
    use pyre_object::generatorobject::*;
    unsafe {
        // generator.py:325-327 — `frame is None: return`.
        if w_generator_is_running(gen_obj) {
            return Err(PyError::value_error("generator already executing"));
        }
        if w_generator_is_exhausted(gen_obj) {
            return Ok(());
        }
        let frame_ptr = w_generator_get_frame(gen_obj) as *mut crate::pyframe::PyFrame;
        if frame_ptr.is_null() {
            w_generator_set_exhausted(gen_obj);
            return Ok(());
        }
        let frame = &mut *frame_ptr;
        // generator.py:328 `pycode = self.pycode` — pyre stashes pycode on
        // the suspended frame; expose it as the JitDriver greenkey.
        let pycode = frame.pycode as PyObjectRef;
        loop {
            // generator.py:330 `jitdriver.jit_merge_point(pycode=pycode)`.
            unpackiterable_driver_jit_merge_point(pycode);
            // generator.py:331 `space = self.space`.
            // generator.py:332-336 `try: w_result =
            //   self._invoke_execute_frame(space.w_None)`.
            //
            // `_invoke_execute_frame(w_arg_or_err)` calls
            // `frame.execute_frame(w_arg_or_err)` (generator.py:131),
            // which feeds `w_arg_or_err` to `resume_execute_frame` —
            // pushing it onto the YIELD result slot.  unpack_into
            // always passes `space.w_None`, both for the never-started
            // case (frame.last_instr == -1: PyPy
            // `resume_execute_frame` skips the push and returns
            // `r_uint(0)`) and for every subsequent resume.  Pyre's
            // earlier `frame.execute_frame(None, None)` skipped the
            // push entirely, so `yield`-expressions that bind the
            // resume value (e.g. `x = yield`) would observe stale
            // stack on the second iteration.
            w_generator_set_started(gen_obj);
            w_generator_set_running(gen_obj, true);
            let result = frame.execute_frame(Some(pyre_object::w_none()), None);
            w_generator_set_running(gen_obj, false);
            match result {
                // generator.py:132-138 `_invoke_execute_frame`'s
                // `finally: self.frame_is_finished()` runs before the
                // OperationError reaches the unpack_into try/except,
                // so by the time PyPy's `if e.match(StopIteration):
                // break` fires the generator is already marked
                // finished.  Pyre's inline `frame.execute_frame` path
                // skips that finally block, so mirror it explicitly.
                Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
                    w_generator_set_exhausted(gen_obj);
                    break;
                }
                Err(e) => {
                    w_generator_set_exhausted(gen_obj);
                    return Err(e);
                }
                Ok(w_result) => {
                    // generator.py:339-341 — frame finished ⇒ RETURNed,
                    // mark exhausted and stop without appending.
                    if frame.frame_finished_execution {
                        w_generator_set_exhausted(gen_obj);
                        break;
                    }
                    // generator.py:342 `results.append(w_result)`.
                    results.push(w_result);
                }
            }
        }
        Ok(())
    }
}

/// pypy/interpreter/baseobjspace.py:1000-1021
/// `_unpackiterable_unknown_length`.
///
/// ```python
/// def _unpackiterable_unknown_length(self, w_iterator, w_iterable):
///     try:
///         items = newlist_hint(self.length_hint(w_iterable, 0))
///     except MemoryError:
///         items = []
///     greenkey = self.iterator_greenkey(w_iterator)
///     while True:
///         unpackiterable_driver.jit_merge_point(greenkey=greenkey)
///         try:
///             w_item = self.next(w_iterator)
///         except OperationError as e:
///             if not e.match(self, self.w_StopIteration):
///                 raise
///             break
///         items.append(w_item)
///     return items
/// ```
fn _unpackiterable_unknown_length(
    w_iterator: PyObjectRef,
    w_iterable: PyObjectRef,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    // baseobjspace.py:1005-1008 — `try: items = newlist_hint(length_hint(...))
    // except MemoryError: items = []`.  Mirror with try_reserve_exact so a
    // hostile / huge `__length_hint__` does not turn into a Rust panic
    // (Vec::with_capacity aborts on capacity overflow).
    let hint = length_hint(w_iterable, 0)?;
    let mut items: Vec<PyObjectRef> = Vec::new();
    if hint > 0 {
        let _ = items.try_reserve_exact(hint as usize);
    }
    // baseobjspace.py:1010 `greenkey = self.iterator_greenkey(w_iterator)`.
    let greenkey = iterator_greenkey(w_iterator);
    loop {
        // baseobjspace.py:1012
        // `unpackiterable_driver.jit_merge_point(greenkey=greenkey)`.
        unpackiterable_driver_jit_merge_point(greenkey);
        match next(w_iterator) {
            Ok(w_item) => items.push(w_item),
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(items)
}

/// pypy/interpreter/baseobjspace.py:1080-1108 `length_hint`.
///
/// Returns the length of an object, consulting its `__length_hint__`
/// method if necessary.  Errors mirror the upstream contract:
/// `len_w`'s TypeError / AttributeError are absorbed; an
/// `__length_hint__` that raises TypeError / AttributeError returns
/// `default`; a NotImplemented return also yields `default`; a
/// negative return raises ValueError "__length_hint__() should return
/// >= 0"; any other exception propagates.
pub fn length_hint(w_obj: PyObjectRef, default: i64) -> Result<i64, crate::PyError> {
    match len_w(w_obj) {
        Ok(n) => return Ok(n),
        Err(e)
            if e.kind == crate::PyErrorKind::TypeError
                || e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    // baseobjspace.py:1093 `w_descr = space.lookup(w_obj, '__length_hint__')`
    // — generic class-MRO lookup, not instance-restricted.
    let w_descr = match unsafe { lookup(w_obj, "__length_hint__") } {
        Some(descr) => descr,
        None => return Ok(default),
    };
    // baseobjspace.py:1095 `space.get_and_call_function(w_descr, w_obj)` —
    // pyre's `call_function_impl_result` returns a Result directly,
    // matching the upstream raise/return discipline without going through
    // the legacy `take_call_error` pending-error stash.
    let w_hint = match crate::call::call_function_impl_result(w_descr, &[w_obj]) {
        Ok(v) => v,
        Err(err) => {
            if err.kind == crate::PyErrorKind::TypeError
                || err.kind == crate::PyErrorKind::AttributeError
            {
                return Ok(default);
            }
            return Err(err);
        }
    };
    if is_w(w_hint, pyre_object::noneobject::w_not_implemented()) {
        return Ok(default);
    }
    let hint = int_w(w_hint)?;
    if hint < 0 {
        return Err(crate::PyError::value_error(
            "__length_hint__() should return >= 0",
        ));
    }
    Ok(hint)
}

/// pypy/objspace/descroperation.py:310-317 `_check_len_result`.
///
/// ```python
/// def _check_len_result(space, w_int):
///     # Will complain if result is too big.
///     assert space.isinstance_w(w_int, space.w_int)
///     if space.is_true(space.lt(w_int, space.newint(0))):
///         raise oefmt(space.w_ValueError, "__len__() should return >= 0")
///     result = space.getindex_w(w_int, space.w_OverflowError)
///     assert result >= 0
///     return result
/// ```
///
/// `int_w` already mirrors `getindex_w(w_int, w_OverflowError)` for the
/// already-int caller contract here: long values that do not fit `i64`
/// raise `OverflowError` ("int too large to convert to int") via
/// `intobject.py:558` / `longobject.py` `_int_w`.
fn _check_len_result(w_int: PyObjectRef) -> Result<i64, crate::PyError> {
    let n = int_w(w_int)?;
    if n < 0 {
        return Err(crate::PyError::value_error("__len__() should return >= 0"));
    }
    Ok(n)
}

/// pypy/objspace/descroperation.py:300-302 `len_w`.
///
/// ```python
/// def len_w(space, w_obj):
///     w_res = space._len(w_obj)
///     return space._check_len_result(space.index(w_res))
/// ```
///
/// pyre's `len()` covers `_len`; the result is then funnelled through
/// `space.index` (descroperation.py:599 `_index` + line 622 `index`)
/// before `_check_len_result` so `__index__` is consulted but `__int__`
/// is NOT — matching PyPy's stricter contract.
pub fn len_w(w_obj: PyObjectRef) -> Result<i64, crate::PyError> {
    let w_res = len(w_obj)?;
    let w_index = space_index(w_res)?;
    _check_len_result(w_index)
}

/// pypy/objspace/descroperation.py:599-620 `_index` + line 622-627 `index`.
///
/// ```python
/// def _index(space, w_obj):
///     if space.isinstance_w(w_obj, space.w_int):
///         return w_obj
///     w_impl = space.lookup(w_obj, '__index__')
///     if w_impl is None:
///         raise oefmt(space.w_TypeError,
///                     "'%T' object cannot be interpreted as an integer", w_obj)
///     w_result = space.get_and_call_function(w_impl, w_obj)
///     if space.is_w(space.type(w_result), space.w_int):
///         return w_result
///     if not space.isinstance_w(w_result, space.w_int):
///         raise oefmt(space.w_TypeError,
///                 "__index__ returned non-int (type %T)", w_result)
///     ...  # subclass-of-int deprecation warning, then return
///     return w_result
/// ```
///
/// `space.index` (line 622) wraps `_index` and additionally re-wraps
/// strict subclass-of-int results into a fresh `W_IntObject` /
/// `W_LongObject`.  Pyre's `int`/`long` are leaf types so the wrap is a
/// no-op; the body below is `_index` line-for-line.
pub fn space_index(obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if obj.is_null() {
        return Err(PyError::type_error("space.index: null object"));
    }
    if unsafe { pyre_object::pyobject::is_int_or_long(obj) } {
        return Ok(obj);
    }
    let Some(method) = (unsafe { lookup(obj, "__index__") }) else {
        return Err(PyError::type_error(format!(
            "'{}' object cannot be interpreted as an integer",
            unsafe { (*(*obj).ob_type).name },
        )));
    };
    let w_result = crate::builtins::call_and_check(method, &[obj])?;
    if unsafe { pyre_object::pyobject::is_int_or_long(w_result) } {
        return Ok(w_result);
    }
    Err(PyError::type_error(format!(
        "__index__ returned non-int (type {})",
        unsafe { (*(*w_result).ob_type).name },
    )))
}

/// `pyframe.py:115-116 self.builtin = space.builtin.pick_builtin(
/// w_globals)`.  Body ports `pypy/module/__builtin__/moduledef.py:89-109
/// pick_builtin`:
///   1. `space.getitem(w_globals, '__builtins__')` (`KeyError` ⇒ default)
///   2. recognise `Module` ⇒ return that Module
///   3. recognise dict (incl. dict subclass) ⇒ wrap as
///      `module.Module(space, None, w_builtin)` (a fresh Module per
///      call, with `module.w_dict = w_builtin`).
///   4. absent / not Module-or-dict ⇒ build a default empty Module
///      with only `None=w_None` defined — matches `moduledef.py:106-108`
///      `builtin = module.Module(space, None); space.setitem(builtin
///      .w_dict, 'None', w_None); return builtin`.
pub fn pick_builtin(
    w_globals: *mut crate::DictStorage,
    exec_ctx: *const crate::PyExecutionContext,
) -> PyObjectRef {
    if !w_globals.is_null() {
        if let Some(w_builtin) = crate::dict_storage_get(unsafe { &*w_globals }, "__builtins__") {
            if !w_builtin.is_null() {
                // moduledef.py:100-101 `if w_builtin is space.builtin: return
                // space.builtin` — Module identity short-circuit.
                if !exec_ctx.is_null() {
                    let space_builtin = unsafe { (*exec_ctx).get_builtin() };
                    if !space_builtin.is_null() && std::ptr::eq(w_builtin, space_builtin) {
                        return w_builtin;
                    }
                }
                // moduledef.py:104 `isinstance(w_builtin, module.Module)`.
                if unsafe { pyre_object::is_module(w_builtin) } {
                    return w_builtin;
                }
                // moduledef.py:102-103 `space.isinstance_w(w_builtin, w_dict)`.
                // PyPy: `return module.Module(space, None, w_builtin)` —
                // a Module wrapping the caller's dict.  `LOAD_GLOBAL`
                // falls through to `space.finditem_str(w_module.w_dict,
                // name)`, dispatching through any dict subclass
                // `__getitem__` override.
                let backing = crate::type_methods::resolve_dict_backing(w_builtin);
                if !backing.is_null() {
                    return pyre_object::w_module_new_aliasing_dict(
                        "",
                        std::ptr::null_mut(),
                        w_builtin,
                    );
                }
                // Fall through — `__builtins__` is not Module/dict (e.g.
                // `42`, a list, ...).  PyPy moduledef.py:106-108 builds
                // the default empty Module here.
            }
        }
    }
    // moduledef.py:106-108 default — anonymous Module with only
    // `None=w_None`.  This is reached when (a) `w_globals` is null,
    // (b) `__builtins__` is absent from globals, or (c) `__builtins__`
    // is not Module/dict.
    build_default_pick_builtin_module()
}

/// Allocate the `moduledef.py:106-108` default Module — empty backing
/// storage with `None=w_None`, anonymous (PyPy passes `name=None` to
/// `Module.__init__`; pyre's `w_module_new` requires a `&str` so use
/// the empty string as the anonymous-name sentinel).
fn build_default_pick_builtin_module() -> PyObjectRef {
    let mut ns = Box::new(crate::DictStorage::new());
    crate::dict_storage_store(&mut ns, "None", unsafe { pyre_object::w_none() });
    ns.fix_ptr();
    let storage: *mut crate::DictStorage = Box::into_raw(ns);
    pyre_object::w_module_new("", storage as *mut u8)
}

/// pypy/interpreter/baseobjspace.py:1031-1053
/// `_unpackiterable_known_length_jitlook`.
///
/// ```python
/// @jit.unroll_safe
/// def _unpackiterable_known_length_jitlook(self, w_iterator, expected_length):
///     items = [None] * expected_length
///     idx = 0
///     while True:
///         try:
///             w_item = self.next(w_iterator)
///         except OperationError as e:
///             if not e.match(self, self.w_StopIteration):
///                 raise
///             break
///         if idx == expected_length:
///             raise oefmt(self.w_ValueError,
///                         "too many values to unpack (expected %d)",
///                         expected_length)
///         items[idx] = w_item
///         idx += 1
///     if idx < expected_length:
///         raise oefmt(self.w_ValueError,
///                     "not enough values to unpack (expected %d, got %d)",
///                     expected_length, idx)
///     return items
/// ```
fn _unpackiterable_known_length_jitlook(
    w_iterator: PyObjectRef,
    expected_length: usize,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let mut items: Vec<PyObjectRef> = Vec::with_capacity(expected_length);
    loop {
        match next(w_iterator) {
            Ok(w_item) => {
                if items.len() == expected_length {
                    return Err(crate::PyError::value_error(format!(
                        "too many values to unpack (expected {expected_length})",
                    )));
                }
                items.push(w_item);
            }
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    if items.len() < expected_length {
        return Err(crate::PyError::value_error(format!(
            "not enough values to unpack (expected {expected_length}, got {got})",
            got = items.len(),
        )));
    }
    Ok(items)
}

/// pypy/interpreter/baseobjspace.py:1159-1163 base default + the
/// `StdObjSpace` override at `pypy/objspace/std/objspace.py:609-617`.
///
/// ```python
/// # baseobjspace.py:1159-1163 (base default)
/// def view_as_kwargs(self, w_dict):
///     """ if w_dict is a kwargs-dict, return two lists, one of unwrapped
///     strings and one of wrapped values. otherwise return (None, None)
///     """
///     return (None, None)
///
/// # objspace.py:609-617 (StdObjSpace override)
/// def view_as_kwargs(self, w_dict):
///     # ... it never fails for dict subclasses; this emulates CPython's
///     # behavior which often won't call custom __iter__() or keys()
///     # methods in dict subclasses.
///     if isinstance(w_dict, W_DictObject):
///         return w_dict.view_as_kwargs()
///     return (None, None)
///
/// # dictmultiobject.py:307-310 (W_DictObject.view_as_kwargs)
/// def view_as_kwargs(self):
///     if not self.user_overridden_class:
///         return self.get_strategy().view_as_kwargs(self)
///     return None, None
///
/// # dictmultiobject.py:1325-1334 (kwargs strategy)
/// def view_as_kwargs(self, w_dict):
///     d = self.unerase(w_dict.dstorage)
///     l = len(d)
///     keys, values = [None] * l, [None] * l
///     i = 0
///     for w_key, val in d.iteritems():
///         keys[i] = w_key
///         values[i] = val
///         i += 1
///     return keys, values
/// ```
///
/// Pyre's `W_DictObject` does not carry the multi-strategy dispatch
/// (Object/Bytes/Int/Unicode/Kwargs), so the strategy-level
/// `view_as_kwargs` is open-coded here: walk the entries vector and
/// require every key to be a unicode string for the fast path to
/// apply, otherwise return `(None, None)` so callers fall through to
/// the slow `keys()` iteration arm at `argument.py:121-150`.
///
/// `user_overridden_class` (typeobject.py term for "type is exact
/// dict, not a subclass") corresponds to pyre's `is_dict(w_dict)` —
/// pyre dict subclasses live as `W_InstanceObject` with a backing
/// dict (`typedef.rs:820 dict_descr_new`), so an exact-type check on
/// the wrapper rules out user subclasses.  Both tuple slots are
/// `Option` so callers distinguish "no fast path" (None) from "fast
/// path with zero entries" (Some(empty)).
pub fn view_as_kwargs(w_dict: PyObjectRef) -> (Option<Vec<PyObjectRef>>, Option<Vec<PyObjectRef>>) {
    if w_dict.is_null() || !unsafe { pyre_object::is_dict(w_dict) } {
        return (None, None);
    }
    // dictmultiobject.py:308 — `if not self.user_overridden_class`.
    // pyre's `is_dict` already screens out subclass instances (those
    // are `W_InstanceObject`), so any wrapper that reached this point
    // is the exact-type case.  The strategy-level fast path (kwargs
    // strategy at dictmultiobject.py:1325) only succeeds when every
    // key is a unicode string — the ObjectStrategy and other
    // non-string strategies return `(None, None)` from their base
    // implementation.
    let entries = unsafe { pyre_object::dictobject::w_dict_items(w_dict) };
    let mut keys: Vec<PyObjectRef> = Vec::with_capacity(entries.len());
    let mut values: Vec<PyObjectRef> = Vec::with_capacity(entries.len());
    for (w_key, w_val) in entries {
        if !unsafe { pyre_object::is_str(w_key) } {
            return (None, None);
        }
        keys.push(w_key);
        values.push(w_val);
    }
    (Some(keys), Some(values))
}

/// pypy/interpreter/baseobjspace.py:2105-2140 `object_functionstr`.
///
/// Full 4-branch port:
///
/// ```python
/// def object_functionstr(self, w_function):
///     from pypy.interpreter.function import Function, _Method
///     if isinstance(w_function, Function):
///         qualname = w_function.qualname
///         w_module = w_function.fget___module__(self)
///         if not self.is_w(w_module, self.w_None):
///             try:
///                 module = self.text_w(w_module)
///                 if module and module != 'builtins':
///                     return module + '.' + qualname + '()'
///             except OperationError:
///                 pass
///         return qualname + '()'
///     if isinstance(w_function, _Method):
///         return self.object_functionstr(w_function.w_function)
///     w_qualname = self.findattr(w_function, self.newtext('__qualname__'))
///     if w_qualname is not None:
///         try:
///             qualname = self.text_w(w_qualname)
///             w_module = self.findattr(w_function, self.newtext('__module__'))
///             if w_module is not None and not self.is_w(w_module, self.w_None):
///                 module = self.text_w(w_module)
///                 if module and module != 'builtins':
///                     return module + '.' + qualname + '()'
///             return qualname + '()'
///         except OperationError:
///             pass
///     try:
///         return self.text_w(self.str(w_function))
///     except OperationError:
///         return self.type(w_function).getname(self) + ' object'
/// ```
///
/// `object_functionstr` uses small private helpers instead of the
/// public `findattr` / `display::py_str` shortcuts because PyPy's
/// control flow is intentionally narrow here:
///
/// - `findattr` suppresses ordinary `OperationError` and returns
///   `None`, but **re-raises** SystemExit / KeyboardInterrupt
///   (`baseobjspace.py:881-884 if e.async(self): raise`).
/// - the final fallback calls `space.str(w_function)` once, then
///   `space.text_w(...)`; it does not try `repr()` after a failing or
///   non-string `__str__`.
///
/// The async-propagation contract is preserved: the `__qualname__`
/// findattr lives outside the inner try, so any async error there
/// surfaces as `Err(PyError)` to `raise_type_error`, which then
/// returns the async error in place of the TypeError prefix.  The
/// `__module__` findattr and the `text_w(...)` calls live inside the
/// PyPy try/except OperationError block — async OR ordinary errors
/// there fall through to the `str(w_function)` fallback, matching
/// PyPy's `except OperationError: pass`.
///
/// `function.py:53` initialises `self.qualname = qualname or self.name`,
/// so `w_function.qualname` returns the dotted form (e.g.
/// `Class.method`) for nested defs and the bare identifier for free
/// functions.  Pyre's `Function` does not carry the field directly;
/// `crate::function::function_get_qualname` reproduces the same
/// precedence (set-attr override → `code.qualname` → `function.name`).
pub fn object_functionstr(w_function: PyObjectRef) -> Result<String, crate::PyError> {
    // baseobjspace.py:2108-2120 — Function fast path (also covers
    // `FunctionWithFixedCode` and `BuiltinFunction`, both subclasses
    // of `Function` per function.py:783,786).  Pyre's `is_function`
    // unifies all three over `FUNCTION_TYPE` + `BUILTIN_FUNCTION_TYPE`.
    if !w_function.is_null() && unsafe { crate::function::is_function(w_function) } {
        // function.py:2108 `qualname = w_function.qualname` — match
        // PyPy's stored `qualname` field via the helper that walks
        // ATTR_TABLE → `code.qualname` → `name`.
        let qualname = unsafe { crate::function::function_get_qualname(w_function) };
        let w_module = unsafe { crate::function::fget___module__(w_function) };
        if !is_w(w_module, w_none()) && unsafe { pyre_object::is_str(w_module) } {
            let module = unsafe { pyre_object::w_str_get_value(w_module) };
            if !module.is_empty() && module != "builtins" {
                return Ok(format!("{module}.{qualname}()"));
            }
        }
        return Ok(format!("{qualname}()"));
    }
    // baseobjspace.py:2121-2122 — `_Method` recursive fast path:
    // unwrap to `w_function.w_function` and recurse.
    if !w_function.is_null() && unsafe { pyre_object::methodobject::is_method(w_function) } {
        let inner = unsafe { pyre_object::methodobject::w_method_get_func(w_function) };
        return object_functionstr(inner);
    }
    // baseobjspace.py:2123 — `w_qualname = self.findattr(...)`.  This
    // findattr lives **outside** the inner try/except, so an async
    // exception (SystemExit/KeyboardInterrupt) here is propagated to
    // the caller via `Err(...)` matching `findattr`'s `e.async(self):
    // raise` re-raise (`baseobjspace.py:881-884`).
    let w_qualname_opt = object_functionstr_findattr(w_function, "__qualname__")?;
    // baseobjspace.py:2125-2135 — `try/except OperationError: pass`.
    // Every fault inside this block (text_w(qualname), findattr(module),
    // text_w(module)) must fall through to the `str(w_function)`
    // fallback rather than propagate.  In particular the second
    // `findattr(__module__)` is **inside** the try, so async errors
    // there are also suppressed — matches PyPy literally.
    'qualname: {
        let Some(w_qualname) = w_qualname_opt else {
            break 'qualname;
        };
        let Ok(qualname) = object_functionstr_text_w(w_qualname) else {
            break 'qualname;
        };
        let w_module = match object_functionstr_findattr(w_function, "__module__") {
            Ok(opt) => opt,
            // try/except OperationError: pass — async findattr suppressed too.
            Err(_) => break 'qualname,
        };
        match w_module {
            // No `__module__` or `__module__ is None`: bare `qualname()`.
            None => return Ok(format!("{qualname}()")),
            Some(w_module) if is_w(w_module, w_none()) => return Ok(format!("{qualname}()")),
            Some(w_module) => {
                // text_w(w_module) — non-string raises in PyPy → except →
                // fall through (do NOT return `qualname()` here, which
                // would mask the OperationError).
                let Ok(module) = object_functionstr_text_w(w_module) else {
                    break 'qualname;
                };
                if !module.is_empty() && module != "builtins" {
                    return Ok(format!("{module}.{qualname}()"));
                }
                // module empty or 'builtins': bare qualname().
                return Ok(format!("{qualname}()"));
            }
        }
    }
    // baseobjspace.py:2137-2140 — `text_w(str(w_function))` fallback,
    // else `type(w_function).getname() + ' object'`.  Both calls live
    // in `try/except OperationError: pass`, so any error (including
    // async) here is swallowed in PyPy — keep the same shape.  PyPy
    // calls `space.str(w_function)`, which dispatches to `__str__`
    // ALONE via `descroperation.str` (it does NOT fall back to
    // `__repr__` — that would require `space.repr(...)`).  Routing
    // through `display::py_str` would mask a failing/non-string
    // `__str__` by calling `__repr__`, producing a different message
    // than upstream.
    if let Ok(w_s) = object_functionstr_str(w_function)
        && unsafe { pyre_object::is_str(w_s) }
    {
        return Ok(unsafe { pyre_object::w_str_get_value(w_s).to_string() });
    }
    Ok(format!(
        "{} object",
        object_functionstr_type_name(w_function)
    ))
}

/// `space.str(w_obj)` — `__str__`-only fast path for
/// `object_functionstr`'s final fallback.
///
/// `pypy/objspace/descroperation.py str(self, space, w_obj)` does
/// `lookup(w_obj, '__str__')` then `space.get_and_call_function(...)`.
/// `__repr__` is never tried here — that would be `space.repr(...)`.
/// Returning `Err` for any of: missing `__str__` slot, descriptor
/// invocation failure, non-string return — caller suppresses to the
/// `<Type> object` fallback per PyPy's `except OperationError`.
fn object_functionstr_str(w_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if w_obj.is_null() {
        return Err(crate::PyError::type_error("NULL object"));
    }
    unsafe {
        if pyre_object::is_str(w_obj) {
            return Ok(w_obj);
        }
        let Some(w_descr) = lookup(w_obj, "__str__") else {
            return Err(crate::PyError::type_error(format!(
                "'{}' object has no __str__",
                object_functionstr_type_name(w_obj),
            )));
        };
        crate::call::call_function_impl_result(w_descr, &[w_obj])
    }
}

/// `object_functionstr`-local version of
/// `baseobjspace.py:878-885 findattr`.
///
/// ```python
/// def findattr(self, w_object, w_name):
///     try:
///         return self.getattr(w_object, w_name)
///     except OperationError as e:
///         # a PyPy extension: let SystemExit and KeyboardInterrupt go through
///         if e.async(self):
///             raise
///         return None
/// ```
///
/// `Err(_)` carries the propagated async exception
/// (`PyErrorKind::SystemExit`, mirroring `OperationError.async`'s
/// SystemExit/KeyboardInterrupt arm — `error.py:62-65`).  Pyre's
/// `PyError` does not yet carry a `KeyboardInterrupt` kind; SystemExit
/// alone covers the propagation contract for the cases pyre raises
/// today.  Ordinary `OperationError`s (AttributeError, NameError,
/// TypeError from descriptors) collapse to `Ok(None)`, matching
/// PyPy's `return None` arm.
fn object_functionstr_findattr(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if unsafe { is_none(obj) } {
        return Ok(None);
    }
    match getattr(obj, name) {
        Ok(value) => Ok(Some(value)),
        Err(e) if e.kind == crate::PyErrorKind::SystemExit => Err(e),
        Err(_) => Ok(None),
    }
}

/// `space.text_w(w_obj)` for the `object_functionstr` try blocks.
fn object_functionstr_text_w(w_obj: PyObjectRef) -> Result<String, crate::PyError> {
    unsafe {
        if pyre_object::is_str(w_obj) {
            Ok(pyre_object::w_str_get_value(w_obj).to_string())
        } else {
            Err(crate::PyError::type_error(format!(
                "expected str, got {} object",
                object_functionstr_type_name(w_obj),
            )))
        }
    }
}

fn object_functionstr_type_name(w_obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "object".to_string(),
        }
    }
}

/// pypy/objspace/descroperation.py:319-326 `is_iterable`.
///
/// ```python
/// def is_iterable(space, w_obj):
///     w_descr = space.lookup(w_obj, '__iter__')
///     if w_descr is None:
///         if space.type(w_obj).flag_map_or_seq != 'M':
///             w_descr = space.lookup(w_obj, '__getitem__')
///         if w_descr is None:
///             return False
///     return True
/// ```
///
/// PyPy's `space.lookup` walks the type's MRO without firing
/// descriptors or `__getattr__`; pyre's `lookup` (`baseobjspace.rs:3945`)
/// has the same MRO-only semantics.  Using `findattr` here would run
/// the descriptor protocol and could surface false positives or
/// side effects in the *args error path, so we route through `lookup`
/// to match upstream exactly.
///
/// `flag_map_or_seq` is read off the resolved `space.type(w_obj)` and
/// gates the `__getitem__` fallback exactly as PyPy does — when
/// `'M'` (mapping) the fallback is skipped so a mapping-shaped type
/// without `__iter__` is reported as not iterable.  Pyre reads the
/// marker from `W_TypeObject`, the same level where PyPy stores
/// `flag_map_or_seq`.
///
/// Builtin shortcuts list/tuple/str/bytes/dict/set/iter/generator/
/// itertools mirror `iter()`'s direct-type arms at
/// `baseobjspace.rs:5158-5208`.
///
/// # Safety
/// Callers may pass any `PyObjectRef`; the function dereferences via
/// the same checks `iter()` uses (null-check, type-tag check) and
/// never reads through a dangling pointer beyond what existing pyre
/// type-tag helpers guarantee.
pub fn is_iterable(w_obj: PyObjectRef) -> bool {
    let obj = unwrap_cell(w_obj);
    if obj.is_null() {
        return false;
    }
    unsafe {
        if is_list(obj)
            || is_tuple(obj)
            || is_str(obj)
            || pyre_object::bytesobject::is_bytes_like(obj)
            || is_dict(obj)
            || pyre_object::is_set_or_frozenset(obj)
            || is_range_iter(obj)
            || is_seq_iter(obj)
            || pyre_object::generatorobject::is_generator(obj)
            || pyre_object::itertoolsmodule::is_count(obj)
            || pyre_object::itertoolsmodule::is_repeat(obj)
        {
            return true;
        }
        // descroperation.py:320 — `space.lookup(w_obj, '__iter__')`.
        // MRO-only walk; no `__getattr__` / descriptor execution.
        if lookup(w_obj, "__iter__").is_some() {
            return true;
        }
        // descroperation.py:322-323 — fallback to `__getitem__` only
        // when `space.type(w_obj).flag_map_or_seq != 'M'` (i.e. the
        // type is not flagged as a mapping).  Mapping types report
        // not-iterable when they don't supply `__iter__`.  The flag
        // lives on `W_TypeObject` (typeobject.py:169) so user-defined
        // `dict`/`list`/`tuple` subclasses inherit the marker via
        // `inherit_flag_map_or_seq` at heap-type construction.
        let w_type = crate::typedef::r#type(w_obj).unwrap_or(std::ptr::null_mut());
        let is_mapping = pyre_object::typeobject::w_type_get_flag_map_or_seq(w_type) == b'M';
        if !is_mapping && lookup(w_obj, "__getitem__").is_some() {
            return true;
        }
    }
    false
}

/// pypy/interpreter/baseobjspace.py:1110-1116 `fixedview`.
///
/// ```python
/// def fixedview(self, w_iterable, expected_length=-1):
///     """ A fixed list view of w_iterable. Don't modify the result """
///     return make_sure_not_resized(self.unpackiterable(w_iterable,
///                                                      expected_length)[:])
///
/// fixedview_unroll = fixedview
/// ```
///
/// Pyre returns a `Vec<PyObjectRef>` directly; the
/// `make_sure_not_resized` annotation is an RPython JIT hint with no
/// runtime effect that translates to "treat the result as immutable
/// at the callsite", which Rust enforces via `&[PyObjectRef]` once
/// the caller binds the return value.
pub fn fixedview(
    w_iterable: PyObjectRef,
    expected_length: isize,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    unpackiterable(w_iterable, expected_length)
}

/// `iter(obj)` — PyPy: space.iter(w_obj)
/// Calls __iter__ on the object if available.
pub fn iter(obj: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    if obj.is_null() {
        return Err(PyError::type_error("'NoneType' object is not iterable"));
    }
    // `pypy/objspace/std/dictproxyobject.py:41 descr_iter` →
    // `space.iter(self.w_mapping)`.
    let obj = unsafe {
        if pyre_object::is_dict_proxy(obj) {
            pyre_object::w_dict_proxy_get_mapping(obj)
        } else {
            obj
        }
    };
    // `pypy/objspace/std/dictmultiobject.py:1701-1741
    // W_BaseDictIterator` line-by-line port — pyre's `W_DictViewIterator`
    // captures the source dict + the version counter seen at iter()
    // time, then on each `next()` step compares against `w_dict.version`
    // and raises `RuntimeError("dictionary changed size during
    // iteration")` if the dict was mutated mid-iteration.
    unsafe {
        if pyre_object::dictviewobject::is_dict_view(obj) {
            let kind = pyre_object::dictviewobject::w_dict_view_get_kind(obj);
            let w_dict = pyre_object::dictviewobject::w_dict_view_get_dict(obj);
            return Ok(pyre_object::dictviewobject::w_dict_view_iterator_new(
                w_dict, kind,
            ));
        }
        // `dict_keyiterator` / `dict_valueiterator` / `dict_itemiterator`
        // — `__iter__` returns self per `dictmultiobject.py:1716-1717
        // W_BaseDictIterator.iter_w`.
        if pyre_object::dictviewobject::is_dict_view_iterator(obj) {
            return Ok(obj);
        }
    }
    unsafe {
        // Builtin iterables
        if is_list(obj) {
            return Ok(pyre_object::w_seq_iter_new(obj, w_list_len(obj)));
        }
        if is_tuple(obj) {
            return Ok(pyre_object::w_seq_iter_new(obj, w_tuple_len(obj)));
        }
        if is_str(obj) {
            let len = w_str_get_value(obj).len();
            return Ok(pyre_object::w_seq_iter_new(obj, len));
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let len = pyre_object::bytesobject::bytes_like_len(obj);
            let mut items = Vec::with_capacity(len);
            for i in 0..len {
                items.push(w_int_new(
                    pyre_object::bytesobject::bytes_like_getitem(obj, i) as i64,
                ));
            }
            let list = pyre_object::w_list_new(items);
            return Ok(pyre_object::w_seq_iter_new(list, len));
        }
        // dict → iterate over keys (dictobject.py __iter__)
        if is_dict(obj) {
            let d = &*(obj as *const pyre_object::dictobject::W_DictObject);
            let entries = &*d.entries;
            let keys: Vec<PyObjectRef> = entries.iter().map(|&(k, _)| k).collect();
            let key_list = pyre_object::w_list_new(keys);
            let len = entries.len();
            return Ok(pyre_object::w_seq_iter_new(key_list, len));
        }
        // set / frozenset → iterate via stable insertion order (PyPy:
        // setobject.py W_BaseSetObject.descr_iter, W_BaseSetIterObject).
        if pyre_object::is_set_or_frozenset(obj) {
            let items = pyre_object::w_set_items(obj);
            let len = items.len();
            let key_list = pyre_object::w_list_new(items);
            return Ok(pyre_object::w_seq_iter_new(key_list, len));
        }
        // Already an iterator
        if is_range_iter(obj) || is_seq_iter(obj) || pyre_object::generatorobject::is_generator(obj)
        {
            return Ok(obj);
        }
        // itertools.count / itertools.repeat — iter_w returns self.
        // PyPy: W_Count.iter_w / W_Repeat.iter_w
        if pyre_object::itertoolsmodule::is_count(obj)
            || pyre_object::itertoolsmodule::is_repeat(obj)
        {
            return Ok(obj);
        }
        // `pypy/module/__builtin__/functional.py:277-278
        // W_Enumerate.descr___iter__` — `return self`.
        if pyre_object::enumerateobject::is_enumerate(obj) {
            return Ok(obj);
        }
        // pypy/objspace/descroperation.py:330-346 `def iter(space, w_obj)`
        // — `space.lookup(w_obj, '__iter__')` is type-MRO-only; PyPy never
        // consults the instance dict for special-method lookup (CPython
        // issue 5985 / typeobject `__iter__` slot resolution).  Earlier
        // pyre revisions also walked `getdict(obj)` and `ATTR_TABLE`,
        // which surfaced per-instance `__iter__` writes (e.g.
        // `obj.__iter__ = method`); those paths are non-orthodox in
        // both CPython and PyPy and have been removed.
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(method) = lookup_in_type_where(w_type, "__iter__") {
                // descroperation.py:339-341 — explicit `__iter__ = None`
                // marks the type as non-iterable even though the lookup
                // succeeds.
                if is_none(method) {
                    return Err(PyError::type_error(format!(
                        "'{}' object is not iterable",
                        (*(*obj).ob_type).name
                    )));
                }
                return Ok(crate::call_function(method, &[obj]));
            }
            // descroperation.py:333-334 — `__getitem__` fallback only when
            // `space.type(w_obj).flag_map_or_seq != 'M'`.  Mapping types
            // without `__iter__` are reported as non-iterable.  Read off
            // the user `W_TypeObject` (typeobject.py:169) so heap-type
            // dict/list/tuple subclasses inherit the marker — see
            // `is_iterable` at baseobjspace.rs:5343 for the same pattern.
            let w_user_type = crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut());
            let is_mapping =
                pyre_object::typeobject::w_type_get_flag_map_or_seq(w_user_type) == b'M';
            if !is_mapping
                && (lookup_in_type_where(w_type, "__getitem__").is_some()
                    || getattr(obj, "__getitem__").is_ok())
            {
                // Try to use __len__ to bound the iteration.
                let mut items = Vec::new();
                if let Ok(len_result) = len(obj) {
                    if is_int(len_result) {
                        let n = w_int_get_value(len_result);
                        for i in 0..n {
                            match getitem(obj, w_int_new(i)) {
                                Ok(item) => items.push(item),
                                Err(_) => break,
                            }
                        }
                        let count = items.len();
                        let list = w_list_new(items);
                        return Ok(pyre_object::w_seq_iter_new(list, count));
                    }
                }
                // No __len__: iterate up to a reasonable bound, breaking on
                // any error (PyPy: descroperation iter_via_getitem with sentinel).
                for i in 0..1_000_000i64 {
                    match getitem(obj, w_int_new(i)) {
                        Ok(item) => items.push(item),
                        Err(_) => break,
                    }
                }
                let count = items.len();
                let list = w_list_new(items);
                return Ok(pyre_object::w_seq_iter_new(list, count));
            }
        }
        // Type object: check metaclass __iter__ (NOT the type's own MRO)
        // PyPy/CPython: iter(X) calls type(X).__iter__(X), not X.__iter__
        // For type objects, type(X) is the metaclass.
        if is_type(obj) {
            // baseobjspace.py:76 — metaclass from w_class
            let w_metaclass = {
                let w_class = (*obj).w_class;
                let w_type_type = crate::typedef::w_type();
                if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                    Some(w_class)
                } else {
                    None
                }
            };
            if let Some(w_metaclass) = w_metaclass {
                if let Some(method) = lookup_in_type_where(w_metaclass, "__iter__") {
                    return Ok(crate::call_function(method, &[obj]));
                }
            }
            // Fallback: check type type's MRO
            if let Some(w_type_type) = crate::typedef::gettypefor(&pyre_object::pyobject::TYPE_TYPE)
            {
                if let Some(method) = lookup_in_type_where(w_type_type, "__iter__") {
                    return Ok(crate::call_function(method, &[obj]));
                }
            }
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not iterable",
        unsafe { (*(*obj).ob_type).name }
    )))
}

/// `next(iterator)` — PyPy: space.next(w_iter)
pub fn next(obj: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    unsafe {
        // Seq iterator
        if is_seq_iter(obj) {
            let iter = &mut *(obj as *mut pyre_object::W_SeqIterator);
            let seq = iter.seq;
            let idx = iter.index;
            let item = if is_list(seq) {
                pyre_object::w_list_getitem(seq, idx)
            } else if is_tuple(seq) {
                pyre_object::w_tuple_getitem(seq, idx)
            } else if is_str(seq) {
                let s = w_str_get_value(seq);
                s.chars().nth(idx as usize).map(|c| {
                    let mut buf = [0u8; 4];
                    w_str_new(c.encode_utf8(&mut buf))
                })
            } else {
                None
            };
            if let Some(v) = item {
                iter.index += 1;
                return Ok(v);
            }
            return Err(PyError {
                kind: PyErrorKind::StopIteration,
                message: "".to_string(),
                exc_object: std::ptr::null_mut(),
                attach_tb: true,
            });
        }
        // Range iterator
        if is_range_iter(obj) {
            let iter = &mut *(obj as *mut pyre_object::rangeobject::W_RangeIterator);
            let has_next = if iter.step > 0 {
                iter.current < iter.stop
            } else if iter.step < 0 {
                iter.current > iter.stop
            } else {
                false
            };
            if has_next {
                let val = w_int_new(iter.current);
                iter.current += iter.step;
                return Ok(val);
            }
            return Err(PyError {
                kind: PyErrorKind::StopIteration,
                message: "".to_string(),
                exc_object: std::ptr::null_mut(),
                attach_tb: true,
            });
        }
        // Generator __next__ — PyPy: generator.py GeneratorIterator.next
        if pyre_object::generatorobject::is_generator(obj) {
            return generator_next(obj);
        }
        // itertools.count.next_w — PyPy interp_itertools.py W_Count.next_w
        //
        //     def next_w(self):
        //         w_c = self.w_c
        //         self.w_c = self.space.add(w_c, self.w_step)
        //         return w_c
        if pyre_object::itertoolsmodule::is_count(obj) {
            let w_c = pyre_object::itertoolsmodule::w_count_get_c(obj);
            let w_step = pyre_object::itertoolsmodule::w_count_get_step(obj);
            let new_c = add(w_c, w_step)?;
            pyre_object::itertoolsmodule::w_count_set_c(obj, new_c);
            return Ok(w_c);
        }
        // itertools.repeat.next_w — PyPy interp_itertools.py W_Repeat.next_w
        //
        //     def next_w(self):
        //         if self.counting:
        //             if self.count <= 0:
        //                 raise OperationError(self.space.w_StopIteration, self.space.w_None)
        //             self.count -= 1
        //         return self.w_obj
        if pyre_object::itertoolsmodule::is_repeat(obj) {
            if pyre_object::itertoolsmodule::w_repeat_get_counting(obj) {
                if pyre_object::itertoolsmodule::w_repeat_get_count(obj) <= 0 {
                    return Err(PyError::stop_iteration());
                }
                pyre_object::itertoolsmodule::w_repeat_dec_count(obj);
            }
            return Ok(pyre_object::itertoolsmodule::w_repeat_get_obj(obj));
        }
        // `pypy/objspace/std/dictmultiobject.py:1719-1741
        // W_BaseDictIterator.descr_next` line-by-line —
        //
        //     def next(self):
        //         if self.len != self.w_dict.length():
        //             raise oefmt(space.w_RuntimeError,
        //                 "dictionary changed size during iteration")
        //         ...
        //
        // Only `len` is compared — re-assigning an existing key
        // (overwrite) is permitted because `length()` is unchanged.
        // Pyre also routes the entry read through `w_dict_items()`
        // so DictStorage-backed dicts (globals(), module dict)
        // see their authoritative storage instead of the local
        // mirror.
        if pyre_object::dictviewobject::is_dict_view_iterator(obj) {
            use pyre_object::dictviewobject as dv;
            let dict = dv::w_dict_view_iterator_get_dict(obj);
            let startlen = dv::w_dict_view_iterator_get_startlen(obj);
            let current_len = pyre_object::dictobject::w_dict_len(dict);
            if startlen != current_len {
                return Err(PyError::new(
                    PyErrorKind::RuntimeError,
                    "dictionary changed size during iteration".to_string(),
                ));
            }
            let index = dv::w_dict_view_iterator_get_index(obj);
            let items = pyre_object::dictobject::w_dict_items(dict);
            if index >= items.len() {
                return Err(PyError::stop_iteration());
            }
            let (k, v) = items[index];
            dv::w_dict_view_iterator_set_index(obj, index + 1);
            return Ok(match dv::w_dict_view_iterator_get_kind(obj) {
                pyre_object::dictviewobject::DictViewKind::Keys => k,
                pyre_object::dictviewobject::DictViewKind::Values => v,
                pyre_object::dictviewobject::DictViewKind::Items => {
                    pyre_object::w_tuple_new(vec![k, v])
                }
            });
        }
        // `pypy/module/__builtin__/functional.py:280-310 W_Enumerate.descr_next`
        // line-by-line port —
        //
        //     def descr_next(self, space):
        //         w_index = self.w_index
        //         w_iter_or_list = self.w_iter_or_list
        //         w_item = None
        //         if w_index is None:
        //             index = self.index
        //             if type(w_iter_or_list) is W_ListObject:
        //                 try:
        //                     w_item = w_iter_or_list.getitem(index)
        //                 except IndexError:
        //                     self.w_iter_or_list = None
        //                     raise OperationError(space.w_StopIteration, space.w_None)
        //                 self.index = index + 1
        //             elif w_iter_or_list is None:
        //                 raise OperationError(space.w_StopIteration, space.w_None)
        //             else:
        //                 try:
        //                     newval = rarithmetic.ovfcheck(index + 1)
        //                 except OverflowError:
        //                     w_index = space.newint(index)
        //                     self.w_index = space.add(w_index, space.newint(1))
        //                     self.index = -1
        //                 else:
        //                     self.index = newval
        //             w_index = space.newint(index)
        //         else:
        //             self.w_index = space.add(w_index, space.newint(1))
        //         if w_item is None:
        //             w_item = space.next(self.w_iter_or_list)
        //         return space.newtuple2(w_index, w_item)
        if pyre_object::enumerateobject::is_enumerate(obj) {
            use pyre_object::enumerateobject as eo;
            let w_index_slot = eo::w_enumerate_get_w_index(obj);
            let mut w_iter_or_list = eo::w_enumerate_get_iter_or_list(obj);
            let mut w_item: PyObjectRef = pyre_object::PY_NULL;
            let w_index: PyObjectRef;
            if w_index_slot.is_null() {
                // i64 fast-path branch.
                let index = eo::w_enumerate_get_index(obj);
                if !w_iter_or_list.is_null() && pyre_object::is_list(w_iter_or_list) {
                    // `:289-294 W_ListObject` fast path — directly
                    // getitem; IndexError marks end-of-iteration and
                    // clears the slot.
                    let list_len = pyre_object::w_list_len(w_iter_or_list) as i64;
                    if index < 0 || index >= list_len {
                        eo::w_enumerate_set_iter_or_list(obj, pyre_object::PY_NULL);
                        return Err(PyError::stop_iteration());
                    }
                    w_item = pyre_object::w_list_getitem(w_iter_or_list, index).unwrap_or(PY_NULL);
                    eo::w_enumerate_set_index(obj, index + 1);
                } else if w_iter_or_list.is_null() {
                    // `:295-296` — slot cleared after a previous
                    // list-getitem stop.
                    return Err(PyError::stop_iteration());
                } else {
                    // General iterator path — `:297-303` ovfcheck.
                    match index.checked_add(1) {
                        Some(next) => eo::w_enumerate_set_index(obj, next),
                        None => {
                            // Promote to bigint slot per `:299-302`.
                            let w_idx =
                                pyre_object::w_long_new(::malachite_bigint::BigInt::from(index));
                            let one =
                                pyre_object::w_long_new(::malachite_bigint::BigInt::from(1i64));
                            let bumped = add(w_idx, one)?;
                            eo::w_enumerate_set_w_index(obj, bumped);
                            eo::w_enumerate_set_index(obj, -1);
                        }
                    }
                }
                w_index = pyre_object::w_int_new(index);
            } else {
                // Bigint slot active — bump via `space.add`.
                let one = pyre_object::w_int_new(1);
                let bumped = add(w_index_slot, one)?;
                eo::w_enumerate_set_w_index(obj, bumped);
                w_index = w_index_slot;
            }
            if w_item.is_null() {
                // Re-read slot — list fast-path already set w_item;
                // otherwise we need to pull from the iterator.
                w_iter_or_list = eo::w_enumerate_get_iter_or_list(obj);
                if w_iter_or_list.is_null() {
                    return Err(PyError::stop_iteration());
                }
                w_item = next(w_iter_or_list)?;
            }
            return Ok(pyre_object::w_tuple_new(vec![w_index, w_item]));
        }
        // Instance __next__
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(method) = lookup_in_type_where(w_type, "__next__") {
                return Ok(crate::call_function(method, &[obj]));
            }
        }
    }
    Err(PyError::type_error("not an iterator"))
}

/// Property setter/getter/deleter helpers — PyPy: W_Property.setter/getter/deleter.
/// args[0] is the owning property (bound via W_Method), args[1] is the new fn.
fn property_setter_impl(args: &[PyObjectRef]) -> PyResult {
    let prop = args[0];
    let new_fn = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    unsafe {
        let fget = w_property_get_fget(prop);
        let fdel = w_property_get_fdel(prop);
        Ok(pyre_object::w_property_new(fget, new_fn, fdel))
    }
}

fn property_getter_impl(args: &[PyObjectRef]) -> PyResult {
    let prop = args[0];
    let new_fn = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    unsafe {
        let fset = w_property_get_fset(prop);
        let fdel = w_property_get_fdel(prop);
        Ok(pyre_object::w_property_new(new_fn, fset, fdel))
    }
}

fn property_deleter_impl(args: &[PyObjectRef]) -> PyResult {
    let prop = args[0];
    let new_fn = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    unsafe {
        let fget = w_property_get_fget(prop);
        let fset = w_property_get_fset(prop);
        Ok(pyre_object::w_property_new(fget, fset, new_fn))
    }
}

// ── Generator methods ────────────────────────────────────────────────
//
// PyPy: pypy/interpreter/generator.py GeneratorIterator
//
// send_ex(w_arg, operr) is the core resume path.
// - __next__() → send_ex(None, None)
// - send(v)    → send_ex(v, None)
// - throw(t,v) → send_ex(None, OperationError(t,v))
// - close()    → throw(GeneratorExit) then check result

/// PyPy: GeneratorIterator._send_ex(w_arg, operr)
///
/// Resume a generator frame: push w_arg (for send/next) or inject operr
/// (for throw), then run the frame until YIELD_VALUE or RETURN_VALUE.
fn generator_send_ex(gen_obj: PyObjectRef, w_arg: PyObjectRef, operr: Option<PyError>) -> PyResult {
    use pyre_object::generatorobject::*;
    unsafe {
        if w_generator_is_running(gen_obj) {
            return Err(PyError::value_error("generator already executing"));
        }

        if w_generator_is_exhausted(gen_obj) {
            if let Some(err) = operr {
                return Err(err);
            }
            return Err(PyError::stop_iteration());
        }

        let frame_ptr = w_generator_get_frame(gen_obj) as *mut crate::pyframe::PyFrame;
        if frame_ptr.is_null() {
            w_generator_set_exhausted(gen_obj);
            if let Some(err) = operr {
                return Err(err);
            }
            return Err(PyError::stop_iteration());
        }
        let frame = &mut *frame_ptr;
        let already_started = w_generator_is_started(gen_obj);

        if !already_started {
            if operr.is_none() && !w_arg.is_null() && !is_none(w_arg) {
                return Err(PyError::type_error(
                    "can't send non-None value to a just-started generator",
                ));
            }
        }
        w_generator_set_started(gen_obj);
        w_generator_set_running(gen_obj, true);

        // generator.py:104 — w_result = frame.execute_frame(w_arg, operr)
        let w_inputvalue = if already_started && operr.is_none() {
            Some(w_arg)
        } else {
            None
        };
        let result = frame.execute_frame(w_inputvalue, operr);

        w_generator_set_running(gen_obj, false);

        match result {
            Ok(value) => {
                // generator.py:109-114 — if the frame marked itself finished,
                // it was RETURNed from; otherwise it YIELDed.
                if frame.frame_finished_execution {
                    w_generator_set_exhausted(gen_obj);
                    // generator.py:117-119 / pyopcode.py RETURN_VALUE in
                    // generator frames — `raise StopIteration(returnvalue)`
                    // so callers can pull the return value off `.value`.
                    // Wrap any non-None return into the exception's args
                    // tuple; bare `return` (or fallthrough → None) keeps
                    // an empty args tuple.
                    Err(stop_iteration_with_value(value))
                } else {
                    Ok(value)
                }
            }
            Err(e) => {
                w_generator_set_exhausted(gen_obj);
                Err(e)
            }
        }
    }
}

/// Build a `StopIteration` carrying `value` on `.value` / `args[0]`.
/// `value == None` (or PY_NULL) keeps the args tuple empty so
/// `next(g)` outside a generator-return context still surfaces a bare
/// `StopIteration()`.
fn stop_iteration_with_value(value: PyObjectRef) -> PyError {
    use pyre_object::excobject::*;
    let exc = w_exception_new(ExcKind::StopIteration, "");
    if !value.is_null() && unsafe { !is_none(value) } {
        // `interp_exceptions.py:121-124 W_BaseException.descr_init`
        // stores `args_w` as a list; pyre matches the shape so that
        // `e.args` materialises a fresh tuple each read.
        let args_list = w_list_new(vec![value]);
        unsafe {
            w_exception_set_args(exc, args_list);
        }
    }
    unsafe { PyError::from_exc_object(exc) }
}

/// PyPy: GeneratorIterator.next() — equivalent to __next__
fn generator_next(gen_obj: PyObjectRef) -> PyResult {
    generator_send_ex(gen_obj, w_none(), None)
}

/// __next__ method wrapper
fn generator_next_method(args: &[PyObjectRef]) -> PyResult {
    let gen_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    generator_next(gen_obj)
}

/// Generic __next__ wrapper for iterators that delegate to `next()`.
/// Used for itertools count/repeat etc.
fn iter_next_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    next(obj)
}

/// PyPy: GeneratorIterator.descr_send(w_arg)
fn generator_send_method(args: &[PyObjectRef]) -> PyResult {
    let gen_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = args.get(1).copied().unwrap_or(w_none());
    generator_send_ex(gen_obj, value, None)
}

/// PyPy: GeneratorIterator.descr_throw(w_type, w_val=None, w_tb=None)
fn generator_throw_method(args: &[PyObjectRef]) -> PyResult {
    let gen_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let w_type = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let w_val = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    // w_tb (args[3]) ignored for now — traceback not yet supported

    let err = normalize_throw_args(w_type, w_val);
    generator_send_ex(gen_obj, w_none(), Some(err))
}

/// PyPy: GeneratorIterator.descr_close()
fn generator_close_method(args: &[PyObjectRef]) -> PyResult {
    let gen_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    unsafe {
        use pyre_object::generatorobject::*;
        if w_generator_is_exhausted(gen_obj) {
            return Ok(w_none());
        }
        if !w_generator_is_started(gen_obj) {
            w_generator_set_exhausted(gen_obj);
            return Ok(w_none());
        }
    }
    let err = PyError {
        kind: PyErrorKind::GeneratorExit,
        message: String::new(),
        exc_object: std::ptr::null_mut(),
        attach_tb: true,
    };
    match generator_send_ex(gen_obj, w_none(), Some(err)) {
        Ok(_) => {
            // Generator yielded after GeneratorExit — RuntimeError
            Err(PyError::runtime_error("generator ignored GeneratorExit"))
        }
        Err(e) if e.kind == PyErrorKind::StopIteration || e.kind == PyErrorKind::GeneratorExit => {
            Ok(w_none())
        }
        Err(e) => Err(e),
    }
}

/// Normalize throw() arguments into a PyError.
///
/// PyPy: generator.py throw() → OperationError(w_type, w_val, tb) + normalize
///
/// Handles:
///   throw(TypeError)         — type → creates instance
///   throw(TypeError("msg"))  — instance → derives type
///   throw(TypeError, "msg")  — type + value → creates instance
fn normalize_throw_args(w_type: PyObjectRef, w_val: PyObjectRef) -> PyError {
    unsafe {
        // If w_type is an exception instance, use it directly
        if !w_type.is_null() && pyre_object::excobject::is_exception(w_type) {
            return PyError::from_exc_object(w_type);
        }

        // If w_type is a type (class), try to create exception from it
        if !w_type.is_null() && pyre_object::is_type(w_type) {
            let type_name = pyre_object::w_type_get_name(w_type);
            if let Some(kind) = pyre_object::excobject::exc_kind_from_name(type_name) {
                let msg = if w_val.is_null() || pyre_object::is_none(w_val) {
                    String::new()
                } else if pyre_object::is_str(w_val) {
                    pyre_object::w_str_get_value(w_val).to_string()
                } else {
                    String::new()
                };
                return PyError {
                    kind: PyError::kind_from_exc(kind),
                    message: msg,
                    exc_object: std::ptr::null_mut(),
                    attach_tb: true,
                };
            }
        }

        // Fallback: TypeError
        PyError::type_error("exceptions must be classes or instances deriving from BaseException")
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

    #[test]
    fn test_setattr_getattr() {
        // PyPy raises AttributeError when setattr targets a non-hasdict
        // type. Use a hasdict instance: a W_InstanceObject of a fresh
        // user class created via type().
        let obj = make_user_instance();
        setattr(obj, "name", w_int_new(100)).unwrap();
        let result = getattr(obj, "name").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 100) };
    }

    #[test]
    fn test_getattr_missing() {
        let obj = w_int_new(1);
        let err = getattr(obj, "missing").unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::AttributeError));
    }

    #[test]
    fn test_setattr_overwrite() {
        let obj = make_user_instance();
        setattr(obj, "x", w_int_new(1)).unwrap();
        setattr(obj, "x", w_int_new(2)).unwrap();
        let result = getattr(obj, "x").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 2) };
    }

    /// Helper for the setattr/getattr tests: build an instance of a fresh
    /// user class so the object has a live W_DictObject backing store
    /// (analogous to PyPy's `_getusercls` instances).
    fn make_user_instance() -> PyObjectRef {
        crate::typedef::init_typeobjects();
        use pyre_object::instanceobject::w_instance_new;
        let cls = crate::typedef::make_builtin_type("TestUserClass", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(cls, true) };
        w_instance_new(cls)
    }

    #[test]
    fn test_module_setattr_getattr() {
        let mut namespace = Box::new(crate::DictStorage::default());
        namespace.fix_ptr();
        let module = pyre_object::moduleobject::w_module_new(
            "test_module",
            Box::into_raw(namespace) as *mut u8,
        );

        setattr(module, "ps1", w_str_new("py> ")).unwrap();
        let result = getattr(module, "ps1").unwrap();
        unsafe { assert_eq!(w_str_get_value(result), "py> ") };
    }

    #[test]
    fn test_module_delattr() {
        let mut namespace = Box::new(crate::DictStorage::default());
        namespace.fix_ptr();
        let module = pyre_object::moduleobject::w_module_new(
            "test_module",
            Box::into_raw(namespace) as *mut u8,
        );

        setattr(module, "ps1", w_str_new("py> ")).unwrap();
        delattr(module, "ps1").unwrap();
        let err = getattr(module, "ps1").unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::AttributeError));
    }

    /// `pypy/interpreter/module.py:77 Module.getdict()` parity invariant:
    /// every call to `dict_storage_to_dict(storage)` must return the
    /// **same** `W_DictObject` (single canonical) for a given storage.
    /// This is the foundation of `f.__globals__ is m.__dict__` and
    /// `globals() is __main__.__dict__` — pyre's split entries Vec /
    /// DictStorage no longer creates fresh snapshot wrappers per call.
    ///
    /// The first call lazy-allocates a W_DictObject and registers it as
    /// the storage's `mirror_target`; subsequent calls retrieve that
    /// same dict via `mirror_target` lookup.
    #[test]
    fn test_dict_storage_to_dict_returns_canonical_w_dict() {
        let mut namespace = Box::new(crate::DictStorage::default());
        namespace.fix_ptr();
        crate::dict_storage_store(&mut namespace, "alpha", w_int_new(7));
        let ns_ptr: *const crate::DictStorage = &*namespace;

        let first = super::dict_storage_to_dict(ns_ptr);
        let second = super::dict_storage_to_dict(ns_ptr);
        assert!(
            std::ptr::eq(first, second),
            "dict_storage_to_dict must return the canonical W_DictObject \
             (storage's mirror_target), not a fresh snapshot",
        );

        // Storage-side write after the canonical has been allocated
        // surfaces in the same W_DictObject via the back-mirror.
        crate::dict_storage_store(&mut namespace, "beta", w_int_new(11));
        unsafe {
            assert_eq!(
                w_int_get_value(pyre_object::w_dict_lookup(first, w_str_new("beta")).unwrap()),
                11,
                "post-canonicalization storage write must mirror into the canonical W_DictObject's entries Vec",
            );
        }
    }

    /// `pypy/interpreter/module.py:77 Module.getdict()` parity invariant
    /// for the new module-creation pattern (canonical W_DictObject reuse
    /// via `dict_storage_to_dict` + `w_module_new_aliasing_dict`):
    /// `module.w_dict` IS the storage's canonical W_DictObject, and
    /// `dict_storage_to_dict(module.dict_storage)` returns that same
    /// object on every call.
    #[test]
    fn test_module_w_dict_is_canonical_for_storage() {
        let mut namespace = Box::new(crate::DictStorage::default());
        namespace.fix_ptr();
        crate::dict_storage_store(&mut namespace, "x", w_int_new(42));
        let ns_ptr = Box::into_raw(namespace);

        // Pattern matches the production module-creation path
        // (executioncontext::get_builtin / importing::init_builtin_module
        //  / pyrex::run_source __main__).
        let canonical = super::dict_storage_to_dict(ns_ptr);
        let module = pyre_object::moduleobject::w_module_new_aliasing_dict(
            "test_canonical",
            ns_ptr as *mut u8,
            canonical,
        );

        // Module's w_dict identity equals the canonical lazy-paired
        // W_DictObject — `f.__globals__ is m.__dict__` invariant.
        let module_w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        assert!(
            std::ptr::eq(module_w_dict, canonical),
            "Module.w_dict must alias the storage's canonical W_DictObject",
        );

        // Repeat lookup of the canonical (e.g. function.__globals__
        // path in production) returns the same object.
        let again = super::dict_storage_to_dict(ns_ptr);
        assert!(
            std::ptr::eq(again, canonical),
            "subsequent dict_storage_to_dict on the same storage must return the same W_DictObject",
        );

        // `module.__dict__["x"]` (resolved via the canonical W_DictObject)
        // sees the storage-pre-populated entry.
        unsafe {
            assert_eq!(
                w_int_get_value(pyre_object::w_dict_lookup(module_w_dict, w_str_new("x")).unwrap()),
                42,
                "canonical W_DictObject must surface storage-side entries that pre-date canonicalization",
            );
        }
    }

    #[test]
    fn test_py_contains_manual_list() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        let needle = w_int_new(1);
        unsafe {
            assert!(
                is_list(list),
                "should be list, got type: {}",
                (*(*list).ob_type).name
            );
        }
        let result = super::contains(list, needle).expect("contains failed");
        assert!(result, "1 should be in [1, 2, 3]");
    }

    /// abstractinst.py:53-72 — `isinstance(5, 6)` must raise TypeError
    /// from `check_class()`, not silently return False from a naive
    /// `isinstance_w` walk. PyPy test: `test_builtin.py:605`.
    #[test]
    fn test_isinstance_non_class_arg2_raises_typeerror() {
        crate::typedef::init_typeobjects();
        let err = super::isinstance(w_int_new(5), w_int_new(6)).unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message.contains("isinstance() arg 2"));
    }

    /// abstractinst.py:108-114 + 53-72 — when one tuple element is not a
    /// class the recursion must surface the TypeError from `check_class`.
    #[test]
    fn test_isinstance_tuple_with_non_class_raises_typeerror() {
        crate::typedef::init_typeobjects();
        let float_type = crate::typedef::r#type(w_float_new(0.0)).unwrap();
        let bad = w_tuple_new(vec![float_type, w_int_new(6)]);
        let err = super::isinstance(w_int_new(5), bad).unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::TypeError));
    }

    /// abstractinst.py:150-169 — `issubclass(5, int)` must raise
    /// TypeError because the first argument is not a class.
    #[test]
    fn test_issubclass_non_class_arg1_raises_typeerror() {
        crate::typedef::init_typeobjects();
        let int_type = crate::typedef::r#type(w_int_new(0)).unwrap();
        let err = super::issubclass(w_int_new(5), int_type).unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message.contains("issubclass() arg 1"));
    }

    /// abstractinst.py:150-169 — `issubclass(int, 6)` must raise
    /// TypeError because the second argument is not a class.
    #[test]
    fn test_issubclass_non_class_arg2_raises_typeerror() {
        crate::typedef::init_typeobjects();
        let int_type = crate::typedef::r#type(w_int_new(0)).unwrap();
        let err = super::issubclass(int_type, w_int_new(6)).unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message.contains("issubclass() arg 2"));
    }

    /// abstractinst.py:127-147 — `p_abstract_issubclass_w` must walk
    /// `__bases__` for pseudo-classes (any object that exposes a tuple
    /// `__bases__` attribute), not just real type objects. We construct
    /// `outer` whose `__bases__` is `(inner,)` and `inner` whose
    /// `__bases__` is the empty tuple, then verify
    /// `issubclass(outer, inner)` returns True via the abstract walk.
    #[test]
    fn test_issubclass_pseudo_class_via_bases() {
        crate::typedef::init_typeobjects();
        let inner_type = crate::typedef::make_builtin_type("PseudoInner", |ns| {
            crate::dict_storage_store(ns, "__bases__", w_tuple_new(vec![]));
        });
        let inner = pyre_object::instanceobject::w_instance_new(inner_type);
        let outer_type = crate::typedef::make_builtin_type("PseudoOuter", |_ns| {
            // closure capture is fine — make_builtin_type runs init eagerly.
        });
        // Stash __bases__ on outer's type dict pointing at the inner instance.
        crate::dict_storage_store(
            unsafe {
                &mut *(pyre_object::w_type_get_dict_ptr(outer_type) as *mut crate::DictStorage)
            },
            "__bases__",
            w_tuple_new(vec![inner]),
        );
        let outer = pyre_object::instanceobject::w_instance_new(outer_type);
        let yes = super::issubclass(outer, inner).expect("issubclass should succeed");
        assert!(yes);
    }

    /// pypy/interpreter/baseobjspace.py:983 `unpackiterable` known-length:
    /// `[1, 2, 3]` with expected_length=3 yields the unpacked items.
    #[test]
    fn unpackiterable_known_length_match() {
        let lst = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        let items = unpackiterable(lst, 3).expect("unpack should succeed");
        assert_eq!(items.len(), 3);
        unsafe {
            assert_eq!(w_int_get_value(items[0]), 1);
            assert_eq!(w_int_get_value(items[1]), 2);
            assert_eq!(w_int_get_value(items[2]), 3);
        }
    }

    /// pypy/interpreter/baseobjspace.py:1049-1052 — `not enough values
    /// to unpack` ValueError when iterator yields fewer items than
    /// expected.
    #[test]
    fn unpackiterable_too_few() {
        let lst = w_list_new(vec![w_int_new(1)]);
        let err = unpackiterable(lst, 3).expect_err("expected ValueError");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        assert!(err.message.contains("not enough values"));
    }

    /// pypy/interpreter/baseobjspace.py:1043-1046 — `too many values
    /// to unpack` ValueError when iterator yields more items than
    /// expected.
    #[test]
    fn unpackiterable_too_many() {
        let lst = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3), w_int_new(4)]);
        let err = unpackiterable(lst, 3).expect_err("expected ValueError");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        assert!(err.message.contains("too many values"));
    }

    /// pypy/interpreter/baseobjspace.py:983-994 — expected_length=-1
    /// accepts any length without validation.
    #[test]
    fn unpackiterable_unknown_length_accepts_any() {
        let lst = w_list_new(vec![w_int_new(10), w_int_new(20)]);
        let items = unpackiterable(lst, -1).expect("unpack should succeed");
        assert_eq!(items.len(), 2);
    }

    /// pypy/interpreter/baseobjspace.py:1110-1116 `fixedview` is a
    /// thin wrapper over `unpackiterable`; verify it dispatches.
    #[test]
    fn fixedview_delegates_to_unpackiterable() {
        let lst = w_list_new(vec![w_int_new(7), w_int_new(8)]);
        let items = fixedview(lst, 2).expect("fixedview should succeed");
        assert_eq!(items.len(), 2);
        unsafe {
            assert_eq!(w_int_get_value(items[0]), 7);
            assert_eq!(w_int_get_value(items[1]), 8);
        }
    }

    /// pypy/objspace/descroperation.py:319-326 `is_iterable`:
    /// list / tuple / dict / str return true via builtin shortcuts.
    #[test]
    fn is_iterable_true_for_builtin_types() {
        assert!(is_iterable(w_list_new(vec![])));
        assert!(is_iterable(pyre_object::w_tuple_new(vec![])));
        assert!(is_iterable(pyre_object::w_str_new("hello")));
    }

    /// pypy/objspace/descroperation.py:319-326 `is_iterable`:
    /// scalar types (int) without `__iter__` / `__getitem__` return false.
    #[test]
    fn is_iterable_false_for_scalar() {
        assert!(!is_iterable(w_int_new(42)));
        assert!(!is_iterable(w_none()));
    }

    /// pypy/objspace/std/objspace.py:609-617 + dictmultiobject.py:307
    /// — exact `dict` with all string keys takes the
    /// strategy-specific fast path and returns parallel
    /// `(Some(keys), Some(values))`.  An empty exact dict goes
    /// through the same path and returns `(Some([]), Some([]))`.
    #[test]
    fn view_as_kwargs_empty_dict_returns_some_empty() {
        let d = pyre_object::dictobject::w_dict_new();
        let (names, values) = view_as_kwargs(d);
        assert_eq!(names.as_ref().map(|v| v.len()), Some(0));
        assert_eq!(values.as_ref().map(|v| v.len()), Some(0));
    }

    /// pypy/objspace/std/dictmultiobject.py:1325 — kwargs strategy
    /// only succeeds when every key is a unicode string; the base
    /// `(None, None)` is returned for non-string keys (e.g. int).
    #[test]
    fn view_as_kwargs_int_key_returns_none() {
        unsafe {
            let d = pyre_object::dictobject::w_dict_new();
            pyre_object::dictobject::w_dict_store(d, w_int_new(1), w_int_new(2));
            let (names, values) = view_as_kwargs(d);
            assert!(names.is_none());
            assert!(values.is_none());
        }
    }

    /// pypy/objspace/std/objspace.py:615 `isinstance(w_dict,
    /// W_DictObject)` — non-dict (e.g. `int`) returns the base
    /// `(None, None)`.
    #[test]
    fn view_as_kwargs_non_dict_returns_none() {
        let (names, values) = view_as_kwargs(w_int_new(42));
        assert!(names.is_none());
        assert!(values.is_none());
    }

    /// pypy/interpreter/baseobjspace.py:2137-2140 `object_functionstr`
    /// fallback path: scalars without `__qualname__` go through
    /// `space.str(w_function)`, which dispatches to the type's
    /// `__str__` slot via `lookup`.  Pyre's `lookup` walks the
    /// W_TypeObject MRO, which is only populated after
    /// `init_typeobjects()` runs.
    #[test]
    fn object_functionstr_scalar_fallback() {
        crate::typedef::init_typeobjects();
        let s = object_functionstr(w_int_new(42)).expect("scalar fallback never propagates async");
        assert_eq!(s, "42");
    }
}

/// `in` operator: check if `needle` is in `haystack`.
/// PyPy: space.contains_w(haystack, needle)
pub fn contains(haystack: PyObjectRef, needle: PyObjectRef) -> Result<bool, PyError> {
    use pyre_object::*;
    // `pypy/objspace/std/dictproxyobject.py:38 descr_contains` →
    // `space.contains(self.w_mapping, w_key)`.
    let haystack = unsafe {
        if pyre_object::is_dict_proxy(haystack) {
            pyre_object::w_dict_proxy_get_mapping(haystack)
        } else {
            haystack
        }
    };
    // `pypy/objspace/std/dictmultiobject.py W_DictMultiViewKeysObject
    // .descr_contains` → `space.contains(self.w_dict, w_key)`.
    // `W_DictMultiViewItemsObject.descr_contains` matches a (k, v)
    // tuple via dict lookup + value equality.  `W_DictMultiViewValues
    // Object` has no `__contains__` slot in PyPy — pyre delegates the
    // fall-through to the standard `iter`-based scan further down so
    // `v in d.values()` still works (as in PyPy where the missing
    // slot triggers the iter fallback).
    unsafe {
        if pyre_object::dictviewobject::is_dict_view(haystack) {
            let kind = pyre_object::dictviewobject::w_dict_view_get_kind(haystack);
            let dict = pyre_object::dictviewobject::w_dict_view_get_dict(haystack);
            if dict.is_null() {
                return Ok(false);
            }
            match kind {
                pyre_object::dictviewobject::DictViewKind::Keys => {
                    return Ok(pyre_object::w_dict_lookup(dict, needle).is_some());
                }
                pyre_object::dictviewobject::DictViewKind::Items => {
                    if !is_tuple(needle) || w_tuple_len(needle) != 2 {
                        return Ok(false);
                    }
                    let k = match w_tuple_getitem(needle, 0) {
                        Some(k) => k,
                        None => return Ok(false),
                    };
                    let want = match w_tuple_getitem(needle, 1) {
                        Some(v) => v,
                        None => return Ok(false),
                    };
                    return Ok(match pyre_object::w_dict_lookup(dict, k) {
                        Some(have) => eq_w(have, want),
                        None => false,
                    });
                }
                pyre_object::dictviewobject::DictViewKind::Values => {
                    // values view: PyPy uses iter-based scan.
                    for (_, v) in pyre_object::w_dict_items(dict) {
                        if eq_w(v, needle) {
                            return Ok(true);
                        }
                    }
                    return Ok(false);
                }
            }
        }
    }
    unsafe {
        if is_list(haystack) {
            let len = w_list_len(haystack);
            for i in 0..len {
                if let Some(item) = w_list_getitem(haystack, i as i64) {
                    if eq_w(item, needle) {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if is_tuple(haystack) {
            let len = w_tuple_len(haystack);
            for i in 0..len {
                if let Some(item) = w_tuple_getitem(haystack, i as i64) {
                    if eq_w(item, needle) {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if is_str(haystack) && is_str(needle) {
            let h = w_str_get_value(haystack);
            let n = w_str_get_value(needle);
            return Ok(h.contains(n));
        }
        // dict: key containment (dictobject.py __contains__)
        if is_dict(haystack) {
            return Ok(w_dict_lookup(haystack, needle).is_some());
        }
        // set / frozenset (setobject.py W_BaseSetObject.descr_contains)
        if pyre_object::is_set_or_frozenset(haystack) {
            return Ok(pyre_object::w_set_contains(haystack, needle));
        }
    }
    // Instance __contains__ — PyPy: descroperation.py contains_w
    unsafe {
        if is_instance(haystack) {
            let w_type = w_instance_get_type(haystack);
            if let Some(method) = lookup_in_type_where(w_type, "__contains__") {
                let result = crate::call_function(method, &[haystack, needle]);
                return Ok(is_true(result));
            }
            // Also check per-instance attributes (ATTR_TABLE)
            if let Ok(method) = getattr(haystack, "__contains__") {
                let result = crate::call_function(method, &[haystack, needle]);
                return Ok(is_true(result));
            }
        }
    }
    // Fallback: try iterating with getitem(obj, i) for i=0,1,...
    let mut i = 0i64;
    loop {
        match getitem(haystack, pyre_object::w_int_new(i)) {
            Ok(item) => {
                if eq_w(item, needle) {
                    return Ok(true);
                }
                i += 1;
            }
            Err(_) => return Ok(false), // IndexError → not found
        }
    }
}

/// Compare two objects for equality (returns bool, not PyObjectRef).
/// baseobjspace.py:823-825 `eq_w`: identity first, then `==` truth.
pub fn eq_w(a: PyObjectRef, b: PyObjectRef) -> bool {
    if a == b {
        return true;
    }
    unsafe {
        use pyre_object::*;
        if is_int_like(a) && is_int_like(b) {
            return w_int_get_value(a) == w_int_get_value(b);
        }
        if is_str(a) && is_str(b) {
            return w_str_get_value(a) == w_str_get_value(b);
        }
    }
    compare(a, b, CompareOp::Eq)
        .map(|r| is_true(r))
        .unwrap_or(false)
}

/// Delete item: `del obj[index]`
///
/// PyPy: descroperation.py delitem → dispatches to type-specific __delitem__.
pub fn delitem(obj: PyObjectRef, index: PyObjectRef) -> Result<(), PyError> {
    use pyre_object::*;
    unsafe {
        // `pypy/objspace/std/dictproxyobject.py` exposes no
        // `__delitem__`, so `space.delitem` on a mappingproxy raises
        // `TypeError: 'mappingproxy' object does not support item
        // deletion`.
        if pyre_object::is_dict_proxy(obj) {
            return Err(PyError::type_error(
                "'mappingproxy' object does not support item deletion",
            ));
        }
        if is_list(obj) {
            if is_int(index) {
                let i = w_int_get_value(index);
                let len = w_list_len(obj) as i64;
                let idx = if i < 0 { len + i } else { i };
                if idx >= 0 && idx < len {
                    w_list_pop(obj, idx);
                    return Ok(());
                }
                return Err(PyError::type_error("list index out of range"));
            }
            if is_slice(index) {
                let len = w_list_len(obj) as i64;
                let start = w_slice_get_start(index);
                let stop = w_slice_get_stop(index);
                let s = if is_none(start) {
                    0
                } else {
                    let v = w_int_get_value(start);
                    if v < 0 { (len + v).max(0) } else { v.min(len) }
                } as usize;
                let e = if is_none(stop) {
                    len
                } else {
                    let v = w_int_get_value(stop);
                    if v < 0 { (len + v).max(0) } else { v.min(len) }
                } as usize;
                w_list_delslice(obj, s, e);
                return Ok(());
            }
        }
        if is_dict(obj) {
            return dict_delitem(obj, index);
        }
    }
    // Instance __delitem__ — PyPy: descroperation.py delitem.  Errors from
    // user `__delitem__` propagate (PyPy `space.delitem` raises directly);
    // pyre's `call_function` stashes errors as PY_NULL so use
    // `call_and_check` to recover them.
    unsafe {
        if pyre_object::is_instance(obj) {
            if let Some(method) =
                lookup_in_type_where(pyre_object::w_instance_get_type(obj), "__delitem__")
            {
                crate::builtins::call_and_check(method, &[obj, index])?;
                return Ok(());
            }
        }
    }
    Err(PyError::type_error("object does not support item deletion"))
}

/// Delete item from dict by key.  When the dict has a `dict_storage_proxy`
/// attached (`pick_builtin` lazy-mirror, live `globals()` view, etc.) the
/// deletion must sync to the storage so subsequent storage-keyed lookups
/// (LOAD_GLOBAL builtins fallback) see the removal.  PyPy stores both
/// shapes in the same W_DictMultiObject so the question doesn't arise.
fn dict_delitem(obj: PyObjectRef, key: PyObjectRef) -> Result<(), PyError> {
    use pyre_object::*;
    unsafe {
        let dict = &mut *(obj as *mut dictobject::W_DictObject);
        let entries = &mut *dict.entries;
        for i in 0..entries.len() {
            let eq = if std::ptr::eq(entries[i].0, key) {
                true
            } else if is_int(entries[i].0) && is_int(key) {
                w_int_get_value(entries[i].0) == w_int_get_value(key)
            } else if is_str(entries[i].0) && is_str(key) {
                w_str_get_value(entries[i].0) == w_str_get_value(key)
            } else {
                false
            };
            if eq {
                entries.remove(i);
                dict.len -= 1;
                let proxy = w_dict_get_dict_storage_proxy(obj);
                if !proxy.is_null() && is_str(key) {
                    let ns = &mut *(proxy as *mut crate::DictStorage);
                    ns.remove(w_str_get_value(key));
                }
                return Ok(());
            }
        }
        // entries Vec missed.  When the dict has a storage proxy
        // (Module.__dict__ over its backing DictStorage), the binding
        // may live only in the proxy storage; PyPy's
        // W_DictMultiObject `dictobject.py:descr_delitem` would
        // remove either way.  Mirrors `w_dict_delitem_str`'s
        // storage-fallback shape for str-keyed deletes.
        if is_str(key) {
            return if dictobject::w_dict_delitem_str(obj, w_str_get_value(key)) {
                Ok(())
            } else {
                Err(PyError::key_error("KeyError"))
            };
        }
    }
    Err(PyError::key_error("KeyError"))
}

// py_str and py_repr are defined in display.rs (with __str__/__repr__ dispatch).
// Re-exported via crate::display::*.
