use crate::PyError;
use crate::bytecode::{BinaryOperator, ComparisonOperator};
use pyre_object::{PyObjectRef, w_bool_from};

use crate::{
    CompareOp, add, and_, compare, floordiv, getitem, invert, is_true, lshift, mod_, mul, neg, or_,
    pow, rshift, sub, truediv, xor,
};

pub fn binary_value(
    a: PyObjectRef,
    b: PyObjectRef,
    op: BinaryOperator,
) -> Result<PyObjectRef, PyError> {
    let a = crate::baseobjspace::unwrap_cell(a);
    let b = crate::baseobjspace::unwrap_cell(b);
    match op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => add(a, b),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => sub(a, b),
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => mul(a, b),
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => floordiv(a, b),
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => mod_(a, b),
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => truediv(a, b),
        BinaryOperator::Power | BinaryOperator::InplacePower => pow(a, b),
        BinaryOperator::Lshift | BinaryOperator::InplaceLshift => lshift(a, b),
        BinaryOperator::Rshift | BinaryOperator::InplaceRshift => rshift(a, b),
        BinaryOperator::And | BinaryOperator::InplaceAnd => and_(a, b),
        BinaryOperator::Or => or_(a, b),
        BinaryOperator::InplaceOr => {
            // `pypy/objspace/std/dictproxyobject.py:67 descr_ior` —
            // mappingproxy `__ior__` raises TypeError unconditionally
            // (the proxy is read-only, in-place merge cannot be
            // honoured even if `__or__` would accept the rhs).  PyPy
            // `descroperation.inplace_or` tries `__ior__` first; for
            // mappingproxy that resolves to the raise, so the
            // surrounding INPLACE_OR opcode propagates.  Pyre's
            // dispatcher otherwise routes Inplace/non-Inplace to the
            // same `or_`, so detect proxy LHS up front to preserve the
            // raise.
            unsafe {
                if pyre_object::is_dict_proxy(a) {
                    return Err(PyError::type_error(
                        "'|=' is not supported by mappingproxy; use '|' instead",
                    ));
                }
            }
            or_(a, b)
        }
        BinaryOperator::Xor | BinaryOperator::InplaceXor => xor(a, b),
        BinaryOperator::Subscr => getitem(a, b),
        _ => Err(PyError::type_error(format!(
            "binary operation {op:?} not yet implemented"
        ))),
    }
}

pub fn binary_value_from_tag(
    a: PyObjectRef,
    b: PyObjectRef,
    op_tag: i64,
) -> Result<PyObjectRef, PyError> {
    match op_tag {
        0 => add(a, b),
        1 => sub(a, b),
        2 => mul(a, b),
        3 => floordiv(a, b),
        4 => mod_(a, b),
        5 => truediv(a, b),
        6 => getitem(a, b),
        7 => pow(a, b),
        8 => lshift(a, b),
        9 => rshift(a, b),
        10 => and_(a, b),
        11 => or_(a, b),
        12 => xor(a, b),
        _ => Err(PyError::type_error(format!(
            "unsupported binary op tag: {op_tag}"
        ))),
    }
}

pub fn compare_value(
    a: PyObjectRef,
    b: PyObjectRef,
    op: ComparisonOperator,
) -> Result<PyObjectRef, PyError> {
    let a = crate::baseobjspace::unwrap_cell(a);
    let b = crate::baseobjspace::unwrap_cell(b);
    let cmp_op = match op {
        ComparisonOperator::Less => CompareOp::Lt,
        ComparisonOperator::LessOrEqual => CompareOp::Le,
        ComparisonOperator::Greater => CompareOp::Gt,
        ComparisonOperator::GreaterOrEqual => CompareOp::Ge,
        ComparisonOperator::Equal => CompareOp::Eq,
        ComparisonOperator::NotEqual => CompareOp::Ne,
    };
    compare(a, b, cmp_op)
}

pub fn compare_value_from_tag(
    a: PyObjectRef,
    b: PyObjectRef,
    op_tag: i64,
) -> Result<PyObjectRef, PyError> {
    let op = match op_tag {
        0 => CompareOp::Lt,
        1 => CompareOp::Le,
        2 => CompareOp::Gt,
        3 => CompareOp::Ge,
        4 => CompareOp::Eq,
        5 => CompareOp::Ne,
        _ => {
            return Err(PyError::type_error(format!(
                "unsupported compare op tag: {op_tag}"
            )));
        }
    };
    compare(a, b, op)
}

pub fn unary_negative_value(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let value = crate::baseobjspace::unwrap_cell(value);
    neg(value)
}

pub fn unary_invert_value(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let value = crate::baseobjspace::unwrap_cell(value);
    invert(value)
}

pub fn truth_value(value: PyObjectRef) -> bool {
    let value = crate::baseobjspace::unwrap_cell(value);
    is_true(value)
}

pub fn bool_value_from_truth(value: bool) -> PyObjectRef {
    w_bool_from(value)
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_truth_value(value: i64) -> i64 {
    truth_value(value as PyObjectRef) as i64
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_bool_value_from_truth(value: i64) -> i64 {
    bool_value_from_truth(value != 0) as i64
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_binary_value_from_tag(a: i64, b: i64, op_tag: i64) -> i64 {
    match binary_value_from_tag(a as PyObjectRef, b as PyObjectRef, op_tag) {
        Ok(value) => value as i64,
        Err(err) => panic!("binary op failed in JIT: {err}"),
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_compare_value_from_tag(a: i64, b: i64, op_tag: i64) -> i64 {
    match compare_value_from_tag(a as PyObjectRef, b as PyObjectRef, op_tag) {
        Ok(value) => value as i64,
        Err(err) => panic!("compare op failed in JIT: {err}"),
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_unary_negative_value(value: i64) -> i64 {
    match unary_negative_value(value as PyObjectRef) {
        Ok(result) => result as i64,
        Err(err) => panic!("unary negative failed in JIT: {err}"),
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_unary_invert_value(value: i64) -> i64 {
    match unary_invert_value(value as PyObjectRef) {
        Ok(result) => result as i64,
        Err(err) => panic!("unary invert failed in JIT: {err}"),
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_getitem(obj: i64, index: i64) -> i64 {
    match getitem(obj as PyObjectRef, index as PyObjectRef) {
        Ok(value) => value as i64,
        Err(err) => panic!("getitem failed in JIT: {err}"),
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_setitem(obj: i64, index: i64, value: i64) -> i64 {
    match crate::setitem(
        obj as PyObjectRef,
        index as PyObjectRef,
        value as PyObjectRef,
    ) {
        Ok(_) => 0,
        Err(err) => panic!("setitem failed in JIT: {err}"),
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_getattr(obj: i64, name_ptr: i64, name_len: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    let name = std::str::from_utf8(bytes).expect("invalid attr name in JIT");
    match crate::getattr(obj as PyObjectRef, name) {
        Ok(value) => value as i64,
        Err(err) => panic!("getattr failed in JIT: {err}"),
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_setattr(obj: i64, name_ptr: i64, name_len: i64, value: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    let name = std::str::from_utf8(bytes).expect("invalid attr name in JIT");
    match crate::setattr(obj as PyObjectRef, name, value as PyObjectRef) {
        Ok(_) => 0,
        Err(err) => panic!("setattr failed in JIT: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_object::{w_bool_get_value, w_int_get_value, w_int_new};

    #[test]
    fn test_binary_value_reuses_objspace_dispatch() {
        let result = binary_value(w_int_new(8), w_int_new(3), BinaryOperator::Subtract)
            .expect("binary dispatch should succeed");
        unsafe {
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_compare_value_reuses_objspace_dispatch() {
        let result = compare_value(w_int_new(2), w_int_new(7), ComparisonOperator::Less)
            .expect("compare dispatch should succeed");
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_truth_and_unary_helpers_share_objspace_semantics() {
        assert!(!truth_value(w_int_new(0)));
        let neg = unary_negative_value(w_int_new(4)).expect("unary negate should succeed");
        let inv = unary_invert_value(w_int_new(5)).expect("unary invert should succeed");
        unsafe {
            assert_eq!(w_int_get_value(neg), -4);
            assert_eq!(w_int_get_value(inv), !5);
        }
    }

    #[test]
    fn test_jit_abi_helpers_share_same_objspace_semantics() {
        assert_eq!(jit_truth_value(w_int_new(0) as i64), 0);
        let neg = jit_unary_negative_value(w_int_new(4) as i64) as PyObjectRef;
        let cmp =
            jit_compare_value_from_tag(w_int_new(2) as i64, w_int_new(7) as i64, 0) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(neg), -4);
            assert!(w_bool_get_value(cmp));
        }
    }

    #[test]
    fn test_jit_getitem_and_setitem_share_objspace_semantics() {
        let list = pyre_object::w_list_new(vec![w_int_new(2), w_int_new(4)]);
        let item = jit_getitem(list as i64, w_int_new(1) as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(item), 4);
        }
        assert_eq!(
            jit_setitem(list as i64, w_int_new(0) as i64, w_int_new(9) as i64),
            0
        );
        let updated = jit_getitem(list as i64, w_int_new(0) as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(updated), 9);
        }
    }
}
