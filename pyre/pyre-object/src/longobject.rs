//! W_LongObject -- arbitrary-precision integer backed by `BigInt`.
//!
//! Used when i64 overflow is detected in `W_IntObject` arithmetic.
//! The JIT never inlines bigint operations; `GuardClass(INT_TYPE)` rejects
//! `W_LongObject` and deoptimizes back to the interpreter.

use malachite_bigint::BigInt;

use crate::pyobject::*;

/// Arbitrary-precision integer object.
///
/// Layout: `[ob_type: *const PyType | value: *mut BigInt]`
/// The `value` pointer owns a heap-allocated `BigInt` (via `Box::into_raw`).
#[repr(C)]
pub struct W_LongObject {
    pub ob_header: PyObject,
    pub value: *mut BigInt,
}

// Safety: BigInt is Send+Sync and W_LongObject only stores a raw pointer
// that is effectively owned.
unsafe impl Send for W_LongObject {}
unsafe impl Sync for W_LongObject {}

/// Field offset of `value` within `W_LongObject`, for potential JIT field access.
pub const LONG_VALUE_OFFSET: usize = std::mem::offset_of!(W_LongObject, value);

/// GC type id assigned to `W_LongObject` at JitDriver init time.
pub const W_LONG_GC_TYPE_ID: u32 = 35;

/// Fixed payload size (`framework.py:811`).
pub const W_LONG_OBJECT_SIZE: usize = std::mem::size_of::<W_LongObject>();

impl crate::lltype::GcType for W_LongObject {
    const TYPE_ID: u32 = W_LONG_GC_TYPE_ID;
    const SIZE: usize = W_LONG_OBJECT_SIZE;
}

/// Allocate a new W_LongObject on the heap.
///
/// Phase 1: uses `Box::leak` (objects are never freed).
pub fn w_long_new(value: BigInt) -> PyObjectRef {
    // W_LongObject shares the `int` type with W_IntObject — the two only
    // differ in their storage layout, not their Python-level identity
    // (PyPy does the same via W_AbstractIntObject's typedef). Wire
    // `w_class` to INT_TYPE.instantiate so `type(x) is int` and
    // `isinstance(x, int)` both hold for long integers.
    let value = crate::lltype::malloc_raw(value);
    crate::lltype::malloc_typed(W_LongObject {
        ob_header: PyObject {
            ob_type: &LONG_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        value,
    }) as PyObjectRef
}

/// Create a W_LongObject from an i64 value.
pub fn w_long_from_i64(v: i64) -> PyObjectRef {
    w_long_new(BigInt::from(v))
}

/// Box a bigint constant into a heap Python int object.
pub fn box_bigint_constant(value: &BigInt) -> PyObjectRef {
    w_long_new(value.clone())
}

/// `W_LongObject._fits_int()` — longobject.py:141 / rbigint.fits_int.
/// True if the value fits in a machine-word integer (i64 on 64-bit).
/// Used by `is_plain_int1` to accept long objects that are in the int range.
#[inline]
pub unsafe fn w_long_fits_int(obj: PyObjectRef) -> bool {
    unsafe {
        let big = w_long_get_value(obj);
        i64::try_from(big).is_ok()
    }
}

/// Extract a reference to the BigInt value from a known W_LongObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_LongObject`.
#[inline]
pub unsafe fn w_long_get_value(obj: PyObjectRef) -> &'static BigInt {
    unsafe {
        let long_obj = obj as *const W_LongObject;
        &*(*long_obj).value
    }
}

/// `rbigint.fits_int()` (`rpython/rlib/rbigint.py:490`) — JIT-callable
/// wrapper. Returns 1 when the W_LongObject's BigInt fits in i64,
/// 0 otherwise. Used as the runtime fits_int guard before
/// `jit_w_long_toint`.
///
/// Unlike `rbigint.toint()`, upstream `fits_int()` is not marked
/// `@jit.elidable`, so keep this call cannot-raise but non-elidable.
pub extern "C" fn jit_w_long_fits_int(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    unsafe { w_long_fits_int(obj) as i64 }
}

/// `W_LongObject.toint()` (`pypy/objspace/std/longobject.py:138`) →
/// `rbigint.toint()` (`rpython/rlib/rbigint.py:465`, `@jit.elidable`).
/// Extract an i64 from a W_LongObject. RPython `toint` raises
/// `OverflowError` when the BigInt does not fit; the elidable
/// trace-time site emits a `fits_int` GUARD_TRUE first
/// (`pypy/objspace/std/listobject.py:2390 is_plain_int1` parity), so
/// the OverflowError path is unreachable in production. Pyre encodes
/// that unreachability as a panic. There is no `_int_w_unsafe` upstream —
/// this is the elidable `toint` after a `fits_int` guard.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_toint(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    unsafe {
        let big = w_long_get_value(obj);
        i64::try_from(big).unwrap_or_else(|_| {
            panic!("jit_w_long_toint: BigInt out of i64 range — fits_int guard violated")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_create_and_read() {
        let obj = w_long_new(BigInt::from(42));
        unsafe {
            assert!(is_long(obj));
            assert!(!is_int(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(42));
        }
    }

    #[test]
    fn test_long_from_i64() {
        let obj = w_long_from_i64(i64::MAX);
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(i64::MAX));
        }
    }

    #[test]
    fn test_long_large_value() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big.clone());
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), big);
        }
    }

    #[test]
    fn test_long_field_offset() {
        assert_eq!(LONG_VALUE_OFFSET, 16);
    }

    #[test]
    fn test_long_type_name_is_int() {
        // Python users see "int" for both W_IntObject and W_LongObject
        assert_eq!(LONG_TYPE.name, "int");
    }

    #[test]
    fn test_jit_w_long_fits_int_in_range() {
        let obj = w_long_from_i64(123);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
        let obj = w_long_from_i64(i64::MAX);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
        let obj = w_long_from_i64(i64::MIN);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
    }

    #[test]
    fn test_jit_w_long_fits_int_out_of_range() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big);
        assert_eq!(jit_w_long_fits_int(obj as i64), 0);
        let big = BigInt::from(i64::MIN) - BigInt::from(1);
        let obj = w_long_new(big);
        assert_eq!(jit_w_long_fits_int(obj as i64), 0);
    }

    #[test]
    fn test_jit_w_long_toint_extracts_i64() {
        let obj = w_long_from_i64(42);
        assert_eq!(jit_w_long_toint(obj as i64), 42);
        let obj = w_long_from_i64(i64::MAX);
        assert_eq!(jit_w_long_toint(obj as i64), i64::MAX);
        let obj = w_long_from_i64(i64::MIN);
        assert_eq!(jit_w_long_toint(obj as i64), i64::MIN);
    }
}
