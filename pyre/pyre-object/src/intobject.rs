//! W_IntObject — Python `int` type backed by i64.
//!
//! Phase 1 uses a fixed i64 representation. BigInt support (like PyPy's
//! `W_LongObject`) will be added in Phase 4.

use std::sync::LazyLock;

use crate::pyobject::*;

/// Python integer object.
///
/// Layout: `[ob_header: PyObject { ob_type, w_class } | intval: i64]`
/// The JIT reads `intval` via `GetfieldGcI` at `INT_INTVAL_OFFSET`.
#[repr(C)]
pub struct W_IntObject {
    pub ob_header: PyObject,
    pub intval: i64,
}

/// Field offset of `intval` within `W_IntObject`, for JIT field access.
pub const INT_INTVAL_OFFSET: usize = std::mem::offset_of!(W_IntObject, intval);

/// GC type id assigned to `W_IntObject` at JitDriver init time. Held
/// as a constant here (rather than runtime-queried) so the allocation
/// hook can reach it without a back-channel. `pyre-jit/src/eval.rs`
/// asserts the same id is returned by `gc.register_type(...)` so any
/// mismatch panics on startup instead of silently misclassifying the
/// type at collection time.
pub const W_INT_GC_TYPE_ID: u32 = 1;

// ── Prebuilt-int cache ───────────────────────────────────────────────
//
// `pypy/config/pypyoption.py:206-213` parity:
//   withprebuiltint  default False — prebuild commonly used int objects
//   prebuiltintfrom  default -5    — lowest integer which is prebuilt
//   prebuiltintto    default 100   — highest integer (stop-exclusive)
//
// `pypy/objspace/std/intobject.py:873-897` `setup_prebuilt` / `wrapint`
// consult these three constants. Pyre keeps them as translation-time
// constants since pyre has no live `space.config` surface — the exact
// PyPy default (`WITHPREBUILTINT=false`) is used to keep `wrapint`'s
// allocation behaviour line-by-line with PyPy default.

pub const WITHPREBUILTINT: bool = false;
pub const PREBUILTINTFROM: i64 = -5;
pub const PREBUILTINTTO: i64 = 100;
pub const W_INT_OBJECT_SIZE: usize = std::mem::size_of::<W_IntObject>();

impl crate::lltype::GcType for W_IntObject {
    const TYPE_ID: u32 = W_INT_GC_TYPE_ID;
    const SIZE: usize = W_INT_OBJECT_SIZE;
}

/// `intobject.py:873-880 setup_prebuilt`. Empty when
/// `WITHPREBUILTINT=false`; populated `[PREBUILTINTFROM, PREBUILTINTTO)`
/// (stop-exclusive) when `true`.
static SMALL_INTS: LazyLock<Vec<W_IntObject>> = LazyLock::new(|| {
    if !WITHPREBUILTINT {
        return Vec::new();
    }
    (PREBUILTINTFROM..PREBUILTINTTO)
        .map(|v| W_IntObject {
            ob_header: PyObject {
                ob_type: &INT_TYPE as *const PyType,
                w_class: std::ptr::null_mut(),
            },
            intval: v,
        })
        .collect()
});

/// `pypy/objspace/std/intobject.py:883-897 wrapint` parity.
///
/// `withprebuiltint=False` (PyPy default) → always allocate fresh,
/// matching upstream `return W_IntObject(x)`. With the flag enabled
/// and `value` inside `[PREBUILTINTFROM, PREBUILTINTTO)` the cache
/// returns the pre-allocated entry; outside the range we allocate
/// (`instantiate(W_IntObject)` upstream).
///
/// The allocation path goes through [`crate::lltype::malloc_typed`]
/// (Task #145 Step 2), which carries `W_INT_GC_TYPE_ID` +
/// `W_INT_OBJECT_SIZE` via the [`crate::lltype::GcType`] impl above —
/// the Rust analog of `gct_fv_gc_malloc`'s compile-time `c_type_id`
/// / `c_size` (`rpython/memory/gctransform/framework.py:807-811`).
/// Phase 1: `malloc_typed` is `Box::into_raw`; future GC integration
/// replaces only that body, this constructor stays unchanged.
#[inline]
pub fn w_int_new(value: i64) -> PyObjectRef {
    if WITHPREBUILTINT && value >= PREBUILTINTFROM && value < PREBUILTINTTO {
        let idx = (value - PREBUILTINTFROM) as usize;
        return (&SMALL_INTS[idx] as *const W_IntObject).cast_mut() as PyObjectRef;
    }
    crate::lltype::malloc_typed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: value,
    }) as PyObjectRef
}

/// Create a W_IntObject bypassing the small-int cache.
///
/// Used for int subclass instances that need unique object identity
/// (so per-object attributes in ATTR_TABLE don't collide).
pub fn w_int_new_unique(value: i64) -> PyObjectRef {
    crate::lltype::malloc_typed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: value,
    }) as PyObjectRef
}

/// Return the address of INT_TYPE for JIT type-id validation.
#[inline]
pub fn w_int_type_id() -> usize {
    &INT_TYPE as *const PyType as usize
}

/// Extract the i64 value from a known W_IntObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_IntObject`.
#[inline]
pub unsafe fn w_int_get_value(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_IntObject)).intval }
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_w_int_new(value: i64) -> i64 {
    w_int_new(value) as i64
}

/// True iff `value` falls inside the prebuilt-int cache range AND
/// the cache is enabled. Mirrors PyPy's `wrapint` in-range branch
/// (`intobject.py:891-895`).
#[inline]
pub fn w_int_small_cached(value: i64) -> bool {
    WITHPREBUILTINT && (PREBUILTINTFROM..PREBUILTINTTO).contains(&value)
}

#[inline]
pub fn w_int_small_cache_base_ptr() -> PyObjectRef {
    SMALL_INTS.as_ptr().cast_mut() as PyObjectRef
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_create_and_read() {
        let obj = w_int_new(42);
        unsafe {
            assert!(is_int(obj));
            assert!(!is_bool(obj));
            assert_eq!(w_int_get_value(obj), 42);
        }
    }

    #[test]
    fn test_int_negative() {
        let obj = w_int_new(-7);
        unsafe {
            assert_eq!(w_int_get_value(obj), -7);
        }
    }

    #[test]
    fn test_int_field_offset() {
        assert_eq!(INT_INTVAL_OFFSET, 16); // after PyObject { ob_type(8) + w_class(8) }
    }

    /// `intobject.py:884 wrapint` parity: with `withprebuiltint=False`
    /// (PyPy default, mirrored by `WITHPREBUILTINT=false`) every call
    /// allocates a fresh `W_IntObject`, so identity is per-call.
    #[test]
    fn test_w_int_new_identity_matches_config() {
        let a = w_int_new(42);
        let b = w_int_new(42);
        if WITHPREBUILTINT && (PREBUILTINTFROM..PREBUILTINTTO).contains(&42) {
            assert_eq!(a, b, "in-cache values share the prebuilt pointer");
        } else {
            assert_ne!(a, b, "no cache: each wrapint allocates fresh");
        }
        unsafe {
            assert_eq!(w_int_get_value(a), 42);
            assert_eq!(w_int_get_value(b), 42);
        }
    }

    /// `intobject.py:891-895 wrapint` in-cache branch — only meaningful
    /// when `WITHPREBUILTINT=true`. Skipped otherwise (no prebuilt
    /// pool means there is no boundary to test).
    #[test]
    fn test_prebuilt_cache_boundary() {
        if !WITHPREBUILTINT {
            return;
        }
        let low = w_int_new(PREBUILTINTFROM);
        let high = w_int_new(PREBUILTINTTO - 1);
        unsafe {
            assert_eq!(w_int_get_value(low), PREBUILTINTFROM);
            assert_eq!(w_int_get_value(high), PREBUILTINTTO - 1);
        }
        assert_eq!(low, w_int_new(PREBUILTINTFROM));
        assert_eq!(high, w_int_new(PREBUILTINTTO - 1));
    }

    #[test]
    fn test_outside_cache_allocates_fresh() {
        let a = w_int_new(1000);
        let b = w_int_new(1000);
        assert_ne!(a, b, "out-of-cache ints allocate fresh per wrapint");
        unsafe {
            assert_eq!(w_int_get_value(a), 1000);
            assert_eq!(w_int_get_value(b), 1000);
        }
    }

    #[test]
    fn test_w_int_gc_type_id_matches_descr() {
        // Guard against drift between the constant colocated with
        // `W_IntObject` and the id that `pyre-jit/src/eval.rs`
        // asserts at JitDriver init. See `descr.rs` re-export.
        assert_eq!(W_INT_GC_TYPE_ID, 1);
    }
}
