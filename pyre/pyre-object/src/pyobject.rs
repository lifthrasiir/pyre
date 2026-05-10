//! Core Python object model with `#[repr(C)]` layout for JIT compatibility.
//!
//! Every Python object starts with a `PyObject` header containing a type pointer.
//! Concrete types (W_IntObject, W_BoolObject, etc.) embed this header as their
//! first field, enabling safe pointer casts between `*mut PyObject` and typed pointers.

use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};

/// Type descriptor for Python objects — corresponds to RPython's OBJECT_VTABLE
/// (rclass.py:167-174).
///
/// Each built-in type has a single static `PyType` instance.
/// The JIT uses `GuardClass` on the `ob_type` pointer to specialize code paths,
/// and `GuardSubclass` via `int_between(cls.min, subcls.min, cls.max)`
/// (rclass.py:1133-1137 `ll_issubclass`).
///
/// Fields match OBJECT_VTABLE layout order:
///   subclassrange_min, subclassrange_max, (rtti omitted), name, (instantiate omitted)
///
/// `AtomicI64`/`AtomicPtr` provide interior mutability for static instances:
/// ranges and instantiate are assigned once at init time,
/// mirroring `assign_inheritance_ids` (normalizecalls.py:373-389).
/// The JIT backend reads them at raw offsets — atomics are layout-
/// compatible with their inner types (same size and alignment).
#[repr(C)]
pub struct PyType {
    pub subclassrange_min: AtomicI64,
    pub subclassrange_max: AtomicI64,
    pub name: &'static str,
    /// rclass.py:172 `('instantiate', Ptr(FuncType([], OBJECTPTR)))`.
    ///
    /// RPython stores an instantiate function pointer; pyre caches
    /// the W_TypeObject pointer here instead. rclass.py:739-743
    /// `new_instance` sets `__class__` at allocation — pyre reads
    /// this cached pointer to set `w_class` at allocation time.
    /// Null until `init_typeobjects()` runs.
    pub instantiate: AtomicPtr<PyObject>,
}

/// Common header for all Python objects.
///
/// RPython rclass.py: OBJECT = GcStruct('object', ('typeptr', CLASSTYPE))
///
/// - `ob_type`: static dispatch tag (like RPython's typeptr for guard_class)
/// - `w_class`: Python class pointer (like RPython's gettypefor(typeptr) result)
///
/// `w_class` is set at allocation time when the type registry is available,
/// or populated lazily by `init_typeobjects()` for static singletons.
#[repr(C)]
pub struct PyObject {
    pub ob_type: *const PyType,
    pub w_class: *mut PyObject,
}

/// The universal Python object reference — a raw pointer to `PyObject`.
///
/// `pyre` currently passes this through the JIT as an integer-sized raw pointer.
/// Phase 1 uses leaked Box allocations; a proper GC will replace this later.
// Safety: PyType instances are read-only static data, safe to share across threads.
unsafe impl Sync for PyType {}
unsafe impl Send for PyType {}

// Safety: PyObject's ob_type points to immutable static PyType instances.
unsafe impl Sync for PyObject {}
unsafe impl Send for PyObject {}

pub type PyObjectRef = *mut PyObject;

/// Null object reference, used as a sentinel for "no value".
pub const PY_NULL: PyObjectRef = std::ptr::null_mut();

// ── Type identity ─────────────────────────────────────────────────────

/// Construct a PyType with zeroed subclass ranges.
/// Ranges are assigned at init time by `assign_subclass_range()`.
pub const fn new_pytype(name: &'static str) -> PyType {
    PyType {
        subclassrange_min: AtomicI64::new(0),
        subclassrange_max: AtomicI64::new(0),
        name,
        instantiate: AtomicPtr::new(std::ptr::null_mut()),
    }
}

/// rclass.py:739-743 parity — cache the W_TypeObject on the PyType
/// so allocators can set `w_class` at allocation time.
///
/// Called by `init_typeobjects()` for each built-in type.
pub fn set_instantiate(tp: &PyType, w_typeobject: PyObjectRef) {
    tp.instantiate.store(w_typeobject, Ordering::Release);
}

/// Read the cached W_TypeObject from a PyType.
///
/// Returns the W_TypeObject (for `w_class`), or null if not yet initialized
/// (bootstrap phase before `init_typeobjects()`).
#[inline]
pub fn get_instantiate(tp: &PyType) -> PyObjectRef {
    tp.instantiate.load(Ordering::Acquire)
}

// Compile-time verification: AtomicI64/AtomicPtr are layout-compatible
// with i64/*mut T so the JIT can read PyType fields at raw offsets.
// Also verify OBJECT_VTABLE field order: subclassrange_min @ 0, max @ 8.
const _: () = {
    assert!(std::mem::size_of::<AtomicI64>() == std::mem::size_of::<i64>());
    assert!(std::mem::align_of::<AtomicI64>() == std::mem::align_of::<i64>());
    assert!(std::mem::size_of::<AtomicPtr<PyObject>>() == std::mem::size_of::<*mut PyObject>());
    assert!(std::mem::offset_of!(PyType, subclassrange_min) == 0);
    assert!(std::mem::offset_of!(PyType, subclassrange_max) == 8);
};

pub static INT_TYPE: PyType = new_pytype("int");
pub static BOOL_TYPE: PyType = new_pytype("bool");
pub static FLOAT_TYPE: PyType = new_pytype("float");
pub static STR_TYPE: PyType = new_pytype("str");
pub static LIST_TYPE: PyType = new_pytype("list");
pub static TUPLE_TYPE: PyType = new_pytype("tuple");
pub static DICT_TYPE: PyType = new_pytype("dict");
pub static LONG_TYPE: PyType = new_pytype("int");
pub static NONE_TYPE: PyType = new_pytype("NoneType");
pub static NOTIMPLEMENTED_TYPE: PyType = new_pytype("NotImplementedType");
pub static ELLIPSIS_TYPE: PyType = new_pytype("ellipsis");
pub static MODULE_TYPE: PyType = new_pytype("module");
pub static MAPPING_PROXY_TYPE: PyType = new_pytype("mappingproxy");
pub static TYPE_TYPE: PyType = new_pytype("type");
pub static INSTANCE_TYPE: PyType = new_pytype("object");

/// Field offset of `ob_type` within PyObject, for JIT field access.
pub const OB_TYPE_OFFSET: usize = std::mem::offset_of!(PyObject, ob_type);

/// Field offset of `w_class` within PyObject, for JIT field access.
/// RPython: this corresponds to reading typeptr + gettypefor (fused into one field).
pub const W_CLASS_OFFSET: usize = std::mem::offset_of!(PyObject, w_class);

/// Field offset of `subclassrange_min` within PyType (OBJECT_VTABLE).
/// rclass.py:168 — first field in OBJECT_VTABLE.
pub const SUBCLASSRANGE_MIN_OFFSET: usize = std::mem::offset_of!(PyType, subclassrange_min);

/// Field offset of `subclassrange_max` within PyType (OBJECT_VTABLE).
/// rclass.py:169 — second field in OBJECT_VTABLE.
pub const SUBCLASSRANGE_MAX_OFFSET: usize = std::mem::offset_of!(PyType, subclassrange_max);

/// rclass.py:1126-1127 `ll_cast_to_object(obj)`.
///
/// In RPython this casts a typed pointer to `OBJECTPTR`. In pyre all
/// objects are already `PyObjectRef`, so this is an identity function
/// kept for structural parity.
#[inline]
pub fn ll_cast_to_object(obj: PyObjectRef) -> PyObjectRef {
    obj
}

/// rclass.py:1130-1131 `ll_type(obj)`.
///
/// Extract the type pointer (CLASSTYPE) from an object.
///
/// # Safety
/// `obj` must be a valid non-null `PyObject`.
#[inline]
pub unsafe fn ll_type(obj: PyObjectRef) -> *const PyType {
    unsafe { (*obj).ob_type }
}

/// rclass.py:1133-1137 `ll_issubclass(subcls, cls)`.
///
/// O(1) subclass check via preorder numbering:
///   `int_between(cls.subclassrange_min, subcls.subclassrange_min, cls.subclassrange_max)`
#[inline]
pub fn ll_issubclass(subcls: &PyType, cls: &PyType) -> bool {
    let cls_min = cls.subclassrange_min.load(Ordering::Relaxed);
    let subcls_min = subcls.subclassrange_min.load(Ordering::Relaxed);
    let cls_max = cls.subclassrange_max.load(Ordering::Relaxed);
    // int_between(a, b, c) ≡ a <= b < c
    cls_min <= subcls_min && subcls_min < cls_max
}

/// rclass.py:1139-1140 `ll_issubclass_const(subcls, minid, maxid)`.
///
/// Variant of `ll_issubclass` where the class bounds are already known
/// constants. Used by the JIT when the target class is constant-folded.
#[inline]
pub fn ll_issubclass_const(subcls: &PyType, minid: i64, maxid: i64) -> bool {
    let subcls_min = subcls.subclassrange_min.load(Ordering::Relaxed);
    // int_between(a, b, c) ≡ a <= b < c
    minid <= subcls_min && subcls_min < maxid
}

/// rclass.py:1143-1147 `ll_isinstance(obj, cls)`.
///
/// RPython-level type check: reads `obj.typeptr` (= `ob_type`) and checks
/// subclass ranges. This checks the **RPython class** (W_IntObject,
/// W_ListObject, etc.), NOT the Python-level class. All user-defined
/// instances share `INSTANCE_TYPE` as their RPython class, just as
/// RPython groups them under W_ObjectObject's vtable.
///
/// For Python-level `isinstance()`, use `issubtype_w` (MRO walk on
/// `w_class`), not this function.
///
/// # Safety
/// `obj` must be a valid non-null `PyObject`.
#[inline]
pub unsafe fn ll_isinstance(obj: PyObjectRef, cls: &PyType) -> bool {
    if obj.is_null() {
        return false;
    }
    let obj_cls = unsafe { &*(*obj).ob_type };
    ll_issubclass(obj_cls, cls)
}

/// rclass.py:1173-1178 `ll_inst_type(obj)`.
///
/// Return the typeptr if obj is non-null, null otherwise.
///
/// # Safety
/// If non-null, `obj` must be a valid `PyObject`.
#[inline]
pub unsafe fn ll_inst_type(obj: PyObjectRef) -> *const PyType {
    unsafe {
        if !obj.is_null() {
            (*obj).ob_type
        } else {
            std::ptr::null()
        }
    }
}

/// Write subclass ranges to a `PyType` instance.
///
/// Mirrors `assign_inheritance_ids` (normalizecalls.py:373-389) which
/// assigns `classdef.minid` / `classdef.maxid` to each vtable entry.
///
/// Uses `Relaxed` ordering: ranges are written once at init time
/// before any concurrent reads.
pub fn assign_subclass_range(tp: &PyType, min: i64, max: i64) {
    tp.subclassrange_min.store(min, Ordering::Relaxed);
    tp.subclassrange_max.store(max, Ordering::Relaxed);
}

/// Every built-in `PyType` static that represents a full `PyObject`
/// subtype (i.e. instances carry `ob_type` at offset 0, matching
/// `rclass.OBJECT` layout), paired with its parent class.
///
/// Modelled on RPython's `assign_inheritance_ids`
/// (normalizecalls.py:373-389) which walks `classdef.getmro()` to build
/// the reversed-MRO witness for each class. The JIT registers each
/// `(type, parent)` pair with the GC via `register_vtable_for_type`,
/// using the parent typeid as `TypeInfo::object_subclass`'s `parent`
/// argument so the resulting `subclassrange_{min,max}` faithfully
/// represents the `rclass.OBJECT` hierarchy. `GUARD_SUBCLASS` then
/// resolves to `int_between(cls.min, subcls.min, cls.max)` per
/// rclass.py:1133-1137 `ll_issubclass`.
///
/// `INSTANCE_TYPE` (the `name = "object"` root) is intentionally
/// absent: it is registered separately as the `rclass.OBJECT` root
/// with no parent. `INT_TYPE` and `FLOAT_TYPE` are also absent: they
/// get their own ids (`W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID`)
/// because the JIT backend allocates W_IntObject / W_FloatObject
/// through NewWithVtable and needs the correct payload size.
pub fn all_foreign_pytypes() -> &'static [(&'static PyType, &'static PyType)] {
    static PYTYPES: &[(&PyType, &PyType)] = &[
        // bool inherits from int (objectobject.py W_BoolObject.typedef).
        (&BOOL_TYPE, &INT_TYPE),
        (&STR_TYPE, &INSTANCE_TYPE),
        (&LIST_TYPE, &INSTANCE_TYPE),
        (&TUPLE_TYPE, &INSTANCE_TYPE),
        (&DICT_TYPE, &INSTANCE_TYPE),
        // longobject.py W_LongObject — Python 3 unifies long under int,
        // but pyre carries a separate static for the BigInt-backed flavour.
        (&LONG_TYPE, &INSTANCE_TYPE),
        (&NONE_TYPE, &INSTANCE_TYPE),
        (&NOTIMPLEMENTED_TYPE, &INSTANCE_TYPE),
        (&ELLIPSIS_TYPE, &INSTANCE_TYPE),
        (&MODULE_TYPE, &INSTANCE_TYPE),
        (&MAPPING_PROXY_TYPE, &INSTANCE_TYPE),
        (&TYPE_TYPE, &INSTANCE_TYPE),
        (&crate::superobject::SUPER_TYPE, &INSTANCE_TYPE),
        (&crate::bytearrayobject::BYTEARRAY_TYPE, &INSTANCE_TYPE),
        (&crate::bytesobject::BYTES_TYPE, &INSTANCE_TYPE),
        (&crate::generatorobject::GENERATOR_TYPE, &INSTANCE_TYPE),
        (&crate::unionobject::UNION_TYPE, &INSTANCE_TYPE),
        (&crate::rangeobject::RANGE_ITER_TYPE, &INSTANCE_TYPE),
        (&crate::rangeobject::SEQ_ITER_TYPE, &INSTANCE_TYPE),
        (&crate::cellobject::CELL_TYPE, &INSTANCE_TYPE),
        (&crate::methodobject::METHOD_TYPE, &INSTANCE_TYPE),
        (&crate::propertyobject::PROPERTY_TYPE, &INSTANCE_TYPE),
        (&crate::propertyobject::STATICMETHOD_TYPE, &INSTANCE_TYPE),
        (&crate::propertyobject::CLASSMETHOD_TYPE, &INSTANCE_TYPE),
        (&crate::excobject::EXCEPTION_TYPE, &INSTANCE_TYPE),
        (&crate::sliceobject::SLICE_TYPE, &INSTANCE_TYPE),
        (&crate::setobject::SET_TYPE, &INSTANCE_TYPE),
        (&crate::setobject::FROZENSET_TYPE, &INSTANCE_TYPE),
        (&crate::memberobject::MEMBER_TYPE, &INSTANCE_TYPE),
    ];
    PYTYPES
}

// ── Type checks ───────────────────────────────────────────────────────

/// Check if an object is of a given type (pointer identity comparison).
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn py_type_check(obj: PyObjectRef, tp: &PyType) -> bool {
    !obj.is_null() && unsafe { std::ptr::eq((*obj).ob_type, tp as *const PyType) }
}

#[inline]
pub unsafe fn is_int(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &INT_TYPE) || py_type_check(obj, &BOOL_TYPE) }
}

#[inline]
pub unsafe fn is_bool(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BOOL_TYPE) }
}

#[inline]
pub unsafe fn is_float(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FLOAT_TYPE) }
}

#[inline]
pub unsafe fn is_long(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LONG_TYPE) }
}

#[inline]
pub unsafe fn is_int_or_long(obj: PyObjectRef) -> bool {
    unsafe { is_int(obj) || is_long(obj) }
}

#[inline]
pub unsafe fn is_list(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LIST_TYPE) }
}

/// Recognise any of the four tuple variants —
/// canonical `W_TupleObject` plus the three `W_SpecialisedTupleObject_*`
/// arity-2 specialisations from
/// `pypy/objspace/std/specialisedtupleobject.py`. All four share the
/// same Python `tuple` typedef in pypy; pyre encodes that by giving
/// each variant a distinct `ob_type` (RPython-vtable equivalent) while
/// `w_class` always resolves to the canonical `tuple` class object.
#[inline]
pub unsafe fn is_tuple(obj: PyObjectRef) -> bool {
    use crate::specialisedtupleobject::{
        SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
    };
    unsafe {
        py_type_check(obj, &TUPLE_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_II_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_FF_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_OO_TYPE)
    }
}

#[inline]
pub unsafe fn is_dict(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DICT_TYPE) }
}

#[inline]
pub unsafe fn is_none(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NONE_TYPE) }
}

#[inline]
pub unsafe fn is_not_implemented(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NOTIMPLEMENTED_TYPE) }
}

#[inline]
pub unsafe fn is_ellipsis(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ELLIPSIS_TYPE) }
}
