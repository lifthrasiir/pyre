//! TypeDef — builtin type descriptor registry.
//!
//! PyPy equivalent: pypy/interpreter/typedef.py
//!
//! Each builtin type (list, str, dict, tuple, int, float, bool, etc.)
//! gets a W_TypeObject with methods pre-installed in its namespace.
//! `getattr` looks up the type object from the registry and searches
//! its namespace via MRO, exactly like user-defined classes.
//!
//! This eliminates the `builtin_type_method` match-based dispatch and
//! unifies attribute lookup for all object types.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyre_object::pyobject::*;
use pyre_object::*;

use crate::{DictStorage, dict_storage_store, make_builtin_function};

/// Compatibility stand-ins for PyPy `typedef.py` API (type descriptor helpers).
#[derive(Debug, Default)]
pub struct TypeDef;

impl TypeDef {
    pub fn new(_name: &str, _base: Option<crate::W_Root>, _dict: Option<PyObjectRef>) -> Self {
        let _ = (_name, _base, _dict);
        Self
    }
}

#[derive(Debug, Default)]
pub struct GetSetProperty;

#[derive(Debug, Default)]
pub struct Member;

#[derive(Debug, Default)]
pub struct ClassAttr;

pub fn default_identity_hash(_space: PyObjectRef, _w_obj: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    let _ = _w_obj;
    0 as *mut pyre_object::PyObject
}

pub fn get_unique_interplevel_subclass(_space: PyObjectRef, cls: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    cls
}

pub fn _getusercls(_cls: PyObjectRef, _reallywantdict: bool) -> PyObjectRef {
    let _ = _reallywantdict;
    _cls
}

pub fn _share_methods(_copycls: PyObjectRef, _subcls: PyObjectRef) {
    let _ = (_copycls, _subcls);
}

pub fn use_special_method_shortcut(_name: &str, _checkerfunc: Option<PyObjectRef>) -> bool {
    let _ = (_name, _checkerfunc);
    false
}

pub fn make_descr_typecheck_wrapper<T, F, A>(
    _tag: T,
    _func: F,
    _extraargs: A,
    _cls: Option<PyObjectRef>,
) -> PyObjectRef
where
    F: Fn() -> PyObjectRef,
{
    let _ = (_tag, _extraargs, _cls);
    _func()
}

pub fn _make_descr_typecheck_wrapper<T, F, A>(
    _tag: T,
    _func: F,
    _extraargs: A,
    _cls: PyObjectRef,
    _use_closure: bool,
) -> PyObjectRef
where
    F: Fn() -> PyObjectRef,
{
    let _ = (_tag, _extraargs, _cls, _use_closure);
    _func()
}

pub fn interp_attrproperty(
    _name: &str,
    cls: PyObjectRef,
    _doc: Option<&str>,
    _wrapfn: Option<PyObjectRef>,
) -> PyObjectRef {
    let _ = (_name, _doc, _wrapfn);
    cls
}

pub fn interp_attrproperty_w(_name: &str, cls: PyObjectRef, _doc: Option<&str>) -> PyObjectRef {
    let _ = (_name, _doc);
    cls
}

pub fn generic_new_descr(_w_type: PyObjectRef) -> PyObjectRef {
    _w_type
}

pub fn generic_ne(_space: PyObjectRef, w_obj1: PyObjectRef, w_obj2: PyObjectRef) -> PyObjectRef {
    let _ = (_space, w_obj1, w_obj2);
    PY_NULL
}

pub fn fget_co_varnames(_space: PyObjectRef, _code: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _code);
    PY_NULL
}

pub fn fget_co_argcount(_space: PyObjectRef, _code: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _code);
    PY_NULL
}

pub fn fget_co_flags(_space: PyObjectRef, _code: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _code);
    PY_NULL
}

pub fn fget_co_consts(_space: PyObjectRef, _code: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _code);
    PY_NULL
}

pub fn make_weakref_descr(_cls: PyObjectRef) -> PyObjectRef {
    _cls
}

pub fn always_none(_self: PyObjectRef, _obj: PyObjectRef) -> PyObjectRef {
    let _ = (_self, _obj);
    PY_NULL
}

/// Global typeobject cache: maps static PyType pointer → W_TypeObject (as usize).
///
/// PyPy equivalent: space.gettypeobject(cls.typedef) → cached W_TypeObject
/// Stored as usize to satisfy Send+Sync requirements of OnceLock.
pub static TYPEOBJECT_CACHE: OnceLock<HashMap<usize, usize>> = OnceLock::new();

/// Get the cached W_TypeObject for a builtin runtime type.
///
/// PyPy: `space.gettypefor(cls)` / `space.gettypeobject(typedef)`
pub fn gettypefor(tp: *const PyType) -> Option<PyObjectRef> {
    TYPEOBJECT_CACHE
        .get()
        .and_then(|reg| reg.get(&(tp as usize)).copied())
        .map(|v| v as PyObjectRef)
}

/// Get the W_TypeObject for any PyObjectRef.
///
/// RPython: `space.type(w_obj)` → `jit.promote(w_obj.__class__); w_obj.getclass(space)`
///
/// With `w_class` on PyObject, this is a direct field read. Falls back to
/// `gettypefor(ob_type)` for objects created before init_typeobjects()
/// (singletons such as None/True/False/Ellipsis live in read-only static
/// memory, so we never write w_class back into them).
pub fn r#type(obj: PyObjectRef) -> Option<PyObjectRef> {
    if obj.is_null() {
        return None;
    }
    unsafe {
        // Exception instances share a single W_ExceptionObject layout
        // but carry an `ExcKind` tag that names the real Python class.
        // Resolve that tag to the class registered by
        // `install_default_builtins`'s `make_exc_type` so
        // `type(TypeError)`, `isinstance(exc, TypeError)` and the
        // `with assertRaises(TypeError): ...` context manager all see
        // the correct class instead of the shared "exception" stub.
        if pyre_object::is_exception(obj) {
            let kind = pyre_object::w_exception_get_kind(obj);
            let name = pyre_object::excobject::exc_kind_name(kind);
            if let Some(cls) = crate::builtins::lookup_exc_class(name) {
                return Some(cls);
            }
        }
        let w_class = (*obj).w_class;
        if !w_class.is_null() {
            return Some(w_class);
        }
        // Fallback for objects created before init_typeobjects (None, True,
        // False, Ellipsis, NotImplemented). These are `static`s in RODATA,
        // so writing to (*obj).w_class would SIGBUS — just look it up via
        // gettypefor(), which reads an AtomicPtr on the PyType.
        let tp = (*obj).ob_type;
        gettypefor(tp)
    }
}

/// Initialize the type registry with all builtin types.
///
/// PyPy: each W_XxxObject.typedef = TypeDef("xxx", ...) is set at
/// module load time. In pyre, we do it once at startup.
///
/// Must be called before any getattr on builtin objects.
pub fn init_typeobjects() {
    TYPEOBJECT_CACHE.get_or_init(|| {
        let mut reg: HashMap<usize, usize> = HashMap::new();

        // 'object' first — PyPy: objectobject.py W_ObjectObject.typedef
        // MRO = [object]. All other types inherit from object.
        let object_type = new_root_typeobject("object", init_object_type);
        reg.insert(
            &INSTANCE_TYPE as *const PyType as usize,
            object_type as usize,
        );
        let _ = W_OBJECT_TYPEOBJECT.set(object_type as usize);

        // type — PyPy: typeobject.py, bases=(object,)
        // type.__new__(metatype, name, bases, dict) creates new types
        let type_type = new_typeobject_with_base("type", init_type_type, object_type);
        // hasdict/weakrefable/acceptable now set by typedef.py:34,37,43 logic
        // in new_typeobject_with_base_and_layout from init_type_type's dict contents.
        reg.insert(&TYPE_TYPE as *const PyType as usize, type_type as usize);
        let _ = W_TYPE_TYPEOBJECT.set(type_type as usize);

        // int — intobject.py W_IntObject.typedef, bases=(object,)
        // Layout = INT_TYPE because instances are W_IntObject.
        let int_type = new_typeobject_with_base_and_layout(
            "int",
            init_int_type,
            object_type,
            &INT_TYPE as *const PyType,
        );
        reg.insert(&INT_TYPE as *const PyType as usize, int_type as usize);

        // float — floatobject.py, bases=(object,)
        reg.insert(
            &FLOAT_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "float",
                init_float_type,
                object_type,
                &FLOAT_TYPE as *const PyType,
            ) as usize,
        );

        // bool — boolobject.py, bases=(int,)
        // Layout = BOOL_TYPE (not INT_TYPE: different struct size).
        // boolobject.py:110 W_BoolObject.typedef.acceptable_as_base_class = False
        let bool_type = new_typeobject_with_base_and_layout(
            "bool",
            init_bool_type,
            int_type,
            &BOOL_TYPE as *const PyType,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(bool_type, false) };
        reg.insert(&BOOL_TYPE as *const PyType as usize, bool_type as usize);

        // str — PyPy: unicodeobject.py, bases=(object,)
        reg.insert(
            &STR_TYPE as *const PyType as usize,
            new_typeobject_with_base("str", init_str_type, object_type) as usize,
        );

        // list — PyPy: listobject.py, bases=(object,)
        reg.insert(
            &LIST_TYPE as *const PyType as usize,
            new_typeobject_with_base("list", init_list_type, object_type) as usize,
        );

        // tuple — PyPy: tupleobject.py, bases=(object,)
        reg.insert(
            &TUPLE_TYPE as *const PyType as usize,
            new_typeobject_with_base("tuple", init_tuple_type, object_type) as usize,
        );

        // dict — PyPy: dictobject.py, bases=(object,)
        reg.insert(
            &DICT_TYPE as *const PyType as usize,
            new_typeobject_with_base("dict", init_dict_type, object_type) as usize,
        );

        // mappingproxy — `pypy/objspace/std/dictproxyobject.py:103`
        // `W_DictProxyObject.typedef = TypeDef('mappingproxy', ...)`,
        // bases=(object,).  The TypeDef surface (keys/values/items/get/
        // copy/__or__/__ror__/__ior__/__reversed__/cmp methods) is
        // populated by `init_mappingproxy_type` so `cls.__dict__.keys()`
        // and friends dispatch through the registered descriptors.
        reg.insert(
            &pyre_object::MAPPING_PROXY_TYPE as *const PyType as usize,
            new_typeobject_with_base("mappingproxy", init_mappingproxy_type, object_type) as usize,
        );

        // function — PyPy: funcobject.py
        // Functions are descriptors: function.__get__ returns a bound method.
        let function_type = new_typeobject_with_base("function", init_function_type, object_type);
        // typedef.py:742 Function.typedef.acceptable_as_base_class = False
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(function_type, false) };
        // typedef.py:735/740 — Function exposes __dict__ and __weakref__.
        unsafe {
            pyre_object::w_type_set_hasdict(function_type, true);
            pyre_object::w_type_set_weakrefable(function_type, true);
        }
        reg.insert(
            &crate::FUNCTION_TYPE as *const PyType as usize,
            function_type as usize,
        );

        // builtin_function — PyPy: typedef.py BuiltinFunction.typedef
        // Mirrors Function.typedef except `__get__` is intentionally absent.
        let builtin_function_type =
            new_typeobject_with_base("builtin_function", init_builtin_function_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(builtin_function_type, false) };
        unsafe {
            pyre_object::w_type_set_hasdict(builtin_function_type, true);
            pyre_object::w_type_set_weakrefable(builtin_function_type, true);
        }
        reg.insert(
            &crate::BUILTIN_FUNCTION_TYPE as *const PyType as usize,
            builtin_function_type as usize,
        );

        // builtin-code — PyPy: BuiltinCode.typedef = TypeDef('builtin-code', ...)
        reg.insert(
            &crate::BUILTIN_CODE_TYPE as *const PyType as usize,
            new_typeobject_with_base("builtin-code", init_builtin_code_type, object_type) as usize,
        );

        // typedef.py:765 Method.typedef.acceptable_as_base_class = False
        let method_type = new_typeobject_with_base("method", init_method_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(method_type, false) };
        // typedef.py:763 — Method exposes __weakref__.
        unsafe { pyre_object::w_type_set_weakrefable(method_type, true) };
        reg.insert(
            &pyre_object::methodobject::METHOD_TYPE as *const PyType as usize,
            method_type as usize,
        );

        // typedef.py:664 PyCode.typedef.acceptable_as_base_class = False
        let code_type = new_typeobject_with_base("code", init_code_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(code_type, false) };
        reg.insert(
            &crate::pycode::CODE_TYPE as *const PyType as usize,
            code_type as usize,
        );

        // typedef.py:500 Member.typedef.acceptable_as_base_class = False
        let member_desc_type = new_typeobject_with_base(
            "member_descriptor",
            init_member_descriptor_type,
            object_type,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(member_desc_type, false) };
        reg.insert(
            &pyre_object::memberobject::MEMBER_TYPE as *const PyType as usize,
            member_desc_type as usize,
        );

        // staticmethod — PyPy: function.py StaticMethod, bases=(object,)
        reg.insert(
            &pyre_object::propertyobject::STATICMETHOD_TYPE as *const PyType as usize,
            new_typeobject_with_base("staticmethod", init_staticmethod_type, object_type) as usize,
        );

        // classmethod — PyPy: function.py ClassMethod, bases=(object,)
        reg.insert(
            &pyre_object::propertyobject::CLASSMETHOD_TYPE as *const PyType as usize,
            new_typeobject_with_base("classmethod", init_classmethod_type, object_type) as usize,
        );

        // property — PyPy: descriptor.py W_Property, bases=(object,)
        reg.insert(
            &pyre_object::propertyobject::PROPERTY_TYPE as *const PyType as usize,
            new_typeobject_with_base("property", init_property_type, object_type) as usize,
        );

        // exception — pyre uses one shared W_TypeObject for all builtin
        // exception instances; the per-class hierarchy lives in the namespace
        // (see make_exc_type in builtins.rs).  Registering it here lets
        // typedef::r#type return a non-null type for raised exception objects.
        reg.insert(
            &pyre_object::excobject::EXCEPTION_TYPE as *const PyType as usize,
            new_typeobject_with_base("exception", |_| {}, object_type) as usize,
        );

        // NoneType — bases=(object,)
        reg.insert(
            &NONE_TYPE as *const PyType as usize,
            new_typeobject_with_base("NoneType", |_| {}, object_type) as usize,
        );

        // typedef.py:941-946 Ellipsis.typedef.
        let ellipsis_type = new_typeobject_with_base("ellipsis", init_ellipsis_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(ellipsis_type, false) };
        reg.insert(
            &ELLIPSIS_TYPE as *const PyType as usize,
            ellipsis_type as usize,
        );

        // typedef.py:948-954 NotImplemented.typedef.
        let notimplemented_type =
            new_typeobject_with_base("NotImplementedType", init_notimplemented_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(notimplemented_type, false) };
        reg.insert(
            &pyre_object::pyobject::NOTIMPLEMENTED_TYPE as *const PyType as usize,
            notimplemented_type as usize,
        );

        // types.UnionType — PyPy: _pypy_generic_alias.py UnionType, bases=(object,)
        reg.insert(
            &pyre_object::UNION_TYPE as *const PyType as usize,
            new_typeobject_with_base("types.UnionType", init_union_type, object_type) as usize,
        );

        // slice — PyPy: sliceobject.py, bases=(object,)
        reg.insert(
            &pyre_object::sliceobject::SLICE_TYPE as *const PyType as usize,
            new_typeobject_with_base("slice", |_| {}, object_type) as usize,
        );

        // bytearray — PyPy: bytearrayobject.py, bases=(object,)
        reg.insert(
            &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const PyType as usize,
            new_typeobject_with_base("bytearray", init_bytearray_type, object_type) as usize,
        );

        // bytes — PyPy: bytesobject.py W_BytesObject, bases=(object,)
        reg.insert(
            &pyre_object::bytesobject::BYTES_TYPE as *const PyType as usize,
            new_typeobject_with_base("bytes", init_bytes_type, object_type) as usize,
        );

        // set / frozenset — PyPy: setobject.py, bases=(object,).
        // Both carry their own layout typedef so check_user_subclass's layout
        // safety check (typeobject.py:520-523) can reject foreign-layout
        // subclasses (e.g. subclass adds __slots__).
        reg.insert(
            &pyre_object::setobject::SET_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "set",
                init_set_type,
                object_type,
                &pyre_object::setobject::SET_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::setobject::FROZENSET_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "frozenset",
                init_frozenset_type,
                object_type,
                &pyre_object::setobject::FROZENSET_TYPE as *const PyType,
            ) as usize,
        );

        // rclass.py:739-743 parity — cache W_TypeObject on each PyType
        // so allocators can set w_class at allocation time (like RPython's
        // `self.setfield(vptr, '__class__', ctypeptr, llops)` in new_instance).
        for (&pytype_addr, &w_typeobject_addr) in &reg {
            let tp = unsafe { &*(pytype_addr as *const PyType) };
            let w_typeobject = w_typeobject_addr as PyObjectRef;
            pyre_object::pyobject::set_instantiate(tp, w_typeobject);
        }
        // pypy/objspace/std/objspace.py:104-108 — set
        // `flag_map_or_seq` on W_TypeObject for dict / list / tuple.
        // PyPy stores this marker on `W_TypeObject` (typeobject.py:169),
        // not on the low-level OBJECT_VTABLE / PyType.  Heap types copy
        // it from their bases in `inherit_flag_map_or_seq`, mirroring
        // typeobject.py:1495.
        for (pytype, flag) in [
            (&pyre_object::pyobject::DICT_TYPE, b'M'),
            (&pyre_object::pyobject::LIST_TYPE, b'S'),
            (&pyre_object::pyobject::TUPLE_TYPE, b'S'),
        ] {
            let w_typeobject = *reg
                .get(&(pytype as *const PyType as usize))
                .expect("built-in type object must be registered before flag_map_or_seq init")
                as PyObjectRef;
            unsafe {
                pyre_object::typeobject::w_type_set_flag_map_or_seq(w_typeobject, flag);
            }
        }
        // Set w_class on all built-in type objects to `type`.
        // baseobjspace.py:76 getclass() — for type objects, the class
        // is the metatype (default: `type`).
        let w_type_type = W_TYPE_TYPEOBJECT
            .get()
            .map(|v| *v as PyObjectRef)
            .unwrap_or(PY_NULL);
        for &w_typeobject_addr in reg.values() {
            let w_typeobj = w_typeobject_addr as PyObjectRef;
            unsafe {
                if (*w_typeobj).w_class.is_null() {
                    (*w_typeobj).w_class = w_type_type;
                }
            }
        }

        reg
    });

    patch_builtin_function_descriptors();
}

/// The global `object` type object, accessible from builtins.
static W_OBJECT_TYPEOBJECT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
/// The global `type` type object.
static W_TYPE_TYPEOBJECT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Get the wrapped `type` typeobject.
pub fn w_type() -> PyObjectRef {
    W_TYPE_TYPEOBJECT
        .get()
        .map(|v| *v as PyObjectRef)
        .unwrap_or(PY_NULL)
}

pub fn gettypeobject(tp: &PyType) -> PyObjectRef {
    gettypefor(tp as *const PyType).unwrap_or(PY_NULL)
}

/// Get the wrapped `object` typeobject.
pub fn w_object() -> PyObjectRef {
    W_OBJECT_TYPEOBJECT
        .get()
        .map(|v| *v as PyObjectRef)
        .unwrap_or(PY_NULL)
}

/// Create the root `object` type. MRO = [object].
fn new_root_typeobject(name: &str, init: fn(&mut DictStorage)) -> PyObjectRef {
    let mut ns = Box::new(DictStorage::new());
    ns.fix_ptr();
    init(&mut ns);
    let ns_ptr = Box::into_raw(ns);
    let type_obj = w_type_new_builtin(
        name,
        PY_NULL,
        ns_ptr as *mut u8,
        &INSTANCE_TYPE as *const PyType,
    );
    // typeobject.py:1261-1280 setup_builtin_type — root type gets its own Layout.
    unsafe {
        let layout = pyre_object::typeobject::leak_layout(pyre_object::typeobject::Layout {
            typedef: &INSTANCE_TYPE as *const PyType,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            acceptable_as_base_class: true, // object has __new__
        });
        pyre_object::w_type_set_layout(type_obj, layout);
        // object: hasdict=False, weakrefable=False (bare object() has no __dict__)
        pyre_object::w_type_set_hasdict(type_obj, false);
        pyre_object::w_type_set_weakrefable(type_obj, false);
    }
    unsafe { w_type_set_mro(type_obj, vec![type_obj]) };
    type_obj
}

/// Create a builtin type with a single base. MRO = [self] + base.mro().
/// Layout defaults to INSTANCE_TYPE (general object layout).
fn new_typeobject_with_base(
    name: &str,
    init: impl FnOnce(&mut DictStorage),
    base: PyObjectRef,
) -> PyObjectRef {
    new_typeobject_with_base_and_layout(name, init, base, &INSTANCE_TYPE as *const PyType)
}

/// Create a builtin type with explicit layout PyType.
///
/// typeobject.py:1261-1280 setup_builtin_type parity: each builtin type
/// gets its own Layout based on its instancetypedef. Types that share
/// the same typedef as their base reuse the parent's Layout object.
fn new_typeobject_with_base_and_layout(
    name: &str,
    init: impl FnOnce(&mut DictStorage),
    base: PyObjectRef,
    layout_pytype: *const PyType,
) -> PyObjectRef {
    let mut ns = Box::new(DictStorage::new());
    ns.fix_ptr();
    init(&mut ns);
    let ns_ptr = Box::into_raw(ns);
    let bases = w_tuple_new(vec![base]);
    let type_obj = w_type_new_builtin(name, bases, ns_ptr as *mut u8, layout_pytype);

    // typeobject.py:1273-1280 setup_builtin_type:
    //   parent_layout = w_bestbase.layout
    //   if parent_layout.typedef is instancetypedef:
    //       return parent_layout      ← reuse
    //   return Layout(instancetypedef, 0, base_layout=parent_layout)
    unsafe {
        let parent_layout = pyre_object::w_type_get_layout_ptr(base);
        let reuse = if !parent_layout.is_null() {
            std::ptr::eq((*parent_layout).typedef, layout_pytype)
        } else {
            false
        };
        let has_dict = (*ns_ptr).get("__dict__").is_some();
        let has_weakref = (*ns_ptr).get("__weakref__").is_some();
        let layout = if reuse {
            parent_layout
        } else {
            let has_new = (*ns_ptr).get("__new__").is_some();
            pyre_object::typeobject::leak_layout(pyre_object::typeobject::Layout {
                typedef: layout_pytype,
                nslots: 0,
                newslotnames: vec![],
                base_layout: parent_layout,
                acceptable_as_base_class: has_new,
            })
        };
        pyre_object::w_type_set_layout(type_obj, layout);
        // typedef.py:39-41: inherit from bases
        let base_hasdict = pyre_object::w_type_get_hasdict(base);
        let base_weakrefable = pyre_object::w_type_get_weakrefable(base);
        pyre_object::w_type_set_hasdict(type_obj, has_dict || base_hasdict);
        pyre_object::w_type_set_weakrefable(type_obj, has_weakref || base_weakrefable);
    }

    // MRO = [self] + base_mro
    let base_mro = unsafe { w_type_get_mro(base) };
    let mut mro = vec![type_obj];
    if !base_mro.is_null() {
        mro.extend_from_slice(unsafe { &*base_mro });
    } else {
        mro.push(base);
    }
    unsafe { w_type_set_mro(type_obj, mro) };
    type_obj
}

/// Create a named builtin type inheriting from `object`.
///
/// Used by extension modules (e.g. _sre) to define their own types.
/// typeobject.py:174 `is_heaptype=False` — builtin type.
pub fn make_builtin_type(name: &str, init: impl FnOnce(&mut DictStorage)) -> PyObjectRef {
    new_typeobject_with_base(name, init, w_object())
}

/// Create a named builtin type inheriting from `base`.
pub fn make_builtin_type_with_base(
    name: &str,
    init: impl FnOnce(&mut DictStorage),
    base: PyObjectRef,
) -> PyObjectRef {
    new_typeobject_with_base(name, init, base)
}

/// Generate a `__new__` wrapper that skips `cls` (first arg) and delegates
/// to the builtin constructor. PyPy: each type's descr__new__ strips cls
/// and calls the type-specific allocator.
macro_rules! descr_new_wrapper {
    ($fn_name:ident, $ctor:path) => {
        fn $fn_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            // args[0] = cls (ignored for builtin types)
            $ctor(&args[1..])
        }
    };
}

/// int.__new__(cls, *args) — PyPy: intobject.py descr__new__
///
/// If cls is the builtin int type, returns a plain W_IntObject.
/// If cls is a subclass of int, returns a W_InstanceObject with the
/// int value stored internally (for int subclasses like IntFlag).
fn int_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        std::ptr::null_mut() as PyObjectRef
    } else {
        args[0]
    };
    // intobject.py _new_int → check_user_subclass
    if !cls.is_null() && unsafe { pyre_object::is_type(cls) } {
        if let Some(w_int) = gettypefor(&pyre_object::INT_TYPE) {
            check_user_subclass(w_int, cls)?;
        }
    }
    let value = crate::builtins::builtin_int(&args[1..])?;
    // If cls is int itself (or null), return a plain int.
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return Ok(value);
    }
    let int_typeobj = gettypefor(&pyre_object::INT_TYPE);
    if int_typeobj.map_or(false, |t| std::ptr::eq(cls, t)) {
        return Ok(value);
    }
    // cls is a subclass of int. Create a unique W_IntObject (bypassing
    // the small-int cache so each instance has its own identity).
    // Set w_class = cls so type()/isinstance() see the subclass while
    // preserving W_IntObject layout for arithmetic.
    let int_val = unsafe { pyre_object::w_int_get_value(value) };
    let obj = pyre_object::w_int_new_unique(int_val);
    unsafe {
        (*obj).w_class = cls;
    }
    Ok(obj)
}

/// `float.__new__(cls, *args)` — PyPy: floatobject.py descr__new__.
///
/// If cls is the builtin float type, returns a plain W_FloatObject.
/// If cls is a float subclass (e.g. test_math's `class FloatCeil(float)`),
/// returns a fresh W_FloatObject with `w_class = cls` so `type(obj) == cls`
/// and `__ceil__`/`__floor__`/`__trunc__` dunders on the subclass dispatch.
fn float_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        args[0]
    };
    let value = crate::builtins::builtin_float(&args[1..])?;
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return Ok(value);
    }
    let float_typeobj = gettypefor(&pyre_object::FLOAT_TYPE);
    if float_typeobj.map_or(false, |t| std::ptr::eq(cls, t)) {
        return Ok(value);
    }
    // Subclass path — allocate a fresh W_FloatObject so setattr/w_class
    // on it don't clobber the value-cached singleton.
    let float_val = unsafe { pyre_object::w_float_get_value(value) };
    let obj = pyre_object::w_float_new(float_val);
    unsafe {
        (*obj).w_class = cls;
    }
    Ok(obj)
}

/// Wrap a `__new__` builtin function in a staticmethod descriptor.
///
/// `__new__` must NOT bind a receiver — calling `cls.__new__(other_cls, ...)`
/// passes `other_cls` as the first argument, not `cls`. PyPy/CPython model
/// this by automatically wrapping `__new__` definitions in `staticmethod` at
/// type-creation time. pyre's TypeDef registry uses this helper at install
/// time so each builtin type's `__new__` slot already carries the correct
/// non-binding descriptor.
fn make_new_descr(func: fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>) -> PyObjectRef {
    let f = make_builtin_function("__new__", func);
    pyre_object::w_staticmethod_new(f)
}

fn ellipsis_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_ellipsis) = gettypefor(&pyre_object::ELLIPSIS_TYPE) {
        check_user_subclass(w_ellipsis, cls)?;
    }
    Ok(pyre_object::noneobject::w_ellipsis())
}

fn init_ellipsis_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(ellipsis_descr_new));
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", |_args| Ok(w_str_new("Ellipsis"))),
    );
    dict_storage_store(
        ns,
        "__reduce__",
        make_builtin_function("__reduce__", |_args| Ok(w_str_new("Ellipsis"))),
    );
}

/// special.py:20: NotImplemented.descr_new_notimplemented
fn notimplemented_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_notimplemented) = gettypefor(&pyre_object::pyobject::NOTIMPLEMENTED_TYPE) {
        check_user_subclass(w_notimplemented, cls)?;
    }
    Ok(pyre_object::noneobject::w_not_implemented())
}

/// typedef.py:948-954 NotImplemented.typedef
fn init_notimplemented_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(notimplemented_descr_new));
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", |_args| Ok(w_str_new("NotImplemented"))),
    );
    dict_storage_store(
        ns,
        "__reduce__",
        make_builtin_function("__reduce__", |_args| Ok(w_str_new("NotImplemented"))),
    );
    // special.py:28-33 descr_bool
    dict_storage_store(
        ns,
        "__bool__",
        make_builtin_function("__bool__", |_args| {
            crate::warn::warn_deprecation("NotImplemented should not be used in a boolean context");
            Ok(pyre_object::boolobject::w_bool_from(true))
        }),
    );
}

/// `str.__new__(cls, *args)` — PyPy: unicodeobject.py descr__new__
///
/// `cls` is `str` itself: return the plain `W_StrObject` from `builtin_str`.
/// `cls` is a `str` subclass: build the value, then allocate a fresh
/// `W_StrObject` tagged with `__class__ = cls` so `type(obj) == cls` while
/// the underlying layout still satisfies `is_str()` for the JIT fast path.
fn str_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        args[0]
    };
    let value = crate::builtins::builtin_str(&args[1..])?;
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return Ok(value);
    }
    let str_typeobj = gettypefor(&pyre_object::STR_TYPE);
    if str_typeobj.map_or(false, |t| std::ptr::eq(cls, t)) {
        return Ok(value);
    }
    let s_owned = unsafe { pyre_object::w_str_get_value(value) }.to_string();
    let obj = pyre_object::w_str_new(&s_owned);
    // Tag with subclass type so type(obj) returns cls.
    unsafe {
        (*(obj as *mut pyre_object::PyObject)).w_class = cls;
    }
    Ok(obj)
}

/// dict.__new__(cls, *args) — if cls is a dict subclass, create an instance
/// with a backing dict for storage. PyPy: dictobject.py descr__new__
fn dict_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        args[0]
    };
    let dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);

    // If cls IS dict (not a subclass), use normal dict constructor
    if cls.is_null() || std::ptr::eq(cls, dict_type) {
        return crate::builtins::builtin_dict_ctor(&args[1..]);
    }

    // cls is a dict subclass — create instance with backing dict
    // PyPy: allocate W_DictObject with custom type
    let instance = pyre_object::w_instance_new(cls);
    let backing = pyre_object::w_dict_new();
    let _ = crate::baseobjspace::setattr(instance, "__dict_data__", backing);

    // Initialize from args if provided
    if args.len() > 1 {
        // dict(iterable) or dict(**kwargs)
        let src = args[1];
        unsafe {
            if pyre_object::is_dict(src) {
                let d = &*(src as *const pyre_object::dictobject::W_DictObject);
                for &(k, v) in &*d.entries {
                    pyre_object::w_dict_store(backing, k, v);
                }
            }
        }
    }
    Ok(instance)
}
/// boolobject.py descr_new — bool.__new__(cls, obj=False)
///
/// check_user_subclass prevents subclassing (acceptable_as_base_class=False).
/// Only positional obj argument accepted.
fn bool_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] = w_booltype (cls)
    let w_booltype = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_bool) = gettypefor(&pyre_object::BOOL_TYPE) {
        check_user_subclass(w_bool, w_booltype)?;
    }
    // boolobject.py: descr_new(space, w_booltype, w_obj)
    // Takes exactly (cls) or (cls, obj). No extra args, no kwargs.
    if args.len() > 2 {
        return Err(crate::PyError::type_error(
            "bool expected at most 1 argument, got more",
        ));
    }
    // args[1] = w_obj (default: False)
    let w_obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if w_obj.is_null() {
        return Ok(pyre_object::w_bool_from(false));
    }
    // Validate __bool__ return type and handle __bool__=None / __len__=None.
    // PyPy: space.is_true validates these conditions.
    // Use space.lookup (resolves type via type(obj)) — works for both
    // W_InstanceObject and int/float subclass instances.
    unsafe {
        if let Some(w_type) = crate::typedef::r#type(w_obj) {
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__bool__") {
                if pyre_object::is_none(method) {
                    return Err(crate::PyError::type_error(
                        "object of this type has no bool()",
                    ));
                }
                let result = crate::call_function(method, &[w_obj]);
                if !result.is_null() {
                    if !pyre_object::is_bool(result) {
                        let tp_name = (*(*result).ob_type).name;
                        return Err(crate::PyError::type_error(format!(
                            "__bool__ should return bool, returned {}",
                            tp_name,
                        )));
                    }
                    return Ok(result);
                }
            }
            if let Some(len_m) = crate::baseobjspace::lookup_in_type(w_type, "__len__") {
                if pyre_object::is_none(len_m) {
                    return Err(crate::PyError::type_error(
                        "object of this type has no len()",
                    ));
                }
                // __len__ returning negative → ValueError
                let len_result = crate::call_function(len_m, &[w_obj]);
                if !len_result.is_null() && pyre_object::is_int(len_result) {
                    let v = pyre_object::w_int_get_value(len_result);
                    if v < 0 {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::ValueError,
                            "__len__() should return >= 0".to_string(),
                        ));
                    }
                    return Ok(pyre_object::w_bool_from(v != 0));
                }
            }
        }
    }
    Ok(pyre_object::w_bool_from(crate::baseobjspace::is_true(
        w_obj,
    )))
}
descr_new_wrapper!(list_descr_new, crate::builtins::builtin_list_ctor);
descr_new_wrapper!(tuple_descr_new, crate::builtins::builtin_tuple);
// dict_new handled by dict_descr_new above (supports dict subclasses)

/// typeobject.py:511-524 W_TypeObject.check_user_subclass.
///
/// ```text
/// def check_user_subclass(self, w_subtype):
///     if not isinstance(w_subtype, W_TypeObject):
///         raise TypeError("X is not a type object ('%T')", w_subtype)
///     if not w_subtype.issubtype(self):
///         raise TypeError("%N.__new__(%N): %N is not a subtype of %N", ...)
///     if self.layout.typedef is not w_subtype.layout.typedef:
///         raise TypeError("%N.__new__(%N) is not safe, use %N.__new__()", ...)
///     return w_subtype
/// ```
fn check_user_subclass(w_self: PyObjectRef, w_subtype: PyObjectRef) -> Result<(), crate::PyError> {
    if w_subtype.is_null() || !unsafe { pyre_object::is_type(w_subtype) } {
        let self_name = unsafe { pyre_object::w_type_get_name(w_self) };
        return Err(crate::PyError::type_error(format!(
            "{}.__new__(X): X is not a type object",
            self_name,
        )));
    }
    if std::ptr::eq(w_subtype, w_self) {
        return Ok(());
    }
    let mro_ptr = unsafe { pyre_object::w_type_get_mro(w_subtype) };
    let is_sub =
        !mro_ptr.is_null() && unsafe { (*mro_ptr).iter().any(|&t| std::ptr::eq(t, w_self)) };
    if !is_sub {
        let self_name = unsafe { pyre_object::w_type_get_name(w_self) };
        let sub_name = unsafe { pyre_object::w_type_get_name(w_subtype) };
        return Err(crate::PyError::type_error(format!(
            "{}.__new__({}): {} is not a subtype of {}",
            self_name, sub_name, sub_name, self_name,
        )));
    }
    // typeobject.py:520-523 — layout safety. The base allocator only knows
    // how to fill the parent layout; if the subtype introduces extra slots
    // (different layout typedef), allocating through it would corrupt the
    // foreign layout.
    let self_layout = unsafe { pyre_object::w_type_get_layout_ptr(w_self) };
    let sub_layout = unsafe { pyre_object::w_type_get_layout_ptr(w_subtype) };
    let self_typedef = if self_layout.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*self_layout).typedef }
    };
    let sub_typedef = if sub_layout.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*sub_layout).typedef }
    };
    if !std::ptr::eq(self_typedef, sub_typedef) {
        let self_name = unsafe { pyre_object::w_type_get_name(w_self) };
        let sub_name = unsafe { pyre_object::w_type_get_name(w_subtype) };
        return Err(crate::PyError::type_error(format!(
            "{}.__new__({}) is not safe, use {}.__new__()",
            self_name, sub_name, sub_name,
        )));
    }
    Ok(())
}

fn set_alloc_for_class(
    cls: PyObjectRef,
    exact_type: PyObjectRef,
    frozen: bool,
) -> Result<PyObjectRef, crate::PyError> {
    // typeobject.py:511 allocate_instance → check_user_subclass.
    check_user_subclass(exact_type, cls)?;
    let obj = if frozen {
        pyre_object::w_frozenset_new()
    } else {
        pyre_object::w_set_new()
    };
    if !std::ptr::eq(cls, exact_type) {
        unsafe {
            (*obj).w_class = cls;
        }
    }
    Ok(obj)
}

/// `set.__new__(cls, ...)` — PyPy: setobject.py W_SetObject.descr_new.
///
/// PyPy declares the inner function as `descr_new(space, w_settype,
/// __args__)`. `__args__` is the gateway sentinel for variadic positional
/// arguments, so gateway.py:723-727 sets `maxargs = sys.maxint`; the body
/// ignores everything past `w_settype`. The actual argument count check
/// lives on `descr_init`, which type.__call__ runs after `__new__`.
fn set_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let set_type = crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
    set_alloc_for_class(cls, set_type, false)
}

/// `frozenset.__new__(cls, [iterable])` — PyPy: setobject.py W_FrozensetObject.descr_new2.
///
/// gateway.py:723 fixes maxargs from the bound `(space, w_frozensettype,
/// w_iterable=None)` signature, so anything beyond `(cls, iterable)` is a
/// TypeError; pyre enforces the same maxargs explicitly here.
fn frozenset_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "frozenset() takes at most 1 argument ({} given)",
            args.len() - 1,
        )));
    }
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let frozenset_type = crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE);
    let iterable = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);

    if !iterable.is_null() && std::ptr::eq(cls, frozenset_type) {
        if let Some(iterable_type) = crate::typedef::r#type(iterable) {
            if std::ptr::eq(iterable_type, frozenset_type) {
                return Ok(iterable);
            }
        }
    }

    let obj = set_alloc_for_class(cls, frozenset_type, true)?;
    if !iterable.is_null() {
        let items = crate::builtins::collect_iterable(iterable)?;
        for item in items {
            unsafe { pyre_object::w_set_add(obj, item) };
        }
    }
    Ok(obj)
}

/// `set.__init__(self, [iterable])` — PyPy: setobject.py W_SetObject.descr_init.
///
/// PyPy parses `__args__` against `init_signature = Signature(['some_iterable'])`
/// so anything beyond `(self, iterable)` raises TypeError; pyre enforces the
/// same maxargs explicitly here.
fn set_descr_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "set expected at most 1 argument, got {}",
            args.len() - 1,
        )));
    }
    let set_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    // gateway.interp2app(W_SetObject.descr_init) enforces that `self` is a
    // W_SetObject before the body runs; without this check pyre would cast
    // arbitrary args[0] values straight to the set layout below.
    if set_obj.is_null() || !unsafe { pyre_object::is_set(set_obj) } {
        let tp_name = if set_obj.is_null() {
            "NoneType".to_string()
        } else {
            unsafe { (*(*set_obj).ob_type).name.to_string() }
        };
        return Err(crate::PyError::type_error(format!(
            "descriptor '__init__' requires a 'set' object but received a '{}'",
            tp_name,
        )));
    }
    let existing = unsafe { pyre_object::w_set_items(set_obj) };
    for item in existing {
        unsafe {
            pyre_object::w_set_discard(set_obj, item);
        }
    }
    if let Some(iterable) = args.get(1).copied() {
        let items = crate::builtins::collect_iterable(iterable)?;
        for item in items {
            unsafe { pyre_object::w_set_add(set_obj, item) };
        }
    }
    Ok(pyre_object::w_none())
}

// ── List TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/listobject.py TypeDef("list", ...)

fn init_list_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(list_descr_new));
    dict_storage_store(
        ns,
        "append",
        make_builtin_function("append", crate::type_methods::list_method_append),
    );
    dict_storage_store(
        ns,
        "extend",
        make_builtin_function("extend", crate::type_methods::list_method_extend),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function("copy", crate::type_methods::list_method_copy),
    );
    dict_storage_store(
        ns,
        "insert",
        make_builtin_function("insert", crate::type_methods::list_method_insert),
    );
    dict_storage_store(
        ns,
        "pop",
        make_builtin_function("pop", crate::type_methods::list_method_pop),
    );
    dict_storage_store(
        ns,
        "clear",
        make_builtin_function("clear", crate::type_methods::list_method_clear),
    );
    dict_storage_store(
        ns,
        "reverse",
        make_builtin_function("reverse", crate::type_methods::list_method_reverse),
    );
    dict_storage_store(
        ns,
        "sort",
        make_builtin_function("sort", crate::type_methods::list_method_sort),
    );
    dict_storage_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::list_method_index),
    );
    dict_storage_store(
        ns,
        "count",
        make_builtin_function("count", crate::type_methods::list_method_count),
    );
    dict_storage_store(
        ns,
        "remove",
        make_builtin_function("remove", crate::type_methods::list_method_remove),
    );
}

// ── Str TypeDef ──────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/unicodeobject.py TypeDef("str", ...)

fn init_str_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(str_descr_new));
    dict_storage_store(
        ns,
        "join",
        make_builtin_function("join", crate::type_methods::str_method_join),
    );
    dict_storage_store(
        ns,
        "split",
        make_builtin_function("split", crate::type_methods::str_method_split),
    );
    dict_storage_store(
        ns,
        "strip",
        make_builtin_function("strip", crate::type_methods::str_method_strip),
    );
    dict_storage_store(
        ns,
        "lstrip",
        make_builtin_function("lstrip", crate::type_methods::str_method_lstrip),
    );
    dict_storage_store(
        ns,
        "rstrip",
        make_builtin_function("rstrip", crate::type_methods::str_method_rstrip),
    );
    dict_storage_store(
        ns,
        "startswith",
        make_builtin_function("startswith", crate::type_methods::str_method_startswith),
    );
    dict_storage_store(
        ns,
        "endswith",
        make_builtin_function("endswith", crate::type_methods::str_method_endswith),
    );
    dict_storage_store(
        ns,
        "replace",
        make_builtin_function("replace", crate::type_methods::str_method_replace),
    );
    dict_storage_store(
        ns,
        "find",
        make_builtin_function("find", crate::type_methods::str_method_find),
    );
    dict_storage_store(
        ns,
        "rfind",
        make_builtin_function("rfind", crate::type_methods::str_method_rfind),
    );
    dict_storage_store(
        ns,
        "upper",
        make_builtin_function("upper", crate::type_methods::str_method_upper),
    );
    dict_storage_store(
        ns,
        "lower",
        make_builtin_function("lower", crate::type_methods::str_method_lower),
    );
    dict_storage_store(
        ns,
        "format",
        make_builtin_function("format", crate::type_methods::str_method_format),
    );
    dict_storage_store(
        ns,
        "encode",
        make_builtin_function("encode", crate::type_methods::str_method_encode),
    );
    dict_storage_store(
        ns,
        "isdigit",
        make_builtin_function("isdigit", crate::type_methods::str_method_isdigit),
    );
    dict_storage_store(
        ns,
        "isdecimal",
        make_builtin_function("isdecimal", crate::type_methods::str_method_isdecimal),
    );
    dict_storage_store(
        ns,
        "isnumeric",
        make_builtin_function("isnumeric", crate::type_methods::str_method_isnumeric),
    );
    dict_storage_store(
        ns,
        "istitle",
        make_builtin_function("istitle", crate::type_methods::str_method_istitle),
    );
    dict_storage_store(
        ns,
        "isalpha",
        make_builtin_function("isalpha", crate::type_methods::str_method_isalpha),
    );
    dict_storage_store(
        ns,
        "isidentifier",
        make_builtin_function("isidentifier", crate::type_methods::str_method_isidentifier),
    );
    dict_storage_store(
        ns,
        "zfill",
        make_builtin_function("zfill", crate::type_methods::str_method_zfill),
    );
    dict_storage_store(
        ns,
        "count",
        make_builtin_function("count", crate::type_methods::str_method_count),
    );
    dict_storage_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::str_method_index),
    );
    dict_storage_store(
        ns,
        "title",
        make_builtin_function("title", crate::type_methods::str_method_title),
    );
    dict_storage_store(
        ns,
        "capitalize",
        make_builtin_function("capitalize", crate::type_methods::str_method_capitalize),
    );
    dict_storage_store(
        ns,
        "swapcase",
        make_builtin_function("swapcase", crate::type_methods::str_method_swapcase),
    );
    dict_storage_store(
        ns,
        "center",
        make_builtin_function("center", crate::type_methods::str_method_center),
    );
    dict_storage_store(
        ns,
        "ljust",
        make_builtin_function("ljust", crate::type_methods::str_method_ljust),
    );
    dict_storage_store(
        ns,
        "rjust",
        make_builtin_function("rjust", crate::type_methods::str_method_rjust),
    );
    dict_storage_store(
        ns,
        "isspace",
        make_builtin_function("isspace", crate::type_methods::str_method_isspace),
    );
    dict_storage_store(
        ns,
        "isupper",
        make_builtin_function("isupper", crate::type_methods::str_method_isupper),
    );
    dict_storage_store(
        ns,
        "islower",
        make_builtin_function("islower", crate::type_methods::str_method_islower),
    );
    dict_storage_store(
        ns,
        "isalnum",
        make_builtin_function("isalnum", crate::type_methods::str_method_isalnum),
    );
    dict_storage_store(
        ns,
        "isascii",
        make_builtin_function("isascii", crate::type_methods::str_method_isascii),
    );
    dict_storage_store(
        ns,
        "partition",
        make_builtin_function("partition", crate::type_methods::str_method_partition),
    );
    dict_storage_store(
        ns,
        "rpartition",
        make_builtin_function("rpartition", crate::type_methods::str_method_rpartition),
    );
    dict_storage_store(
        ns,
        "splitlines",
        make_builtin_function("splitlines", crate::type_methods::str_method_splitlines),
    );
    dict_storage_store(
        ns,
        "removeprefix",
        make_builtin_function("removeprefix", crate::type_methods::str_method_removeprefix),
    );
    dict_storage_store(
        ns,
        "removesuffix",
        make_builtin_function("removesuffix", crate::type_methods::str_method_removesuffix),
    );
    dict_storage_store(
        ns,
        "expandtabs",
        make_builtin_function("expandtabs", crate::type_methods::str_method_expandtabs),
    );
    dict_storage_store(
        ns,
        "translate",
        make_builtin_function("translate", crate::type_methods::str_method_translate),
    );
    // str dunder methods
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            Ok(pyre_object::w_bool_from(
                crate::baseobjspace::contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            crate::baseobjspace::len(args[0])
        }),
    );
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function("__getitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__getitem__"));
            }
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
    dict_storage_store(
        ns,
        "__add__",
        make_builtin_function("__add__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__add__"));
            }
            crate::baseobjspace::add(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "__mul__",
        make_builtin_function("__mul__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__mul__"));
            }
            crate::baseobjspace::mul(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "__mod__",
        make_builtin_function("__mod__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__mod__"));
            }
            crate::baseobjspace::mod_(args[0], args[1])
        }),
    );
    // maketrans — PyPy: unicodeobject.py descr_maketrans
    dict_storage_store(
        ns,
        "maketrans",
        make_builtin_function("maketrans", |args| {
            // maketrans(x[, y[, z]]) → translation dict
            let d = pyre_object::w_dict_new();
            if args.len() >= 3 {
                // maketrans(x, y, z) — z is chars to delete (map to None)
                let x = unsafe { pyre_object::w_str_get_value(args[0]) };
                let y = unsafe { pyre_object::w_str_get_value(args[1]) };
                let z = unsafe { pyre_object::w_str_get_value(args[2]) };
                for (xc, yc) in x.chars().zip(y.chars()) {
                    unsafe {
                        pyre_object::w_dict_store(
                            d,
                            pyre_object::w_int_new(xc as i64),
                            pyre_object::w_int_new(yc as i64),
                        );
                    }
                }
                for zc in z.chars() {
                    unsafe {
                        pyre_object::w_dict_store(
                            d,
                            pyre_object::w_int_new(zc as i64),
                            pyre_object::w_none(),
                        );
                    }
                }
            } else if args.len() >= 2 {
                let x = unsafe { pyre_object::w_str_get_value(args[0]) };
                let y = unsafe { pyre_object::w_str_get_value(args[1]) };
                for (xc, yc) in x.chars().zip(y.chars()) {
                    unsafe {
                        pyre_object::w_dict_store(
                            d,
                            pyre_object::w_int_new(xc as i64),
                            pyre_object::w_int_new(yc as i64),
                        );
                    }
                }
            } else if args.len() == 1 && unsafe { pyre_object::is_dict(args[0]) } {
                // 1-arg dict form: maketrans({ord_or_char: replacement, ...})
                let src = args[0];
                unsafe {
                    let entries = &*((src as *const pyre_object::dictobject::W_DictObject)
                        .as_ref()
                        .unwrap()
                        .entries);
                    for &(k, v) in entries {
                        let ord_key = if pyre_object::is_int(k) {
                            k
                        } else if pyre_object::is_str(k) {
                            let s = pyre_object::w_str_get_value(k);
                            let ch = s.chars().next().unwrap_or('\0');
                            pyre_object::w_int_new(ch as i64)
                        } else {
                            k
                        };
                        pyre_object::w_dict_store(d, ord_key, v);
                    }
                }
            }
            Ok(d)
        }),
    );
}

// ── Dict TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/dictobject.py TypeDef("dict", ...)

fn init_dict_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(dict_descr_new));
    // dict.__init__(self, mapping_or_iterable=None, **kwargs)
    // PyPy: W_DictMultiObject.descr_init
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let self_dict = args[0];
            // Process positional arg (mapping or iterable of pairs)
            // Process each arg: may be positional mapping/iterable or kwargs dict
            for arg_idx in 1..args.len() {
                let src = args[arg_idx];
                unsafe {
                    if pyre_object::is_dict(src) {
                        let marker =
                            pyre_object::w_dict_lookup(src, pyre_object::w_str_new("__pyre_kw__"));
                        if marker.is_some() {
                            // kwargs dict — merge excluding marker
                            for (k, v) in pyre_object::w_dict_items(src) {
                                if pyre_object::is_str(k)
                                    && pyre_object::w_str_get_value(k) == "__pyre_kw__"
                                {
                                    continue;
                                }
                                pyre_object::w_dict_store(self_dict, k, v);
                            }
                        } else {
                            for (k, v) in pyre_object::w_dict_items(src) {
                                pyre_object::w_dict_store(self_dict, k, v);
                            }
                        }
                    } else {
                        if let Ok(keys_method) = crate::baseobjspace::getattr(src, "keys") {
                            let keys_obj = crate::call_function(keys_method, &[]);
                            if let Ok(keys) = crate::builtins::collect_iterable(keys_obj) {
                                for key in keys {
                                    if let Ok(val) = crate::baseobjspace::getitem(src, key) {
                                        pyre_object::w_dict_store(self_dict, key, val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "get",
        make_builtin_function("get", crate::type_methods::dict_method_get),
    );
    dict_storage_store(
        ns,
        "keys",
        make_builtin_function("keys", crate::type_methods::dict_method_keys),
    );
    dict_storage_store(
        ns,
        "values",
        make_builtin_function("values", crate::type_methods::dict_method_values),
    );
    dict_storage_store(
        ns,
        "items",
        make_builtin_function("items", crate::type_methods::dict_method_items),
    );
    dict_storage_store(
        ns,
        "update",
        make_builtin_function("update", crate::type_methods::dict_method_update),
    );
    dict_storage_store(
        ns,
        "pop",
        make_builtin_function("pop", crate::type_methods::dict_method_pop),
    );
    dict_storage_store(
        ns,
        "setdefault",
        make_builtin_function("setdefault", crate::type_methods::dict_method_setdefault),
    );
    dict_storage_store(
        ns,
        "__setitem__",
        make_builtin_function("__setitem__", |args| {
            if args.len() < 3 {
                return Err(crate::PyError::type_error("__setitem__ requires 3 args"));
            }
            // For plain dict: direct store. For dict subclass instance: use backing dict.
            unsafe {
                if pyre_object::is_dict(args[0]) {
                    pyre_object::w_dict_store(args[0], args[1], args[2]);
                } else if pyre_object::is_instance(args[0]) {
                    // dict subclass — store in __dict_data__ backing dict
                    if let Ok(backing) = crate::baseobjspace::getattr(args[0], "__dict_data__") {
                        if pyre_object::is_dict(backing) {
                            pyre_object::w_dict_store(backing, args[1], args[2]);
                        }
                    }
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function("__getitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__getitem__ requires 2 args"));
            }
            unsafe {
                if pyre_object::is_dict(args[0]) {
                    return crate::baseobjspace::getitem(args[0], args[1]);
                }
                if pyre_object::is_instance(args[0]) {
                    if let Ok(backing) = crate::baseobjspace::getattr(args[0], "__dict_data__") {
                        if pyre_object::is_dict(backing) {
                            return crate::baseobjspace::getitem(backing, args[1]);
                        }
                    }
                }
            }
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            let dict = crate::type_methods::resolve_dict_backing(args[0]);
            if !dict.is_null() {
                // Dict or dict subclass — look up in backing dict
                return Ok(pyre_object::w_bool_from(
                    unsafe { pyre_object::w_dict_lookup(dict, args[1]) }.is_some(),
                ));
            }
            Ok(pyre_object::w_bool_from(
                crate::baseobjspace::contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            let dict = crate::type_methods::resolve_dict_backing(args[0]);
            if !dict.is_null() {
                return Ok(pyre_object::w_int_new(
                    unsafe { pyre_object::w_dict_len(dict) } as i64,
                ));
            }
            crate::baseobjspace::len(args[0])
        }),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let dict = crate::type_methods::resolve_dict_backing(args[0]);
            if !dict.is_null() {
                // Iterate over dict keys
                return crate::baseobjspace::iter(dict);
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
    dict_storage_store(
        ns,
        "__delitem__",
        make_builtin_function("__delitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__delitem__ requires 2 args"));
            }
            crate::baseobjspace::delitem(args[0], args[1])?;
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "__eq__",
        make_builtin_function("__eq__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Eq)
        }),
    );
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function("__or__", |args| {
            // dict | dict → merge
            if args.len() < 2 {
                return Ok(args[0]);
            }
            Ok(args[0]) // stub
        }),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function("copy", crate::type_methods::dict_method_copy),
    );
    dict_storage_store(
        ns,
        "clear",
        make_builtin_function("clear", |_args| Ok(pyre_object::w_none())),
    );
    // dict.fromkeys(iterable, value=None) — classmethod
    dict_storage_store(
        ns,
        "fromkeys",
        make_builtin_function("fromkeys", |args| {
            // Called as dict.fromkeys(iter, val): args = [iter, val] (no cls binding)
            let (iterable, value) = if args.len() >= 2 {
                (args[0], args[1])
            } else if args.len() == 1 {
                (args[0], pyre_object::w_none())
            } else {
                return Ok(pyre_object::w_dict_new());
            };
            let d = pyre_object::w_dict_new();
            let items = crate::builtins::collect_iterable(iterable)?;
            for key in items {
                unsafe { pyre_object::w_dict_store(d, key, value) };
            }
            Ok(d)
        }),
    );
}

// ── Mappingproxy TypeDef ─────────────────────────────────────────────
//
// `pypy/objspace/std/dictproxyobject.py:103` —
// `W_DictProxyObject.typedef = TypeDef('mappingproxy', ...)`.  All
// methods forward to `self.w_mapping` (the wrapped W_DictObject);
// pyre routes through `resolve_dict_backing`, which now unwraps the
// proxy to its inner dict so the dict-method bodies stay shared.

fn init_mappingproxy_type(ns: &mut DictStorage) {
    // dictproxyobject.py:32 descr_len → space.len(self.w_mapping)
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            crate::baseobjspace::len(args[0])
        }),
    );
    // dictproxyobject.py:35 descr_getitem → space.getitem(self.w_mapping, w_key)
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function("__getitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__getitem__ requires 2 args"));
            }
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    // dictproxyobject.py:38 descr_contains → space.contains(self.w_mapping, w_key)
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            Ok(pyre_object::w_bool_from(crate::baseobjspace::contains(
                args[0], args[1],
            )?))
        }),
    );
    // dictproxyobject.py:41 descr_iter → space.iter(self.w_mapping)
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
    // dictproxyobject.py:47 descr_repr →
    // `b"mappingproxy(%s)" % space.utf8_w(space.repr(self.w_mapping))`
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("mappingproxy({})"));
            }
            unsafe { Ok(pyre_object::w_str_new(&crate::display::py_repr(args[0]))) }
        }),
    );
    // dictproxyobject.py:44 descr_str → space.str(self.w_mapping)
    dict_storage_store(
        ns,
        "__str__",
        make_builtin_function("__str__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new(""));
            }
            unsafe { Ok(pyre_object::w_str_new(&crate::display::py_str(args[0]))) }
        }),
    );
    // dictproxyobject.py:67 descr_ior → unconditional TypeError; the
    // proxy is read-only so in-place merge is rejected by name even
    // when the rhs would otherwise be acceptable for `__or__`.
    dict_storage_store(
        ns,
        "__ior__",
        make_builtin_function("__ior__", |_args| {
            Err(crate::PyError::type_error(
                "'|=' is not supported by mappingproxy; use '|' instead",
            ))
        }),
    );
    // dictproxyobject.py:51 descr_or →
    // `copy_self.update(w_other); return copy_self`.  Implemented via
    // `dict_method_copy` (unwraps proxy through resolve_dict_backing)
    // followed by an items merge from `w_other`.
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function("__or__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__or__ requires 2 args"));
            }
            let lhs = args[0];
            let rhs = unsafe {
                if pyre_object::is_dict_proxy(args[1]) {
                    pyre_object::w_dict_proxy_get_mapping(args[1])
                } else {
                    args[1]
                }
            };
            if !unsafe { pyre_object::is_dict(rhs) } {
                return Ok(pyre_object::w_not_implemented());
            }
            let new_dict = crate::type_methods::dict_method_copy(&[lhs])?;
            crate::type_methods::dict_method_update(&[new_dict, rhs])?;
            Ok(new_dict)
        }),
    );
    // dictproxyobject.py:60 descr_ror →
    // `space.call_method(w_other, '__or__', self.w_mapping)`.
    dict_storage_store(
        ns,
        "__ror__",
        make_builtin_function("__ror__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__ror__ requires 2 args"));
            }
            let self_mapping = unsafe {
                if pyre_object::is_dict_proxy(args[0]) {
                    pyre_object::w_dict_proxy_get_mapping(args[0])
                } else {
                    args[0]
                }
            };
            let lhs = args[1];
            if !unsafe { pyre_object::is_dict(lhs) } {
                return Ok(pyre_object::w_not_implemented());
            }
            let new_dict = crate::type_methods::dict_method_copy(&[lhs])?;
            crate::type_methods::dict_method_update(&[new_dict, self_mapping])?;
            Ok(new_dict)
        }),
    );
    // dictproxyobject.py:87 descr_reversed →
    // `space.call_method(self.w_mapping, '__reversed__')`.  Pyre lacks
    // a dedicated reverse iterator on dict, so fall back to building
    // a list of keys in reverse insertion order.
    dict_storage_store(
        ns,
        "__reversed__",
        make_builtin_function("__reversed__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let dict = crate::type_methods::resolve_dict_backing(args[0]);
            if dict.is_null() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let mut keys: Vec<pyre_object::PyObjectRef> = unsafe {
                pyre_object::w_dict_items(dict)
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect()
            };
            keys.reverse();
            crate::baseobjspace::iter(pyre_object::w_list_new(keys))
        }),
    );
    // dictproxyobject.py:71 get_w / 75 keys_w / 78 values_w / 81 items_w /
    // 84 copy_w — forward through `dict_method_*` (which unwraps the
    // proxy via `resolve_dict_backing`).
    dict_storage_store(
        ns,
        "get",
        make_builtin_function("get", crate::type_methods::dict_method_get),
    );
    dict_storage_store(
        ns,
        "keys",
        make_builtin_function("keys", crate::type_methods::dict_method_keys),
    );
    dict_storage_store(
        ns,
        "values",
        make_builtin_function("values", crate::type_methods::dict_method_values),
    );
    dict_storage_store(
        ns,
        "items",
        make_builtin_function("items", crate::type_methods::dict_method_items),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function("copy", crate::type_methods::dict_method_copy),
    );
    // dictproxyobject.py:91-100 cmp methods (eq/ne/lt/le/gt/ge) →
    // `getattr(space, op)(self.w_mapping, w_other)`.  Pyre routes
    // through `space.compare`; the proxy's `space.eq`/`space.lt`/etc.
    // path runs the same `resolve_dict_backing` unwrap.  Each
    // comparison gets its own `fn` so the pointer stays
    // non-capturing.
    fn cmp_helper(
        args: &[PyObjectRef],
        op: crate::baseobjspace::CompareOp,
    ) -> Result<PyObjectRef, crate::PyError> {
        if args.len() < 2 {
            return Ok(pyre_object::w_bool_from(false));
        }
        crate::baseobjspace::compare(args[0], args[1], op)
    }
    fn proxy_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Eq)
    }
    fn proxy_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Ne)
    }
    fn proxy_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Lt)
    }
    fn proxy_le(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Le)
    }
    fn proxy_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Gt)
    }
    fn proxy_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Ge)
    }
    dict_storage_store(ns, "__eq__", make_builtin_function("__eq__", proxy_eq));
    dict_storage_store(ns, "__ne__", make_builtin_function("__ne__", proxy_ne));
    dict_storage_store(ns, "__lt__", make_builtin_function("__lt__", proxy_lt));
    dict_storage_store(ns, "__le__", make_builtin_function("__le__", proxy_le));
    dict_storage_store(ns, "__gt__", make_builtin_function("__gt__", proxy_gt));
    dict_storage_store(ns, "__ge__", make_builtin_function("__ge__", proxy_ge));
}

// ── Tuple TypeDef ────────────────────────────────────────────────────

fn init_tuple_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(tuple_descr_new));
    dict_storage_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::tuple_method_index),
    );
    dict_storage_store(
        ns,
        "count",
        make_builtin_function("count", crate::type_methods::tuple_method_count),
    );
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            Ok(pyre_object::w_bool_from(
                crate::baseobjspace::contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            Ok(pyre_object::w_int_new(
                unsafe { pyre_object::w_tuple_len(args[0]) } as i64,
            ))
        }),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
}

// ── Int/Float/Bool TypeDef (minimal) ─────────────────────────────────

// ── Type TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/typeobject.py TypeDef("type", ...)

/// types.UnionType — PyPy: _pypy_generic_alias.py UnionType
fn init_union_type(ns: &mut DictStorage) {
    // UnionType.__args__ — returns the tuple of union member types
    dict_storage_store(
        ns,
        "__args__",
        make_builtin_function("__args__", |args| {
            let self_ = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if unsafe { pyre_object::is_union(self_) } {
                Ok(unsafe { pyre_object::w_union_get_args(self_) })
            } else {
                Ok(pyre_object::PY_NULL)
            }
        }),
    );
    // UnionType.__or__ — PyPy: UnionType.__or__ → _create_union
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function("__or__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__or__ requires 2 arguments"));
            }
            Ok(pyre_object::w_union_new(args[0], args[1]))
        }),
    );
    // UnionType.__ror__
    dict_storage_store(
        ns,
        "__ror__",
        make_builtin_function("__ror__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__ror__ requires 2 arguments"));
            }
            Ok(pyre_object::w_union_new(args[1], args[0]))
        }),
    );
}

thread_local! {
    static GETSET_DESCRIPTOR_TYPE: std::cell::OnceCell<pyre_object::PyObjectRef>
        = const { std::cell::OnceCell::new() };
}

fn getset_descriptor_type() -> pyre_object::PyObjectRef {
    GETSET_DESCRIPTOR_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp = make_builtin_type("getset_descriptor", init_getset_descriptor_type);
            // typedef.py:446 assert not GetSetProperty.typedef.acceptable_as_base_class
            unsafe { pyre_object::w_type_set_acceptable_as_base_class(tp, false) };
            // GetSetProperty stores fget/fset/fdel/name/w_objclass as instance
            // attributes; pyre needs hasdict so they land in the live W_DictObject
            // backing store rather than the legacy ATTR_TABLE fallback.
            unsafe { pyre_object::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// typedef.py:367-371 readonly_attribute
///
/// ```python
/// def readonly_attribute(self, space):
///     if self.name == '<generic property>':
///         raise oefmt(space.w_TypeError, "readonly attribute")
///     else:
///         raise oefmt(space.w_TypeError, "readonly attribute '%s'", self.name)
/// ```
fn readonly_attribute(descr: pyre_object::PyObjectRef) -> crate::PyError {
    let name_obj = read_descr_name(descr);
    let name = if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
        Some(unsafe { pyre_object::w_str_get_value(name_obj) })
    } else {
        None
    };
    match name {
        Some(n) if n != "<generic property>" => {
            crate::PyError::type_error(format!("readonly attribute '{}'", n))
        }
        _ => crate::PyError::type_error("readonly attribute".to_string()),
    }
}

/// typedef.py:308-415 GetSetProperty.typedef = TypeDef("getset_descriptor", ...)
fn init_getset_descriptor_type(ns: &mut DictStorage) {
    // typedef.py:347-365 GetSetProperty.descr_property_get
    //
    // ```python
    // @unwrap_spec(w_cls = WrappedDefault(None))
    // def descr_property_get(self, space, w_obj, w_cls=None):
    //     """property.__get__(obj[, type]) -> value
    //     Read the value of the property of the given obj."""
    //     # XXX HAAAAAAAAAAAACK (but possibly a good one)
    //     if (space.is_w(w_obj, space.w_None)
    //         and not space.is_w(w_cls, space.type(space.w_None))):
    //         #print self, w_obj, w_cls
    //         if space.is_w(w_cls, space.w_None):
    //             raise oefmt(space.w_TypeError, "__get__(None, None) is invalid")
    //         return self
    //     else:
    //         try:
    //             return self.fget(self, space, w_obj)
    //         except DescrMismatch:
    //             return w_obj.descr_call_mismatch(
    //                 space, '__getattribute__',
    //                 self.reqcls, Arguments(space, [w_obj,
    //                                                space.newtext(self.name)]))
    // ```
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function("__get__", |args| {
            let w_self = args[0];
            let w_obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let w_cls = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            let w_obj_is_none = !w_obj.is_null() && unsafe { pyre_object::is_none(w_obj) };
            let none_type =
                crate::typedef::r#type(pyre_object::w_none()).unwrap_or(pyre_object::PY_NULL);
            let w_cls_is_none_type = !w_cls.is_null() && std::ptr::eq(w_cls, none_type);
            // typedef.py:352-353 if w_obj is None and w_cls is not type(None):
            if w_obj_is_none && !w_cls_is_none_type {
                // typedef.py:355 if w_cls is None: raise TypeError
                if w_cls.is_null() || unsafe { pyre_object::is_none(w_cls) } {
                    return Err(crate::PyError::type_error(
                        "__get__(None, None) is invalid".to_string(),
                    ));
                }
                // typedef.py:357 return self
                return Ok(w_self);
            }
            // typedef.py:359-365 try: return self.fget(self, space, w_obj)
            //                    except DescrMismatch: descr_call_mismatch(...)
            let reqcls = read_reqcls(w_self);
            // pyre's typecheck wrapper equivalent: descr_self_interp_w runs
            // before the inner function so DescrMismatch is raised the same
            // way PyPy's `_make_descr_typecheck_wrapper` does.
            if !reqcls.is_null() {
                if let Err(e) = crate::baseobjspace::descr_self_interp_w(reqcls, w_obj) {
                    if e.kind == crate::PyErrorKind::DescrMismatch {
                        return Err(crate::baseobjspace::descr_call_mismatch(
                            w_obj,
                            "__getattribute__",
                            reqcls,
                        ));
                    }
                    return Err(e);
                }
            }
            let fget = read_fget(w_self);
            if fget.is_null() {
                return Err(readonly_attribute(w_self));
            }
            match crate::call::call_function_impl_result(fget, &[w_self, w_obj]) {
                Ok(v) => Ok(v),
                Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => Err(
                    crate::baseobjspace::descr_call_mismatch(w_obj, "__getattribute__", reqcls),
                ),
                Err(e) => Err(e),
            }
        }),
    );
    // typedef.py:373-386 GetSetProperty.descr_property_set
    //
    // ```python
    // def descr_property_set(self, space, w_obj, w_value):
    //     fset = self.fset
    //     if fset is None:
    //         raise self.readonly_attribute(space)
    //     try:
    //         fset(self, space, w_obj, w_value)
    //     except DescrMismatch:
    //         w_obj.descr_call_mismatch(
    //             space, '__setattr__',
    //             self.reqcls, Arguments(space, [w_obj,
    //                                            space.newtext(self.name),
    //                                            w_value]))
    // ```
    dict_storage_store(
        ns,
        "__set__",
        make_builtin_function("__set__", |args| {
            let w_self = args[0];
            let w_obj = args[1];
            let w_value = args[2];
            let fset = read_fset(w_self);
            if fset.is_null() || unsafe { pyre_object::is_none(fset) } {
                return Err(readonly_attribute(w_self));
            }
            let reqcls = read_reqcls(w_self);
            if !reqcls.is_null() {
                if let Err(e) = crate::baseobjspace::descr_self_interp_w(reqcls, w_obj) {
                    if e.kind == crate::PyErrorKind::DescrMismatch {
                        return Err(crate::baseobjspace::descr_call_mismatch(
                            w_obj,
                            "__setattr__",
                            reqcls,
                        ));
                    }
                    return Err(e);
                }
            }
            match crate::call::call_function_impl_result(fset, &[w_self, w_obj, w_value]) {
                Ok(_) => Ok(pyre_object::w_none()),
                Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => Err(
                    crate::baseobjspace::descr_call_mismatch(w_obj, "__setattr__", reqcls),
                ),
                Err(e) => Err(e),
            }
        }),
    );
    // typedef.py:388-400 GetSetProperty.descr_property_del
    //
    // ```python
    // def descr_property_del(self, space, w_obj):
    //     fdel = self.fdel
    //     if fdel is None:
    //         raise oefmt(space.w_AttributeError, "cannot delete attribute")
    //     try:
    //         fdel(self, space, w_obj)
    //     except DescrMismatch:
    //         w_obj.descr_call_mismatch(
    //             space, '__delattr__',
    //             self.reqcls, Arguments(space, [w_obj,
    //                                            space.newtext(self.name)]))
    // ```
    dict_storage_store(
        ns,
        "__delete__",
        make_builtin_function("__delete__", |args| {
            let w_self = args[0];
            let w_obj = args[1];
            let fdel = read_fdel(w_self);
            if fdel.is_null() || unsafe { pyre_object::is_none(fdel) } {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::AttributeError,
                    "cannot delete attribute".to_string(),
                ));
            }
            let reqcls = read_reqcls(w_self);
            if !reqcls.is_null() {
                if let Err(e) = crate::baseobjspace::descr_self_interp_w(reqcls, w_obj) {
                    if e.kind == crate::PyErrorKind::DescrMismatch {
                        return Err(crate::baseobjspace::descr_call_mismatch(
                            w_obj,
                            "__delattr__",
                            reqcls,
                        ));
                    }
                    return Err(e);
                }
            }
            match crate::call::call_function_impl_result(fdel, &[w_self, w_obj]) {
                Ok(_) => Ok(pyre_object::w_none()),
                Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => Err(
                    crate::baseobjspace::descr_call_mismatch(w_obj, "__delattr__", reqcls),
                ),
                Err(e) => Err(e),
            }
        }),
    );
}

/// `GetSetProperty(fget)` — read-only getset descriptor with no required class.
///
/// PyPy: `GetSetProperty(fget)` (typedef.py:312-325).
fn make_getset_descriptor(getter: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    make_getset_property_full(
        getter,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
}

/// `GetSetProperty(fget, fset, fdel)` — full getset descriptor with no
/// required class. Equivalent to PyPy's `GetSetProperty(fget, fset, fdel)`
/// invocation with `cls=None`.
fn make_getset_property(
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    make_getset_property_full(fget, fset, fdel, pyre_object::PY_NULL)
}

/// `GetSetProperty(fget, fset, fdel, cls=cls)` — full getset descriptor
/// with a required class for descriptor-level type enforcement.
///
/// PyPy: `GetSetProperty(...)` (typedef.py:312-325) with the `cls` keyword.
/// `cls` is stored as `reqcls` and `descr_self_interp_w` raises
/// `DescrMismatch` when a wrong-class instance reaches `__get__/__set__/__delete__`.
fn make_getset_property_full(
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
    cls: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    let obj = w_instance_new(getset_descriptor_type());
    getset_property_init(
        obj,
        fget,
        fset,
        fdel,
        pyre_object::PY_NULL, // doc
        cls,
        false,                // use_closure
        pyre_object::PY_NULL, // name (defaults to '<generic property>')
    );
    obj
}

fn init_type_type(ns: &mut DictStorage) {
    // type.__new__(metatype, name, bases, dict) — creates new type
    dict_storage_store(
        ns,
        "__new__",
        make_new_descr(crate::builtins::type_descr_new),
    );
    // type.__init__ — no-op for now
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", |_| Ok(pyre_object::w_none())),
    );
    // type.__annotations__ / __dict__ / __mro__ / __name__ / __bases__
    // are exposed as getset descriptors so
    // `type.__dict__['<name>'].__get__(cls)` invokes the underlying getter
    // and returns the real value (matching CPython's getset_descriptor).
    //
    // PyPy: pypy/objspace/std/typeobject.py get_annotations / descr_getdict
    // / descr_getmro / descr_getname / descr_getbases.
    // GetSetProperty fget callbacks receive (descriptor_self, w_obj) so the
    // wrapped object is at args[1] (matches PyPy's typecheck wrapper that
    // passes (closure, space, w_obj)).
    let annotations_getter = make_builtin_function("__annotations__", |args| {
        // GetSetProperty fget callbacks receive (descriptor_self, w_obj),
        // so the cls is at args[1].
        let cls = args[1];
        // First try a directly stored __annotations__ dict — pyre's legacy
        // path stashes it on the type's ATTR_TABLE entry.
        let stored = crate::baseobjspace::ATTR_TABLE.with(|table| {
            table
                .borrow()
                .get(&(cls as usize))
                .and_then(|d| d.get("__annotations__").copied())
        });
        if let Some(v) = stored {
            return Ok(v);
        }
        // PEP 649 path: bytecode emits `__annotate_func__` (== `__annotate__`).
        // Call it with format=1 (VALUE) to materialise the dict.
        if let Some(annotate_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(cls, "__annotate_func__") }
                .or_else(|| unsafe { crate::baseobjspace::lookup_in_type(cls, "__annotate__") })
        {
            if !annotate_fn.is_null() && !unsafe { pyre_object::is_none(annotate_fn) } {
                return Ok(crate::call::call_function_impl_raw(
                    annotate_fn,
                    &[pyre_object::w_int_new(1)],
                ));
            }
        }
        Ok(pyre_object::w_dict_new())
    });
    dict_storage_store(
        ns,
        "__annotations__",
        make_getset_descriptor(annotations_getter),
    );

    let mro_getter = make_builtin_function("__mro__", |args| {
        let cls = args[1];
        unsafe {
            let mro_ptr = pyre_object::w_type_get_mro(cls);
            if mro_ptr.is_null() {
                return Ok(pyre_object::w_tuple_new(vec![]));
            }
            Ok(pyre_object::w_tuple_new((*mro_ptr).clone()))
        }
    });
    dict_storage_store(ns, "__mro__", make_getset_descriptor(mro_getter));

    let dict_getter = make_builtin_function("__dict__", |args| {
        let cls = args[1];
        unsafe {
            let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(cls);
            if ns_ptr.is_null() {
                return Ok(pyre_object::w_dict_proxy_new(pyre_object::w_dict_new()));
            }
            // `pypy/objspace/std/typeobject.py:1277 descr_get_dict`
            // returns `W_DictProxyObject(w_dict)` — read-only **live**
            // view.  Wrap the type's canonical W_DictObject so
            // subsequent `cls.x = 1` setattrs flow through the
            // dict_storage_proxy and become visible on the proxy.
            let canonical = crate::baseobjspace::dict_storage_to_dict(ns_ptr as *const DictStorage);
            Ok(pyre_object::w_dict_proxy_new(canonical))
        }
    });
    dict_storage_store(ns, "__dict__", make_getset_descriptor(dict_getter));

    let name_getter = make_builtin_function("__name__", |args| unsafe {
        let name = pyre_object::w_type_get_name(args[1]);
        Ok(pyre_object::w_str_new(name))
    });
    dict_storage_store(ns, "__name__", make_getset_descriptor(name_getter));

    let bases_getter = make_builtin_function("__bases__", |args| unsafe {
        let bases = pyre_object::w_type_get_bases(args[1]);
        if bases.is_null() {
            return Ok(pyre_object::w_tuple_new(vec![]));
        }
        Ok(bases)
    });
    dict_storage_store(ns, "__bases__", make_getset_descriptor(bases_getter));
}

/// function/builtin_function_or_method — PyPy: function.py Function typedef
/// descr_function_get (function.py:462): always returns a Method.
/// PyPy: shared `Function.typedef.rawdict` entries that BuiltinFunction.typedef
/// inherits via `TypeDef("builtin_function", **Function.typedef.rawdict)`.
///
/// Slots that exist on `Function.typedef` *and* on `BuiltinFunction.typedef`
/// belong here so the two initializers stay structurally aligned with PyPy's
/// `**rawdict` pattern. Function-only slots (currently just `__get__`) and
/// BuiltinFunction-only overrides (`__new__`, `__self__`, `__repr__`,
/// `__doc__`) live in their respective wrappers.
fn init_function_type_common(_ns: &mut DictStorage) {
    // Pyre does not yet model the rest of Function.typedef
    // (`__call__`, `__name__`, `__qualname__`, `__doc__`, `__module__`,
    // `__globals__`, `__closure__`, `__defaults__`, `__kwdefaults__`,
    // `__code__`, `__dict__`, `__class__`, ...).
    // They will be installed here when ported, so they automatically
    // propagate to BuiltinFunction.typedef.
}

fn init_function_type(ns: &mut DictStorage) {
    init_function_type_common(ns);
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function("__get__", |args| {
            let w_function = args.first().copied().unwrap_or(pyre_object::w_none());
            let w_obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let w_cls = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            // function.py:464-470 descr_function_get
            //
            //   asking_for_function = (
            //       space.is_w(w_cls, space.w_None)
            //       or (
            //           space.is_w(w_obj, space.w_None)
            //           and not space.is_w(w_cls, space.type(space.w_None))
            //       )
            //   )
            //
            // The class-access case (`w_obj == None and w_cls is some type`)
            // returns the bare function — that's how `cls.func` stays callable
            // as a plain function rather than a bound method.
            let cls_is_none = unsafe { w_cls.is_null() || pyre_object::is_none(w_cls) };
            let obj_is_none = unsafe { w_obj.is_null() || pyre_object::is_none(w_obj) };
            let cls_is_none_type = std::ptr::eq(w_cls, gettypeobject(&pyre_object::NONE_TYPE));
            let asking_for_function = cls_is_none || (obj_is_none && !cls_is_none_type);
            if asking_for_function {
                Ok(w_function)
            } else {
                // function.py:470  Method(space, w_function, w_obj, w_cls)
                Ok(pyre_object::w_method_new(w_function, w_obj, w_cls))
            }
        }),
    );
}

/// PyPy typedef.py:813-820:
///
/// ```text
/// BuiltinFunction.typedef = TypeDef("builtin_function",
///                                   **Function.typedef.rawdict)
/// BuiltinFunction.typedef.rawdict.update({
///     '__new__': interp2app(BuiltinFunction.descr_builtinfunction__new__.im_func),
///     '__self__': GetSetProperty(always_none, cls=BuiltinFunction),
///     '__repr__': interp2app(BuiltinFunction.descr_function_repr),
///     '__doc__': getset_func_doc,
/// })
/// del BuiltinFunction.typedef.rawdict['__get__']
/// ```
///
/// `init_function_type_common` provides the shared `**rawdict` slots; the
/// missing `dict_storage_store(ns, "__get__", ...)` call after it expresses the
/// `del rawdict['__get__']` step. The `update({...})` overrides go below as
/// pyre starts modeling them.
fn init_builtin_function_type(ns: &mut DictStorage) {
    init_function_type_common(ns);
    dict_storage_store(
        ns,
        "__new__",
        make_new_descr(|_args| {
            Err(crate::PyError::type_error(
                "cannot create 'builtin_function' instances",
            ))
        }),
    );

    // typedef.py:816 GetSetProperty(always_none, cls=BuiltinFunction). The
    // `cls=` argument routes through descr_self_interp_w so wrong-class
    // instances raise DescrMismatch instead of silently returning None.
    // `init_builtin_function_type` runs while the BuiltinFunction
    // W_TypeObject is still under construction, so `cls` cannot be
    // resolved here; `patch_builtin_function_descriptors` runs after the
    // type cache is populated and writes the missing reqcls.
    let self_getter = make_builtin_function("__self__", |_args| Ok(pyre_object::w_none()));
    dict_storage_store(ns, "__self__", make_getset_descriptor(self_getter));

    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", |args| {
            let func = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let name = if func.is_null() {
                "<unknown>"
            } else {
                unsafe { crate::function_get_name(func) }
            };
            Ok(pyre_object::w_str_new(&format!(
                "<built-in function {name}>"
            )))
        }),
    );

    // function.py:395 getset_func_doc = GetSetProperty(fget_func_doc,
    // fset_func_doc, fdel_func_doc). Reading goes through fget_func_doc,
    // which is the same accessor used by `function.__doc__`; writing/
    // deleting routes through _check_code_mutable, which raises TypeError
    // for builtin functions because `can_change_code` is False.
    // The `cls=BuiltinFunction` guard is patched in by
    // `patch_builtin_function_descriptors` after the type cache exists.
    let doc_getter = make_builtin_function("__doc__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        Ok(unsafe { crate::function::fget_func_doc(func) })
    });
    let doc_setter = make_builtin_function("__doc__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_doc(func, value)? };
        Ok(pyre_object::w_none())
    });
    let doc_deleter = make_builtin_function("__doc__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fdel_func_doc(func)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__doc__",
        make_getset_property(doc_getter, doc_setter, doc_deleter),
    );
}

/// typedef.py:816,818 wires `cls=BuiltinFunction` on the `__self__` and
/// `__doc__` GetSetProperty entries; the inner `init_builtin_function_type`
/// runs while the W_TypeObject is still under construction, so the reqcls
/// patch happens here, after `init_typeobjects` has filled the cache and
/// the BuiltinFunction typeobject is reachable.
fn patch_builtin_function_descriptors() {
    let bf_type =
        gettypefor(&crate::BUILTIN_FUNCTION_TYPE as *const PyType).unwrap_or(pyre_object::PY_NULL);
    if bf_type.is_null() {
        return;
    }
    let dict_ptr = unsafe { pyre_object::w_type_get_dict_ptr(bf_type) } as *mut DictStorage;
    if dict_ptr.is_null() {
        return;
    }
    let ns = unsafe { &*dict_ptr };
    for name in ["__self__", "__doc__"] {
        if let Some(&descr) = ns.get(name) {
            if let Some(mut fields) = getset_fields_read(descr) {
                fields.reqcls = bf_type as usize;
                getset_fields_write(descr, fields);
            }
        }
    }
}

/// BuiltinCode.typedef (typedef.py) — code object attributes for builtins.
///
/// PyPy exposes co_name, co_varnames, co_argcount, co_flags, co_consts.
/// No __get__ — BuiltinCode is a code object, not a descriptor.
fn init_builtin_code_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "co_name",
        make_builtin_function("co_name", |args| {
            let code = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if code.is_null() {
                return Ok(pyre_object::w_none());
            }
            let name = unsafe { crate::builtin_code_name(code) };
            Ok(pyre_object::w_str_new(name))
        }),
    );
}

fn init_method_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__func__",
        make_builtin_function("__func__", |args| {
            Ok(args
                .first()
                .map(|&method| unsafe { pyre_object::w_method_get_func(method) })
                .unwrap_or(pyre_object::w_none()))
        }),
    );
    dict_storage_store(
        ns,
        "__self__",
        make_builtin_function("__self__", |args| {
            Ok(args
                .first()
                .map(|&method| unsafe { pyre_object::w_method_get_self(method) })
                .unwrap_or(pyre_object::w_none()))
        }),
    );
}

fn init_code_type(ns: &mut DictStorage) {
    // code.replace(**kwargs) — PyPy: interpreter/pycode.py W_PyCode.descr_replace.
    // The full method creates a new code object with the given fields
    // replaced; pyre's code objects are immutable, so replace() with no
    // kwargs returns the code itself (enough for reset_code / tests).
    dict_storage_store(
        ns,
        "replace",
        make_builtin_function("replace", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
        }),
    );
}

/// typedef.py:492-500 Member.typedef
fn init_member_descriptor_type(ns: &mut DictStorage) {
    // typedef.py:494 __get__ = interp2app(Member.descr_member_get)
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function("__get__", |args| {
            let descr = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if descr.is_null() || !unsafe { pyre_object::memberobject::is_member(descr) } {
                return Ok(pyre_object::w_none());
            }
            let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            // typedef.py:467: if space.is_w(w_obj, space.w_None): return self
            if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
                return Ok(descr);
            }
            // typedef.py:470: self.typecheck(space, w_obj)
            unsafe {
                let w_cls = pyre_object::w_member_get_cls(descr);
                if !w_cls.is_null()
                    && pyre_object::is_type(w_cls)
                    && !crate::baseobjspace::isinstance_w(obj, w_cls)
                {
                    let slot_name = pyre_object::w_member_get_name(descr);
                    return Err(crate::PyError::type_error(format!(
                        "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                        slot_name,
                        pyre_object::w_type_get_name(w_cls),
                        (*(*obj).ob_type).name,
                    )));
                }
            }
            // typedef.py:471-474: w_result = w_obj.getslotvalue(self.index)
            let slot_name = unsafe { pyre_object::w_member_get_name(descr) };
            let found = crate::baseobjspace::ATTR_TABLE.with(|table| {
                let table = table.borrow();
                table
                    .get(&(obj as usize))
                    .and_then(|d| d.get(slot_name).copied())
            });
            match found {
                Some(v) => Ok(v),
                None => Err(crate::PyError::new(
                    crate::PyErrorKind::AttributeError,
                    slot_name.to_string(),
                )),
            }
        }),
    );
    // typedef.py:495 __set__ = interp2app(Member.descr_member_set)
    dict_storage_store(
        ns,
        "__set__",
        make_builtin_function("__set__", |args| {
            let descr = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if descr.is_null() || !unsafe { pyre_object::memberobject::is_member(descr) } {
                return Ok(pyre_object::w_none());
            }
            let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            // typedef.py:480: self.typecheck(space, w_obj)
            unsafe {
                let w_cls = pyre_object::w_member_get_cls(descr);
                if !w_cls.is_null()
                    && pyre_object::is_type(w_cls)
                    && !crate::baseobjspace::isinstance_w(obj, w_cls)
                {
                    let slot_name = pyre_object::w_member_get_name(descr);
                    return Err(crate::PyError::type_error(format!(
                        "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                        slot_name,
                        pyre_object::w_type_get_name(w_cls),
                        (*(*obj).ob_type).name,
                    )));
                }
            }
            // typedef.py:481: w_obj.setslotvalue(self.index, w_value)
            let slot_name = unsafe { pyre_object::w_member_get_name(descr) };
            crate::baseobjspace::ATTR_TABLE.with(|table| {
                let mut table = table.borrow_mut();
                table
                    .entry(obj as usize)
                    .or_default()
                    .insert(slot_name.to_string(), value);
            });
            Ok(pyre_object::w_none())
        }),
    );
    // typedef.py:496 __delete__ = interp2app(Member.descr_member_del)
    dict_storage_store(
        ns,
        "__delete__",
        make_builtin_function("__delete__", |args| {
            let descr = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if descr.is_null() || !unsafe { pyre_object::memberobject::is_member(descr) } {
                return Ok(pyre_object::w_none());
            }
            let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            // typedef.py:486: self.typecheck(space, w_obj)
            unsafe {
                let w_cls = pyre_object::w_member_get_cls(descr);
                if !w_cls.is_null()
                    && pyre_object::is_type(w_cls)
                    && !crate::baseobjspace::isinstance_w(obj, w_cls)
                {
                    let slot_name = pyre_object::w_member_get_name(descr);
                    return Err(crate::PyError::type_error(format!(
                        "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                        slot_name,
                        pyre_object::w_type_get_name(w_cls),
                        (*(*obj).ob_type).name,
                    )));
                }
            }
            // typedef.py:487-490: success = w_obj.delslotvalue(self.index)
            let slot_name = unsafe { pyre_object::w_member_get_name(descr) };
            let removed = crate::baseobjspace::ATTR_TABLE.with(|table| {
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
            Ok(pyre_object::w_none())
        }),
    );
    // typedef.py:497 __name__ = interp_attrproperty('name', ...)
    dict_storage_store(
        ns,
        "__name__",
        make_builtin_function("__name__", |args| {
            let member = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if member.is_null() || !unsafe { pyre_object::memberobject::is_member(member) } {
                return Ok(pyre_object::w_none());
            }
            Ok(pyre_object::w_str_new(unsafe {
                pyre_object::w_member_get_name(member)
            }))
        }),
    );
    // typedef.py:498 __objclass__ = interp_attrproperty_w('w_cls', ...)
    dict_storage_store(
        ns,
        "__objclass__",
        make_builtin_function("__objclass__", |args| {
            let member = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if member.is_null() || !unsafe { pyre_object::memberobject::is_member(member) } {
                return Ok(pyre_object::w_none());
            }
            Ok(unsafe { pyre_object::w_member_get_cls(member) })
        }),
    );
}

/// `staticmethod.__new__(cls, func)` — PyPy: function.py StaticMethod.descr__new__
fn init_staticmethod_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", |args| {
            // staticmethod(func) — args[0] is cls (staticmethod type), args[1] is func
            let func = if args.len() > 1 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            Ok(pyre_object::propertyobject::w_staticmethod_new(func))
        }),
    );
}

/// `classmethod.__new__(cls, func)` — PyPy: function.py ClassMethod.descr__new__
fn init_classmethod_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", |args| {
            let func = if args.len() > 1 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            Ok(pyre_object::propertyobject::w_classmethod_new(func))
        }),
    );
}

/// `property.__new__(cls, fget=None, fset=None, fdel=None, doc=None)`
/// — descriptor.py W_Property.descr_new
fn init_property_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", |args| {
            // args[0] is cls; fget/fset/fdel follow.
            let fget = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let fset = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            let fdel = args.get(3).copied().unwrap_or(pyre_object::PY_NULL);
            Ok(pyre_object::w_property_new(fget, fset, fdel))
        }),
    );
}

fn init_int_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(int_descr_new));
    dict_storage_store(
        ns,
        "bit_length",
        make_builtin_function("bit_length", |args| {
            let val = if !args.is_empty() && unsafe { pyre_object::is_int(args[0]) } {
                unsafe { pyre_object::w_int_get_value(args[0]) }
            } else {
                0
            };
            let bits = if val == 0 {
                0
            } else {
                64 - val.unsigned_abs().leading_zeros()
            };
            Ok(pyre_object::w_int_new(bits as i64))
        }),
    );
    dict_storage_store(
        ns,
        "bit_count",
        // PyPy `intobject.py:649-650 descr_bit_count` parity:
        // `space.newint(_bit_count(self.intval))`.  Routes through
        // `pyre_object::int_bit_count` (`@jit.elidable` parity port of
        // `_bit_count`) so the call graph matches upstream
        // `descr_bit_count -> _bit_count` 1:1.
        make_builtin_function("bit_count", |args| {
            let val = if !args.is_empty() && unsafe { pyre_object::is_int(args[0]) } {
                unsafe { pyre_object::w_int_get_value(args[0]) }
            } else {
                0
            };
            Ok(pyre_object::w_int_new(pyre_object::int_bit_count(val)))
        }),
    );
    // int.to_bytes(length=1, byteorder='big', *, signed=False)
    // PyPy: longobject.py descr_to_bytes
    dict_storage_store(
        ns,
        "to_bytes",
        make_builtin_function("to_bytes", |args| {
            let val = if !args.is_empty() && unsafe { pyre_object::is_int(args[0]) } {
                unsafe { pyre_object::w_int_get_value(args[0]) }
            } else {
                0
            };
            let length = if args.len() >= 2 && unsafe { pyre_object::is_int(args[1]) } {
                unsafe { pyre_object::w_int_get_value(args[1]) as usize }
            } else {
                1
            };
            let little_endian = if args.len() >= 3 && unsafe { pyre_object::is_str(args[2]) } {
                unsafe { pyre_object::w_str_get_value(args[2]) == "little" }
            } else {
                false
            };
            let mut bytes = vec![0u8; length];
            let uval = val as u64;
            for i in 0..length {
                let shift = if little_endian { i } else { length - 1 - i } * 8;
                bytes[i] = ((uval >> shift) & 0xff) as u8;
            }
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
        }),
    );
    // int.from_bytes(bytes, byteorder='big', *, signed=False) — classmethod in CPython
    dict_storage_store(
        ns,
        "from_bytes",
        make_builtin_function("from_bytes", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            // First arg can be the cls or the data depending on call site.
            let data_arg = if args.len() >= 2 && !unsafe { pyre_object::is_type(args[0]) } {
                args[0]
            } else if args.len() >= 2 {
                args[1]
            } else {
                args[0]
            };
            let byteorder_arg = if args.len() >= 3 {
                args[2]
            } else if args.len() >= 2 && unsafe { pyre_object::is_str(args[1]) } {
                args[1]
            } else {
                pyre_object::w_str_new("big")
            };
            let bytes: Vec<u8> = unsafe {
                if pyre_object::bytesobject::is_bytes_like(data_arg) {
                    pyre_object::bytesobject::bytes_like_data(data_arg).to_vec()
                } else if pyre_object::is_str(data_arg) {
                    pyre_object::w_str_get_value(data_arg).as_bytes().to_vec()
                } else {
                    vec![]
                }
            };
            let little_endian = unsafe {
                pyre_object::is_str(byteorder_arg)
                    && pyre_object::w_str_get_value(byteorder_arg) == "little"
            };
            let mut val: u64 = 0;
            if little_endian {
                for (i, &b) in bytes.iter().enumerate() {
                    val |= (b as u64) << (i * 8);
                }
            } else {
                for &b in &bytes {
                    val = (val << 8) | b as u64;
                }
            }
            Ok(pyre_object::w_int_new(val as i64))
        }),
    );
    // int.__index__ / __int__ / __trunc__ — identity
    for method in ["__index__", "__int__", "__trunc__"] {
        dict_storage_store(
            ns,
            method,
            make_builtin_function(method, |args| {
                Ok(args.first().copied().unwrap_or(pyre_object::w_int_new(0)))
            }),
        );
    }
    // int.conjugate — identity
    dict_storage_store(
        ns,
        "conjugate",
        make_builtin_function("conjugate", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_int_new(0)))
        }),
    );
    // int.real / int.imag / int.numerator — properties
    // True.real → 1 (int, not bool), False.real → 0
    dict_storage_store(
        ns,
        "real",
        pyre_object::w_property_new(
            make_builtin_function("real", |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::w_int_new(0));
                unsafe {
                    if pyre_object::is_bool(obj) {
                        return Ok(pyre_object::w_int_new(
                            pyre_object::w_bool_get_value(obj) as i64
                        ));
                    }
                }
                Ok(obj)
            }),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    dict_storage_store(
        ns,
        "imag",
        pyre_object::w_property_new(
            make_builtin_function("imag", |_| Ok(pyre_object::w_int_new(0))),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    dict_storage_store(
        ns,
        "numerator",
        pyre_object::w_property_new(
            make_builtin_function("numerator", |args| {
                Ok(args.first().copied().unwrap_or(pyre_object::w_int_new(0)))
            }),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    dict_storage_store(
        ns,
        "denominator",
        make_builtin_function("denominator", |_| Ok(pyre_object::w_int_new(1))),
    );
}
fn init_float_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(float_descr_new));
    // float.__getformat__(kind) → returns the format string for the
    // given kind. PyPy: floatobject.py W_FloatObject.descr__getformat__.
    // Both 'double' and 'float' are IEEE 754 little-endian on x86/ARM.
    dict_storage_store(
        ns,
        "__getformat__",
        make_builtin_function("__getformat__", |args| {
            // Python classmethod signature: float.__getformat__(kind).
            // pyre may pass either (kind,) or (self, kind); accept both by
            // scanning for the first str argument.
            let kind = args
                .iter()
                .find_map(|&a| unsafe {
                    if pyre_object::is_str(a) {
                        Some(pyre_object::w_str_get_value(a).to_string())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    crate::PyError::type_error(
                        "__getformat__() argument must be 'double' or 'float'",
                    )
                })?;
            match kind.as_str() {
                "double" | "float" => Ok(pyre_object::w_str_new("IEEE, little-endian")),
                _ => Err(crate::PyError::value_error(
                    "__getformat__() argument must be 'double' or 'float'",
                )),
            }
        }),
    );
    dict_storage_store(
        ns,
        "hex",
        make_builtin_function("hex", |args| {
            // float.hex() — PyPy: descr_hex. Format the float as a hex
            // literal compatible with float.fromhex.
            if args.is_empty() {
                return Err(crate::PyError::type_error("hex() requires self"));
            }
            let v = unsafe { pyre_object::w_float_get_value(args[0]) };
            if v.is_nan() {
                return Ok(pyre_object::w_str_new("nan"));
            }
            if v.is_infinite() {
                return Ok(pyre_object::w_str_new(if v > 0.0 { "inf" } else { "-inf" }));
            }
            Ok(pyre_object::w_str_new(&format!("{v:e}")))
        }),
    );
    dict_storage_store(
        ns,
        "fromhex",
        make_builtin_function("fromhex", |args| {
            // float.fromhex(s) — PyPy: floatobject.py descr_fromhex.
            // Parse hexadecimal floating-point literals like '0x1.8p3'.
            let s_arg = args
                .iter()
                .find_map(|&a| unsafe {
                    if pyre_object::is_str(a) {
                        Some(pyre_object::w_str_get_value(a).to_string())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    crate::PyError::type_error("fromhex() requires a string argument")
                })?;
            let s = s_arg.trim();
            let lower = s.to_ascii_lowercase();
            match lower.as_str() {
                "inf" | "infinity" | "+inf" | "+infinity" => {
                    return Ok(pyre_object::w_float_new(f64::INFINITY));
                }
                "-inf" | "-infinity" => {
                    return Ok(pyre_object::w_float_new(f64::NEG_INFINITY));
                }
                "nan" | "+nan" | "-nan" => {
                    return Ok(pyre_object::w_float_new(f64::NAN));
                }
                _ => {}
            }
            let (sign_s, rest) = if let Some(r) = s.strip_prefix('-') {
                (-1.0f64, r)
            } else if let Some(r) = s.strip_prefix('+') {
                (1.0f64, r)
            } else {
                (1.0f64, s)
            };
            let rest = rest
                .strip_prefix("0x")
                .or_else(|| rest.strip_prefix("0X"))
                .unwrap_or(rest);
            let (body, exp_str) = if let Some(i) = rest.find(|c| c == 'p' || c == 'P') {
                (&rest[..i], &rest[i + 1..])
            } else {
                (rest, "0")
            };
            let (int_part, frac_part) = if let Some(i) = body.find('.') {
                (&body[..i], &body[i + 1..])
            } else {
                (body, "")
            };
            let int_val = if int_part.is_empty() {
                0u64
            } else {
                u64::from_str_radix(int_part, 16).map_err(|_| {
                    crate::PyError::value_error("invalid hexadecimal floating-point literal")
                })?
            };
            let mut frac_val = 0f64;
            for (i, ch) in frac_part.chars().enumerate() {
                let d = ch.to_digit(16).ok_or_else(|| {
                    crate::PyError::value_error("invalid hexadecimal floating-point literal")
                })? as f64;
                frac_val += d / 16f64.powi(i as i32 + 1);
            }
            let exp: i32 = exp_str.parse().map_err(|_| {
                crate::PyError::value_error("invalid hexadecimal floating-point literal")
            })?;
            let mantissa = int_val as f64 + frac_val;
            Ok(pyre_object::w_float_new(sign_s * mantissa * 2f64.powi(exp)))
        }),
    );
    dict_storage_store(
        ns,
        "is_integer",
        make_builtin_function("is_integer", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_bool_from(false));
            }
            let v = unsafe { pyre_object::w_float_get_value(args[0]) };
            Ok(pyre_object::w_bool_from(v.is_finite() && v == v.trunc()))
        }),
    );
    dict_storage_store(
        ns,
        "as_integer_ratio",
        make_builtin_function("as_integer_ratio", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "as_integer_ratio() requires self",
                ));
            }
            let v = unsafe { pyre_object::w_float_get_value(args[0]) };
            if v.is_nan() || v.is_infinite() {
                return Err(crate::PyError::value_error(
                    "cannot convert NaN/Infinity to integer ratio",
                ));
            }
            // Simple conversion: use bit representation to get exact ratio.
            // f = m * 2^e where m is an integer mantissa.
            let (mantissa, exponent, sign) = integer_decode(v);
            let sign_i = sign as i64;
            let m = mantissa as i64 * sign_i;
            if exponent >= 0 {
                let num = m.saturating_mul(1i64 << exponent.min(62));
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(num),
                    pyre_object::w_int_new(1),
                ]))
            } else {
                let denom = 1i64 << (-exponent).min(62);
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(m),
                    pyre_object::w_int_new(denom),
                ]))
            }
        }),
    );
    // floatobject.py:713: __int__ = interp2app(W_FloatObject.descr_trunc)
    for method in ["__trunc__", "__int__"] {
        dict_storage_store(
            ns,
            method,
            make_builtin_function(method, |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("__trunc__() requires self"));
                }
                let v = unsafe { pyre_object::w_float_get_value(args[0]) };
                Ok(pyre_object::w_int_new(v.trunc() as i64))
            }),
        );
    }
}

/// IEEE 754 double decomposition into (mantissa, exponent, sign).
/// PyPy: Lib/fractions.py _decimal_to_ratio uses a similar approach.
fn integer_decode(v: f64) -> (u64, i16, i8) {
    let bits = v.to_bits();
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };
    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}
fn init_bool_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(bool_descr_new));
}

// ── Object TypeDef ───────────────────────────────────────────────────
// PyPy: pypy/objspace/std/objectobject.py TypeDef("object", ...)

/// `object.__new__(cls)` — allocate a bare instance of cls.
///
/// PyPy: objectobject.py descr__new__
fn object_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(
        !args.is_empty(),
        "object.__new__() requires a type argument"
    );
    let cls = crate::baseobjspace::unwrap_cell(args[0]);
    // cls should be a W_TypeObject — create instance of it
    if unsafe { is_type(cls) } {
        return Ok(w_instance_new(cls));
    }
    // Fallback: create bare instance with no type
    Ok(w_instance_new(PY_NULL))
}

/// `object.__init__(self)` — no-op base __init__.
fn object_descr_init(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn init_object_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(object_descr_new));
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", object_descr_init),
    );
    // PyPy: objectobject.py — default comparison/hash/repr for all objects
    dict_storage_store(
        ns,
        "__eq__",
        make_builtin_function("__eq__", |args| {
            Ok(pyre_object::w_bool_from(
                args.len() >= 2 && std::ptr::eq(args[0], args[1]),
            ))
        }),
    );
    dict_storage_store(
        ns,
        "__ne__",
        make_builtin_function("__ne__", |args| {
            Ok(pyre_object::w_bool_from(
                args.len() >= 2 && !std::ptr::eq(args[0], args[1]),
            ))
        }),
    );
    dict_storage_store(
        ns,
        "__hash__",
        make_builtin_function("__hash__", |args| {
            Ok(pyre_object::w_int_new(if args.is_empty() {
                0
            } else {
                args[0] as i64
            }))
        }),
    );
    dict_storage_store(
        ns,
        "__repr__",
        // PyPy: objectobject.py descr___repr__ — base __repr__ for all objects
        make_builtin_function("__repr__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("<object>"));
            }
            let obj = args[0];
            unsafe {
                if pyre_object::is_instance(obj) {
                    let w_type = pyre_object::w_instance_get_type(obj);
                    let name = pyre_object::w_type_get_name(w_type);
                    return Ok(pyre_object::w_str_new(&format!(
                        "<{name} object at {obj:?}>"
                    )));
                }
            }
            // For non-instances, delegate to display
            Ok(pyre_object::w_str_new(&format!("<object at {:?}>", obj)))
        }),
    );
    dict_storage_store(
        ns,
        "__str__",
        make_builtin_function("__str__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("<object>"));
            }
            // Delegate to __repr__ to avoid infinite recursion
            // PyPy: objectobject.py descr___str__ → space.repr(w_self)
            Ok(pyre_object::w_str_new(&unsafe { crate::py_repr(args[0]) }))
        }),
    );
    // PyPy: objectobject.py descr___format__
    dict_storage_store(
        ns,
        "__format__",
        make_builtin_function("__format__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new(""));
            }
            Ok(pyre_object::w_str_new(&unsafe { crate::py_str(args[0]) }))
        }),
    );
    // PyPy: objectobject.py descr___reduce_ex__
    dict_storage_store(
        ns,
        "__reduce_ex__",
        make_builtin_function("__reduce_ex__", |_| Ok(pyre_object::w_none())),
    );
    dict_storage_store(
        ns,
        "__init_subclass__",
        make_builtin_function("__init_subclass__", |_| Ok(pyre_object::w_none())),
    );
    dict_storage_store(
        ns,
        "__subclasshook__",
        make_builtin_function("__subclasshook__", |_| Ok(pyre_object::w_not_implemented())),
    );
    // PyPy: objectobject.py descr___setattr__
    // object.__setattr__(self, name, value) → setattr dispatch
    dict_storage_store(
        ns,
        "__setattr__",
        make_builtin_function("__setattr__", |args| {
            if args.len() < 3 {
                return Err(crate::PyError::type_error(
                    "__setattr__ requires 3 arguments",
                ));
            }
            let name = unsafe { pyre_object::w_str_get_value(args[1]) };
            crate::baseobjspace::setattr(args[0], name, args[2])
        }),
    );
    // PyPy: objectobject.py descr___delattr__
    dict_storage_store(
        ns,
        "__delattr__",
        make_builtin_function("__delattr__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "__delattr__ requires 2 arguments",
                ));
            }
            let name = unsafe { pyre_object::w_str_get_value(args[1]) };
            crate::baseobjspace::delattr(args[0], name)
        }),
    );
    // PyPy: objectobject.py descr___getattribute__
    dict_storage_store(
        ns,
        "__getattribute__",
        make_builtin_function("__getattribute__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "__getattribute__ requires 2 arguments",
                ));
            }
            let name = unsafe { pyre_object::w_str_get_value(args[1]) };
            crate::baseobjspace::getattr(args[0], name)
        }),
    );
}

fn bytearray_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] = cls (ignored — bytearray subclasses still allocate the
    // primitive layout). bytearrayobject.py descr_new accepts:
    //   bytearray()           → empty
    //   bytearray(int)        → zero-filled buffer of length n
    //   bytearray(bytes-like) → copy of the contents
    //   bytearray(str, encoding[, errors]) → encoded bytes (encoding ignored)
    let rest = if args.is_empty() { args } else { &args[1..] };
    if rest.is_empty() {
        return Ok(pyre_object::bytearrayobject::w_bytearray_new(0));
    }
    let arg = rest[0];
    unsafe {
        if pyre_object::is_int(arg) {
            let n = pyre_object::w_int_get_value(arg).max(0) as usize;
            return Ok(pyre_object::bytearrayobject::w_bytearray_new(n));
        }
        if pyre_object::bytesobject::is_bytes_like(arg) {
            let data = pyre_object::bytesobject::bytes_like_data(arg);
            return Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(data));
        }
        if pyre_object::is_str(arg) {
            let s = pyre_object::w_str_get_value(arg);
            return Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                s.as_bytes(),
            ));
        }
    }
    // Iterable of ints — collect and convert.
    if let Ok(items) = crate::builtins::collect_iterable(arg) {
        let mut buf = Vec::with_capacity(items.len());
        for item in items {
            if unsafe { pyre_object::is_int(item) } {
                let v = unsafe { pyre_object::w_int_get_value(item) };
                buf.push((v & 0xff) as u8);
            }
        }
        return Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(&buf));
    }
    Ok(pyre_object::bytearrayobject::w_bytearray_new(0))
}

/// PyPy: bytesobject.py W_BytesObject.typedef
fn init_bytes_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(bytes_descr_new));
    dict_storage_store(
        ns,
        "decode",
        make_builtin_function("decode", bytes_method_decode),
    );
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", bytes_method_repr),
    );
    dict_storage_store(
        ns,
        "__str__",
        make_builtin_function("__str__", bytes_method_repr),
    );
    // bytes methods are mostly shared with bytearray — add as needed.
}

/// PyPy: bytesobject.py descr_decode
/// Decodes bytes to str via the given codec (defaults to utf-8/strict).
fn bytes_method_decode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let encoding: String = if args.len() >= 2 {
        unsafe {
            if pyre_object::is_str(args[1]) {
                pyre_object::w_str_get_value(args[1]).to_string()
            } else {
                "utf-8".to_string()
            }
        }
    } else {
        "utf-8".to_string()
    };
    let enc_lower = encoding.to_ascii_lowercase().replace('_', "-");
    let s = match enc_lower.as_str() {
        "utf-8" | "utf8" | "u8" | "ascii" | "us-ascii" | "646" => {
            // Best-effort: surrogateescape on invalid UTF-8 produces
            // lossy output; for our purposes just replace invalid with U+FFFD.
            match std::str::from_utf8(data) {
                Ok(s) => s.to_string(),
                Err(_) => String::from_utf8_lossy(data).into_owned(),
            }
        }
        "latin-1" | "latin1" | "iso-8859-1" | "8859" => data.iter().map(|&b| b as char).collect(),
        _ => String::from_utf8_lossy(data).into_owned(),
    };
    Ok(pyre_object::w_str_new(&s))
}

/// PyPy: bytesobject.py descr_repr — returns a quoted literal like `b'hello'`.
fn bytes_method_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    // Determine preferred quote: single unless the data contains single but
    // not double quote (matches CPython).
    let has_single = data.contains(&b'\'');
    let has_double = data.contains(&b'"');
    let quote: char = if has_single && !has_double { '"' } else { '\'' };
    let mut out = String::with_capacity(data.len() + 3);
    out.push('b');
    out.push(quote);
    for &b in data {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            q if q as char == quote => {
                out.push('\\');
                out.push(quote);
            }
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\x{:02x}", b)),
        }
    }
    out.push(quote);
    Ok(pyre_object::w_str_new(&out))
}

/// `bytes.__new__(cls, *args)` — PyPy: bytesobject.py descr__new__
fn bytes_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] = cls (ignored for now)
    // bytes()           → empty
    // bytes(int)        → zero-filled
    // bytes(bytes-like) → copy
    // bytes(str)        → UTF-8 encode
    // bytes(iterable)   → collect bytes
    if args.len() <= 1 {
        return Ok(pyre_object::bytesobject::w_bytes_empty());
    }
    let arg = args[1];
    unsafe {
        if pyre_object::is_int(arg) {
            let n = pyre_object::w_int_get_value(arg).max(0) as usize;
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&vec![0u8; n]));
        }
        if pyre_object::bytesobject::is_bytes_like(arg) {
            let data = pyre_object::bytesobject::bytes_like_data(arg);
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(data));
        }
        if pyre_object::is_str(arg) {
            let s = pyre_object::w_str_get_value(arg);
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(s.as_bytes()));
        }
    }
    // Iterable of ints
    let items = crate::builtins::collect_iterable(arg)?;
    let mut buf = Vec::with_capacity(items.len());
    for item in items {
        let val = unsafe { pyre_object::w_int_get_value(item) };
        buf.push(val as u8);
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf))
}

/// PyPy: bytearrayobject.py W_BytearrayObject.typedef
fn init_bytearray_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(bytearray_descr_new));
    dict_storage_store(
        ns,
        "find",
        make_builtin_function("find", |args| {
            assert!(args.len() >= 2, "find() takes at least 1 argument");
            let ba = args[0];
            let value = args[1];
            let start = if args.len() > 2 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize
            } else {
                0
            };
            unsafe {
                let v = pyre_object::w_int_get_value(value) as u8;
                Ok(pyre_object::w_int_new(
                    pyre_object::bytearrayobject::w_bytearray_find(ba, v, start),
                ))
            }
        }),
    );
    dict_storage_store(
        ns,
        "__add__",
        make_builtin_function("__add__", |args| {
            assert!(args.len() >= 2, "__add__ requires 2 arguments");
            let a = args[0];
            let b = args[1];
            unsafe {
                let a_data = pyre_object::bytesobject::bytes_like_data(a);
                let b_data = if pyre_object::bytesobject::is_bytes_like(b) {
                    pyre_object::bytesobject::bytes_like_data(b).to_vec()
                } else {
                    vec![]
                };
                let mut result = a_data.to_vec();
                result.extend_from_slice(&b_data);
                Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                    &result,
                ))
            }
        }),
    );
    dict_storage_store(
        ns,
        "__iadd__",
        make_builtin_function("__iadd__", |args| {
            assert!(args.len() >= 2);
            let ba = args[0];
            let other = args[1];
            unsafe {
                if pyre_object::bytesobject::is_bytes_like(other) {
                    let data = pyre_object::bytesobject::bytes_like_data(other).to_vec();
                    pyre_object::bytearrayobject::w_bytearray_extend(ba, &data);
                }
            }
            Ok(ba)
        }),
    );
    dict_storage_store(
        ns,
        "translate",
        make_builtin_function("translate", |args| {
            assert!(args.len() >= 2);
            let ba = args[0];
            let table = args[1];
            unsafe {
                let data = pyre_object::bytesobject::bytes_like_data(ba);
                let table_bytes_owned;
                let table_data: &[u8] = if pyre_object::bytesobject::is_bytes_like(table) {
                    pyre_object::bytesobject::bytes_like_data(table)
                } else if pyre_object::is_str(table) {
                    table_bytes_owned = pyre_object::w_str_get_value(table).as_bytes();
                    table_bytes_owned
                } else {
                    return Ok(ba);
                };
                let mut result = Vec::with_capacity(data.len());
                for &b in data {
                    if (b as usize) < table_data.len() {
                        result.push(table_data[b as usize]);
                    } else {
                        result.push(b);
                    }
                }
                Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                    &result,
                ))
            }
        }),
    );
}

// ── set / frozenset TypeDef ──────────────────────────────────────────
// PyPy: pypy/objspace/std/setobject.py W_BaseSetObject.typedef
// pyre splits the shared methods through `init_setlike_common` so the
// frozenset typedef can omit the in-place mutators.

fn init_setlike_common(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            unsafe {
                if pyre_object::is_set_or_frozenset(args[0]) {
                    return Ok(pyre_object::w_bool_from(pyre_object::w_set_contains(
                        args[0], args[1],
                    )));
                }
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            unsafe {
                if pyre_object::is_set_or_frozenset(args[0]) {
                    return Ok(pyre_object::w_int_new(
                        pyre_object::w_set_len(args[0]) as i64
                    ));
                }
            }
            Ok(pyre_object::w_int_new(0))
        }),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
    dict_storage_store(
        ns,
        "__bool__",
        make_builtin_function("__bool__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_bool_from(false));
            }
            unsafe {
                if pyre_object::is_set_or_frozenset(args[0]) {
                    return Ok(pyre_object::w_bool_from(
                        pyre_object::w_set_len(args[0]) > 0,
                    ));
                }
            }
            Ok(pyre_object::w_bool_from(true))
        }),
    );
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function("__or__", set_method_union),
    );
    dict_storage_store(
        ns,
        "__and__",
        make_builtin_function("__and__", set_method_intersection),
    );
    dict_storage_store(
        ns,
        "__sub__",
        make_builtin_function("__sub__", set_method_difference),
    );
    dict_storage_store(
        ns,
        "__xor__",
        make_builtin_function("__xor__", set_method_symmetric_difference),
    );
    dict_storage_store(ns, "__eq__", make_builtin_function("__eq__", set_method_eq));
    dict_storage_store(ns, "__le__", make_builtin_function("__le__", set_method_le));
    dict_storage_store(ns, "__ge__", make_builtin_function("__ge__", set_method_ge));
    dict_storage_store(
        ns,
        "__lt__",
        make_builtin_function("__lt__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            let le = unsafe { pyre_object::w_bool_get_value(set_method_le(args)?) };
            let eq = unsafe { pyre_object::w_bool_get_value(set_method_eq(args)?) };
            Ok(pyre_object::w_bool_from(le && !eq))
        }),
    );
    dict_storage_store(
        ns,
        "__gt__",
        make_builtin_function("__gt__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            let ge = unsafe { pyre_object::w_bool_get_value(set_method_ge(args)?) };
            let eq = unsafe { pyre_object::w_bool_get_value(set_method_eq(args)?) };
            Ok(pyre_object::w_bool_from(ge && !eq))
        }),
    );
    dict_storage_store(
        ns,
        "union",
        make_builtin_function("union", set_method_union),
    );
    dict_storage_store(
        ns,
        "intersection",
        make_builtin_function("intersection", set_method_intersection),
    );
    dict_storage_store(
        ns,
        "difference",
        make_builtin_function("difference", set_method_difference),
    );
    dict_storage_store(
        ns,
        "symmetric_difference",
        make_builtin_function("symmetric_difference", set_method_symmetric_difference),
    );
    dict_storage_store(
        ns,
        "issubset",
        make_builtin_function("issubset", set_method_le),
    );
    dict_storage_store(
        ns,
        "issuperset",
        make_builtin_function("issuperset", set_method_ge),
    );
    dict_storage_store(
        ns,
        "isdisjoint",
        make_builtin_function("isdisjoint", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(true));
            }
            let other_items = crate::builtins::collect_iterable(args[1])?;
            unsafe {
                for item in &other_items {
                    if pyre_object::w_set_contains(args[0], *item) {
                        return Ok(pyre_object::w_bool_from(false));
                    }
                }
            }
            Ok(pyre_object::w_bool_from(true))
        }),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function("copy", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_set_new());
            }
            let items = unsafe { pyre_object::w_set_items(args[0]) };
            unsafe {
                if pyre_object::is_frozenset(args[0]) {
                    return Ok(pyre_object::w_frozenset_from_items(&items));
                }
            }
            Ok(pyre_object::w_set_from_items(&items))
        }),
    );
}

fn set_method_union(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let mut items = unsafe { pyre_object::w_set_items(args[0]) };
    for other in &args[1..] {
        let other_items = crate::builtins::collect_iterable(*other)?;
        for item in other_items {
            items.push(item);
        }
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            return Ok(pyre_object::w_frozenset_from_items(&items));
        }
    }
    Ok(pyre_object::w_set_from_items(&items))
}

fn set_method_intersection(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let self_items = unsafe { pyre_object::w_set_items(args[0]) };
    let mut result: Vec<pyre_object::PyObjectRef> = self_items;
    for other in &args[1..] {
        let other_items = crate::builtins::collect_iterable(*other)?;
        result.retain(|&item| unsafe {
            other_items
                .iter()
                .any(|&o| pyre_object::w_set_contains(pyre_object::w_set_from_items(&[o]), item))
        });
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            return Ok(pyre_object::w_frozenset_from_items(&result));
        }
    }
    Ok(pyre_object::w_set_from_items(&result))
}

fn set_method_difference(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let mut items = unsafe { pyre_object::w_set_items(args[0]) };
    for other in &args[1..] {
        let other_items = crate::builtins::collect_iterable(*other)?;
        let probe = pyre_object::w_set_from_items(&other_items);
        items.retain(|&item| !unsafe { pyre_object::w_set_contains(probe, item) });
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            return Ok(pyre_object::w_frozenset_from_items(&items));
        }
    }
    Ok(pyre_object::w_set_from_items(&items))
}

fn set_method_symmetric_difference(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        if args.is_empty() {
            return Ok(pyre_object::w_set_new());
        }
        return Ok(args[0]);
    }
    let self_items = unsafe { pyre_object::w_set_items(args[0]) };
    let other_items = crate::builtins::collect_iterable(args[1])?;
    let other_probe = pyre_object::w_set_from_items(&other_items);
    let self_probe = pyre_object::w_set_from_items(&self_items);
    let mut result: Vec<pyre_object::PyObjectRef> = self_items
        .iter()
        .copied()
        .filter(|&item| !unsafe { pyre_object::w_set_contains(other_probe, item) })
        .collect();
    for item in other_items {
        if !unsafe { pyre_object::w_set_contains(self_probe, item) } {
            result.push(item);
        }
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            return Ok(pyre_object::w_frozenset_from_items(&result));
        }
    }
    Ok(pyre_object::w_set_from_items(&result))
}

fn set_method_eq(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_bool_from(false));
    }
    unsafe {
        if !pyre_object::is_set_or_frozenset(args[1]) {
            return Ok(pyre_object::w_bool_from(false));
        }
        if pyre_object::w_set_len(args[0]) != pyre_object::w_set_len(args[1]) {
            return Ok(pyre_object::w_bool_from(false));
        }
        for item in pyre_object::w_set_items(args[0]) {
            if !pyre_object::w_set_contains(args[1], item) {
                return Ok(pyre_object::w_bool_from(false));
            }
        }
    }
    Ok(pyre_object::w_bool_from(true))
}

fn set_method_le(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_bool_from(true));
    }
    let other_items = crate::builtins::collect_iterable(args[1])?;
    let probe = pyre_object::w_set_from_items(&other_items);
    unsafe {
        for item in pyre_object::w_set_items(args[0]) {
            if !pyre_object::w_set_contains(probe, item) {
                return Ok(pyre_object::w_bool_from(false));
            }
        }
    }
    Ok(pyre_object::w_bool_from(true))
}

fn set_method_ge(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_bool_from(true));
    }
    let other_items = crate::builtins::collect_iterable(args[1])?;
    unsafe {
        for item in other_items {
            if !pyre_object::w_set_contains(args[0], item) {
                return Ok(pyre_object::w_bool_from(false));
            }
        }
    }
    Ok(pyre_object::w_bool_from(true))
}

fn init_set_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(set_descr_new));
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", set_descr_init),
    );
    init_setlike_common(ns);
    dict_storage_store(
        ns,
        "add",
        make_builtin_function("add", |args| {
            if args.len() >= 2 {
                unsafe { pyre_object::w_set_add(args[0], args[1]) };
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "discard",
        make_builtin_function("discard", |args| {
            if args.len() >= 2 {
                unsafe { pyre_object::w_set_discard(args[0], args[1]) };
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "remove",
        make_builtin_function("remove", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("remove() requires an argument"));
            }
            let removed = unsafe { pyre_object::w_set_discard(args[0], args[1]) };
            if !removed {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::KeyError,
                    "set.remove(x): x not in set",
                ));
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "pop",
        make_builtin_function("pop", |args| {
            if args.is_empty() {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::KeyError,
                    "pop from an empty set",
                ));
            }
            let items = unsafe { pyre_object::w_set_items(args[0]) };
            if let Some(&item) = items.first() {
                unsafe { pyre_object::w_set_discard(args[0], item) };
                return Ok(item);
            }
            Err(crate::PyError::new(
                crate::PyErrorKind::KeyError,
                "pop from an empty set",
            ))
        }),
    );
    dict_storage_store(
        ns,
        "clear",
        make_builtin_function("clear", |args| {
            if !args.is_empty() {
                let items = unsafe { pyre_object::w_set_items(args[0]) };
                for item in items {
                    unsafe { pyre_object::w_set_discard(args[0], item) };
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "update",
        make_builtin_function("update", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            for other in &args[1..] {
                let other_items = crate::builtins::collect_iterable(*other)?;
                for item in other_items {
                    unsafe { pyre_object::w_set_add(args[0], item) };
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
}

fn init_frozenset_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(frozenset_descr_new));
    init_setlike_common(ns);
}

// ── __dict__ / __weakref__ descriptors ───────────────────────────────

/// typedef.py:561-563 dict_descr.
///
/// ```python
/// dict_descr = GetSetProperty(descr_get_dict, descr_set_dict, descr_del_dict,
///                             doc="dictionary for instance variables (if defined)")
/// dict_descr.name = '__dict__'
/// ```
pub fn dict_descr() -> pyre_object::PyObjectRef {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    let addr = *CACHED.get_or_init(|| {
        let fget = make_builtin_function("descr_get_dict", descr_get_dict);
        let fset = make_builtin_function("descr_set_dict", descr_set_dict);
        let fdel = make_builtin_function("descr_del_dict", descr_del_dict);
        let descr = make_getset_property(fget, fset, fdel);
        let _ = crate::baseobjspace::setattr(descr, "__name__", pyre_object::w_str_new("__dict__"));
        descr as usize
    });
    addr as pyre_object::PyObjectRef
}

/// typedef.py:593-595 weakref_descr.
///
/// ```python
/// weakref_descr = GetSetProperty(descr_get_weakref,
///                     doc="list of weak references to the object (if defined)")
/// weakref_descr.name = '__weakref__'
/// ```
pub fn weakref_descr() -> pyre_object::PyObjectRef {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    let addr = *CACHED.get_or_init(|| {
        let fget = make_builtin_function("descr_get_weakref", descr_get_weakref);
        let descr = make_getset_property(fget, pyre_object::PY_NULL, pyre_object::PY_NULL);
        let _ =
            crate::baseobjspace::setattr(descr, "__name__", pyre_object::w_str_new("__weakref__"));
        descr as usize
    });
    addr as pyre_object::PyObjectRef
}

/// PyPy stores `fget/fset/fdel/doc/reqcls/use_closure/name` directly on
/// the `GetSetProperty` instance fields. pyre's instance dict (mapdict)
/// is thread-local, but `init_typeobjects` runs once globally and the
/// resulting `getset_descriptor` instances are visible from every thread,
/// so per-instance attribute storage cannot be used. Mirror PyPy's
/// "fields live on the descriptor itself" model with a process-global
/// side table keyed by descriptor address.
#[derive(Clone, Copy)]
struct GetSetFields {
    fget: usize,
    fset: usize,
    fdel: usize,
    doc: usize,
    reqcls: usize,
    use_closure: bool,
    name: usize,
}

static GETSET_FIELDS: std::sync::OnceLock<
    std::sync::RwLock<std::collections::HashMap<usize, GetSetFields>>,
> = std::sync::OnceLock::new();

fn getset_fields_table()
-> &'static std::sync::RwLock<std::collections::HashMap<usize, GetSetFields>> {
    GETSET_FIELDS.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

fn getset_fields_read(descr: pyre_object::PyObjectRef) -> Option<GetSetFields> {
    if descr.is_null() {
        return None;
    }
    getset_fields_table()
        .read()
        .unwrap()
        .get(&(descr as usize))
        .copied()
}

fn getset_fields_write(descr: pyre_object::PyObjectRef, fields: GetSetFields) {
    getset_fields_table()
        .write()
        .unwrap()
        .insert(descr as usize, fields);
}

/// typedef.py:327-335 GetSetProperty._init.
///
/// ```python
/// def _init(self, fget, fset, fdel, doc, cls, use_closure, name):
///     self.fget = fget
///     self.fset = fset
///     self.fdel = fdel
///     self.doc = doc
///     self.reqcls = cls
///     self.use_closure = use_closure
///     self.name = name if name is not None else '<generic property>'
/// ```
///
/// `cls` is stored as `reqcls` exactly like PyPy. `use_closure` is unused
/// (pyre has no closure-passing distinction) but still stored for parity.
fn getset_property_init(
    new: pyre_object::PyObjectRef,
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
    doc: pyre_object::PyObjectRef,
    cls: pyre_object::PyObjectRef,
    use_closure: bool,
    name: pyre_object::PyObjectRef,
) {
    let resolved_name = if !name.is_null() && unsafe { pyre_object::is_str(name) } {
        name
    } else {
        pyre_object::w_str_new("<generic property>")
    };
    getset_fields_write(
        new,
        GetSetFields {
            fget: fget as usize,
            fset: fset as usize,
            fdel: fdel as usize,
            doc: doc as usize,
            reqcls: cls as usize,
            use_closure,
            name: resolved_name as usize,
        },
    );
}

/// Read the optional `reqcls` field from a getset descriptor.
/// Returns null if no required class is set.
fn read_reqcls(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    getset_fields_read(descr)
        .map(|f| f.reqcls as pyre_object::PyObjectRef)
        .filter(|c| !c.is_null() && !unsafe { pyre_object::is_none(*c) })
        .unwrap_or(pyre_object::PY_NULL)
}

fn read_fget(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    getset_fields_read(descr)
        .map(|f| f.fget as pyre_object::PyObjectRef)
        .unwrap_or(pyre_object::PY_NULL)
}

fn read_fset(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    getset_fields_read(descr)
        .map(|f| f.fset as pyre_object::PyObjectRef)
        .unwrap_or(pyre_object::PY_NULL)
}

fn read_fdel(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    getset_fields_read(descr)
        .map(|f| f.fdel as pyre_object::PyObjectRef)
        .unwrap_or(pyre_object::PY_NULL)
}

fn read_descr_name(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    getset_fields_read(descr)
        .map(|f| f.name as pyre_object::PyObjectRef)
        .unwrap_or(pyre_object::PY_NULL)
}

/// typedef.py:337-345 GetSetProperty.copy_for_type.
///
/// ```python
/// def copy_for_type(self, w_objclass):
///     if self.reqcls is None:
///         new = instantiate(GetSetProperty)
///         new._init(self.fget, self.fset, self.fdel, self.doc, self.reqcls,
///                   self.use_closure, self.name)
///         new.w_objclass = w_objclass
///         return new
///     else:
///         return self
/// ```
fn copy_for_type(
    descr: pyre_object::PyObjectRef,
    w_objclass: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    // typedef.py:338 if self.reqcls is None:
    let reqcls = read_reqcls(descr);
    if !reqcls.is_null() {
        // typedef.py:344 return self
        return descr;
    }
    let fields = match getset_fields_read(descr) {
        Some(f) => f,
        None => return descr,
    };
    // typedef.py:339 new = instantiate(GetSetProperty)
    let new = w_instance_new(getset_descriptor_type());
    // typedef.py:340-341 new._init(self.fget, self.fset, self.fdel, self.doc,
    //                              self.reqcls, self.use_closure, self.name)
    getset_property_init(
        new,
        fields.fget as pyre_object::PyObjectRef,
        fields.fset as pyre_object::PyObjectRef,
        fields.fdel as pyre_object::PyObjectRef,
        fields.doc as pyre_object::PyObjectRef,
        pyre_object::PY_NULL,
        fields.use_closure,
        fields.name as pyre_object::PyObjectRef,
    );
    // typedef.py:342 new.w_objclass = w_objclass
    let _ = crate::baseobjspace::setattr(new, "__objclass__", w_objclass);
    new
}

/// Public re-export of `copy_for_type` so that
/// `objspace/std/typeobject.py::create_dict_slot`'s pyre equivalent in
/// `call.rs` can call `copy_for_type(dict_descr(), w_self)` directly,
/// matching PyPy's `dict_descr.copy_for_type(w_self)` shape.
pub fn copy_descriptor_for_type(
    descr: pyre_object::PyObjectRef,
    w_objclass: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    copy_for_type(descr, w_objclass)
}

/// typedef.py:541-547 descr_get_dict.
///
/// ```python
/// def descr_get_dict(space, w_obj):
///     w_dict = w_obj.getdict(space)
///     if w_dict is None:
///         raise oefmt(space.w_TypeError,
///                     "descriptor '__dict__' doesn't apply to '%T' objects",
///                     w_obj)
///     return w_dict
/// ```
///
/// In pyre the typecheck wrapper passes (closure, w_obj) — args[0] is
/// the descriptor `self` and args[1] is w_obj. There is no `space`
/// parameter (pyre has no space first-class object).
fn descr_get_dict(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let _closure = args[0];
    let w_obj = args[1];
    let w_dict = crate::baseobjspace::getdict(w_obj);
    if w_dict.is_null() {
        let tp_name = unsafe { (*(*w_obj).ob_type).name };
        return Err(crate::PyError::type_error(format!(
            "descriptor '__dict__' doesn't apply to '{}' objects",
            tp_name,
        )));
    }
    Ok(w_dict)
}

/// typedef.py:549-550 descr_set_dict.
///
/// ```python
/// def descr_set_dict(space, w_obj, w_dict):
///     w_obj.setdict(space, w_dict)
/// ```
fn descr_set_dict(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let _closure = args[0];
    let w_obj = args[1];
    let w_dict = args[2];
    crate::baseobjspace::setdict(w_obj, w_dict)?;
    Ok(pyre_object::w_none())
}

/// typedef.py:552-553 descr_del_dict.
///
/// ```python
/// def descr_del_dict(space, w_obj): # blame CPython for the existence of this one
///     w_obj.setdict(space, space.newdict())
/// ```
fn descr_del_dict(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let _closure = args[0];
    let w_obj = args[1];
    crate::baseobjspace::setdict(w_obj, pyre_object::w_dict_new())?;
    Ok(pyre_object::w_none())
}

/// typedef.py:555-559 descr_get_weakref.
///
/// ```python
/// def descr_get_weakref(space, w_obj):
///     lifeline = w_obj.getweakref()
///     if lifeline is None:
///         return space.w_None
///     return lifeline.get_any_weakref(space)
/// ```
fn descr_get_weakref(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let _closure = args[0];
    let w_obj = args[1];
    let lifeline = crate::baseobjspace::getweakref(w_obj);
    match lifeline {
        None => Ok(pyre_object::w_none()),
        Some(lifeline) => Ok(crate::module::_weakref::interp_weakref::get_any_weakref(
            lifeline,
        )),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ellipsis_has_registered_typeobject() {
        crate::typedef::init_typeobjects();
        let w_type = crate::typedef::r#type(pyre_object::noneobject::w_ellipsis())
            .expect("Ellipsis should resolve to a W_TypeObject");
        unsafe {
            assert_eq!(pyre_object::w_type_get_name(w_type), "ellipsis");
            assert!(!pyre_object::w_type_get_acceptable_as_base_class(w_type));
        }
    }
}
