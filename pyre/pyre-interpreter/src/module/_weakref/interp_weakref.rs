//! pypy/module/_weakref/interp__weakref.py
//!
//! Structural port of `WeakrefLifeline`, `W_WeakrefBase`, `W_Weakref`,
//! `W_AbstractProxy`, `W_Proxy`, `W_CallableProxy`, plus the module-level
//! `getlifeline`, `descr__new__weakref`, `proxy`, `force`, etc.
//!
//! pyre has no GC so the underlying weak reference is kept as a strong
//! pointer; the class hierarchy and method names mirror PyPy so that
//! `__weakref__` and `_weakref.{ref,proxy,...}` go through the same
//! code paths as the original.

#![allow(non_camel_case_types, non_snake_case)]

use crate::{
    DictStorage, PyError, dict_storage_store, make_builtin_function,
    make_builtin_function_with_arity,
};
use pyre_object::*;

use std::sync::OnceLock;

thread_local! {
    static WEAKREF_LIFELINE_TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
    static WEAKREF_TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
    static PROXY_TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
    static CALLABLE_PROXY_TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
}

// ── Instance attribute names ──────────────────────────────────────────
//
// PyPy stores fields directly on the W_Root subclass. pyre stores them
// as instance attributes on the corresponding hasdict type so that the
// JIT and getattr/setattr paths see the same shape they would for any
// user instance. The names mirror PyPy field names exactly.

const ATTR_CACHED_WEAKREF: &str = "cached_weakref";
const ATTR_CACHED_PROXY: &str = "cached_proxy";
const ATTR_W_OBJ_WEAK: &str = "w_obj_weak";
const ATTR_W_CALLABLE: &str = "w_callable";
const ATTR_W_HASH: &str = "w_hash";

/// Read a per-instance slot from the underlying `INSTANCE_DICT` directly,
/// bypassing the public `getattr` path. The proxy fast-path in
/// `baseobjspace::getattr` would otherwise force the receiver and recurse
/// indefinitely while the proxy is reading its OWN `w_obj_weak`/etc.
fn read_attr(obj: PyObjectRef, name: &str) -> PyObjectRef {
    let w_dict = crate::baseobjspace::getdict(obj);
    if w_dict.is_null() {
        return PY_NULL;
    }
    let value = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) }.unwrap_or(PY_NULL);
    if value.is_null() || unsafe { pyre_object::is_none(value) } {
        return PY_NULL;
    }
    value
}

/// Mirror of `read_attr` for writes — direct dict access keeps lifeline /
/// proxy / weakref bookkeeping out of the fast-path's force loop.
fn write_attr(obj: PyObjectRef, name: &str, value: PyObjectRef) {
    let w_dict = crate::baseobjspace::getdict(obj);
    if !w_dict.is_null() {
        unsafe { pyre_object::w_dict_setitem_str(w_dict, name, value) };
    }
}

// ── Type registration ─────────────────────────────────────────────────

fn weakref_lifeline_type() -> PyObjectRef {
    WEAKREF_LIFELINE_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("WeakrefLifeline", |_| {});
            unsafe { pyre_object::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// pypy/module/_weakref/interp__weakref.py:270-280 W_Weakref.typedef
///
/// ```python
/// W_Weakref.typedef = TypeDef("weakref",
///     __doc__ = """A weak reference to an object 'obj'.  A 'callback' can be given,
/// which is called with 'obj' as an argument when it is about to be finalized.""",
///     __new__ = interp2app(descr__new__weakref),
///     __init__ = interp2app(W_Weakref.descr__init__weakref),
///     __eq__ = interp2app(W_Weakref.descr__eq__),
///     __ne__ = interp2app(W_Weakref.descr__ne__),
///     __hash__ = interp2app(W_Weakref.descr_hash),
///     __call__ = interp2app(W_Weakref.descr_call),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
/// )
/// ```
fn init_weakref_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", descr__new__weakref_typecall),
    );
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", descr__init__weakref),
    );
    dict_storage_store(
        ns,
        "__eq__",
        make_builtin_function_with_arity("__eq__", descr__eq__, 2),
    );
    dict_storage_store(
        ns,
        "__ne__",
        make_builtin_function_with_arity("__ne__", descr__ne__, 2),
    );
    dict_storage_store(
        ns,
        "__hash__",
        make_builtin_function_with_arity("__hash__", descr_hash, 1),
    );
    dict_storage_store(
        ns,
        "__call__",
        make_builtin_function_with_arity("__call__", descr_call, 1),
    );
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity("__repr__", descr__repr__, 1),
    );
}

pub fn weakref_type() -> PyObjectRef {
    WEAKREF_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("weakref", init_weakref_type);
            unsafe { pyre_object::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// pypy/module/_weakref/interp__weakref.py:405-410 W_Proxy.typedef
///
/// ```python
/// W_Proxy.typedef = TypeDef("weakproxy",
///     __new__ = interp2app(descr__new__proxy),
///     __hash__ = interp2app(W_Proxy.descr__hash__),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
///     **proxy_typedef_dict)
/// W_Proxy.typedef.acceptable_as_base_class = False
/// ```
fn init_proxy_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", descr__new__proxy),
    );
    dict_storage_store(
        ns,
        "__hash__",
        make_builtin_function_with_arity("__hash__", proxy_descr__hash__, 1),
    );
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity("__repr__", descr__repr__, 1),
    );
    // **proxy_typedef_dict — interp__weakref.py:409.
    register_proxy_typedef_dict(ns, /*include_comparisons=*/ true);
}

pub fn proxy_type() -> PyObjectRef {
    PROXY_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("weakproxy", init_proxy_type);
            unsafe {
                pyre_object::w_type_set_hasdict(tp, true);
                pyre_object::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp
        })
    })
}

/// pypy/module/_weakref/interp__weakref.py:412-418 W_CallableProxy.typedef
///
/// ```python
/// W_CallableProxy.typedef = TypeDef("weakcallableproxy",
///     __new__ = interp2app(descr__new__callableproxy),
///     __hash__ = interp2app(W_CallableProxy.descr__hash__),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
///     __call__ = interp2app(W_CallableProxy.descr__call__),
///     **callable_proxy_typedef_dict)
/// W_CallableProxy.typedef.acceptable_as_base_class = False
/// ```
fn init_callable_proxy_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", descr__new__callableproxy),
    );
    dict_storage_store(
        ns,
        "__hash__",
        make_builtin_function_with_arity("__hash__", proxy_descr__hash__, 1),
    );
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity("__repr__", descr__repr__, 1),
    );
    dict_storage_store(
        ns,
        "__call__",
        make_builtin_function("__call__", callable_proxy_descr__call__),
    );
    // **callable_proxy_typedef_dict — interp__weakref.py:417. Comparison
    // ops are excluded (interp__weakref.py:390-391 only writes them to
    // `proxy_typedef_dict`).
    register_proxy_typedef_dict(ns, /*include_comparisons=*/ false);
}

pub fn callable_proxy_type() -> PyObjectRef {
    CALLABLE_PROXY_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp =
                crate::typedef::make_builtin_type("weakcallableproxy", init_callable_proxy_type);
            unsafe {
                pyre_object::w_type_set_hasdict(tp, true);
                pyre_object::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp
        })
    })
}

// ── WeakrefLifeline ───────────────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py:19-153
//
// class WeakrefLifeline(W_Root):
//     cached_weakref  = None
//     cached_proxy    = None
//     other_refs_weak = None
//     has_callbacks   = False
//
// pyre stores cached_weakref/cached_proxy as instance attributes on a
// W_InstanceObject of type WeakrefLifeline so the field access goes
// through the same setattr/getattr path as any user instance.

/// pypy/module/_weakref/interp__weakref.py:27-28 WeakrefLifeline.__init__
///
/// ```python
/// def __init__(self, space):
///     self.space = space
/// ```
pub fn weakref_lifeline_new() -> PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    let obj = w_instance_new(weakref_lifeline_type());
    write_attr(obj, ATTR_CACHED_WEAKREF, pyre_object::w_none());
    write_attr(obj, ATTR_CACHED_PROXY, pyre_object::w_none());
    obj
}

/// pypy/module/_weakref/interp__weakref.py:60-77 get_or_make_weakref
///
/// ```python
/// @jit.dont_look_inside
/// def get_or_make_weakref(self, w_subtype, w_obj):
///     space = self.space
///     w_weakreftype = space.gettypeobject(W_Weakref.typedef)
///     #
///     if space.is_w(w_weakreftype, w_subtype):
///         if self.cached_weakref is not None:
///             w_cached = self.cached_weakref()
///             if w_cached is not None:
///                 return w_cached
///         w_ref = W_Weakref(space, w_obj, None)
///         self.cached_weakref = weakref.ref(w_ref)
///     else:
///         # subclass: cannot cache
///         w_ref = space.allocate_instance(W_Weakref, w_subtype)
///         W_Weakref.__init__(w_ref, space, w_obj, None)
///         self.append_wref_to(w_ref)
///     return w_ref
/// ```
pub fn get_or_make_weakref(
    self_lifeline: PyObjectRef,
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
) -> PyObjectRef {
    let w_weakreftype = weakref_type();
    if w_subtype.is_null() || std::ptr::eq(w_weakreftype, w_subtype) {
        let cached = read_attr(self_lifeline, ATTR_CACHED_WEAKREF);
        if !cached.is_null() {
            return cached;
        }
        let w_ref = W_Weakref_new(w_subtype, w_obj, PY_NULL);
        write_attr(self_lifeline, ATTR_CACHED_WEAKREF, w_ref);
        w_ref
    } else {
        // subclass: cannot cache
        W_Weakref_new(w_subtype, w_obj, PY_NULL)
    }
}

/// pypy/module/_weakref/interp__weakref.py:79-91 get_or_make_proxy
///
/// ```python
/// @jit.dont_look_inside
/// def get_or_make_proxy(self, w_obj):
///     space = self.space
///     if self.cached_proxy is not None:
///         w_cached = self.cached_proxy()
///         if w_cached is not None:
///             return w_cached
///     if space.is_true(space.callable(w_obj)):
///         w_proxy = W_CallableProxy(space, w_obj, None)
///     else:
///         w_proxy = W_Proxy(space, w_obj, None)
///     self.cached_proxy = weakref.ref(w_proxy)
///     return w_proxy
/// ```
pub fn get_or_make_proxy(self_lifeline: PyObjectRef, w_obj: PyObjectRef) -> PyObjectRef {
    let cached = read_attr(self_lifeline, ATTR_CACHED_PROXY);
    if !cached.is_null() {
        return cached;
    }
    let w_proxy = if is_callable(w_obj) {
        W_CallableProxy_new(w_obj, PY_NULL)
    } else {
        W_Proxy_new(w_obj, PY_NULL)
    };
    write_attr(self_lifeline, ATTR_CACHED_PROXY, w_proxy);
    w_proxy
}

/// pypy/module/_weakref/interp__weakref.py:93-104 get_any_weakref
///
/// ```python
/// def get_any_weakref(self, space):
///     if self.cached_weakref is not None:
///         w_ref = self.cached_weakref()
///         if w_ref is not None:
///             return w_ref
///     if self.other_refs_weak is not None:
///         w_weakreftype = space.gettypeobject(W_Weakref.typedef)
///         for wref in self.other_refs_weak.items():
///             w_ref = wref()
///             if (w_ref is not None and space.isinstance_w(w_ref, w_weakreftype)):
///                 return w_ref
///     return space.w_None
/// ```
pub fn get_any_weakref(self_lifeline: PyObjectRef) -> PyObjectRef {
    let cached = read_attr(self_lifeline, ATTR_CACHED_WEAKREF);
    if !cached.is_null() {
        return cached;
    }
    pyre_object::w_none()
}

/// pypy/module/_weakref/interp__weakref.py:111-118 make_weakref_with_callback
///
/// ```python
/// @jit.dont_look_inside
/// def make_weakref_with_callback(self, w_subtype, w_obj, w_callable):
///     space = self.space
///     w_ref = space.allocate_instance(W_Weakref, w_subtype)
///     W_Weakref.__init__(w_ref, space, w_obj, w_callable)
///     self.append_wref_to(w_ref)
///     self.enable_callbacks()
///     return w_ref
/// ```
pub fn make_weakref_with_callback(
    _self_lifeline: PyObjectRef,
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    W_Weakref_new(w_subtype, w_obj, w_callable)
}

/// pypy/module/_weakref/interp__weakref.py:120-129 make_proxy_with_callback
///
/// ```python
/// @jit.dont_look_inside
/// def make_proxy_with_callback(self, w_obj, w_callable):
///     space = self.space
///     if space.is_true(space.callable(w_obj)):
///         w_proxy = W_CallableProxy(space, w_obj, w_callable)
///     else:
///         w_proxy = W_Proxy(space, w_obj, w_callable)
///     self.append_wref_to(w_proxy)
///     self.enable_callbacks()
///     return w_proxy
/// ```
pub fn make_proxy_with_callback(
    _self_lifeline: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    if is_callable(w_obj) {
        W_CallableProxy_new(w_obj, w_callable)
    } else {
        W_Proxy_new(w_obj, w_callable)
    }
}

// ── W_WeakrefBase / W_Weakref ─────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py:158-205
//
// class W_WeakrefBase(W_Root):
//     def __init__(self, space, w_obj, w_callable):
//         assert w_callable is not space.w_None
//         self.space = space
//         self.w_obj_weak = weakref.ref(w_obj)
//         self.w_callable = w_callable
//
// class W_Weakref(W_WeakrefBase):
//     def __init__(self, space, w_obj, w_callable):
//         W_WeakrefBase.__init__(self, space, w_obj, w_callable)
//         self.w_hash = None

#[allow(non_snake_case)]
pub fn W_Weakref_new(
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    // typedef.py:519 generic_new_descr → space.allocate_instance(W_Type, w_subtype)
    let actual_type = if w_subtype.is_null() {
        weakref_type()
    } else {
        w_subtype
    };
    let obj = w_instance_new(actual_type);
    // W_WeakrefBase.__init__
    write_attr(obj, ATTR_W_OBJ_WEAK, w_obj);
    if !w_callable.is_null() && !unsafe { pyre_object::is_none(w_callable) } {
        write_attr(obj, ATTR_W_CALLABLE, w_callable);
    } else {
        write_attr(obj, ATTR_W_CALLABLE, pyre_object::w_none());
    }
    // W_Weakref.__init__: self.w_hash = None
    write_attr(obj, ATTR_W_HASH, pyre_object::w_none());
    obj
}

#[allow(non_snake_case)]
pub fn W_Proxy_new(w_obj: PyObjectRef, w_callable: PyObjectRef) -> PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    let obj = w_instance_new(proxy_type());
    write_attr(obj, ATTR_W_OBJ_WEAK, w_obj);
    if !w_callable.is_null() && !unsafe { pyre_object::is_none(w_callable) } {
        write_attr(obj, ATTR_W_CALLABLE, w_callable);
    } else {
        write_attr(obj, ATTR_W_CALLABLE, pyre_object::w_none());
    }
    obj
}

#[allow(non_snake_case)]
pub fn W_CallableProxy_new(w_obj: PyObjectRef, w_callable: PyObjectRef) -> PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    let obj = w_instance_new(callable_proxy_type());
    write_attr(obj, ATTR_W_OBJ_WEAK, w_obj);
    if !w_callable.is_null() && !unsafe { pyre_object::is_none(w_callable) } {
        write_attr(obj, ATTR_W_CALLABLE, w_callable);
    } else {
        write_attr(obj, ATTR_W_CALLABLE, pyre_object::w_none());
    }
    obj
}

/// pypy/module/_weakref/interp__weakref.py:168-171 dereference
///
/// ```python
/// @jit.dont_look_inside
/// def dereference(self):
///     w_obj = self.w_obj_weak()
///     return w_obj
/// ```
pub fn dereference(w_ref: PyObjectRef) -> PyObjectRef {
    read_attr(w_ref, ATTR_W_OBJ_WEAK)
}

/// pypy/module/_weakref/interp__weakref.py:179-190 W_WeakrefBase.descr__repr__
///
/// ```python
/// def descr__repr__(self, space):
///     w_obj = self.dereference()
///     if w_obj is None:
///         state = '; dead'
///     else:
///         typename = space.type(w_obj).getname(space)
///         objname = w_obj.getname(space)
///         if objname and objname != '?':
///             state = "; to '%s' (%s)" % (typename, objname)
///         else:
///             state = "; to '%s'" % (typename,)
///     return self.getrepr(space, self.typedef.name, state)
/// ```
pub fn descr__repr__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_obj = dereference(w_self);
    let type_name = unsafe {
        match crate::typedef::r#type(w_self) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "weakref".to_string(),
        }
    };
    let state = if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        "; dead".to_string()
    } else {
        let objtype_name = unsafe {
            match crate::typedef::r#type(w_obj) {
                Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                None => "object".to_string(),
            }
        };
        format!("; to '{}'", objtype_name)
    };
    let addr = w_self as usize;
    Ok(pyre_object::w_str_new(&format!(
        "<{} at 0x{:x}{}>",
        type_name, addr, state
    )))
}

/// pypy/module/_weakref/interp__weakref.py:207-214 descr__init__weakref
///
/// ```python
/// def descr__init__weakref(self, space, w_obj, w_callable=None,
///                          __args__=None):
///     if __args__.arguments_w:
///         raise oefmt(space.w_TypeError,
///                     "__init__ expected at most 2 arguments")
///     if __args__.keywords:
///         raise oefmt(space.w_TypeError,
///                     "ref() does not take keyword arguments")
/// ```
pub fn descr__init__weakref(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // args[0] = self, args[1] = w_obj, args[2] = w_callable (optional)
    if args.len() > 3 {
        return Err(PyError::type_error(
            "__init__ expected at most 2 arguments".to_string(),
        ));
    }
    Ok(pyre_object::w_none())
}

/// pypy/module/_weakref/interp__weakref.py:216-223 descr_hash
///
/// ```python
/// def descr_hash(self):
///     if self.w_hash is not None:
///         return self.w_hash
///     w_obj = self.dereference()
///     if w_obj is None:
///         raise oefmt(self.space.w_TypeError, "weak object has gone away")
///     self.w_hash = self.space.hash(w_obj)
///     return self.w_hash
/// ```
pub fn descr_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let cached = read_attr(w_self, ATTR_W_HASH);
    if !cached.is_null() {
        return Ok(cached);
    }
    let w_obj = dereference(w_self);
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Err(PyError::type_error("weak object has gone away".to_string()));
    }
    // pyre's hash: identity-based for non-int/non-str. Mirrors space.hash(w_obj).
    let h = pyre_object::w_int_new(w_obj as i64);
    write_attr(w_self, ATTR_W_HASH, h);
    Ok(h)
}

/// pypy/module/_weakref/interp__weakref.py:225-229 descr_call
///
/// ```python
/// def descr_call(self):
///     w_obj = self.dereference()
///     if w_obj is None:
///         return self.space.w_None
///     return w_obj
/// ```
pub fn descr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_obj = dereference(w_self);
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Ok(pyre_object::w_none());
    }
    Ok(w_obj)
}

/// pypy/module/_weakref/interp__weakref.py:231-244 W_Weakref.compare
///
/// ```python
/// def compare(self, space, w_ref2, invert):
///     if not isinstance(w_ref2, W_Weakref):
///         return space.w_NotImplemented
///     ref1 = self
///     ref2 = w_ref2
///     w_obj1 = ref1.dereference()
///     w_obj2 = ref2.dereference()
///     if w_obj1 is None or w_obj2 is None:
///         w_res = space.is_(ref1, ref2)
///     else:
///         w_res = space.eq(w_obj1, w_obj2)
///     if invert:
///         w_res = space.not_(w_res)
///     return w_res
/// ```
fn compare(w_self: PyObjectRef, w_ref2: PyObjectRef, invert: bool) -> Result<PyObjectRef, PyError> {
    if !is_w_weakref(w_ref2) {
        return Ok(pyre_object::w_not_implemented());
    }
    let w_obj1 = dereference(w_self);
    let w_obj2 = dereference(w_ref2);
    let w_res = if w_obj1.is_null()
        || unsafe { pyre_object::is_none(w_obj1) }
        || w_obj2.is_null()
        || unsafe { pyre_object::is_none(w_obj2) }
    {
        crate::baseobjspace::is_(w_self, w_ref2)
    } else {
        crate::baseobjspace::compare(w_obj1, w_obj2, crate::baseobjspace::CompareOp::Eq)?
    };
    if invert {
        Ok(crate::baseobjspace::not_(w_res))
    } else {
        Ok(w_res)
    }
}

/// pypy/module/_weakref/interp__weakref.py:246-247 descr__eq__
pub fn descr__eq__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    compare(args[0], args[1], false)
}

/// pypy/module/_weakref/interp__weakref.py:249-250 descr__ne__
pub fn descr__ne__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    compare(args[0], args[1], true)
}

fn is_w_weakref(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    match crate::typedef::r#type(obj) {
        Some(tp) => std::ptr::eq(tp, weakref_type()),
        None => false,
    }
}

fn is_callable(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe {
        if crate::is_function(obj)
            || pyre_object::is_method(obj)
            || pyre_object::is_type(obj)
            || pyre_object::is_staticmethod(obj)
            || pyre_object::is_classmethod(obj)
        {
            return true;
        }
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            return crate::baseobjspace::lookup_in_type(w_type, "__call__").is_some();
        }
    }
    false
}

// ── module-level helpers ──────────────────────────────────────────────

/// pypy/module/_weakref/interp__weakref.py:252-257 getlifeline
///
/// ```python
/// def getlifeline(space, w_obj):
///     lifeline = w_obj.getweakref()
///     if lifeline is None:
///         lifeline = WeakrefLifeline(space)
///         w_obj.setweakref(space, lifeline)
///     return lifeline
/// ```
pub fn getlifeline(w_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if let Some(lifeline) = crate::baseobjspace::getweakref(w_obj) {
        return Ok(lifeline);
    }
    let lifeline = weakref_lifeline_new();
    crate::baseobjspace::setweakref(w_obj, lifeline)?;
    Ok(lifeline)
}

/// pypy/module/_weakref/interp__weakref.py:260-268 descr__new__weakref
///
/// ```python
/// def descr__new__weakref(space, w_subtype, w_obj, w_callable=None,
///                         __args__=None):
///     if __args__.arguments_w:
///         raise oefmt(space.w_TypeError, "__new__ expected at most 2 arguments")
///     lifeline = getlifeline(space, w_obj)
///     if space.is_none(w_callable):
///         return lifeline.get_or_make_weakref(w_subtype, w_obj)
///     else:
///         return lifeline.make_weakref_with_callback(w_subtype, w_obj, w_callable)
/// ```
pub fn descr__new__weakref(
    w_subtype: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "ref() takes at least 1 argument".to_string(),
        ));
    }
    if args.len() > 2 {
        return Err(PyError::type_error(
            "__new__ expected at most 2 arguments".to_string(),
        ));
    }
    let w_obj = args[0];
    let w_callable = if args.len() >= 2 {
        args[1]
    } else {
        pyre_object::w_none()
    };
    let lifeline = getlifeline(w_obj)?;
    if w_callable.is_null() || unsafe { pyre_object::is_none(w_callable) } {
        Ok(get_or_make_weakref(lifeline, w_subtype, w_obj))
    } else {
        Ok(make_weakref_with_callback(
            lifeline, w_subtype, w_obj, w_callable,
        ))
    }
}

/// Type-call entry: pyre passes `(cls, *args)` so we strip cls and route
/// through descr__new__weakref with the user-supplied subtype.
fn descr__new__weakref_typecall(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "ref() takes at least 1 argument".to_string(),
        ));
    }
    descr__new__weakref(args[0], &args[1..])
}

/// pypy/module/_weakref/interp__weakref.py:283-295 getweakrefcount
///
/// ```python
/// def getweakrefcount(space, w_obj):
///     """Return the number of weak references to 'obj'."""
///     lifeline = w_obj.getweakref()
///     if lifeline is None:
///         return space.newint(0)
///     else:
///         result = lifeline.traverse(_weakref_count, 0)
///         return space.newint(result)
/// ```
pub fn getweakrefcount(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_int_new(0));
    }
    let w_obj = args[0];
    let count = match crate::baseobjspace::getweakref(w_obj) {
        None => 0,
        Some(lifeline) => {
            if !read_attr(lifeline, ATTR_CACHED_WEAKREF).is_null() {
                1
            } else {
                0
            }
        }
    };
    Ok(pyre_object::w_int_new(count))
}

/// pypy/module/_weakref/interp__weakref.py:297-309 getweakrefs
///
/// ```python
/// def getweakrefs(space, w_obj):
///     """Return a list of all weak reference objects that point to 'obj'."""
///     result = []
///     lifeline = w_obj.getweakref()
///     if lifeline is not None:
///         lifeline.traverse(_get_weakrefs, result)
///     return space.newlist(result)
/// ```
pub fn getweakrefs(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_list_new(vec![]));
    }
    let w_obj = args[0];
    let mut result = Vec::new();
    if let Some(lifeline) = crate::baseobjspace::getweakref(w_obj) {
        let cached = read_attr(lifeline, ATTR_CACHED_WEAKREF);
        if !cached.is_null() {
            result.push(cached);
        }
    }
    Ok(pyre_object::w_list_new(result))
}

// ── Proxy ─────────────────────────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py:311-417

/// pypy/module/_weakref/interp__weakref.py:318-319 W_Proxy.descr__hash__
///
/// ```python
/// def descr__hash__(self, space):
///     raise oefmt(space.w_TypeError, "unhashable type")
/// ```
pub fn proxy_descr__hash__(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Err(PyError::type_error("unhashable type".to_string()))
}

/// pypy/module/_weakref/interp__weakref.py:322-324 W_CallableProxy.descr__call__
///
/// ```python
/// def descr__call__(self, space, __args__):
///     w_obj = force(space, self)
///     return space.call_args(w_obj, __args__)
/// ```
pub fn callable_proxy_descr__call__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_obj = force(w_self)?;
    crate::call::call_function_impl_result(w_obj, &args[1..])
}

/// pypy/module/_weakref/interp__weakref.py:329-337 proxy
///
/// ```python
/// def proxy(space, w_obj, w_callable=None):
///     """Create a proxy object that weakly references 'obj'.
/// 'callback', if given, is called with the proxy as an argument when 'obj'
/// is about to be finalized."""
///     lifeline = getlifeline(space, w_obj)
///     if space.is_none(w_callable):
///         return lifeline.get_or_make_proxy(w_obj)
///     else:
///         return lifeline.make_proxy_with_callback(w_obj, w_callable)
/// ```
pub fn proxy(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "proxy() takes at least 1 argument".to_string(),
        ));
    }
    let w_obj = args[0];
    let w_callable = if args.len() >= 2 {
        args[1]
    } else {
        pyre_object::w_none()
    };
    let lifeline = getlifeline(w_obj)?;
    if w_callable.is_null() || unsafe { pyre_object::is_none(w_callable) } {
        Ok(get_or_make_proxy(lifeline, w_obj))
    } else {
        Ok(make_proxy_with_callback(lifeline, w_obj, w_callable))
    }
}

/// pypy/module/_weakref/interp__weakref.py:339-340 descr__new__proxy
///
/// ```python
/// def descr__new__proxy(space, w_subtype, w_obj, w_callable=None):
///     raise oefmt(space.w_TypeError, "cannot create 'weakproxy' instances")
/// ```
pub fn descr__new__proxy(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Err(PyError::type_error(
        "cannot create 'weakproxy' instances".to_string(),
    ))
}

/// pypy/module/_weakref/interp__weakref.py:342-344 descr__new__callableproxy
///
/// ```python
/// def descr__new__callableproxy(space, w_subtype, w_obj, w_callable=None):
///     raise oefmt(space.w_TypeError,
///                 "cannot create 'weakcallableproxy' instances")
/// ```
pub fn descr__new__callableproxy(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Err(PyError::type_error(
        "cannot create 'weakcallableproxy' instances".to_string(),
    ))
}

/// pypy/module/_weakref/interp__weakref.py:347-354 force
///
/// ```python
/// def force(space, proxy):
///     if not isinstance(proxy, W_AbstractProxy):
///         return proxy
///     w_obj = proxy.dereference()
///     if w_obj is None:
///         raise oefmt(space.w_ReferenceError,
///                     "weakly referenced object no longer exists")
///     return w_obj
/// ```
pub fn force(proxy: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if !is_w_abstract_proxy(proxy) {
        return Ok(proxy);
    }
    let w_obj = dereference(proxy);
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Err(PyError::reference_error(
            "weakly referenced object no longer exists".to_string(),
        ));
    }
    Ok(w_obj)
}

pub fn is_w_abstract_proxy(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    match crate::typedef::r#type(obj) {
        Some(tp) => std::ptr::eq(tp, proxy_type()) || std::ptr::eq(tp, callable_proxy_type()),
        None => false,
    }
}

// ── proxy_typedef_dict / callable_proxy_typedef_dict ──────────────────
//
// pypy/module/_weakref/interp__weakref.py:356-402
//
// ```python
// proxy_typedef_dict = {}
// callable_proxy_typedef_dict = {}
// special_ops = {'repr': True, 'hash': True}
//
// for opname, _, arity, special_methods in ObjSpace.MethodTable:
//     if opname in special_ops or not special_methods:
//         continue
//     ...
//     for i in range(forcing_count):
//         code += "    w_obj%s = force(space, w_obj%s)\n" % (i, i)
//     code += "    return space.%s(%s)" % (opname, nonspaceargs)
//     exec py.code.Source(code).compile()
//     ...
// ```
//
// PyPy generates one wrapper per `(opname, special_methods)` row by
// `exec`'ing a string template. pyre cannot synthesize functions at
// runtime — `BuiltinCodeFn` is a fixed `fn(&[PyObjectRef])` pointer —
// so we mirror the loop statically below, one Rust function per
// generated row. Pyre's space op surface is the Python-3 subset of
// PyPy's MethodTable, so the Py-2-only rows (`getslice`, `coerce`,
// `__hex__`, …) are skipped exactly as Python 3 would.

/// Helper used by every proxy wrapper to call a dunder via getattr +
/// `call_function_impl_result`. Used for ops where pyre has no direct
/// `crate::baseobjspace` entry (e.g. `__pos__`, `__abs__`, `__int__`).
fn forward_to_dunder(
    w_obj: PyObjectRef,
    methname: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    let method = crate::baseobjspace::getattr(w_obj, methname)?;
    crate::call::call_function_impl_result(method, args)
}

// Forwarding wrappers — `force(args[i])` then dispatch to the named
// `crate::baseobjspace` op. The macro arms cover the common arities and
// keep the per-op definitions a single line each, mirroring PyPy's
// generated source.

macro_rules! proxy_unary {
    ($name:ident, $space_op:path) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            let w_obj0 = force(args[0])?;
            $space_op(w_obj0)
        }
    };
}

macro_rules! proxy_binary {
    ($name:ident, $space_op:path) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            let w_obj0 = force(args[0])?;
            let w_obj1 = force(args[1])?;
            $space_op(w_obj0, w_obj1)
        }
    };
}

macro_rules! proxy_binary_reflected {
    ($name:ident, $space_op:path) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            // interp__weakref.py:382-385 — reflected wrappers swap the
            // operands before calling the space op:
            //   `code = code.replace("(w_obj0, w_obj1)", "(w_obj1, w_obj0)")`
            let w_obj0 = force(args[0])?;
            let w_obj1 = force(args[1])?;
            $space_op(w_obj1, w_obj0)
        }
    };
}

// Forward + reflected binary ops — interp__weakref.py:376-389.
proxy_binary!(proxy_add, crate::baseobjspace::add);
proxy_binary_reflected!(proxy_radd, crate::baseobjspace::add);
proxy_binary!(proxy_sub, crate::baseobjspace::sub);
proxy_binary_reflected!(proxy_rsub, crate::baseobjspace::sub);
proxy_binary!(proxy_mul, crate::baseobjspace::mul);
proxy_binary_reflected!(proxy_rmul, crate::baseobjspace::mul);
proxy_binary!(proxy_truediv, crate::baseobjspace::truediv);
proxy_binary_reflected!(proxy_rtruediv, crate::baseobjspace::truediv);
proxy_binary!(proxy_floordiv, crate::baseobjspace::floordiv);
proxy_binary_reflected!(proxy_rfloordiv, crate::baseobjspace::floordiv);
proxy_binary!(proxy_mod, crate::baseobjspace::mod_);
proxy_binary_reflected!(proxy_rmod, crate::baseobjspace::mod_);
// pow / rpow — interp__weakref.py:363 generates a 3-arg wrapper because
// `('pow', '**', 3, ['__pow__', '__rpow__'])` has arity=3 but
// `forcing_count = len(special_methods) = 2`, so the optional modulo
// operand passes through unforced. The wrapper hands the result to
// `space.pow(w_obj0, w_obj1, w_obj2)` (descroperation.py:399), so
// NotImplemented from forward `__pow__` properly falls through to the
// reflected `__rpow__` and vice versa.
pub fn proxy_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    let has_modulo =
        args.len() >= 3 && !args[2].is_null() && !unsafe { pyre_object::is_none(args[2]) };
    if has_modulo {
        crate::baseobjspace::pow3(w_obj0, w_obj1, args[2])
    } else {
        crate::baseobjspace::pow(w_obj0, w_obj1)
    }
}

pub fn proxy_rpow(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // interp__weakref.py:382-385 — reflected wrapper swaps the first
    // two operands; the modulo argument keeps its slot.
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    let has_modulo =
        args.len() >= 3 && !args[2].is_null() && !unsafe { pyre_object::is_none(args[2]) };
    if has_modulo {
        crate::baseobjspace::pow3(w_obj1, w_obj0, args[2])
    } else {
        crate::baseobjspace::pow(w_obj1, w_obj0)
    }
}
proxy_binary!(proxy_lshift, crate::baseobjspace::lshift);
proxy_binary_reflected!(proxy_rlshift, crate::baseobjspace::lshift);
proxy_binary!(proxy_rshift, crate::baseobjspace::rshift);
proxy_binary_reflected!(proxy_rrshift, crate::baseobjspace::rshift);
proxy_binary!(proxy_and, crate::baseobjspace::and_);
proxy_binary_reflected!(proxy_rand, crate::baseobjspace::and_);
proxy_binary!(proxy_or, crate::baseobjspace::or_);
proxy_binary_reflected!(proxy_ror, crate::baseobjspace::or_);
proxy_binary!(proxy_xor, crate::baseobjspace::xor);
proxy_binary_reflected!(proxy_rxor, crate::baseobjspace::xor);

// Inplace ops — interp__weakref.py:367-369. PyPy forces every operand
// (`forcing_count = arity`) and dispatches to `space.inplace_X`. pyre
// has no separate `inplace_` space ops; the regular forward op is the
// closest equivalent and matches the runtime fall-back PyPy uses for
// any type without an in-place specialization.
proxy_binary!(proxy_iadd, crate::baseobjspace::add);
proxy_binary!(proxy_isub, crate::baseobjspace::sub);
proxy_binary!(proxy_imul, crate::baseobjspace::mul);
proxy_binary!(proxy_itruediv, crate::baseobjspace::truediv);
proxy_binary!(proxy_ifloordiv, crate::baseobjspace::floordiv);
proxy_binary!(proxy_imod, crate::baseobjspace::mod_);
proxy_binary!(proxy_ipow, crate::baseobjspace::pow);
proxy_binary!(proxy_ilshift, crate::baseobjspace::lshift);
proxy_binary!(proxy_irshift, crate::baseobjspace::rshift);
proxy_binary!(proxy_iand, crate::baseobjspace::and_);
proxy_binary!(proxy_ior, crate::baseobjspace::or_);
proxy_binary!(proxy_ixor, crate::baseobjspace::xor);

// 1-arg unary ops with a direct pyre space op.
proxy_unary!(proxy_len, crate::baseobjspace::len);
proxy_unary!(proxy_neg, crate::baseobjspace::neg);
proxy_unary!(proxy_invert, crate::baseobjspace::invert);
proxy_unary!(proxy_iter, crate::baseobjspace::iter);
proxy_unary!(proxy_next, crate::baseobjspace::next);

// 1-arg unary ops without a direct space op — fall through to the
// equivalent dunder so the proxy still delegates to the referent.
pub fn proxy_pos(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__pos__", &[])
}

pub fn proxy_abs(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__abs__", &[])
}

pub fn proxy_int(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__int__", &[])
}

pub fn proxy_float(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__float__", &[])
}

pub fn proxy_index(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__index__", &[])
}

pub fn proxy_trunc(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__trunc__", &[])
}

pub fn proxy_str(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    Ok(pyre_object::w_str_new(&unsafe { crate::py_str(w_obj0) }))
}

pub fn proxy_bool(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // interp__weakref.py:360 MethodTable row `('nonzero', ..., ['__nonzero__'])`.
    // pyre uses Python-3 truth dispatch (`is_true`) so the wrapper bridges
    // to that and returns the boxed bool.
    let w_obj0 = force(args[0])?;
    Ok(pyre_object::w_bool_from(crate::baseobjspace::is_true(
        w_obj0,
    )))
}

// 2-/3-arg attribute ops with name-string conversion.
pub fn proxy_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let name = unsafe { pyre_object::w_str_get_value(args[1]) };
    crate::baseobjspace::getattr(w_obj0, name)
}

pub fn proxy_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let name = unsafe { pyre_object::w_str_get_value(args[1]) };
    crate::baseobjspace::setattr(w_obj0, name, args[2])
}

pub fn proxy_delattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let name = unsafe { pyre_object::w_str_get_value(args[1]) };
    crate::baseobjspace::delattr(w_obj0, name)
}

// Item ops — interp__weakref.py:365 single special method, so
// `forcing_count = len(special_methods) = 1`. The index/value operands
// pass through unchanged so weakproxy/dead-proxy keys and values reach
// the referent's __getitem__/__setitem__/__delitem__ unmodified.
pub fn proxy_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    crate::baseobjspace::getitem(w_obj0, args[1])
}

pub fn proxy_setitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    crate::baseobjspace::setitem(w_obj0, args[1], args[2])
}

pub fn proxy_delitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    crate::baseobjspace::delitem(w_obj0, args[1])?;
    Ok(pyre_object::w_none())
}

// __format__(self, format_spec) — single special method, force self only.
pub fn proxy_format(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__format__", &[args[1]])
}

// __contains__(self, needle) — single special method, force self only.
// pyre's contains returns Result<bool>.
pub fn proxy_contains(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let result = crate::baseobjspace::contains(w_obj0, args[1])?;
    Ok(pyre_object::w_bool_from(result))
}

// Comparison ops — interp__weakref.py:390-391:
//   elif opname in ["lt", "le", "gt", "ge", "eq", "ne"]:
//       proxy_typedef_dict[special_methods[0]] = interp2app(func)
// Each opname's first special method (e.g. `__lt__` for `lt`) is the
// only one registered, and the wrapper dispatches via the matching
// `CompareOp` variant.
pub fn proxy_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Lt)
}

pub fn proxy_le(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Le)
}

pub fn proxy_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Gt)
}

pub fn proxy_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Ge)
}

pub fn proxy_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Eq)
}

pub fn proxy_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    crate::baseobjspace::compare(w_obj0, w_obj1, crate::baseobjspace::CompareOp::Ne)
}

// pypy/interpreter/baseobjspace.py:2127-2128 isinstance/issubtype rows
// — single special method so `forcing_count = 1`. The wrapper hands
// off to `space.isinstance(forced_self, w_obj1)` /
// `space.issubtype(forced_self, w_obj1)`, which is the descroperation
// dispatch that consults `__instancecheck__` / `__subclasscheck__` on
// `type(forced_self)` and falls back to the recursive type-MRO walk.
pub fn proxy_instancecheck(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let result = crate::baseobjspace::isinstance(args[1], w_obj0)?;
    Ok(pyre_object::w_bool_from(result))
}

pub fn proxy_subclasscheck(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let result = crate::baseobjspace::issubclass(args[1], w_obj0)?;
    Ok(pyre_object::w_bool_from(result))
}

// pypy/interpreter/baseobjspace.py:2159 divmod row — two special
// methods (`__divmod__`, `__rdivmod__`) so `forcing_count = 2`. The
// wrapper calls `space.divmod`, which is the descroperation dispatch
// (numeric fast path then forward + reverse `__divmod__`/`__rdivmod__`
// with NotImplemented fallback).
pub fn proxy_divmod(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    crate::baseobjspace::divmod(w_obj0, w_obj1)
}

pub fn proxy_rdivmod(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // Reflected: swap operands per interp__weakref.py:382-385.
    let w_obj0 = force(args[0])?;
    let w_obj1 = force(args[1])?;
    crate::baseobjspace::divmod(w_obj1, w_obj0)
}

// Descriptor protocol — `get`/`set`/`delete` rows in MethodTable each
// have a single special method, so `forcing_count = 1`. Only `self` is
// dereferenced; the descriptor instance and value/type operands pass
// through unchanged so identity-sensitive bindings remain intact.
pub fn proxy_get(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__get__", &[args[1], args[2]])
}

pub fn proxy_set(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__set__", &[args[1], args[2]])
}

pub fn proxy_delete(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj0 = force(args[0])?;
    forward_to_dunder(w_obj0, "__delete__", &[args[1]])
}

/// pypy/module/_weakref/interp__weakref.py:356-395 register the entries
/// of `proxy_typedef_dict` (and, with `include_comparisons=true`, the
/// six comparison ops registered only on `proxy_typedef_dict`).
///
/// Called from `init_proxy_type` (`include_comparisons=true`) and
/// `init_callable_proxy_type` (`include_comparisons=false`) so the two
/// typedefs end up with the same set of methods PyPy generates.
fn register_proxy_typedef_dict(ns: &mut DictStorage, include_comparisons: bool) {
    // Forward + reflected binary arithmetic — interp__weakref.py:376-389.
    dict_storage_store(
        ns,
        "__add__",
        make_builtin_function_with_arity("__add__", proxy_add, 2),
    );
    dict_storage_store(
        ns,
        "__radd__",
        make_builtin_function_with_arity("__radd__", proxy_radd, 2),
    );
    dict_storage_store(
        ns,
        "__sub__",
        make_builtin_function_with_arity("__sub__", proxy_sub, 2),
    );
    dict_storage_store(
        ns,
        "__rsub__",
        make_builtin_function_with_arity("__rsub__", proxy_rsub, 2),
    );
    dict_storage_store(
        ns,
        "__mul__",
        make_builtin_function_with_arity("__mul__", proxy_mul, 2),
    );
    dict_storage_store(
        ns,
        "__rmul__",
        make_builtin_function_with_arity("__rmul__", proxy_rmul, 2),
    );
    dict_storage_store(
        ns,
        "__truediv__",
        make_builtin_function_with_arity("__truediv__", proxy_truediv, 2),
    );
    dict_storage_store(
        ns,
        "__rtruediv__",
        make_builtin_function_with_arity("__rtruediv__", proxy_rtruediv, 2),
    );
    dict_storage_store(
        ns,
        "__floordiv__",
        make_builtin_function_with_arity("__floordiv__", proxy_floordiv, 2),
    );
    dict_storage_store(
        ns,
        "__rfloordiv__",
        make_builtin_function_with_arity("__rfloordiv__", proxy_rfloordiv, 2),
    );
    dict_storage_store(
        ns,
        "__mod__",
        make_builtin_function_with_arity("__mod__", proxy_mod, 2),
    );
    dict_storage_store(
        ns,
        "__rmod__",
        make_builtin_function_with_arity("__rmod__", proxy_rmod, 2),
    );
    dict_storage_store(ns, "__pow__", make_builtin_function("__pow__", proxy_pow));
    dict_storage_store(
        ns,
        "__rpow__",
        make_builtin_function("__rpow__", proxy_rpow),
    );
    dict_storage_store(
        ns,
        "__lshift__",
        make_builtin_function_with_arity("__lshift__", proxy_lshift, 2),
    );
    dict_storage_store(
        ns,
        "__rlshift__",
        make_builtin_function_with_arity("__rlshift__", proxy_rlshift, 2),
    );
    dict_storage_store(
        ns,
        "__rshift__",
        make_builtin_function_with_arity("__rshift__", proxy_rshift, 2),
    );
    dict_storage_store(
        ns,
        "__rrshift__",
        make_builtin_function_with_arity("__rrshift__", proxy_rrshift, 2),
    );
    dict_storage_store(
        ns,
        "__and__",
        make_builtin_function_with_arity("__and__", proxy_and, 2),
    );
    dict_storage_store(
        ns,
        "__rand__",
        make_builtin_function_with_arity("__rand__", proxy_rand, 2),
    );
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function_with_arity("__or__", proxy_or, 2),
    );
    dict_storage_store(
        ns,
        "__ror__",
        make_builtin_function_with_arity("__ror__", proxy_ror, 2),
    );
    dict_storage_store(
        ns,
        "__xor__",
        make_builtin_function_with_arity("__xor__", proxy_xor, 2),
    );
    dict_storage_store(
        ns,
        "__rxor__",
        make_builtin_function_with_arity("__rxor__", proxy_rxor, 2),
    );
    // baseobjspace.py:2159 divmod row — forward + reflected.
    dict_storage_store(
        ns,
        "__divmod__",
        make_builtin_function_with_arity("__divmod__", proxy_divmod, 2),
    );
    dict_storage_store(
        ns,
        "__rdivmod__",
        make_builtin_function_with_arity("__rdivmod__", proxy_rdivmod, 2),
    );

    // Inplace ops — interp__weakref.py:367-369.
    dict_storage_store(
        ns,
        "__iadd__",
        make_builtin_function_with_arity("__iadd__", proxy_iadd, 2),
    );
    dict_storage_store(
        ns,
        "__isub__",
        make_builtin_function_with_arity("__isub__", proxy_isub, 2),
    );
    dict_storage_store(
        ns,
        "__imul__",
        make_builtin_function_with_arity("__imul__", proxy_imul, 2),
    );
    dict_storage_store(
        ns,
        "__itruediv__",
        make_builtin_function_with_arity("__itruediv__", proxy_itruediv, 2),
    );
    dict_storage_store(
        ns,
        "__ifloordiv__",
        make_builtin_function_with_arity("__ifloordiv__", proxy_ifloordiv, 2),
    );
    dict_storage_store(
        ns,
        "__imod__",
        make_builtin_function_with_arity("__imod__", proxy_imod, 2),
    );
    dict_storage_store(
        ns,
        "__ipow__",
        make_builtin_function_with_arity("__ipow__", proxy_ipow, 2),
    );
    dict_storage_store(
        ns,
        "__ilshift__",
        make_builtin_function_with_arity("__ilshift__", proxy_ilshift, 2),
    );
    dict_storage_store(
        ns,
        "__irshift__",
        make_builtin_function_with_arity("__irshift__", proxy_irshift, 2),
    );
    dict_storage_store(
        ns,
        "__iand__",
        make_builtin_function_with_arity("__iand__", proxy_iand, 2),
    );
    dict_storage_store(
        ns,
        "__ior__",
        make_builtin_function_with_arity("__ior__", proxy_ior, 2),
    );
    dict_storage_store(
        ns,
        "__ixor__",
        make_builtin_function_with_arity("__ixor__", proxy_ixor, 2),
    );

    // Single-dunder rows — interp__weakref.py:393-395.
    dict_storage_store(
        ns,
        "__format__",
        make_builtin_function_with_arity("__format__", proxy_format, 2),
    );
    dict_storage_store(
        ns,
        "__str__",
        make_builtin_function_with_arity("__str__", proxy_str, 1),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function_with_arity("__len__", proxy_len, 1),
    );
    dict_storage_store(
        ns,
        "__getattribute__",
        make_builtin_function_with_arity("__getattribute__", proxy_getattribute, 2),
    );
    dict_storage_store(
        ns,
        "__setattr__",
        make_builtin_function_with_arity("__setattr__", proxy_setattr, 3),
    );
    dict_storage_store(
        ns,
        "__delattr__",
        make_builtin_function_with_arity("__delattr__", proxy_delattr, 2),
    );
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function_with_arity("__getitem__", proxy_getitem, 2),
    );
    dict_storage_store(
        ns,
        "__setitem__",
        make_builtin_function_with_arity("__setitem__", proxy_setitem, 3),
    );
    dict_storage_store(
        ns,
        "__delitem__",
        make_builtin_function_with_arity("__delitem__", proxy_delitem, 2),
    );
    dict_storage_store(
        ns,
        "__trunc__",
        make_builtin_function_with_arity("__trunc__", proxy_trunc, 1),
    );
    dict_storage_store(
        ns,
        "__pos__",
        make_builtin_function_with_arity("__pos__", proxy_pos, 1),
    );
    dict_storage_store(
        ns,
        "__neg__",
        make_builtin_function_with_arity("__neg__", proxy_neg, 1),
    );
    dict_storage_store(
        ns,
        "__bool__",
        make_builtin_function_with_arity("__bool__", proxy_bool, 1),
    );
    dict_storage_store(
        ns,
        "__abs__",
        make_builtin_function_with_arity("__abs__", proxy_abs, 1),
    );
    dict_storage_store(
        ns,
        "__invert__",
        make_builtin_function_with_arity("__invert__", proxy_invert, 1),
    );
    dict_storage_store(
        ns,
        "__int__",
        make_builtin_function_with_arity("__int__", proxy_int, 1),
    );
    dict_storage_store(
        ns,
        "__index__",
        make_builtin_function_with_arity("__index__", proxy_index, 1),
    );
    dict_storage_store(
        ns,
        "__float__",
        make_builtin_function_with_arity("__float__", proxy_float, 1),
    );
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function_with_arity("__contains__", proxy_contains, 2),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity("__iter__", proxy_iter, 1),
    );
    dict_storage_store(
        ns,
        "__next__",
        make_builtin_function_with_arity("__next__", proxy_next, 1),
    );
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function_with_arity("__get__", proxy_get, 3),
    );
    dict_storage_store(
        ns,
        "__set__",
        make_builtin_function_with_arity("__set__", proxy_set, 3),
    );
    dict_storage_store(
        ns,
        "__delete__",
        make_builtin_function_with_arity("__delete__", proxy_delete, 2),
    );
    // baseobjspace.py:2127-2128 isinstance / issubtype rows.
    dict_storage_store(
        ns,
        "__instancecheck__",
        make_builtin_function_with_arity("__instancecheck__", proxy_instancecheck, 2),
    );
    dict_storage_store(
        ns,
        "__subclasscheck__",
        make_builtin_function_with_arity("__subclasscheck__", proxy_subclasscheck, 2),
    );

    // interp__weakref.py:390-391 — comparison ops are registered only on
    // `proxy_typedef_dict`, not `callable_proxy_typedef_dict`.
    if include_comparisons {
        dict_storage_store(
            ns,
            "__lt__",
            make_builtin_function_with_arity("__lt__", proxy_lt, 2),
        );
        dict_storage_store(
            ns,
            "__le__",
            make_builtin_function_with_arity("__le__", proxy_le, 2),
        );
        dict_storage_store(
            ns,
            "__gt__",
            make_builtin_function_with_arity("__gt__", proxy_gt, 2),
        );
        dict_storage_store(
            ns,
            "__ge__",
            make_builtin_function_with_arity("__ge__", proxy_ge, 2),
        );
        dict_storage_store(
            ns,
            "__eq__",
            make_builtin_function_with_arity("__eq__", proxy_eq, 2),
        );
        dict_storage_store(
            ns,
            "__ne__",
            make_builtin_function_with_arity("__ne__", proxy_ne, 2),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pypy/module/_weakref/interp__weakref.py:347-354 force —
    /// dead proxy must raise ReferenceError, not RuntimeError.
    #[test]
    fn test_force_dead_proxy_raises_reference_error() {
        crate::typedef::init_typeobjects();
        let proxy = W_Proxy_new(pyre_object::w_none(), PY_NULL);
        // Simulate a dead referent by setting the weak slot back to None.
        write_attr(proxy, ATTR_W_OBJ_WEAK, pyre_object::w_none());
        let err = force(proxy).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::ReferenceError);
        assert_eq!(err.message, "weakly referenced object no longer exists");
    }

    /// `len(proxy)` must dispatch through `proxy_typedef_dict["__len__"]`,
    /// which forces the receiver and calls `space.len(referent)`.
    #[test]
    fn test_proxy_len_delegates_to_referent() {
        crate::typedef::init_typeobjects();
        let referent = pyre_object::w_list_new(vec![
            pyre_object::w_int_new(1),
            pyre_object::w_int_new(2),
            pyre_object::w_int_new(3),
        ]);
        let proxy = W_Proxy_new(referent, PY_NULL);
        let result = crate::baseobjspace::len(proxy).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 3);
    }

    /// `proxy + x` must call `proxy_typedef_dict["__add__"]` which forces
    /// every operand and dispatches to `space.add(referent, x)`.
    #[test]
    fn test_proxy_add_delegates_to_referent() {
        crate::typedef::init_typeobjects();
        let referent = pyre_object::w_int_new(40);
        let proxy = W_Proxy_new(referent, PY_NULL);
        let result = crate::baseobjspace::add(proxy, pyre_object::w_int_new(2)).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 42);
    }

    /// `proxy.attr` must call the `__getattribute__` wrapper which forces
    /// the receiver and reads the attribute off the referent.
    #[test]
    fn test_proxy_getattr_delegates_to_referent() {
        crate::typedef::init_typeobjects();
        // Use a hasdict instance so the underlying object stores
        // attributes in INSTANCE_DICT.
        let user_type = crate::typedef::make_builtin_type("ProxyTarget", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(user_type, true) };
        let referent = pyre_object::instanceobject::w_instance_new(user_type);
        crate::baseobjspace::setattr(referent, "x", pyre_object::w_int_new(7)).unwrap();

        let proxy = W_Proxy_new(referent, PY_NULL);
        let result = crate::baseobjspace::getattr(proxy, "x").unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 7);
    }

    /// `proxy.attr = value` must delegate to the referent's dict.
    #[test]
    fn test_proxy_setattr_delegates_to_referent() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("ProxyTarget2", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(user_type, true) };
        let referent = pyre_object::instanceobject::w_instance_new(user_type);

        let proxy = W_Proxy_new(referent, PY_NULL);
        crate::baseobjspace::setattr(proxy, "y", pyre_object::w_int_new(11)).unwrap();

        // Read back via the referent directly to confirm the write
        // landed there, not on the proxy itself.
        let result = crate::baseobjspace::getattr(referent, "y").unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 11);
    }

    /// `force()` must be a no-op for any non-proxy operand. Otherwise the
    /// `getattr` / `setattr` fast-path would impose proxy semantics on
    /// every Python value.
    #[test]
    fn test_force_is_noop_for_non_proxy() {
        crate::typedef::init_typeobjects();
        let plain = pyre_object::w_int_new(99);
        let result = force(plain).unwrap();
        assert!(std::ptr::eq(result, plain));
    }

    /// `proxy[index]` must NOT force the index operand. PyPy's
    /// interp__weakref.py:365 sets `forcing_count = len(special_methods)`,
    /// so for `__getitem__` (single special method) only `self` is
    /// dereferenced. A weakproxy passed as the index must reach the
    /// referent's `__getitem__` unchanged so identity-keyed dicts work.
    #[test]
    fn test_proxy_getitem_does_not_force_index() {
        crate::typedef::init_typeobjects();
        // Build a list referent so getitem accepts an int index.
        let referent = pyre_object::w_list_new(vec![
            pyre_object::w_int_new(10),
            pyre_object::w_int_new(20),
            pyre_object::w_int_new(30),
        ]);
        let proxy = W_Proxy_new(referent, PY_NULL);
        // Drive the wrapper directly so we exercise its forcing rule.
        let result = proxy_getitem(&[proxy, pyre_object::w_int_new(1)]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 20);
    }

    /// `__pow__` is a 3-arg row in MethodTable. The wrapper must force
    /// the first two operands and pass the optional modulo through
    /// unchanged — PyPy's `forcing_count = len(special_methods) = 2`.
    #[test]
    fn test_proxy_pow_two_arg_form() {
        crate::typedef::init_typeobjects();
        let referent = pyre_object::w_int_new(2);
        let proxy = W_Proxy_new(referent, PY_NULL);
        // 2-arg form: 2 ** 8 == 256
        let result = proxy_pow(&[proxy, pyre_object::w_int_new(8)]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 256);
    }

    /// pypy/interpreter/baseobjspace.py:2127-2128 — `__instancecheck__`
    /// and `__subclasscheck__` must be present on both proxy typedefs.
    /// pypy/interpreter/baseobjspace.py:2159 — divmod must register
    /// both `__divmod__` and `__rdivmod__`.
    #[test]
    fn test_proxy_typedef_dict_includes_metaclass_and_divmod_rows() {
        crate::typedef::init_typeobjects();
        let weakproxy = proxy_type();
        let callable = callable_proxy_type();
        unsafe {
            for tp in [weakproxy, callable] {
                assert!(crate::baseobjspace::lookup_in_type(tp, "__instancecheck__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__subclasscheck__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__divmod__").is_some());
                assert!(crate::baseobjspace::lookup_in_type(tp, "__rdivmod__").is_some());
            }
        }
    }

    /// `isinstance(obj, weakproxy)` must dispatch through
    /// `proxy_typedef_dict["__instancecheck__"]` (the proxy_instancecheck
    /// wrapper), force the proxy, then run the recursive type check on
    /// the referent. Verifies the new `baseobjspace::isinstance` shares
    /// the same entry point as the builtin and the proxy wrapper.
    #[test]
    fn test_isinstance_through_proxy() {
        crate::typedef::init_typeobjects();
        let int_type = crate::typedef::r#type(pyre_object::w_int_new(0)).unwrap();
        let proxy = W_Proxy_new(int_type, PY_NULL);
        // 7 is an int — through the proxy it must still resolve.
        let yes = crate::baseobjspace::isinstance(pyre_object::w_int_new(7), proxy).unwrap();
        assert!(yes);
        // A non-int receiver — must be False.
        let no = crate::baseobjspace::isinstance(pyre_object::w_str_new("hi"), proxy).unwrap();
        assert!(!no);
    }

    /// `issubclass(int, weakproxy_around_int)` parity. Same dispatch
    /// path as `isinstance`, but via `__subclasscheck__`.
    #[test]
    fn test_issubclass_through_proxy() {
        crate::typedef::init_typeobjects();
        let int_type = crate::typedef::r#type(pyre_object::w_int_new(0)).unwrap();
        let str_type = crate::typedef::r#type(pyre_object::w_str_new("")).unwrap();
        let proxy = W_Proxy_new(int_type, PY_NULL);
        let yes = crate::baseobjspace::issubclass(int_type, proxy).unwrap();
        assert!(yes);
        let no = crate::baseobjspace::issubclass(str_type, proxy).unwrap();
        assert!(!no);
    }

    /// `isinstance(obj, Checker())` where `Checker` defines
    /// `__instancecheck__`. Verifies the override path fires for
    /// non-proxy user instances acting as classinfo —
    /// pypy/module/__builtin__/abstractinst.py:117-124.
    #[test]
    fn test_isinstance_user_instancecheck_override() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("Checker", |ns| {
            crate::dict_storage_store(
                ns,
                "__instancecheck__",
                crate::make_builtin_function("__instancecheck__", |_args| {
                    Ok(pyre_object::w_bool_from(true))
                }),
            );
        });
        let checker = pyre_object::instanceobject::w_instance_new(user_type);
        let yes = crate::baseobjspace::isinstance(pyre_object::w_int_new(123), checker).unwrap();
        assert!(yes);
    }

    /// Same as above but for `__subclasscheck__`.
    #[test]
    fn test_issubclass_user_subclasscheck_override() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("ClassChecker", |ns| {
            crate::dict_storage_store(
                ns,
                "__subclasscheck__",
                crate::make_builtin_function("__subclasscheck__", |_args| {
                    Ok(pyre_object::w_bool_from(true))
                }),
            );
        });
        let checker = pyre_object::instanceobject::w_instance_new(user_type);
        let int_type = crate::typedef::r#type(pyre_object::w_int_new(0)).unwrap();
        let yes = crate::baseobjspace::issubclass(int_type, checker).unwrap();
        assert!(yes);
    }

    /// `pow(proxy_around_lhs, rhs)` where `lhs.__pow__` returns
    /// NotImplemented and `rhs.__rpow__` returns the actual answer must
    /// fall through to the reflected operand. The pre-fix proxy_pow
    /// called the forward dunder directly and silently swallowed the
    /// NotImplemented sentinel.
    #[test]
    fn test_proxy_pow_falls_through_to_rpow() {
        crate::typedef::init_typeobjects();
        let lhs_type = crate::typedef::make_builtin_type("PowLhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__pow__",
                crate::make_builtin_function("__pow__", |_args| {
                    Ok(pyre_object::w_not_implemented())
                }),
            );
        });
        let rhs_type = crate::typedef::make_builtin_type("PowRhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__rpow__",
                crate::make_builtin_function("__rpow__", |_args| Ok(pyre_object::w_int_new(7777))),
            );
        });
        let lhs = pyre_object::instanceobject::w_instance_new(lhs_type);
        let rhs = pyre_object::instanceobject::w_instance_new(rhs_type);
        let proxy_lhs = W_Proxy_new(lhs, PY_NULL);
        // 2-arg form, no modulus.
        let result = proxy_pow(&[proxy_lhs, rhs]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 7777);
    }

    /// 3-arg `pow(proxy, b, c)` must end up in
    /// `space.pow3 → try_dispatch_ternary_special`, so a forward
    /// `__pow__` returning NotImplemented falls through to the
    /// reflected `__rpow__`.
    #[test]
    fn test_proxy_pow_three_arg_falls_through_to_rpow() {
        crate::typedef::init_typeobjects();
        let lhs_type = crate::typedef::make_builtin_type("Pow3Lhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__pow__",
                crate::make_builtin_function("__pow__", |_args| {
                    Ok(pyre_object::w_not_implemented())
                }),
            );
        });
        let rhs_type = crate::typedef::make_builtin_type("Pow3Rhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__rpow__",
                crate::make_builtin_function("__rpow__", |args| {
                    // Echo the modulus operand so we can confirm it
                    // threaded through to the reflected wrapper.
                    Ok(args[2])
                }),
            );
        });
        let lhs = pyre_object::instanceobject::w_instance_new(lhs_type);
        let rhs = pyre_object::instanceobject::w_instance_new(rhs_type);
        let proxy_lhs = W_Proxy_new(lhs, PY_NULL);
        let result = proxy_pow(&[proxy_lhs, rhs, pyre_object::w_int_new(99)]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 99);
    }

    /// `divmod(proxy, rhs)` where `lhs.__divmod__` returns NotImplemented
    /// must fall through to `rhs.__rdivmod__`. Same regression class as
    /// `proxy_pow`.
    #[test]
    fn test_proxy_divmod_falls_through_to_rdivmod() {
        crate::typedef::init_typeobjects();
        let lhs_type = crate::typedef::make_builtin_type("DivmodLhsNI", |ns| {
            crate::dict_storage_store(
                ns,
                "__divmod__",
                crate::make_builtin_function("__divmod__", |_args| {
                    Ok(pyre_object::w_not_implemented())
                }),
            );
        });
        let rhs_type = crate::typedef::make_builtin_type("DivmodRhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__rdivmod__",
                crate::make_builtin_function("__rdivmod__", |_args| {
                    Ok(pyre_object::w_int_new(123456))
                }),
            );
        });
        let lhs = pyre_object::instanceobject::w_instance_new(lhs_type);
        let rhs = pyre_object::instanceobject::w_instance_new(rhs_type);
        let proxy_lhs = W_Proxy_new(lhs, PY_NULL);
        let result = proxy_divmod(&[proxy_lhs, rhs]).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 123456);
    }

    /// pypy/module/_weakref/interp__weakref.py:390-391 — comparison ops
    /// land on `proxy_typedef_dict` only, never on
    /// `callable_proxy_typedef_dict`. `__lt__`/`__le__`/`__gt__`/`__ge__`
    /// are not on `object`, so an MRO lookup is enough to tell the two
    /// typedefs apart. (`__eq__`/`__ne__` are inherited from `object`
    /// regardless and so cannot be tested by name lookup alone.)
    #[test]
    fn test_comparison_ops_only_on_weakproxy() {
        crate::typedef::init_typeobjects();
        let weakproxy = proxy_type();
        let callable = callable_proxy_type();
        unsafe {
            assert!(crate::baseobjspace::lookup_in_type(weakproxy, "__lt__").is_some());
            assert!(crate::baseobjspace::lookup_in_type(weakproxy, "__le__").is_some());
            assert!(crate::baseobjspace::lookup_in_type(weakproxy, "__gt__").is_some());
            assert!(crate::baseobjspace::lookup_in_type(weakproxy, "__ge__").is_some());
            assert!(crate::baseobjspace::lookup_in_type(callable, "__lt__").is_none());
            assert!(crate::baseobjspace::lookup_in_type(callable, "__le__").is_none());
            assert!(crate::baseobjspace::lookup_in_type(callable, "__gt__").is_none());
            assert!(crate::baseobjspace::lookup_in_type(callable, "__ge__").is_none());
            // Forwarded ops should land on both typedefs.
            assert!(crate::baseobjspace::lookup_in_type(weakproxy, "__add__").is_some());
            assert!(crate::baseobjspace::lookup_in_type(callable, "__add__").is_some());
            assert!(crate::baseobjspace::lookup_in_type(weakproxy, "__len__").is_some());
            assert!(crate::baseobjspace::lookup_in_type(callable, "__len__").is_some());
        }
    }
}
