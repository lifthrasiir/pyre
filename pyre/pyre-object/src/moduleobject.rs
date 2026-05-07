//! W_ModuleObject — Python `module` type.
//!
//! PyPy equivalent: pypy/interpreter/module.py → Module
//!
//! A module holds a name (str) and a pointer to its backing dict storage.
//! The storage holds all names defined in the module after execution.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python module object.
///
/// Layout: `[ob_type | name | dict | w_dict]`
///
/// `dict` is a raw pointer to `pyre_interpreter::DictStorage`, the internal
/// storage pyre uses where PyPy would expose a string-keyed module dict.
/// We store it as `*mut u8` to avoid a circular dependency on pyre-interpreter.
///
/// `w_dict` mirrors PyPy `module.py:20 self.w_dict = w_dict` — every
/// Module owns a non-null `W_DictObject` (or dict subclass instance
/// for the user-supplied wrap case at `moduledef.py:102-103`).  For
/// storage-only Modules pyre constructs a `W_DictObject` whose
/// `dict_storage_proxy` points at `dict`, so reads on the wrapper
/// fall through to the storage and `getdict(space)` returns a stable
/// identity across calls.  For the user-supplied case `w_dict` is the
/// caller's object directly, preserving subclass identity for
/// `space.finditem_str` dispatch.
#[repr(C)]
pub struct W_ModuleObject {
    pub ob_header: PyObject,
    /// Heap-allocated module name string.
    pub name: *mut String,
    /// Raw pointer to the module's backing dict storage (globals after execution).
    pub dict: *mut u8,
    /// Authoritative dict object (`PyPy module.w_dict`).  Always non-null
    /// after construction.
    pub w_dict: PyObjectRef,
}

/// GC type id assigned to `W_ModuleObject` at JitDriver init time.
pub const W_MODULE_GC_TYPE_ID: u32 = 36;

/// Fixed payload size (`framework.py:811`).
pub const W_MODULE_OBJECT_SIZE: usize = std::mem::size_of::<W_ModuleObject>();

/// Byte offset of the inline `w_dict: PyObjectRef` slot — the GC must
/// trace the aliased `W_DictObject` (`pypy/interpreter/module.py:22
/// self.w_dict = w_dict`) so a Module surviving a minor collection
/// keeps the user-supplied dict alive.  `name`/`dict` are non-PyObject
/// raw heap pointers and are intentionally absent; they are owned via
/// `lltype::malloc_raw` and traced through their own type ids.
pub const W_MODULE_GC_PTR_OFFSETS: [usize; 1] = [std::mem::offset_of!(W_ModuleObject, w_dict)];

impl crate::lltype::GcType for W_ModuleObject {
    const TYPE_ID: u32 = W_MODULE_GC_TYPE_ID;
    const SIZE: usize = W_MODULE_OBJECT_SIZE;
}

/// Allocate a new W_ModuleObject backed by a `DictStorage`.  Use this
/// for `space.builtin`, freshly-imported modules, REPL `__main__`, and
/// other Modules whose authoritative dict IS the storage.  The Module
/// owns a `W_DictObject` whose `dict_storage_proxy` points at `dict_ptr`,
/// so reads on the wrapper fall through to the storage and
/// `getdict(space)` returns a stable identity across calls.
///
/// `module.py:24` — `if w_name is not None: setitem(w_dict, '__name__',
/// w_name)`.  Pyre seeds `__name__` through `w_dict_setitem_str` which
/// also propagates the entry into the proxy storage (so storage-keyed
/// readers observe it without going through the `W_DictObject`).
///
/// `name` — the module name (e.g. "math", "os.path"); empty string is
///   the anonymous-name sentinel for `pick_builtin`'s default Module
///   case (`moduledef.py:106-108`, PyPy `Module(space, None, ...)`)
///   in which `Module.__init__` skips the `__name__` setitem.
/// `dict_ptr` — raw pointer to the module's backing dict storage; may
///   be null only for tests and the anonymous default Module.
pub fn w_module_new(name: &str, dict_ptr: *mut u8) -> PyObjectRef {
    let name_box = crate::lltype::malloc_raw(name.to_string());
    let w_dict = crate::dictobject::w_dict_new_with_dict_storage(dict_ptr);
    if !name.is_empty() {
        unsafe {
            crate::dictobject::w_dict_setitem_str(w_dict, "__name__", crate::w_str_new(name));
        }
    }
    crate::lltype::malloc_typed(W_ModuleObject {
        ob_header: PyObject {
            ob_type: &MODULE_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_TYPE),
        },
        name: name_box,
        dict: dict_ptr,
        w_dict,
    }) as PyObjectRef
}

/// Allocate a `W_ModuleObject` aliasing a user-supplied `W_DictObject`.
/// Mirrors `pypy/module/__builtin__/moduledef.py:102-103
/// module.Module(space, None, w_builtin)`: the Module's dict identity
/// IS the user dict (PyPy `module.w_dict = w_builtin`).
///
/// `dict_ptr` is the storage-proxy mirror that callers attach to the
/// user dict (see `w_dict_set_dict_storage_proxy` +
/// `maybe_sync_dict_storage_{store,delete}` for the sync contract); the
/// storage-keyed fastpath (`LOAD_GLOBAL` builtins fallback,
/// `frame.builtin`) reaches that proxy via the `dict` field, while
/// observers (`getdict(space)`, `__dict__`, `f_builtins`) prefer
/// `w_dict` so the user dict identity survives.
pub fn w_module_new_aliasing_dict(
    name: &str,
    dict_ptr: *mut u8,
    w_dict_object: PyObjectRef,
) -> PyObjectRef {
    let name = crate::lltype::malloc_raw(name.to_string());
    crate::lltype::malloc_typed(W_ModuleObject {
        ob_header: PyObject {
            ob_type: &MODULE_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_TYPE),
        },
        name,
        dict: dict_ptr,
        w_dict: w_dict_object,
    }) as PyObjectRef
}

/// Get the module name.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleObject`.
pub unsafe fn w_module_get_name(obj: PyObjectRef) -> &'static str {
    let module = &*(obj as *const W_ModuleObject);
    &*module.name
}

/// Get the module's namespace pointer (as *mut u8).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleObject`.
pub unsafe fn w_module_get_dict_ptr(obj: PyObjectRef) -> *mut u8 {
    let module = &*(obj as *const W_ModuleObject);
    module.dict
}

/// Get the aliased `W_DictObject` (`PY_NULL` when storage-only).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleObject`.
pub unsafe fn w_module_get_w_dict(obj: PyObjectRef) -> PyObjectRef {
    let module = &*(obj as *const W_ModuleObject);
    module.w_dict
}

/// pypy/interpreter/module.py:Module.getdictvalue —
/// `space.finditem_str(self.w_dict, attr)`.  When `w_dict` is a real
/// `W_DictObject` pyre routes through `w_dict_getitem_str` (which
/// honours the storage-proxy read-through, so storage-only Modules
/// surface storage entries via the same call).  When `w_dict` is a
/// dict subclass instance the caller must take the
/// `space.finditem_str` dispatch path itself (subclass `__getitem__`
/// override) — pyre-object can't reach the interpreter's dispatcher,
/// so we return `None` and rely on the storage fallback at the
/// caller (`eval.rs:load_global_value`).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleObject`.
pub unsafe fn w_module_alias_getitem_str(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let module = &*(obj as *const W_ModuleObject);
    if module.w_dict.is_null() {
        return None;
    }
    if !crate::is_dict(module.w_dict) {
        return None;
    }
    crate::dictobject::w_dict_getitem_str(module.w_dict, name)
}

/// Check if an object is a module.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_module(obj: PyObjectRef) -> bool {
    py_type_check(obj, &MODULE_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_create_and_check() {
        let obj = w_module_new("test_mod", std::ptr::null_mut());
        unsafe {
            assert!(is_module(obj));
            assert!(!is_int(obj));
            assert_eq!(w_module_get_name(obj), "test_mod");
            assert!(w_module_get_dict_ptr(obj).is_null());
        }
    }
}
