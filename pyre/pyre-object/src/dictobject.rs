//! W_DictObject — Python `dict` type.
//!
//! PyPy equivalent: pypy/objspace/std/dictobject.py
//!
//! Supports arbitrary PyObjectRef keys (int, str, etc.) with
//! equality comparison via pointer identity and type-specific checks.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python dict object.
///
/// Layout: `[ob_type | entries | len | dict_storage_proxy]`
///
/// Keys are PyObjectRef compared by dict_keys_equal.
/// PyPy uses multiple dict strategies; pyre uses a single Vec for simplicity.
///
/// `dict_storage_proxy`: when non-null, mutations to this dict are also
/// written to the backing `DictStorage`. This is pyre's internal stand-in for
/// the string-keyed dict state that PyPy keeps in `W_DictMultiObject` /
/// `ModuleDictStrategy` or type-level `dict_w`.
#[repr(C)]
pub struct W_DictObject {
    pub ob_header: PyObject,
    pub entries: *mut Vec<(PyObjectRef, PyObjectRef)>,
    pub len: usize,
    pub dict_storage_proxy: *mut u8,
}

/// Field offset of `len` within `W_DictObject`, for JIT field access.
pub const DICT_LEN_OFFSET: usize = std::mem::offset_of!(W_DictObject, len);

/// GC type id assigned to `W_DictObject` at JitDriver init time.
pub const W_DICT_GC_TYPE_ID: u32 = 29;

/// Fixed payload size (`framework.py:811`).
pub const W_DICT_OBJECT_SIZE: usize = std::mem::size_of::<W_DictObject>();

impl crate::lltype::GcType for W_DictObject {
    const TYPE_ID: u32 = W_DICT_GC_TYPE_ID;
    const SIZE: usize = W_DICT_OBJECT_SIZE;
}

/// Allocate a new empty dict.
pub fn w_dict_new() -> PyObjectRef {
    let entries = crate::lltype::malloc_raw(Vec::new());
    crate::lltype::malloc_typed(W_DictObject {
        ob_header: PyObject {
            ob_type: &DICT_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_TYPE),
        },
        entries,
        len: 0,
        dict_storage_proxy: std::ptr::null_mut(),
    }) as PyObjectRef
}

/// Allocate a dict backed by a `DictStorage` (for `globals()` and similar
/// live dict views). Mutations to this dict also update the backing storage.
pub fn w_dict_new_with_dict_storage(ns: *mut u8) -> PyObjectRef {
    let entries = crate::lltype::malloc_raw(Vec::new());
    crate::lltype::malloc_typed(W_DictObject {
        ob_header: PyObject {
            ob_type: &DICT_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_TYPE),
        },
        entries,
        len: 0,
        dict_storage_proxy: ns,
    }) as PyObjectRef
}

/// Compare two dict keys for equality.
///
/// PyPy: uses space.eq_w which dispatches to type-specific comparison.
/// pyre handles the hashable builtin types directly: int, str, bool, tuple,
/// frozenset, plus pointer identity for everything else.
pub(crate) unsafe fn dict_keys_equal(a: PyObjectRef, b: PyObjectRef) -> bool {
    if std::ptr::eq(a, b) {
        return true;
    }
    if a.is_null() || b.is_null() {
        return false;
    }
    // Mixed numeric: bool ↔ int (Python: True == 1 and False == 0).
    let a_is_bool = crate::is_bool(a);
    let b_is_bool = crate::is_bool(b);
    let a_is_int = crate::is_int(a);
    let b_is_int = crate::is_int(b);
    if (a_is_int || a_is_bool) && (b_is_int || b_is_bool) {
        let av = if a_is_bool {
            crate::w_bool_get_value(a) as i64
        } else {
            crate::w_int_get_value(a)
        };
        let bv = if b_is_bool {
            crate::w_bool_get_value(b) as i64
        } else {
            crate::w_int_get_value(b)
        };
        return av == bv;
    }
    // Str keys
    if crate::is_str(a) && crate::is_str(b) {
        return crate::w_str_get_value(a) == crate::w_str_get_value(b);
    }
    // Bytes keys — compare byte contents.
    if crate::bytesobject::is_bytes(a) && crate::bytesobject::is_bytes(b) {
        return crate::bytesobject::w_bytes_data(a) == crate::bytesobject::w_bytes_data(b);
    }
    // Tuple keys — element-wise compare via dict_keys_equal recursively.
    if crate::is_tuple(a) && crate::is_tuple(b) {
        let la = crate::w_tuple_len(a);
        let lb = crate::w_tuple_len(b);
        if la != lb {
            return false;
        }
        for i in 0..la {
            let ea = crate::w_tuple_getitem(a, i as i64).unwrap_or(std::ptr::null_mut());
            let eb = crate::w_tuple_getitem(b, i as i64).unwrap_or(std::ptr::null_mut());
            if !dict_keys_equal(ea, eb) {
                return false;
            }
        }
        return true;
    }
    // frozenset / set: element-wise containment via the same equality.
    if crate::is_frozenset(a) && crate::is_frozenset(b) {
        let ai = crate::w_set_items(a);
        let bi = crate::w_set_items(b);
        if ai.len() != bi.len() {
            return false;
        }
        return ai
            .iter()
            .all(|&x| bi.iter().any(|&y| dict_keys_equal(x, y)));
    }
    false
}

/// Get a value by PyObjectRef key.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_lookup(obj: PyObjectRef, key: PyObjectRef) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    for &(ref k, v) in entries {
        if dict_keys_equal(*k, key) {
            return Some(v);
        }
    }
    None
}

/// Set a value by PyObjectRef key.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_store(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *dict.entries;
    for entry in entries.iter_mut() {
        if dict_keys_equal(entry.0, key) {
            entry.1 = value;
            // Storage proxy sync: if this dict is backed by a DictStorage
            // (typical for globals()), propagate the update back so that
            // module-level assignments via `globals()[name] = value` appear
            // in the frame's backing dict storage.
            maybe_sync_dict_storage_store(dict.dict_storage_proxy, key, value);
            return;
        }
    }
    entries.push((key, value));
    dict.len += 1;
    maybe_sync_dict_storage_store(dict.dict_storage_proxy, key, value);
}

/// Write a str-keyed assignment back to the dict's backing DictStorage,
/// if any. Declared in pyre-interpreter and re-exported via an `extern`
/// hook registered at startup to avoid a circular dependency.
unsafe fn maybe_sync_dict_storage_store(ns_ptr: *mut u8, key: PyObjectRef, value: PyObjectRef) {
    if ns_ptr.is_null() || !crate::is_str(key) {
        return;
    }
    if let Some(hook) = DICT_STORAGE_STORE_HOOK
        .load(std::sync::atomic::Ordering::Acquire)
        .as_ref()
    {
        let name = crate::w_str_get_value(key);
        hook(ns_ptr, name, value);
    }
}

/// Mirror of `maybe_sync_dict_storage_store` for deletions.  When a dict
/// with a backing storage drops a str-keyed entry, propagate the
/// deletion so storage-keyed lookups (LOAD_GLOBAL builtins fallback)
/// stop seeing the stale entry.  PyPy keeps everything in one
/// `W_DictMultiObject` so this asymmetry is pyre-only.
unsafe fn maybe_sync_dict_storage_delete(ns_ptr: *mut u8, key_str: &str) {
    if ns_ptr.is_null() {
        return;
    }
    if let Some(hook) = DICT_STORAGE_DELETE_HOOK
        .load(std::sync::atomic::Ordering::Acquire)
        .as_ref()
    {
        hook(ns_ptr, key_str);
    }
}

type NamespaceStoreHook = unsafe fn(*mut u8, &str, PyObjectRef);
type NamespaceDeleteHook = unsafe fn(*mut u8, &str);

struct AtomicHookPtr(std::sync::atomic::AtomicPtr<NamespaceStoreHook>);

impl AtomicHookPtr {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
    }

    fn store(&self, hook: NamespaceStoreHook) {
        // Leak a boxed function pointer so the pointer lives for the entire
        // process lifetime; this matches PyPy's one-time interp init.
        // `flavor='raw'` because this is host-side dispatch state, not a
        // GC-managed Python object.
        let raw = crate::lltype::malloc_raw(hook);
        self.0.store(raw, std::sync::atomic::Ordering::Release);
    }

    fn load(&self, order: std::sync::atomic::Ordering) -> *const NamespaceStoreHook {
        self.0.load(order) as *const NamespaceStoreHook
    }
}

struct AtomicDeleteHookPtr(std::sync::atomic::AtomicPtr<NamespaceDeleteHook>);

impl AtomicDeleteHookPtr {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
    }

    fn store(&self, hook: NamespaceDeleteHook) {
        let raw = crate::lltype::malloc_raw(hook);
        self.0.store(raw, std::sync::atomic::Ordering::Release);
    }

    fn load(&self, order: std::sync::atomic::Ordering) -> *const NamespaceDeleteHook {
        self.0.load(order) as *const NamespaceDeleteHook
    }
}

static DICT_STORAGE_STORE_HOOK: AtomicHookPtr = AtomicHookPtr::new();
static DICT_STORAGE_DELETE_HOOK: AtomicDeleteHookPtr = AtomicDeleteHookPtr::new();

/// Register the interpreter-level hook that writes (name, value) into a
/// DictStorage. Called once during interpreter startup.
pub fn register_dict_storage_store_hook(hook: NamespaceStoreHook) {
    DICT_STORAGE_STORE_HOOK.store(hook);
}

/// Register the interpreter-level hook that deletes `name` from a
/// DictStorage. Called once during interpreter startup.
pub fn register_dict_storage_delete_hook(hook: NamespaceDeleteHook) {
    DICT_STORAGE_DELETE_HOOK.store(hook);
}

/// Get the dict_storage_proxy pointer from a dict (used by interpreter for
/// live globals sync).
pub unsafe fn w_dict_get_dict_storage_proxy(obj: PyObjectRef) -> *mut u8 {
    (*(obj as *const W_DictObject)).dict_storage_proxy
}

/// Attach a `DictStorage` proxy to an existing dict so subsequent
/// mutations sync into the storage.  Used by interpreter-level
/// `pick_builtin` (`pypy/module/__builtin__/moduledef.py:102-103`)
/// to lift an arbitrary user-supplied `__builtins__` dict into pyre's
/// storage-keyed lookup model — counterpart of the no-op assignment
/// `module.Module(space, None, w_builtin)` does in PyPy by aliasing
/// `module.w_dict = w_builtin`.
pub unsafe fn w_dict_set_dict_storage_proxy(obj: PyObjectRef, ns: *mut u8) {
    (*(obj as *mut W_DictObject)).dict_storage_proxy = ns;
}

/// Get a value by int key (convenience wrapper).
pub unsafe fn w_dict_getitem(obj: PyObjectRef, key: i64) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    for &(ref k, v) in entries {
        if crate::is_int(*k) && crate::w_int_get_value(*k) == key {
            return Some(v);
        }
    }
    None
}

/// Set a value by int key (convenience wrapper).
pub unsafe fn w_dict_setitem(obj: PyObjectRef, key: i64, value: PyObjectRef) {
    w_dict_store(obj, crate::w_int_new(key), value)
}

/// Get a value by str key (convenience wrapper).
pub unsafe fn w_dict_getitem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    for &(ref k, v) in entries {
        if crate::is_str(*k) && crate::w_str_get_value(*k) == key {
            return Some(v);
        }
    }
    None
}

/// Set a value by str key (convenience wrapper).
pub unsafe fn w_dict_setitem_str(obj: PyObjectRef, key: &str, value: PyObjectRef) {
    w_dict_store(obj, crate::w_str_new(key), value)
}

/// Remove an entry by str key. Returns true if the key was found.
pub unsafe fn w_dict_delitem_str(obj: PyObjectRef, key: &str) -> bool {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *dict.entries;
    if let Some(idx) = entries
        .iter()
        .position(|(k, _)| crate::is_str(*k) && crate::w_str_get_value(*k) == key)
    {
        entries.remove(idx);
        dict.len -= 1;
        // Storage proxy sync: keep `pick_builtin`'s lazy mirror /
        // `globals()` view in step with explicit `del dict[name]`.
        maybe_sync_dict_storage_delete(dict.dict_storage_proxy, key);
        true
    } else {
        false
    }
}

/// Get the number of entries.
pub unsafe fn w_dict_len(obj: PyObjectRef) -> usize {
    (*(obj as *const W_DictObject)).len
}

/// Iterate over all (key, value) pairs without type assumptions.
pub unsafe fn w_dict_items(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    let dict = &*(obj as *const W_DictObject);
    (*dict.entries).clone()
}

/// Iterate over (key_str, value) pairs. Keys must be str objects.
pub unsafe fn w_dict_str_entries(obj: PyObjectRef) -> Vec<(String, PyObjectRef)> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    entries
        .iter()
        .map(|&(k, v)| (crate::w_str_get_value(k).to_string(), v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::{w_int_get_value, w_int_new};

    #[test]
    fn test_dict_int_key() {
        let dict = w_dict_new();
        unsafe {
            assert!(is_dict(dict));
            w_dict_setitem(dict, 1, w_int_new(100));
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 1).unwrap()), 100);
        }
    }

    #[test]
    fn test_dict_str_key() {
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem_str(dict, "hello", w_int_new(42));
            assert_eq!(
                w_int_get_value(w_dict_getitem_str(dict, "hello").unwrap()),
                42
            );
            assert!(w_dict_getitem_str(dict, "world").is_none());
        }
    }

    #[test]
    fn test_dict_pyobj_key() {
        let dict = w_dict_new();
        let key = crate::w_str_new("test");
        unsafe {
            w_dict_store(dict, key, w_int_new(99));
            assert_eq!(w_int_get_value(w_dict_lookup(dict, key).unwrap()), 99);
        }
    }

    #[test]
    fn test_dict_overwrite() {
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem(dict, 1, w_int_new(10));
            w_dict_setitem(dict, 1, w_int_new(20));
            assert_eq!(w_dict_len(dict), 1);
        }
    }

    #[test]
    fn w_dict_gc_type_id_matches_descr() {
        assert_eq!(W_DICT_GC_TYPE_ID, 29);
        assert_eq!(
            <W_DictObject as crate::lltype::GcType>::TYPE_ID,
            W_DICT_GC_TYPE_ID
        );
        assert_eq!(
            <W_DictObject as crate::lltype::GcType>::SIZE,
            W_DICT_OBJECT_SIZE
        );
    }
}
