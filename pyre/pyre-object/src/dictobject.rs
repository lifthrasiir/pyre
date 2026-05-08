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
/// When `dict_storage_proxy` is attached, the storage is treated as
/// authoritative for str keys: lookup checks the storage FIRST, so a
/// transient proxy dict whose `entries` Vec carries a stale snapshot
/// (`dict_storage_to_dict` materialisation, `w_module_new`
/// pre-population that was later mutated by a STORE_GLOBAL on the
/// shared storage) returns the live value rather than the cached
/// stale one.  Non-str keys live only in the entries Vec because
/// `DictStorage` is str-keyed by construction.
///
/// PyPy parity: `pypy/interpreter/module.py:77 Module.getdict()`
/// returns the live `W_DictMultiObject` whose state IS the module's
/// dict — there is no stale snapshot to worry about because there is
/// only one map.  Pyre's split (entries Vec + DictStorage) mirrors
/// the same single-source-of-truth shape only when the storage side
/// wins for the key types it represents.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_lookup(obj: PyObjectRef, key: PyObjectRef) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    if !dict.dict_storage_proxy.is_null() && crate::is_str(key) {
        if let Some(v) =
            maybe_lookup_dict_storage(dict.dict_storage_proxy, crate::w_str_get_value(key))
        {
            return Some(v);
        }
    }
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

/// Storage-proxy read-through.  PyPy keeps every dict-backed lookup
/// inside the same `W_DictMultiObject`, so reads see entries
/// regardless of which interpreter side wrote them.  Pyre splits the
/// dict's `entries` Vec from the `DictStorage` proxy, so a dict whose
/// authoritative state lives in the storage (`Module.w_dict` over
/// `space.builtin`'s storage, `globals()` view) must surface storage
/// entries on read.  Returns `None` when no proxy is attached or the
/// hook is unregistered.
unsafe fn maybe_lookup_dict_storage(ns_ptr: *mut u8, key_str: &str) -> Option<PyObjectRef> {
    if ns_ptr.is_null() {
        return None;
    }
    let ptr = DICT_STORAGE_LOOKUP_HOOK.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        return None;
    }
    (*ptr)(ns_ptr, key_str)
}

type NamespaceStoreHook = unsafe fn(*mut u8, &str, PyObjectRef);
type NamespaceDeleteHook = unsafe fn(*mut u8, &str);
type NamespaceLookupHook = unsafe fn(*mut u8, &str) -> Option<PyObjectRef>;
type NamespaceItemsHook = unsafe fn(*mut u8) -> Vec<(String, PyObjectRef)>;

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

struct AtomicLookupHookPtr(std::sync::atomic::AtomicPtr<NamespaceLookupHook>);

impl AtomicLookupHookPtr {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
    }

    fn store(&self, hook: NamespaceLookupHook) {
        let raw = crate::lltype::malloc_raw(hook);
        self.0.store(raw, std::sync::atomic::Ordering::Release);
    }

    fn load(&self, order: std::sync::atomic::Ordering) -> *const NamespaceLookupHook {
        self.0.load(order) as *const NamespaceLookupHook
    }
}

struct AtomicItemsHookPtr(std::sync::atomic::AtomicPtr<NamespaceItemsHook>);

impl AtomicItemsHookPtr {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
    }

    fn store(&self, hook: NamespaceItemsHook) {
        let raw = crate::lltype::malloc_raw(hook);
        self.0.store(raw, std::sync::atomic::Ordering::Release);
    }

    fn load(&self, order: std::sync::atomic::Ordering) -> *const NamespaceItemsHook {
        self.0.load(order) as *const NamespaceItemsHook
    }
}

static DICT_STORAGE_STORE_HOOK: AtomicHookPtr = AtomicHookPtr::new();
static DICT_STORAGE_DELETE_HOOK: AtomicDeleteHookPtr = AtomicDeleteHookPtr::new();
static DICT_STORAGE_LOOKUP_HOOK: AtomicLookupHookPtr = AtomicLookupHookPtr::new();
static DICT_STORAGE_ITEMS_HOOK: AtomicItemsHookPtr = AtomicItemsHookPtr::new();

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

/// Register the interpreter-level hook that looks up `name` in a
/// DictStorage and returns its value (or `None`).  Called once during
/// interpreter startup so dicts with a `dict_storage_proxy` surface
/// storage entries on read miss.
pub fn register_dict_storage_lookup_hook(hook: NamespaceLookupHook) {
    DICT_STORAGE_LOOKUP_HOOK.store(hook);
}

/// Register the interpreter-level hook that enumerates all str-keyed
/// `(name, value)` pairs from a DictStorage.  Used by `w_dict_len`,
/// `w_dict_items`, `w_dict_str_entries` and `w_dict_delitem_str` to keep
/// the full dict protocol (`len(module.__dict__)`, `module.__dict__.items()`,
/// `del module.__dict__[name]`) consistent with PyPy's
/// `Module.getdict()` returning the live W_DictMultiObject — pyre splits
/// the proxy off the entries Vec and would otherwise miss every storage
/// entry not yet mirrored into the W_DictObject.
pub fn register_dict_storage_items_hook(hook: NamespaceItemsHook) {
    DICT_STORAGE_ITEMS_HOOK.store(hook);
}

/// Read-side counterpart of `maybe_sync_dict_storage_store`: enumerate
/// the str-keyed entries currently in the backing storage.
///
/// Returns `None` when the items hook has not yet been registered (the
/// hookless case fires for direct `w_module_new` callers and for unit
/// tests that exercise dict surfaces before `register_dict_storage_*_hook`
/// runs).  Callers must distinguish "hook missing" (storage view is
/// indeterminate, fall back to entries Vec) from "hook installed, storage
/// empty" (authoritative empty result) — collapsing the two would silently
/// drop entries Vec str keys for proxied dicts whose hook arrives later
/// in the bootstrap.
unsafe fn maybe_items_dict_storage(ns_ptr: *mut u8) -> Option<Vec<(String, PyObjectRef)>> {
    if ns_ptr.is_null() {
        return Some(Vec::new());
    }
    let ptr = DICT_STORAGE_ITEMS_HOOK.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        return None;
    }
    Some((*ptr)(ns_ptr))
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

/// Get a value by str key (convenience wrapper).  Mirrors
/// `w_dict_lookup`'s storage-first contract for proxied dicts so
/// stale `entries` snapshots (e.g. `dict_storage_to_dict`
/// materialisation) don't shadow live storage updates.
pub unsafe fn w_dict_getitem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    if !dict.dict_storage_proxy.is_null() {
        if let Some(v) = maybe_lookup_dict_storage(dict.dict_storage_proxy, key) {
            return Some(v);
        }
    }
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

/// Set a value by str key WITHOUT firing the dict_storage_proxy
/// store hook.  Used by the storage→W_DictObject back-mirror so a
/// `dict_storage_store` on a storage that has a registered mirror
/// target updates the W_DictObject's entries Vec without
/// re-entering `maybe_sync_dict_storage_store` (which would feed
/// the same write right back into the storage and create an
/// observable double-invalidation of slot watchers).
///
/// PyPy keeps everything in one `W_DictMultiObject`, so the
/// asymmetric "entries Vec write must skip storage notification"
/// shape is pyre-only; the no-proxy variant is the structural
/// adapter for the bidirectional sync that PyPy gets for free.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_setitem_str_no_proxy(obj: PyObjectRef, key: &str, value: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *dict.entries;
    for entry in entries.iter_mut() {
        if crate::is_str(entry.0) && crate::w_str_get_value(entry.0) == key {
            entry.1 = value;
            return;
        }
    }
    entries.push((crate::w_str_new(key), value));
    dict.len += 1;
}

/// Remove an entry by str key WITHOUT firing the dict_storage_proxy
/// delete hook.  Counterpart of `w_dict_setitem_str_no_proxy`; see
/// that doc-comment for the back-mirror rationale.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_delitem_str_no_proxy(obj: PyObjectRef, key: &str) -> bool {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *dict.entries;
    if let Some(idx) = entries
        .iter()
        .position(|(k, _)| crate::is_str(*k) && crate::w_str_get_value(*k) == key)
    {
        entries.remove(idx);
        dict.len -= 1;
        true
    } else {
        false
    }
}

/// Remove an entry by str key. Returns true if the key was found.
pub unsafe fn w_dict_delitem_str(obj: PyObjectRef, key: &str) -> bool {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *dict.entries;
    let mut hit = false;
    if let Some(idx) = entries
        .iter()
        .position(|(k, _)| crate::is_str(*k) && crate::w_str_get_value(*k) == key)
    {
        entries.remove(idx);
        dict.len -= 1;
        hit = true;
    }
    // Storage proxy sync: keep `pick_builtin`'s lazy mirror / `globals()`
    // view in step with explicit `del dict[name]`.  PyPy's
    // `W_DictMultiObject` keeps a single map so a `del module.__dict__[k]`
    // removes the entry whether it lives in entries Vec or storage; pyre
    // mirrors that by always issuing the delete-through hook even when
    // the entries Vec missed (proxy storage may still own the binding).
    if !dict.dict_storage_proxy.is_null() {
        if !hit {
            // Probe storage so we report whether anything was actually
            // removed; PyPy raises KeyError when neither side knows the
            // key.
            if maybe_lookup_dict_storage(dict.dict_storage_proxy, key).is_some() {
                hit = true;
            }
        }
        maybe_sync_dict_storage_delete(dict.dict_storage_proxy, key);
    }
    hit
}

/// Get the number of entries.
///
/// Storage-authoritative for str keys when proxy is attached:
/// returns the storage's str-key count plus any non-str-keyed
/// `entries` Vec slots (storage is str-keyed by construction).  This
/// avoids the stale-cache double-count `dict_storage_to_dict` would
/// otherwise produce when a STORE_GLOBAL through the shared storage
/// replaces a pre-existing entry — the entries Vec might still hold
/// the old version, but storage owns the live count.
///
/// PyPy parity: `pypy/interpreter/module.py:77 Module.getdict()`
/// returns the live `W_DictMultiObject`; pyre's split arrangement
/// reproduces that view by treating storage as the source of truth
/// for the keys it represents.
pub unsafe fn w_dict_len(obj: PyObjectRef) -> usize {
    let dict = &*(obj as *const W_DictObject);
    if dict.dict_storage_proxy.is_null() {
        return dict.len;
    }
    let entries = &*dict.entries;
    // Hook missing → storage view is indeterminate; fall back to the
    // entries Vec (keeps str keys visible during bootstrap before
    // `register_dict_storage_items_hook` lands).  Hook installed → use
    // its str-key authoritative count plus any non-str entries Vec slots.
    let Some(storage_items) = maybe_items_dict_storage(dict.dict_storage_proxy) else {
        return dict.len;
    };
    let non_str = entries.iter().filter(|(k, _)| !crate::is_str(*k)).count();
    storage_items.len() + non_str
}

/// Iterate over all (key, value) pairs without type assumptions.
///
/// Storage-authoritative for str keys when proxy is attached: emits
/// the storage's str-keyed entries first, then any non-str-keyed
/// `entries` Vec slots.  Stale str entries cached in the entries Vec
/// (e.g. `dict_storage_to_dict` snapshot taken before a STORE_GLOBAL
/// on the shared storage) are dropped in favour of the storage's
/// live values.
pub unsafe fn w_dict_items(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    if dict.dict_storage_proxy.is_null() {
        return entries.clone();
    }
    // Hook missing → fall back to the entries Vec; hook installed →
    // storage owns str keys and entries Vec contributes non-str slots.
    let Some(storage_items) = maybe_items_dict_storage(dict.dict_storage_proxy) else {
        return entries.clone();
    };
    let mut out: Vec<(PyObjectRef, PyObjectRef)> = storage_items
        .into_iter()
        .map(|(name, value)| (crate::w_str_new(&name), value))
        .collect();
    for &(k, v) in entries.iter() {
        if !crate::is_str(k) {
            out.push((k, v));
        }
    }
    out
}

/// Iterate over (key_str, value) pairs. Keys must be str objects.
///
/// Storage-authoritative for str keys when proxy is attached;
/// non-str entries Vec slots are filtered out per the str-keyed
/// signature.
pub unsafe fn w_dict_str_entries(obj: PyObjectRef) -> Vec<(String, PyObjectRef)> {
    let dict = &*(obj as *const W_DictObject);
    // Hook installed → storage authoritative for str keys.  Hook missing
    // (or proxy unattached) → entries Vec is the only str-key surface.
    if !dict.dict_storage_proxy.is_null() {
        if let Some(items) = maybe_items_dict_storage(dict.dict_storage_proxy) {
            return items;
        }
    }
    let entries = &*dict.entries;
    entries
        .iter()
        .filter(|(k, _)| crate::is_str(*k))
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

    /// `dict_storage_proxy` attached but `register_dict_storage_items_hook`
    /// has not yet been called (direct `w_module_new` callers,
    /// hookless unit/integration tests).  The pre-fix code returned an
    /// empty storage view from `maybe_items_dict_storage` and combined it
    /// only with the *non-str* entries Vec slots, which silently dropped
    /// every str key written through `w_dict_setitem_str`.  The current
    /// behaviour treats the missing hook as "storage view indeterminate"
    /// and falls back to the entries Vec, matching PyPy's
    /// `W_DictMultiObject` single-source-of-truth semantics during the
    /// hookless bootstrap window.
    ///
    /// pyre-object alone has no caller of
    /// `register_dict_storage_items_hook`, so within `cargo test -p
    /// pyre-object` the hook stays null for the lifetime of the test
    /// process — the assertion is therefore stable here.
    #[test]
    fn test_w_dict_proxied_hookless_falls_back_to_entries_vec() {
        let dict = w_dict_new();
        unsafe {
            // Non-null sentinel; the hook never fires because no hook
            // has been registered, so the pointer's pointee is never
            // dereferenced.
            let sentinel: *mut u8 = 0xdead_beef_usize as *mut u8;
            w_dict_set_dict_storage_proxy(dict, sentinel);

            w_dict_setitem_str(dict, "alpha", w_int_new(1));
            w_dict_setitem_str(dict, "beta", w_int_new(2));

            assert_eq!(
                w_dict_len(dict),
                2,
                "hookless proxied dict must expose the entries Vec count, not 0",
            );

            let items = w_dict_items(dict);
            assert_eq!(items.len(), 2);
            let mut keys: Vec<&str> = items
                .iter()
                .map(|&(k, _)| crate::w_str_get_value(k))
                .collect();
            keys.sort();
            assert_eq!(keys, vec!["alpha", "beta"]);

            let mut str_entries = w_dict_str_entries(dict);
            str_entries.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(str_entries.len(), 2);
            assert_eq!(str_entries[0].0, "alpha");
            assert_eq!(w_int_get_value(str_entries[0].1), 1);
            assert_eq!(str_entries[1].0, "beta");
            assert_eq!(w_int_get_value(str_entries[1].1), 2);
        }
    }
}
