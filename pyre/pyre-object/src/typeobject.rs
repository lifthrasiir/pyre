//! W_TypeObject — Python `type` object for user-defined classes.
//!
//! PyPy equivalent: pypy/objspace/std/typeobject.py → W_TypeObject
//!
//! A type object holds the class name, tuple of base types, and a namespace
//! dict containing class-level attributes and methods.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// typeobject.py:103-129 Layout object.
///
/// Immutable after creation. Shared between types that have the same
/// instance layout (e.g. a class without __slots__ shares its base's layout).
/// Identity comparison via pointer equality.
pub struct Layout {
    /// typeobject.py:113 — the typedef (PyType) that this layout is for.
    pub typedef: *const PyType,
    /// typeobject.py:114 — total number of extra slots.
    pub nslots: u32,
    /// typeobject.py:115 — sorted list of slot names introduced by this class.
    pub newslotnames: Vec<String>,
    /// typeobject.py:116 — parent layout (identity comparison).
    pub base_layout: *const Layout,
    /// typedef.py:43 — `acceptable_as_base_class = '__new__' in rawdict`.
    /// Stored on Layout (≈ TypeDef level) so the check goes through
    /// w_bestbase.layout, not the type object itself.
    pub acceptable_as_base_class: bool,
}

impl Layout {
    /// typeobject.py:118-123 issublayout(parent):
    ///   while self is not parent:
    ///       self = self.base_layout
    ///       if self is None: return False
    ///   return True
    pub fn issublayout(&self, parent: *const Layout) -> bool {
        let mut current = self as *const Layout;
        while current != parent {
            let cur = unsafe { &*current };
            if cur.base_layout.is_null() {
                return false;
            }
            current = cur.base_layout;
        }
        true
    }

    /// typeobject.py:125-129 expand(hasdict, weakrefable):
    ///   return (self.typedef, self.newslotnames, self.base_layout,
    ///           hasdict, weakrefable)
    ///
    /// Two types have compatible layouts iff their expand() tuples are equal.
    pub fn expands_equal(
        a: *const Layout,
        a_hasdict: bool,
        a_weakrefable: bool,
        b: *const Layout,
        b_hasdict: bool,
        b_weakrefable: bool,
    ) -> bool {
        if a == b {
            // Same Layout object → typedef, newslotnames, base_layout all identical.
            return a_hasdict == b_hasdict && a_weakrefable == b_weakrefable;
        }
        if a.is_null() || b.is_null() {
            return false;
        }
        let la = unsafe { &*a };
        let lb = unsafe { &*b };
        std::ptr::eq(la.typedef, lb.typedef)
            && la.newslotnames == lb.newslotnames
            && la.base_layout == lb.base_layout
            && a_hasdict == b_hasdict
            && a_weakrefable == b_weakrefable
    }
}

/// Python type object (user-defined class).
///
/// PyPy: pypy/objspace/std/typeobject.py W_TypeObject
#[repr(C)]
pub struct W_TypeObject {
    pub ob_header: PyObject,
    /// Class name (heap-allocated, leaked).
    pub name: *mut String,
    /// Tuple of base type objects (PyObjectRef → W_TupleObject or PY_NULL).
    pub bases: PyObjectRef,
    /// Raw pointer to the class dict backing storage (`dict_w` analogue).
    pub dict: *mut u8,
    /// Cached C3 MRO — W_TypeObject.mro_w.
    pub mro_w: *mut Vec<PyObjectRef>,
    /// typeobject.py:184 `flag_heaptype` — immutable after creation.
    pub flag_heaptype: bool,
    /// typeobject.py:195 `layout` — pointer to shared Layout object.
    pub layout: *const Layout,
    /// typeobject.py:179 `hasdict` — True when instances have __dict__.
    pub hasdict: bool,
    /// typeobject.py:181 `weakrefable` — True when instances support weakrefs.
    pub weakrefable: bool,
    /// typeobject.py:169 `flag_map_or_seq` (`'?'`, `'M'`, `'S'`).
    ///
    /// Default `'?'` per typeobject.py:216.  Inherited from base
    /// classes during heap-type construction (typeobject.py:1495):
    /// when self's flag is `'?'` and a base's flag is non-`'?'`, copy.
    /// Used by `descroperation.py:319-326 is_iterable` and `:330-346
    /// iter` to skip the `__getitem__` fallback for mapping-typed
    /// classes.  Stored on `W_TypeObject` (not the low-level
    /// `PyType`) so user-defined `dict`/`list`/`tuple` subclasses
    /// inherit the marker the same way PyPy does.
    pub flag_map_or_seq: std::sync::atomic::AtomicU8,
}

/// Field offset of `bases` within `W_TypeObject`.
pub const TYPE_BASES_OFFSET: usize = std::mem::offset_of!(W_TypeObject, bases);

/// GC type id assigned to `W_TypeObject` at JitDriver init time.
pub const W_TYPE_GC_TYPE_ID: u32 = 33;

/// Fixed payload size (`framework.py:811`).
pub const W_TYPE_OBJECT_SIZE: usize = std::mem::size_of::<W_TypeObject>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
/// Only `bases` is a `PyObjectRef` — `name` (`*mut String`),
/// `dict` (`*mut u8`), `mro_w` (`*mut Vec<PyObjectRef>`), and `layout`
/// (`*const Layout`) are non-PyObject heap allocations.
pub const W_TYPE_GC_PTR_OFFSETS: [usize; 1] = [TYPE_BASES_OFFSET];

impl crate::lltype::GcType for W_TypeObject {
    const TYPE_ID: u32 = W_TYPE_GC_TYPE_ID;
    const SIZE: usize = W_TYPE_OBJECT_SIZE;
}

/// Leak a Layout to get a 'static pointer for sharing.
pub fn leak_layout(layout: Layout) -> *const Layout {
    crate::lltype::malloc_raw(layout)
}

/// Allocate a new W_TypeObject with `flag_heaptype = true`.
///
/// typeobject.py:174 `__init__(..., is_heaptype=True)`.
/// Layout is set to null initially; caller must set it via set_layout
/// after running create_all_slots / setup_builtin_type.
pub fn w_type_new(name: &str, bases: PyObjectRef, dict_ptr: *mut u8) -> PyObjectRef {
    let name = crate::lltype::malloc_raw(name.to_string());
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(bases);

    let w_self = crate::lltype::malloc_typed(W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        mro_w: std::ptr::null_mut(),
        name,
        bases,
        dict: dict_ptr,
        flag_heaptype: true,
        layout: std::ptr::null(),
        hasdict: false,
        weakrefable: false,
        // typeobject.py:216 default — inheritance walk below may
        // overwrite from a non-`?` base.
        flag_map_or_seq: std::sync::atomic::AtomicU8::new(b'?'),
    }) as PyObjectRef;
    // typeobject.py:1493-1496 — `for w_base in w_self.bases_w: if
    // w_self.flag_map_or_seq == '?': w_self.flag_map_or_seq =
    // w_base.flag_map_or_seq`.  Heap-type construction inherits the
    // marker from the first non-`?` base so user-defined dict / list
    // / tuple subclasses see PyPy's `M` / `S` flag.
    unsafe { inherit_flag_map_or_seq(w_self, bases) };
    w_self
}

/// typeobject.py:1493-1496 — copy `flag_map_or_seq` from the first
/// W_TypeObject base whose flag is non-`?`.  No-op if self's flag is
/// already non-`?` (explicit value wins) or if no base has a non-`?`
/// flag.
unsafe fn inherit_flag_map_or_seq(w_self: PyObjectRef, bases: PyObjectRef) {
    if w_self.is_null() || bases.is_null() {
        return;
    }
    let self_ref = &*(w_self as *const W_TypeObject);
    if self_ref
        .flag_map_or_seq
        .load(std::sync::atomic::Ordering::Acquire)
        != b'?'
    {
        return;
    }
    let n = crate::w_tuple_len(bases);
    for i in 0..n as i64 {
        let Some(w_base) = crate::w_tuple_getitem(bases, i) else {
            continue;
        };
        if w_base.is_null() || !is_w_type(w_base) {
            continue;
        }
        let base_ref = &*(w_base as *const W_TypeObject);
        let base_flag = base_ref
            .flag_map_or_seq
            .load(std::sync::atomic::Ordering::Acquire);
        if base_flag != b'?' {
            self_ref
                .flag_map_or_seq
                .store(base_flag, std::sync::atomic::Ordering::Release);
            return;
        }
    }
}

/// Identity test mirroring `pyre_object::is_type` (the user-facing
/// helper in `pyobject.rs`).  Inlined here to avoid a circular module
/// dependency.
unsafe fn is_w_type(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    std::ptr::eq((*obj).ob_type, &TYPE_TYPE as *const PyType)
}

/// Allocate a new W_TypeObject with `flag_heaptype = false`.
///
/// typeobject.py:174 `__init__(..., is_heaptype=False)`.
pub fn w_type_new_builtin(
    name: &str,
    bases: PyObjectRef,
    dict_ptr: *mut u8,
    _layout_pytype: *const PyType,
) -> PyObjectRef {
    let name = crate::lltype::malloc_raw(name.to_string());
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(bases);

    crate::lltype::malloc_typed(W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        mro_w: std::ptr::null_mut(),
        name,
        bases,
        dict: dict_ptr,
        flag_heaptype: false,
        layout: std::ptr::null(),
        hasdict: false,
        weakrefable: false,
        // typeobject.py:216 default; built-in dict/list/tuple
        // override via `w_type_set_flag_map_or_seq` at typedef
        // registration time (see `typedef.rs`).
        flag_map_or_seq: std::sync::atomic::AtomicU8::new(b'?'),
    }) as PyObjectRef
}

/// typeobject.py:169 — `flag_map_or_seq` accessor on a `W_TypeObject`.
/// Returns `'?'` if `w_type` is null, not a type object, or never had
/// the marker assigned.
pub unsafe fn w_type_get_flag_map_or_seq(w_type: PyObjectRef) -> u8 {
    if w_type.is_null() || !is_w_type(w_type) {
        return b'?';
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_map_or_seq.load(std::sync::atomic::Ordering::Acquire)
}

/// typeobject.py:169 — `flag_map_or_seq` setter.  Used by
/// `init_typeobjects` to mark dict / list / tuple W_TypeObjects at
/// registration time (objspace.py:104-108).
pub unsafe fn w_type_set_flag_map_or_seq(w_type: PyObjectRef, flag: u8) {
    if w_type.is_null() || !is_w_type(w_type) {
        return;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_map_or_seq
        .store(flag, std::sync::atomic::Ordering::Release);
}

// ── Layout accessors ─────────────────────────────────────────────────

/// Set the Layout pointer on a type object.
pub unsafe fn w_type_set_layout(obj: PyObjectRef, layout: *const Layout) {
    (*(obj as *mut W_TypeObject)).layout = layout;
}

/// Get the Layout pointer from a type object.
pub unsafe fn w_type_get_layout_ptr(obj: PyObjectRef) -> *const Layout {
    (*(obj as *const W_TypeObject)).layout
}

/// typeobject.py:336-337 get_full_instance_layout(self).
/// Returns the Layout.typedef pointer (the PyType describing instance struct).
/// For backward-compat with existing code that compares PyType pointers.
#[inline]
pub unsafe fn w_type_get_layout(obj: PyObjectRef) -> *const PyType {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        &INSTANCE_TYPE as *const PyType
    } else {
        (*layout).typedef
    }
}

/// Get nslots from the Layout.
pub unsafe fn w_type_get_nslots(obj: PyObjectRef) -> u32 {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        0
    } else {
        (*layout).nslots
    }
}

/// Backward-compat: set nslots on the Layout (creates a new Layout if needed).
pub unsafe fn w_type_set_nslots(_obj: PyObjectRef, _n: u32) {
    // No-op: nslots is set via Layout construction in create_all_slots.
}

/// Get newslotnames from the Layout.
pub unsafe fn w_type_get_newslotnames(obj: PyObjectRef) -> &'static [String] {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        &[]
    } else {
        &(*layout).newslotnames
    }
}

/// Backward-compat alias.
pub unsafe fn w_type_set_newslotnames(_obj: PyObjectRef, _names: Vec<String>) {
    // No-op: newslotnames set via Layout construction.
}

/// Get base_layout pointer for identity comparison.
pub unsafe fn w_type_get_base_layout(obj: PyObjectRef) -> *const Layout {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        std::ptr::null()
    } else {
        (*layout).base_layout
    }
}

/// typeobject.py:179 `hasdict` getter/setter.
pub unsafe fn w_type_get_hasdict(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).hasdict
}
pub unsafe fn w_type_set_hasdict(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).hasdict = v;
}

/// typeobject.py:181 `weakrefable` getter/setter.
pub unsafe fn w_type_get_weakrefable(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).weakrefable
}
pub unsafe fn w_type_set_weakrefable(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).weakrefable = v;
}

// ── Other accessors ──────────────────────────────────────────────────

/// Get the class name.
pub unsafe fn w_type_get_name(obj: PyObjectRef) -> &'static str {
    &*(*(obj as *const W_TypeObject)).name
}

/// Get the bases tuple.
pub unsafe fn w_type_get_bases(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_TypeObject)).bases
}

/// Get the class namespace pointer (as *mut u8).
pub unsafe fn w_type_get_dict_ptr(obj: PyObjectRef) -> *mut u8 {
    (*(obj as *const W_TypeObject)).dict
}

/// Get the cached MRO, or null if not yet set.
pub unsafe fn w_type_get_mro(obj: PyObjectRef) -> *mut Vec<PyObjectRef> {
    (*(obj as *const W_TypeObject)).mro_w
}

/// Set the cached MRO.
pub unsafe fn w_type_set_mro(obj: PyObjectRef, mro: Vec<PyObjectRef>) {
    (*(obj as *mut W_TypeObject)).mro_w = crate::lltype::malloc_raw(mro);
}

/// Check if an object is a type (user-defined class).
#[inline]
pub unsafe fn is_type(obj: PyObjectRef) -> bool {
    py_type_check(obj, &TYPE_TYPE)
}

/// typeobject.py:543-544 `is_heaptype(self)`.
#[inline]
pub unsafe fn w_type_is_heaptype(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).flag_heaptype
}

/// typedef.py:43 `acceptable_as_base_class` — read from Layout level.
/// typeobject.py:1116: w_bestbase.layout.typedef.acceptable_as_base_class
pub unsafe fn w_type_get_acceptable_as_base_class(obj: PyObjectRef) -> bool {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        true
    } else {
        (*layout).acceptable_as_base_class
    }
}
/// Override acceptable_as_base_class by cloning the Layout.
/// typedef.py:742,765,664 explicit overrides after initial creation.
/// Layouts may be shared (reused from parent), so we clone to avoid
/// corrupting the parent type's flag.
pub unsafe fn w_type_set_acceptable_as_base_class(obj: PyObjectRef, v: bool) {
    let old_layout = (*(obj as *const W_TypeObject)).layout;
    if old_layout.is_null() {
        return;
    }
    let old = &*old_layout;
    if old.acceptable_as_base_class == v {
        return; // already correct
    }
    // Clone with new value to avoid mutating shared Layout.
    let new_layout = leak_layout(Layout {
        typedef: old.typedef,
        nslots: old.nslots,
        newslotnames: old.newslotnames.clone(),
        base_layout: old.base_layout,
        acceptable_as_base_class: v,
    });
    (*(obj as *mut W_TypeObject)).layout = new_layout;
}

// Backward-compat no-ops for removed direct field setters.
pub unsafe fn w_type_set_base_layout(_obj: PyObjectRef, _base: PyObjectRef) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_create_and_check() {
        let obj = w_type_new("Foo", PY_NULL, std::ptr::null_mut());
        unsafe {
            assert!(is_type(obj));
            assert!(!is_int(obj));
            assert_eq!(w_type_get_name(obj), "Foo");
            assert!(w_type_get_dict_ptr(obj).is_null());
        }
    }

    #[test]
    fn test_layout_issublayout() {
        let root = leak_layout(Layout {
            typedef: &INSTANCE_TYPE,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            acceptable_as_base_class: true,
        });
        let child = leak_layout(Layout {
            typedef: &INSTANCE_TYPE,
            nslots: 1,
            newslotnames: vec!["x".to_string()],
            base_layout: root,
            acceptable_as_base_class: true,
        });
        unsafe {
            assert!((*child).issublayout(root));
            assert!((*root).issublayout(root));
            assert!(!(*root).issublayout(child));
        }
    }

    #[test]
    fn test_layout_expand_equality() {
        let root = leak_layout(Layout {
            typedef: &INSTANCE_TYPE,
            nslots: 1,
            newslotnames: vec!["x".to_string()],
            base_layout: std::ptr::null(),
            acceptable_as_base_class: true,
        });
        // Same Layout pointer → equal
        assert!(Layout::expands_equal(root, true, true, root, true, true));
        // Different hasdict → not equal
        assert!(!Layout::expands_equal(root, true, true, root, false, true));
    }

    #[test]
    fn w_type_gc_type_id_matches_descr() {
        assert_eq!(W_TYPE_GC_TYPE_ID, 33);
        assert_eq!(
            <W_TypeObject as crate::lltype::GcType>::TYPE_ID,
            W_TYPE_GC_TYPE_ID
        );
        assert_eq!(
            <W_TypeObject as crate::lltype::GcType>::SIZE,
            W_TYPE_OBJECT_SIZE
        );
    }
}
