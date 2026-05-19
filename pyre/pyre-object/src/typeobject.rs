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
    /// PRE-EXISTING-ADAPTATION: in RPython this lives on TypeDef, accessed
    /// via `layout.typedef.acceptable_as_base_class`. Stored on Layout
    /// here because Rust has no TypeDef struct yet — Layout.typedef is
    /// `*const PyType` (≈ CLASSTYPE), and many types share INSTANCE_TYPE
    /// but need different acceptable_as_base_class values.
    /// Convergence: introduce a Rust TypeDef struct, move this field there.
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
    /// typeobject.py:171 `compares_by_identity_status?` —
    /// `UNKNOWN=0`, `COMPARES_BY_IDENTITY=1`,
    /// `OVERRIDES_EQ_CMP_OR_HASH=2`.  Cached result of
    /// `W_TypeObject.compares_by_identity` (`:353-371`); UNKNOWN
    /// until first lookup forces a `__eq__` / `__hash__` MRO walk.
    ///
    /// Invalidated by `baseobjspace::setattr` /
    /// `baseobjspace::delattr` whenever a type-dict entry changes
    /// (matches `typeobject.py:280 mutated()`), which walks
    /// `weak_subclasses` and recurses, so a base-class mutation
    /// eagerly resets cached subclasses.
    pub compares_by_identity_status: std::sync::atomic::AtomicU8,
    /// typeobject.py:640-689 `weak_subclasses` —
    /// per-type list of subclass references populated by
    /// `add_subclass` at heaptype creation time
    /// (`typeobject.py:373-377 ready()` and
    /// `:1604-1613 _add_mro_classes_as_subclasses`).
    ///
    /// PyPy stores `weakref.ref(w_subclass)` entries so subclasses
    /// can be garbage-collected.  Pyre now follows the rweakref
    /// path via `pyre_object::weakref::Weakref` — each slot is a
    /// `*mut Weakref` whose `weakptr` is invalidated by the GC
    /// when the target subclass becomes unreachable
    /// (gctypelayout.py:587, incminimark.py:3058-3126).  The outer
    /// `Vec` is heap-allocated (`Box::into_raw`); the GC's
    /// custom-trace hook registered for `W_TYPE_GC_TYPE_ID` keeps
    /// each `Weakref` struct alive across collections (`pyre-jit
    /// ::eval`).  Null when no subclasses have been registered.
    pub weak_subclasses: *mut Vec<*mut crate::weakref::Weakref>,
}

/// GC type id assigned to `W_TypeObject` at JitDriver init time.
pub const W_TYPE_GC_TYPE_ID: u32 = 33;

/// Fixed payload size (`framework.py:811`).
pub const W_TYPE_OBJECT_SIZE: usize = std::mem::size_of::<W_TypeObject>();

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

    crate::lltype::malloc_typed(W_TypeObject {
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
        flag_map_or_seq: std::sync::atomic::AtomicU8::new(b'?'),
        compares_by_identity_status: std::sync::atomic::AtomicU8::new(COMPARES_BY_IDENTITY_UNKNOWN),
        weak_subclasses: std::ptr::null_mut(),
    }) as PyObjectRef
}

/// typeobject.py:1507-1508 in setup_user_defined_type — copy
/// `flag_map_or_seq` from the first base whose flag is non-`?`.
pub unsafe fn inherit_flag_map_or_seq(w_self: PyObjectRef, bases: PyObjectRef) {
    if w_self.is_null() || bases.is_null() || !is_type(w_self) {
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
        if w_base.is_null() || !is_type(w_base) {
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
        compares_by_identity_status: std::sync::atomic::AtomicU8::new(COMPARES_BY_IDENTITY_UNKNOWN),
        weak_subclasses: std::ptr::null_mut(),
    }) as PyObjectRef
}

/// `dictmultiobject.py:153 UNKNOWN` — cache miss; recompute via
/// `compares_by_identity` lookup.
pub const COMPARES_BY_IDENTITY_UNKNOWN: u8 = 0;
/// `dictmultiobject.py:154 COMPARES_BY_IDENTITY` — type uses
/// object-default `__eq__`/`__hash__`; identity comparison is
/// observable-equivalent.
pub const COMPARES_BY_IDENTITY_YES: u8 = 1;
/// `dictmultiobject.py:155 OVERRIDES_EQ_CMP_OR_HASH` — type defines a
/// custom `__eq__` or `__hash__`; identity comparison is not safe.
pub const COMPARES_BY_IDENTITY_NO: u8 = 2;

/// `typeobject.py:353-371 W_TypeObject.compares_by_identity` —
/// status reader.  Returns the cached value directly without
/// recomputation; callers that need the fresh value invoke the
/// `dict_eq_hook::COMPARES_BY_IDENTITY_HOOK` trampoline which
/// forwards to pyre-interpreter for the MRO walk.
///
/// # Safety
/// `w_type` must be a valid PyObjectRef pointing at a `W_TypeObject`.
pub unsafe fn w_type_compares_by_identity_status(w_type: PyObjectRef) -> u8 {
    if w_type.is_null() || !is_type(w_type) {
        return COMPARES_BY_IDENTITY_NO;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.compares_by_identity_status
        .load(std::sync::atomic::Ordering::Acquire)
}

/// Write-side companion to [`w_type_compares_by_identity_status`].
///
/// # Safety
/// Same as the reader; called by pyre-interpreter's lookup after
/// resolving `__eq__` / `__hash__`.
pub unsafe fn w_type_set_compares_by_identity_status(w_type: PyObjectRef, status: u8) {
    if w_type.is_null() || !is_type(w_type) {
        return;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.compares_by_identity_status
        .store(status, std::sync::atomic::Ordering::Release);
}

/// typeobject.py:169 — `flag_map_or_seq` accessor on a `W_TypeObject`.
/// Returns `'?'` if `w_type` is null, not a type object, or never had
/// the marker assigned.
pub unsafe fn w_type_get_flag_map_or_seq(w_type: PyObjectRef) -> u8 {
    if w_type.is_null() || !is_type(w_type) {
        return b'?';
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_map_or_seq.load(std::sync::atomic::Ordering::Acquire)
}

/// typeobject.py:169 — `flag_map_or_seq` setter.  Used by
/// `init_typeobjects` to mark dict / list / tuple W_TypeObjects at
/// registration time (objspace.py:104-108).
pub unsafe fn w_type_set_flag_map_or_seq(w_type: PyObjectRef, flag: u8) {
    if w_type.is_null() || !is_type(w_type) {
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

/// Get newslotnames from the Layout.
pub unsafe fn w_type_get_newslotnames(obj: PyObjectRef) -> &'static [String] {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        &[]
    } else {
        &(*layout).newslotnames
    }
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

// ── Subclass tree (typeobject.py:640-689) ────────────────────────────

/// `typeobject.py:640-662 W_TypeObject.add_subclass`.
///
/// Records `w_subclass` in `w_parent.weak_subclasses` if not
/// already present.  In PyPy this stores `weakref.ref(w_subclass)`
/// so subclass GC isn't blocked; under `not rweakref` PyPy
/// degrades to a strong-ref list and warns "ALL CLASSES LEAK"
/// (`:642-650`).  Pyre follows the strong-ref fallback because
/// `W_TypeObject` has no weakref wiring yet — a future weakref
/// port can switch this to a weak ref without changing call
/// sites.
///
/// # Safety
/// `w_parent` must point at a valid `W_TypeObject`.  `w_subclass`
/// likewise; the function does not type-check the argument since
/// `ready()` already filters non-type bases (`:374-376`).
pub unsafe fn w_type_add_subclass(w_parent: PyObjectRef, w_subclass: PyObjectRef) {
    if w_parent.is_null() || w_subclass.is_null() {
        return;
    }
    if !is_type(w_parent) || !is_type(w_subclass) {
        return;
    }
    let parent = &mut *(w_parent as *mut W_TypeObject);
    if parent.weak_subclasses.is_null() {
        parent.weak_subclasses = Box::into_raw(Box::new(Vec::new()));
    }
    let subs = &mut *parent.weak_subclasses;
    // typeobject.py:651-660 — `newref = weakref.ref(w_subclass);
    // for i in range(...): if ref() is w_subclass: return; if ref()
    // is None: self.weak_subclasses[i] = newref; return;
    // else: self.weak_subclasses.append(newref)`.
    let newref = crate::weakref::w_weakref_new(w_subclass);
    for i in 0..subs.len() {
        let existing = crate::weakref::w_weakref_deref(subs[i]);
        if existing == w_subclass {
            return;
        }
        if existing.is_null() {
            subs[i] = newref;
            return;
        }
    }
    subs.push(newref);
}

/// `typeobject.py:664-670 W_TypeObject.remove_subclass`.
///
/// Removes `w_subclass` from `w_parent.weak_subclasses` if
/// present; no-op otherwise.  Pointer equality matches PyPy's
/// `ref() is w_subclass`.
///
/// # Safety
/// Same as [`w_type_add_subclass`].
pub unsafe fn w_type_remove_subclass(w_parent: PyObjectRef, w_subclass: PyObjectRef) {
    if w_parent.is_null() || w_subclass.is_null() {
        return;
    }
    if !is_type(w_parent) {
        return;
    }
    let parent = &mut *(w_parent as *mut W_TypeObject);
    if parent.weak_subclasses.is_null() {
        return;
    }
    let subs = &mut *parent.weak_subclasses;
    // typeobject.py:665-669 — `for i in range(len(self
    // .weak_subclasses)): ref = self.weak_subclasses[i]; if ref()
    // is w_subclass: del self.weak_subclasses[i]; return`.
    for i in 0..subs.len() {
        if crate::weakref::w_weakref_deref(subs[i]) == w_subclass {
            subs.remove(i);
            return;
        }
    }
}

/// `typeobject.py:672-689 W_TypeObject.get_subclasses`.
///
/// Returns the recorded direct subclasses.  Under PyPy's weakref
/// path, dead refs are filtered; pyre's strong-ref fallback has
/// no dead entries to filter so the result is a copy of the
/// stored vector.  The `only_real_subclasses` flag from PyPy
/// (`:672-688`) — used by `descr___subclasses__` to filter
/// metaclass-mro override leaks — is omitted because pyre has no
/// `_add_mro_classes_as_subclasses` call site yet; invalidation
/// callers only need the inclusive list.
///
/// # Safety
/// `w_parent` must point at a valid `W_TypeObject`.
pub unsafe fn w_type_get_subclasses(w_parent: PyObjectRef) -> Vec<PyObjectRef> {
    if w_parent.is_null() || !is_type(w_parent) {
        return Vec::new();
    }
    let parent = &*(w_parent as *const W_TypeObject);
    if parent.weak_subclasses.is_null() {
        return Vec::new();
    }
    // typeobject.py:683-686 — `for ref in self.weak_subclasses: w_ob
    // = ref(); if w_ob is not None: subclasses_w.append(w_ob)`.
    let subs = &*parent.weak_subclasses;
    let mut alive: Vec<PyObjectRef> = Vec::with_capacity(subs.len());
    for &slot in subs.iter() {
        let target = crate::weakref::w_weakref_deref(slot);
        if !target.is_null() {
            alive.push(target);
        }
    }
    alive
}

/// `typeobject.py:373-377 W_TypeObject.ready` — register `w_self`
/// as a direct subclass on each W_TypeObject base.  Called once
/// per heap type after `bases` is set, so the subclass tree
/// reflects the class declaration before any attribute lookup.
///
/// # Safety
/// `w_self.bases` must be a valid tuple (or `PY_NULL`).
pub unsafe fn w_type_ready(w_self: PyObjectRef) {
    if w_self.is_null() || !is_type(w_self) {
        return;
    }
    let bases = (*(w_self as *const W_TypeObject)).bases;
    if bases.is_null() {
        return;
    }
    let n = crate::w_tuple_len(bases);
    for i in 0..n as i64 {
        let Some(w_base) = crate::w_tuple_getitem(bases, i) else {
            continue;
        };
        if w_base.is_null() || !is_type(w_base) {
            continue;
        }
        w_type_add_subclass(w_base, w_self);
    }
}

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
