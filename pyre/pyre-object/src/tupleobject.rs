//! W_TupleObject — Python `tuple` type.
//!
//! Upstream `pypy/objspace/std/tupleobject.py:376-390` `W_TupleObject`
//! stores `wrappeditems: list` with `_immutable_fields_ =
//! ['wrappeditems[*]']`. After RPython translation `wrappeditems`
//! becomes `Ptr(GcArray(OBJECTPTR))`; the `[*]` annotation marks both
//! the list and its contents as immutable so the JIT can hoist loads.
//! pyre stores the array via `*mut ItemsBlock` (shared GcArray body
//! layer with `W_ListObject`). Length comes directly from the
//! GcArray header — `arraylen_gc(wrappeditems)` on the JIT side and
//! `items_block_capacity(wrappeditems)` on the host side.
//!
//! Arity-2 tuples are routed through specialised variants
//! (`W_SpecialisedTupleObject_{ii,ff,oo}` per
//! `pypy/objspace/std/specialisedtupleobject.py`) — see
//! `makespecialisedtuple2` below. Polymorphic readers
//! (`w_tuple_len`, `w_tuple_getitem`, `w_tuple_items_copy_as_vec`)
//! dispatch on `ob_type` so callers see a uniform tuple API.
//!
//! Tuples are immutable after creation. `wrappeditems` is allocated
//! once via `alloc_tuple_items_block` (exact-size; empty tuple yields
//! a 0-cap header-only block) and never resized.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::floatobject::{w_float_get_value, w_float_new};
use crate::intobject::w_int_new;
use crate::listobject::{is_plain_int1, plain_int_w};
use crate::object_array::{
    ItemsBlock, alloc_tuple_items_block, items_block_capacity, items_block_items_base,
};
use crate::pyobject::*;
use crate::specialisedtupleobject::{
    SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
    W_SpecialisedTupleObject_ff, W_SpecialisedTupleObject_ii, W_SpecialisedTupleObject_oo,
    is_specialised_tuple_ff, is_specialised_tuple_ii, is_specialised_tuple_oo,
    w_specialised_tuple_ff_new, w_specialised_tuple_ii_new, w_specialised_tuple_oo_new,
};

/// Python tuple object — array-backed default representation.
///
/// Layout mirrors `pypy/objspace/std/tupleobject.py:376-390` after
/// RPython translation: `{wrappeditems: Ptr(GcArray(OBJECTPTR))}`.
/// `_immutable_fields_ = ['wrappeditems[*]']` is reflected via
/// `immutable: true` on the `wrappeditems` field descr; the array
/// items are loaded as `getfield_gc_pure_r` and the array length
/// comes from `arraylen_gc` against the GcArray header.
#[repr(C)]
pub struct W_TupleObject {
    pub ob_header: PyObject,
    /// `Ptr(GcArray(OBJECTPTR))` — items body, allocation-immutable
    /// per `_immutable_fields_ = ['wrappeditems[*]']`. The GcArray
    /// header `capacity` IS the live tuple length (rlist.py:251
    /// `len(l.items)`); empty tuples carry a 0-cap header-only
    /// allocation (non-null pointer).
    pub wrappeditems: *mut ItemsBlock,
}

/// GC type id assigned to `W_TupleObject` at `JitDriver` init time.
/// Held as a constant here (rather than runtime-queried) so
/// pyre-object's host-side allocator can reach it without a
/// back-channel; `pyre-jit/src/eval.rs` asserts the same id is
/// returned by `gc.register_type(...)` so any drift panics on
/// startup. Re-exported from `pyre_jit_trace::descr` for existing
/// call sites.
pub const W_TUPLE_GC_TYPE_ID: u32 = 8;
pub const W_TUPLE_OBJECT_SIZE: usize = std::mem::size_of::<W_TupleObject>();

/// Allocate a new tuple from a Vec of items.
///
/// Arity-2 tuples are routed through `makespecialisedtuple2`
/// (`pypy/objspace/std/specialisedtupleobject.py:161-167`); other
/// arities use the array-backed `W_TupleObject`.
pub fn w_tuple_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    if items.len() == 2 {
        return makespecialisedtuple2(items[0], items[1]);
    }
    w_tuple_new_array_backed(items)
}

/// Allocate the array-backed `W_TupleObject` directly, bypassing
/// arity-2 specialisation. Useful for tests and call sites that need
/// the canonical layout.
///
/// Carries `gc_ptr_offsets = [offset_of(wrappeditems)]`
/// (`eval.rs:289`). `wrappeditems` still points at a `std::alloc`'d
/// `ItemsBlock`; the mark walker's
/// `is_managed_heap_object` guard (collector.rs:991/1008) keeps that
/// stepping-stone correctness-safe.
pub fn w_tuple_new_array_backed(items: Vec<PyObjectRef>) -> PyObjectRef {
    // First caller of the `gct_fv_gc_malloc` bracket pattern
    // (`framework.py:853-856`):
    //   livevars = self.push_roots(hop)
    //   v_alloc = hop.genop("direct_call", [malloc_fast_ptr, ...])
    //   self.pop_roots(hop, livevars)
    // Each `items[i]` is a live GC pointer that must survive the
    // potential collection triggered inside `try_gc_alloc_stable`
    // below; the `RootScope` guard truncates the shadow stack on
    // function return mirroring `pop_roots`.
    let _roots = crate::gc_roots::push_roots();
    for &item in &items {
        crate::gc_roots::pin_root(item);
    }

    let items_block = unsafe { alloc_tuple_items_block(&items) };
    let header = PyObject {
        ob_type: &TUPLE_TYPE as *const PyType,
        w_class: get_instantiate(&TUPLE_TYPE),
    };
    if let Some(raw) = crate::gc_hook::try_gc_alloc_stable(W_TUPLE_GC_TYPE_ID, W_TUPLE_OBJECT_SIZE)
        .filter(|p| !p.is_null())
    {
        unsafe {
            std::ptr::write(
                raw as *mut W_TupleObject,
                W_TupleObject {
                    ob_header: header,
                    wrappeditems: items_block,
                },
            );
        }
        return raw as PyObjectRef;
    }
    Box::into_raw(Box::new(W_TupleObject {
        ob_header: header,
        wrappeditems: items_block,
    })) as PyObjectRef
}

/// `pypy/objspace/std/specialisedtupleobject.py:169-179
/// makespecialisedtuple2`. Picks the most specific variant for two
/// args; falls through to `Cls_oo` when neither operand qualifies for
/// the int-int / float-float fast paths.
///
/// Predicates: `listobject.py:2390 is_plain_int1` accepts exact
/// `W_IntObject` (not bool, not int subclass) AND fits-int
/// `W_LongObject`; `type(w) is W_FloatObject` is strict identity.
pub fn makespecialisedtuple2(w_arg1: PyObjectRef, w_arg2: PyObjectRef) -> PyObjectRef {
    unsafe {
        if is_plain_int1(w_arg1) && is_plain_int1(w_arg2) {
            return w_specialised_tuple_ii_new(plain_int_w(w_arg1), plain_int_w(w_arg2));
        }
        if is_plain_float_strict(w_arg1) && is_plain_float_strict(w_arg2) {
            return w_specialised_tuple_ff_new(
                w_float_get_value(w_arg1),
                w_float_get_value(w_arg2),
            );
        }
        w_specialised_tuple_oo_new(w_arg1, w_arg2)
    }
}

/// `type(w) is W_FloatObject`. Strict identity, no subclass match —
/// `specialisedtupleobject.py:176` uses `type(w_arg1) is W_FloatObject`
/// directly with no fits-* extension (no `is_plain_float1` helper
/// upstream).
#[inline]
pub unsafe fn is_plain_float_strict(obj: PyObjectRef) -> bool {
    if !py_type_check(obj, &FLOAT_TYPE) {
        return false;
    }
    let float_typeobj = get_instantiate(&FLOAT_TYPE);
    if float_typeobj.is_null() {
        return (*obj).w_class.is_null();
    }
    let w_class = (*obj).w_class;
    w_class.is_null() || std::ptr::eq(w_class, float_typeobj)
}

/// Get the item at the given index from a tuple — polymorphic over
/// `W_TupleObject` and the three specialised variants.
///
/// Supports negative indexing. Returns None if out of bounds. For
/// `Cls_ii` / `Cls_ff` the unboxed payload is wrapped via
/// `w_int_new` / `w_float_new` (mirrors
/// `specialisedtupleobject.py:138-141 wraps[i](self.space, value)`).
///
/// # Safety
/// `obj` must point to a valid tuple of any of the four variants.
pub unsafe fn w_tuple_getitem(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let len = w_tuple_len(obj) as i64;
    let idx = if index < 0 { index + len } else { index };
    if idx < 0 || idx >= len {
        return None;
    }
    Some(w_tuple_getitem_known(obj, idx as usize))
}

/// Internal: read a tuple item at a known-in-bounds index. Splitting
/// this out lets `w_tuple_items_copy_as_vec` reuse the dispatch.
#[inline]
unsafe fn w_tuple_getitem_known(obj: PyObjectRef, idx: usize) -> PyObjectRef {
    let ob_type = (*obj).ob_type;
    if std::ptr::eq(ob_type, &TUPLE_TYPE) {
        let tuple = &*(obj as *const W_TupleObject);
        let base = items_block_items_base(tuple.wrappeditems);
        *base.add(idx)
    } else if std::ptr::eq(ob_type, &SPECIALISED_TUPLE_II_TYPE) {
        let t = &*(obj as *const W_SpecialisedTupleObject_ii);
        match idx {
            0 => w_int_new(t.value0),
            1 => w_int_new(t.value1),
            _ => unreachable!("bounds guard above"),
        }
    } else if std::ptr::eq(ob_type, &SPECIALISED_TUPLE_FF_TYPE) {
        let t = &*(obj as *const W_SpecialisedTupleObject_ff);
        match idx {
            0 => w_float_new(t.value0),
            1 => w_float_new(t.value1),
            _ => unreachable!("bounds guard above"),
        }
    } else {
        debug_assert!(std::ptr::eq(ob_type, &SPECIALISED_TUPLE_OO_TYPE));
        let t = &*(obj as *const W_SpecialisedTupleObject_oo);
        match idx {
            0 => t.value0,
            1 => t.value1,
            _ => unreachable!("bounds guard above"),
        }
    }
}

/// Get the length of a tuple — polymorphic over all four variants.
/// `Cls_ii` / `Cls_ff` / `Cls_oo` are arity-2 by construction. The
/// canonical `W_TupleObject` reads `len(wrappeditems)` directly from
/// the GcArray header per upstream `tupleobject.py:376-390` (no
/// inline length cache; `_immutable_fields_ = ['wrappeditems[*]']`).
///
/// # Safety
/// `obj` must point to a valid tuple of any of the four variants.
pub unsafe fn w_tuple_len(obj: PyObjectRef) -> usize {
    if is_specialised_tuple_ii(obj) || is_specialised_tuple_ff(obj) || is_specialised_tuple_oo(obj)
    {
        return 2;
    }
    let tuple = &*(obj as *const W_TupleObject);
    items_block_capacity(tuple.wrappeditems)
}

/// Snapshot the tuple's items as an owned `Vec<PyObjectRef>`.
/// Polymorphic over all four variants — `Cls_ii` / `Cls_ff` re-box
/// their inline payloads via `w_int_new` / `w_float_new`.
///
/// # Safety
/// `obj` must point to a valid tuple of any of the four variants.
pub unsafe fn w_tuple_items_copy_as_vec(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let n = w_tuple_len(obj);
    if std::ptr::eq((*obj).ob_type, &TUPLE_TYPE) {
        // Fast path: shared backing array, just copy the slice.
        let tuple = &*(obj as *const W_TupleObject);
        let base = items_block_items_base(tuple.wrappeditems);
        return std::slice::from_raw_parts(base, n).to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(w_tuple_getitem_known(obj, i));
    }
    out
}

#[majit_macros::elidable]
pub extern "C" fn jit_tuple_getitem(tuple: i64, index: i64) -> i64 {
    unsafe {
        match w_tuple_getitem(tuple as PyObjectRef, index) {
            Some(value) => value as i64,
            None => panic!("tuple index out of range in JIT"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    #[test]
    fn test_tuple_create_and_access() {
        let items = vec![w_int_new(1), w_int_new(2), w_int_new(3)];
        let tup = w_tuple_new(items);
        unsafe {
            assert!(is_tuple(tup));
            assert_eq!(w_tuple_len(tup), 3);
            let item = w_tuple_getitem(tup, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 1);
            let item = w_tuple_getitem(tup, 2).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 3);
        }
    }

    #[test]
    fn test_tuple_negative_index() {
        let items = vec![w_int_new(10), w_int_new(20)];
        let tup = w_tuple_new(items);
        unsafe {
            // Arity-2 int-int specialisation is active.
            assert!(is_specialised_tuple_ii(tup));
            assert!(is_tuple(tup));
            let item = w_tuple_getitem(tup, -1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 20);
        }
    }

    #[test]
    fn test_tuple_out_of_bounds() {
        let tup = w_tuple_new(vec![w_int_new(1)]);
        unsafe {
            assert!(w_tuple_getitem(tup, 5).is_none());
            assert!(w_tuple_getitem(tup, -5).is_none());
        }
    }

    #[test]
    fn test_jit_tuple_getitem_shares_tuple_semantics() {
        let tup = w_tuple_new(vec![w_int_new(3), w_int_new(5)]);
        unsafe {
            assert_eq!(
                crate::intobject::w_int_get_value(jit_tuple_getitem(tup as i64, 1) as PyObjectRef),
                5
            );
        }
    }

    #[test]
    fn test_arity2_int_int_routes_to_specialised_ii() {
        let tup = w_tuple_new(vec![w_int_new(7), w_int_new(11)]);
        unsafe {
            assert!(is_specialised_tuple_ii(tup));
            assert!(is_tuple(tup));
            assert_eq!(w_tuple_len(tup), 2);
            let v0 = w_tuple_getitem(tup, 0).unwrap();
            let v1 = w_tuple_getitem(tup, 1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v0), 7);
            assert_eq!(crate::intobject::w_int_get_value(v1), 11);
        }
    }

    #[test]
    fn test_arity2_float_float_routes_to_specialised_ff() {
        let tup = w_tuple_new(vec![
            crate::floatobject::w_float_new(1.5),
            crate::floatobject::w_float_new(2.25),
        ]);
        unsafe {
            assert!(is_specialised_tuple_ff(tup));
            assert!(is_tuple(tup));
            let v0 = w_tuple_getitem(tup, 0).unwrap();
            assert_eq!(crate::floatobject::w_float_get_value(v0), 1.5);
        }
    }

    /// `specialisedtupleobject.py:176` checks `type(w) is W_FloatObject`.
    /// A float subclass keeps the W_FloatObject payload shape but has a
    /// different Python-level `w_class`, so it must fall through to `Cls_oo`.
    #[test]
    fn test_arity2_float_subclass_pair_falls_through_to_oo() {
        let lhs = crate::floatobject::w_float_new(1.5);
        let rhs = crate::floatobject::w_float_new(2.25);
        let fake_subclass = &INSTANCE_TYPE as *const PyType as PyObjectRef;
        unsafe {
            (*lhs).w_class = fake_subclass;
            (*rhs).w_class = fake_subclass;
        }

        let tup = w_tuple_new(vec![lhs, rhs]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(!is_specialised_tuple_ff(tup));
            assert_eq!(w_tuple_getitem(tup, 0).unwrap(), lhs);
            assert_eq!(w_tuple_getitem(tup, 1).unwrap(), rhs);
        }
    }

    /// `listobject.py:2390` checks `type(w) is W_LongObject`, so a
    /// W_LongObject carrying an app-level int subclass must not take the
    /// int-int specialised tuple path.
    #[test]
    fn test_arity2_long_subclass_pair_falls_through_to_oo() {
        let lhs = crate::longobject::w_long_from_i64(7);
        let rhs = crate::longobject::w_long_from_i64(11);
        let fake_subclass = &INSTANCE_TYPE as *const PyType as PyObjectRef;
        unsafe {
            (*lhs).w_class = fake_subclass;
            (*rhs).w_class = fake_subclass;
        }

        let tup = w_tuple_new(vec![lhs, rhs]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(!is_specialised_tuple_ii(tup));
            assert_eq!(w_tuple_getitem(tup, 0).unwrap(), lhs);
            assert_eq!(w_tuple_getitem(tup, 1).unwrap(), rhs);
        }
    }

    #[test]
    fn test_arity2_mixed_falls_through_to_specialised_oo() {
        let tup = w_tuple_new(vec![w_int_new(7), crate::floatobject::w_float_new(2.0)]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(is_tuple(tup));
            assert_eq!(w_tuple_len(tup), 2);
        }
    }

    #[test]
    fn test_arity_other_uses_array_backing() {
        let tup0 = w_tuple_new(vec![]);
        let tup1 = w_tuple_new(vec![w_int_new(1)]);
        let tup3 = w_tuple_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(is_tuple(tup0));
            assert!(is_tuple(tup1));
            assert!(is_tuple(tup3));
            assert!(!is_specialised_tuple_ii(tup0));
            assert!(!is_specialised_tuple_ii(tup1));
            assert!(!is_specialised_tuple_ii(tup3));
            assert_eq!(w_tuple_len(tup0), 0);
            assert_eq!(w_tuple_len(tup1), 1);
            assert_eq!(w_tuple_len(tup3), 3);
        }
    }

    #[test]
    fn test_copy_as_vec_reboxes_unboxed_values() {
        let tup = w_tuple_new(vec![w_int_new(7), w_int_new(11)]);
        unsafe {
            let items = w_tuple_items_copy_as_vec(tup);
            assert_eq!(items.len(), 2);
            assert_eq!(crate::intobject::w_int_get_value(items[0]), 7);
            assert_eq!(crate::intobject::w_int_get_value(items[1]), 11);
        }
    }

    /// `specialisedtupleobject.py:172-175` `is_plain_int1` accepts both
    /// exact `W_IntObject` and fits-int `W_LongObject`. A tuple of two
    /// fits-int longs must therefore land on `Cls_ii`, with the inline
    /// payload carrying the unwrapped i64 values.
    #[test]
    fn test_arity2_long_long_fits_int_routes_to_specialised_ii() {
        use crate::longobject::w_long_new;
        use malachite_bigint::BigInt;
        let tup = w_tuple_new(vec![
            w_long_new(BigInt::from(7)),
            w_long_new(BigInt::from(11)),
        ]);
        unsafe {
            assert!(is_specialised_tuple_ii(tup));
            let v0 = w_tuple_getitem(tup, 0).unwrap();
            let v1 = w_tuple_getitem(tup, 1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v0), 7);
            assert_eq!(crate::intobject::w_int_get_value(v1), 11);
        }
    }

    /// A `W_LongObject` whose value does not fit in a machine int
    /// rejects the int-int specialisation (per `is_plain_int1`'s
    /// `_fits_int()` check) and must fall through to `Cls_oo`.
    #[test]
    fn test_arity2_overflow_long_falls_through_to_oo() {
        use crate::longobject::w_long_new;
        use malachite_bigint::BigInt;
        let huge = BigInt::from(i64::MAX) * BigInt::from(2);
        let tup = w_tuple_new(vec![w_long_new(huge), w_int_new(0)]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(!is_specialised_tuple_ii(tup));
        }
    }

    /// `bool` is a subclass of `int` but `is_plain_int1` rejects it
    /// (`type(w) is W_IntObject`). A `(True, False)` pair therefore
    /// falls through to `Cls_oo`, not `Cls_ii`.
    #[test]
    fn test_arity2_bool_pair_falls_through_to_oo() {
        use crate::boolobject::w_bool_from;
        let tup = w_tuple_new(vec![w_bool_from(true), w_bool_from(false)]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(!is_specialised_tuple_ii(tup));
        }
    }
}
