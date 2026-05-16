//! W_ListObject — Python `list` with a minimal PyPy-style strategy split.
//!
//! Homogeneous integer and float lists keep unboxed storage, matching PyPy's
//! `IntegerListStrategy` / `FloatListStrategy` direction. Mixed lists fall back
//! to object storage.
//! The JIT's current raw-array fast path only handles object storage.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use crate::object_array::{
    ItemsBlock, alloc_list_items_block, dealloc_list_items_block, grow_list_items_block,
    items_block_capacity, items_block_items_base,
};
use crate::pyobject::*;
use crate::{
    FloatArray, IntArray, floatobject::w_float_get_value, floatobject::w_float_new,
    intobject::w_int_get_value, intobject::w_int_new, longobject::w_long_fits_int,
    longobject::w_long_get_value,
};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListStrategy {
    Object = 0,
    Integer = 1,
    Float = 2,
    /// listobject.py:1092 EmptyListStrategy — newly created or cleared list
    /// without any storage yet. First append picks a typed strategy via
    /// switch_to_correct_strategy.
    Empty = 3,
}

/// Python list object.
///
/// Layout matches upstream `rpython/rtyper/lltypesystem/rlist.py:116`
/// `GcStruct("list", ("length", Signed), ("items",
/// Ptr(GcArray(OBJECTPTR))))`. The JIT parity-field pair is
/// `(length, items)`; `items` points at an `ItemsBlock` whose
/// offset-0 header holds the allocated capacity
/// (upstream `len(l.items)` per rlist.py:251).
///
/// `strategy`, `int_items`, `float_items` are pyre-only
/// TODOs for PyPy's list strategy split
/// (`pypy/objspace/std/listobject.py`). Only the Object strategy
/// reads/writes `length` + `items`; Integer/Float strategies operate
/// on their own typed arrays and keep `length = 0`, `items = null`.
#[repr(C)]
pub struct W_ListObject {
    pub ob_header: PyObject,
    /// Live length under the Object strategy. Upstream `l.length`
    /// (rlist.py:116). Under Integer/Float strategies this mirrors
    /// `int_items.len()` / `float_items.len()` only when a strategy
    /// switch rewrites both together — typed-strategy operations do
    /// NOT update this field. Callers must read length via
    /// `w_list_len()` which dispatches on strategy.
    pub length: usize,
    /// `Ptr(GcArray(OBJECTPTR))` — rlist.py:116 `l.items`. Points at
    /// the `ItemsBlock` whose offset-0 header is the allocated
    /// capacity (= upstream `len(l.items)` per rlist.py:251). Null
    /// when the list is in a non-Object strategy (Empty/Integer/
    /// Float); lazily allocated on strategy switch.
    pub items: *mut ItemsBlock,
    pub strategy: ListStrategy,
    pub int_items: IntArray,
    pub float_items: FloatArray,
}

/// GC type id assigned to `W_ListObject` at `JitDriver` init time.
/// Held as a constant here (rather than runtime-queried) so
/// pyre-object's host-side allocator can reach it without a
/// back-channel; `pyre-jit/src/eval.rs` asserts the same id is
/// returned by `gc.register_type(...)` so any drift panics on
/// startup. Re-exported from `pyre_jit_trace::descr` for existing
/// call sites.
pub const W_LIST_GC_TYPE_ID: u32 = 7;
pub const W_LIST_OBJECT_SIZE: usize = std::mem::size_of::<W_ListObject>();

impl W_ListObject {
    /// Borrow a slice over object-strategy items. Must only be called
    /// when `self.strategy == ListStrategy::Object`.
    #[inline]
    unsafe fn object_items_slice(&self) -> &[PyObjectRef] {
        let base = items_block_items_base(self.items);
        std::slice::from_raw_parts(base, self.length)
    }

    #[inline]
    unsafe fn object_items_slice_mut(&mut self) -> &mut [PyObjectRef] {
        let base = items_block_items_base(self.items);
        std::slice::from_raw_parts_mut(base, self.length)
    }

    #[inline]
    unsafe fn object_items_capacity(&self) -> usize {
        items_block_capacity(self.items)
    }

    #[inline]
    unsafe fn object_spare_capacity(&self) -> usize {
        self.object_items_capacity().saturating_sub(self.length)
    }

    /// Grow `items` to accommodate at least `min_cap` slots. Upstream
    /// `_ll_list_resize_really` (rlist.py:262-267) — allocate fresh,
    /// copy, swap, free.
    unsafe fn object_grow(&mut self, min_cap: usize) {
        let current_cap = self.object_items_capacity();
        let target_cap = min_cap.max(current_cap.saturating_mul(2).max(4));
        self.items = grow_list_items_block(self.items, target_cap, self.length);
    }

    /// Upstream list.append equivalent for the object strategy.
    /// (listobject.py:1695 `AbstractUnwrappedStrategy.append` for the
    /// Object case: no unwrap, just append.)
    unsafe fn object_push(&mut self, value: PyObjectRef) {
        if self.length == self.object_items_capacity() {
            self.object_grow(self.length + 1);
        }
        let base = items_block_items_base(self.items);
        *base.add(self.length) = value;
        self.length += 1;
    }

    unsafe fn object_insert(&mut self, index: usize, value: PyObjectRef) {
        assert!(index <= self.length);
        if self.length == self.object_items_capacity() {
            self.object_grow(self.length + 1);
        }
        let base = items_block_items_base(self.items);
        let p = base.add(index);
        std::ptr::copy(p, p.add(1), self.length - index);
        *p = value;
        self.length += 1;
    }

    unsafe fn object_remove(&mut self, index: usize) -> PyObjectRef {
        assert!(index < self.length);
        let base = items_block_items_base(self.items);
        let value = *base.add(index);
        let p = base.add(index);
        std::ptr::copy(p.add(1), p, self.length - index - 1);
        self.length -= 1;
        value
    }

    unsafe fn object_pop(&mut self) -> PyObjectRef {
        assert!(self.length > 0);
        let base = items_block_items_base(self.items);
        let value = *base.add(self.length - 1);
        self.length -= 1;
        value
    }

    unsafe fn object_reverse(&mut self) {
        self.object_items_slice_mut().reverse();
    }

    unsafe fn object_drain(&mut self, range: std::ops::Range<usize>) {
        let start = range.start;
        let end = range.end;
        assert!(start <= end && end <= self.length);
        let count = end - start;
        if count == 0 {
            return;
        }
        let base = items_block_items_base(self.items);
        let p = base.add(start);
        std::ptr::copy(p.add(count), p, self.length - end);
        self.length -= count;
    }

    unsafe fn object_splice(
        &mut self,
        start: usize,
        remove_count: usize,
        new_values: &[PyObjectRef],
    ) {
        let old_len = self.length;
        let s = start.min(old_len);
        let slicelength = remove_count.min(old_len - s);
        let len2 = new_values.len();
        let new_len = old_len - slicelength + len2;
        if len2 > slicelength {
            if new_len > self.object_items_capacity() {
                self.object_grow(new_len);
            }
            let base = items_block_items_base(self.items);
            std::ptr::copy(
                base.add(s + slicelength),
                base.add(s + len2),
                old_len - s - slicelength,
            );
            self.length = new_len;
        } else if slicelength > len2 {
            let base = items_block_items_base(self.items);
            std::ptr::copy(
                base.add(s + slicelength),
                base.add(s + len2),
                old_len - s - slicelength,
            );
            self.length = new_len;
        }
        if len2 > 0 {
            self.object_items_slice_mut()[s..s + len2].copy_from_slice(new_values);
        }
    }

    unsafe fn object_to_vec(&self) -> Vec<PyObjectRef> {
        self.object_items_slice().to_vec()
    }

    /// Free the current `items` block and install a freshly allocated
    /// one populated with `values`. `length` is reset to `values.len()`.
    unsafe fn set_object_items_from_vec(&mut self, values: Vec<PyObjectRef>) {
        dealloc_list_items_block(self.items);
        self.items = alloc_list_items_block(&values);
        self.length = values.len();
    }

    /// Drop the object-strategy backing (used when switching to a typed
    /// strategy). Sets `items = null` and `length = 0`.
    unsafe fn drop_object_items(&mut self) {
        dealloc_list_items_block(self.items);
        self.items = std::ptr::null_mut();
        self.length = 0;
    }
}

/// listobject.py:2390-2392 is_plain_int1(w_obj)
///
/// Accepts exact W_IntObject (not bool, not int subclass) or W_LongObject
/// whose value fits in a machine-word integer. Shared with
/// `specialisedtupleobject.py:172-175 makespecialisedtuple2` and the
/// `IntegerListStrategy.is_correct_type` strategy gate
/// (`listobject.py:1957-1958`).
#[inline]
pub unsafe fn is_plain_int1(item: PyObjectRef) -> bool {
    if item.is_null() {
        return false;
    }
    if is_int(item) && !is_bool(item) {
        // type(w_obj) is W_IntObject — reject int subclasses.
        // Subclass instances share ob_type == &INT_TYPE but have w_class
        // overwritten to the subclass type object (typedef.rs:673).
        let int_typeobj = get_instantiate(&INT_TYPE);
        let w_class = (*item).w_class;
        if int_typeobj.is_null() {
            return w_class.is_null();
        }
        if !w_class.is_null() && !std::ptr::eq(w_class, int_typeobj) {
            return false;
        }
        return true;
    }
    if is_long(item) {
        // `type(w_obj) is W_LongObject` — reject app-level int
        // subclasses that reuse the W_LongObject payload layout.
        let int_typeobj = get_instantiate(&INT_TYPE);
        let w_class = (*item).w_class;
        if int_typeobj.is_null() {
            return w_class.is_null() && w_long_fits_int(item);
        }
        if !w_class.is_null() && !std::ptr::eq(w_class, int_typeobj) {
            return false;
        }
        return w_long_fits_int(item);
    }
    false
}

/// listobject.py:2394-2398 `plain_int_w(space, w_obj)`. Unwraps a plain
/// int value from W_IntObject or W_LongObject. Caller must ensure
/// `is_plain_int1(item)` returned true (which for `W_LongObject`
/// implies `_fits_int()`). RPython routes through `w_obj._int_w(space)`
/// (`longobject.py:157`) which raises `OverflowError` on out-of-range
/// values; pyre treats that path as unreachable and panics on
/// precondition violation rather than silently returning 0.
#[inline]
pub(crate) unsafe fn plain_int_w(item: PyObjectRef) -> i64 {
    if is_int(item) {
        w_int_get_value(item)
    } else {
        i64::try_from(w_long_get_value(item)).unwrap_or_else(|_| {
            panic!(
                "plain_int_w: W_LongObject out of i64 range — \
                 is_plain_int1/_fits_int precondition violated \
                 (listobject.py:2394 / longobject.py:157)"
            )
        })
    }
}

/// Check if all items are plain ints for IntegerListStrategy.
fn all_ints(items: &[PyObjectRef]) -> bool {
    items.iter().all(|&item| unsafe { is_plain_int1(item) })
}

fn all_floats(items: &[PyObjectRef]) -> bool {
    items
        .iter()
        .all(|&item| !item.is_null() && unsafe { is_float(item) })
}

fn boxed_from_ints(values: &[i64]) -> Vec<PyObjectRef> {
    values.iter().map(|&value| w_int_new(value)).collect()
}

fn boxed_from_floats(values: &[f64]) -> Vec<PyObjectRef> {
    values.iter().map(|&value| w_float_new(value)).collect()
}

unsafe fn switch_to_object_strategy(list: &mut W_ListObject) {
    if list.strategy == ListStrategy::Object {
        return;
    }
    let seed: Vec<PyObjectRef> = match list.strategy {
        ListStrategy::Integer => boxed_from_ints(list.int_items.as_slice()),
        ListStrategy::Float => boxed_from_floats(list.float_items.as_slice()),
        ListStrategy::Object | ListStrategy::Empty => Vec::new(),
    };
    list.set_object_items_from_vec(seed);
    list.int_items = IntArray::from_vec(Vec::new());
    list.int_items.fix_ptr();
    list.float_items = FloatArray::from_vec(Vec::new());
    list.float_items.fix_ptr();
    list.strategy = ListStrategy::Object;
}

/// listobject.py:1154-1168 EmptyListStrategy.switch_to_correct_strategy
///
/// First append on an empty list picks the typed strategy that matches
/// the appended item, then installs an empty typed storage. Caller is
/// expected to perform the actual append immediately afterward.
unsafe fn switch_to_correct_strategy(list: &mut W_ListObject, w_item: PyObjectRef) {
    if is_plain_int1(w_item) {
        list.int_items = IntArray::from_vec(Vec::new());
        list.int_items.fix_ptr();
        list.strategy = ListStrategy::Integer;
    } else if !w_item.is_null() && is_float(w_item) {
        list.float_items = FloatArray::from_vec(Vec::new());
        list.float_items.fix_ptr();
        list.strategy = ListStrategy::Float;
    } else {
        list.set_object_items_from_vec(Vec::new());
        list.strategy = ListStrategy::Object;
    }
}

/// Allocate a new W_ListObject from a Vec of items.
pub fn w_list_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`):
    // pin every PyObjectRef in `items` before the GC malloc paths
    // below (`alloc_list_items_block`, `try_gc_alloc_stable`) so the
    // shadow stack walker sees them if a collection fires inside the
    // allocator. The Empty / Integer / Float strategies still hold
    // PyObjectRef pointers in `items` until each element is unboxed
    // (`plain_int_w`, `w_float_get_value`); pinning all of them at
    // function entry covers every strategy uniformly.
    let _roots = crate::gc_roots::push_roots();
    for &item in &items {
        crate::gc_roots::pin_root(item);
    }

    // listobject.py:1092 EmptyListStrategy: a freshly created list with no
    // items uses Empty until first append picks a typed strategy.
    let strategy = if items.is_empty() {
        ListStrategy::Empty
    } else if all_ints(&items) {
        ListStrategy::Integer
    } else if all_floats(&items) {
        ListStrategy::Float
    } else {
        ListStrategy::Object
    };
    let (length, items_block, int_items, float_items) = match strategy {
        ListStrategy::Empty | ListStrategy::Integer | ListStrategy::Float => {
            let int_items = if let ListStrategy::Integer = strategy {
                let values = items
                    .iter()
                    .map(|&item| unsafe { plain_int_w(item) })
                    .collect();
                IntArray::from_vec(values)
            } else {
                IntArray::from_vec(Vec::new())
            };
            let float_items = if let ListStrategy::Float = strategy {
                let values = items
                    .iter()
                    .map(|&item| unsafe { w_float_get_value(item) })
                    .collect();
                FloatArray::from_vec(values)
            } else {
                FloatArray::from_vec(Vec::new())
            };
            (0usize, std::ptr::null_mut(), int_items, float_items)
        }
        ListStrategy::Object => {
            let length = items.len();
            let items_block = unsafe { alloc_list_items_block(&items) };
            (
                length,
                items_block,
                IntArray::from_vec(Vec::new()),
                FloatArray::from_vec(Vec::new()),
            )
        }
    };
    let header = PyObject {
        ob_type: &LIST_TYPE as *const PyType,
        w_class: get_instantiate(&LIST_TYPE),
    };
    // Allocate body via GC old-gen (mark-sweep, non-moving). The
    // `items` field carries `gc_ptr_offsets = [offset_of(items)]`
    // (`eval.rs:274`) and still points at a `std::alloc`'d
    // `ItemsBlock`; the mark walker's
    // `is_managed_heap_object` guard (collector.rs:991/1008) keeps
    // that stepping-stone correctness-safe.
    let raw = match crate::gc_hook::try_gc_alloc_stable(W_LIST_GC_TYPE_ID, W_LIST_OBJECT_SIZE)
        .filter(|p| !p.is_null())
    {
        Some(p) => p,
        None => {
            let mut boxed = Box::new(W_ListObject {
                ob_header: header,
                length,
                items: items_block,
                strategy,
                int_items,
                float_items,
            });
            boxed.int_items.fix_ptr();
            boxed.float_items.fix_ptr();
            return Box::into_raw(boxed) as PyObjectRef;
        }
    };
    unsafe {
        std::ptr::write(
            raw as *mut W_ListObject,
            W_ListObject {
                ob_header: header,
                length,
                items: items_block,
                strategy,
                int_items,
                float_items,
            },
        );
        let obj = &mut *(raw as *mut W_ListObject);
        obj.int_items.fix_ptr();
        obj.float_items.fix_ptr();
    }
    raw as PyObjectRef
}

/// Get the item at the given index from a list.
///
/// Supports negative indexing. Returns None if out of bounds.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_getitem(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // listobject.py:1134 EmptyListStrategy.getitem raises IndexError.
        ListStrategy::Empty => None,
        ListStrategy::Object => {
            let items = list.object_items_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(items[idx as usize])
        }
        ListStrategy::Integer => {
            let items = list.int_items.as_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_int_new(items[idx as usize]))
        }
        ListStrategy::Float => {
            let items = list.float_items.as_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_float_new(items[idx as usize]))
        }
    }
}

/// Set the item at the given index in a list.
///
/// Supports negative indexing. Returns false if out of bounds.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_setitem(obj: PyObjectRef, index: i64, value: PyObjectRef) -> bool {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // listobject.py:1185 EmptyListStrategy.setitem raises IndexError.
        ListStrategy::Empty => false,
        ListStrategy::Object => {
            let items = list.object_items_slice_mut();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            items[idx as usize] = value;
            true
        }
        ListStrategy::Integer => {
            let len = list.int_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            // AbstractUnwrappedStrategy.setitem (listobject.py:1737): plain_int_w (unwrap)
            if is_plain_int1(value) {
                list.int_items[idx as usize] = plain_int_w(value);
                true
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(obj, index, value)
            }
        }
        ListStrategy::Float => {
            let len = list.float_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            if !value.is_null() && is_float(value) {
                list.float_items[idx as usize] = w_float_get_value(value);
                true
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(obj, index, value)
            }
        }
    }
}

/// Append an item to a list.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_append(obj: PyObjectRef, value: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // listobject.py:1170 EmptyListStrategy.append: pick the matching
        // typed strategy first, then fall through to its append.
        ListStrategy::Empty => {
            switch_to_correct_strategy(list, value);
            w_list_append(obj, value);
        }
        // AbstractUnwrappedStrategy.append (listobject.py:1695):
        //   if self.is_correct_type(w_item): l.append(self.unwrap(w_item)); return
        //   self.switch_to_next_strategy(w_list, w_item); w_list.append(w_item)
        ListStrategy::Object => list.object_push(value),
        ListStrategy::Integer => {
            if is_plain_int1(value) {
                list.int_items.push(plain_int_w(value));
            } else {
                switch_to_object_strategy(list);
                list.object_push(value);
            }
        }
        ListStrategy::Float => {
            if !value.is_null() && is_float(value) {
                list.float_items.push(w_float_get_value(value));
            } else {
                switch_to_object_strategy(list);
                list.object_push(value);
            }
        }
    }
}

/// Get the length of a list.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_len(obj: PyObjectRef) -> usize {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // listobject.py:1131 EmptyListStrategy.length returns 0.
        ListStrategy::Empty => 0,
        ListStrategy::Object => list.length,
        ListStrategy::Integer => list.int_items.len(),
        ListStrategy::Float => list.float_items.len(),
    }
}

/// Check whether appending one element can complete without reallocating.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_can_append_without_realloc(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // EmptyListStrategy holds no array yet — first append always reallocates.
        ListStrategy::Empty => false,
        ListStrategy::Object => list.object_spare_capacity() > 0,
        ListStrategy::Integer => list.int_items.spare_capacity() > 0,
        ListStrategy::Float => list.float_items.spare_capacity() > 0,
    }
}

/// Check whether the list is currently using inline array storage.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_is_inline_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // EmptyListStrategy.lstorage = self.erase(None) — no backing array.
        ListStrategy::Empty => false,
        // Object strategy stores items in a GC-shaped `ItemsBlock`, never
        // an inline allocation — upstream rlist.py doesn't have an
        // "inline" bit either.
        ListStrategy::Object => false,
        ListStrategy::Integer => list.int_items.is_inline(),
        ListStrategy::Float => list.float_items.is_inline(),
    }
}

pub unsafe fn w_list_uses_object_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Object
}

pub unsafe fn w_list_uses_int_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Integer
}

pub unsafe fn w_list_uses_float_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Float
}

pub unsafe fn w_list_uses_empty_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Empty
}

/// Rebuild the list's object storage from a Vec.
unsafe fn rebuild_object_items(list: &mut W_ListObject, items: Vec<PyObjectRef>) {
    list.set_object_items_from_vec(items);
}

/// Snapshot all items of a list as a `Vec<PyObjectRef>`, regardless of
/// strategy. Integer/Float items are wrapped into `W_IntObject` /
/// `W_FloatObject`, matching listobject.py:363-371
/// `_temporarily_as_objects()`. Used by callers outside `pyre-object`
/// (e.g. the interpreter's unpack / set-update / list-to-tuple paths)
/// that need a uniform object view.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_items_copy_as_vec(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let list = &*(obj as *const W_ListObject);
    temporarily_as_objects(list)
}

/// listobject.py:363-371 _temporarily_as_objects()
///
/// Returns wrapped object items without mutating the source list's strategy.
/// PyPy creates a temporary W_ListObject with ObjectListStrategy; Rust
/// returns a Vec<PyObjectRef> copy instead.
unsafe fn temporarily_as_objects(list: &W_ListObject) -> Vec<PyObjectRef> {
    match list.strategy {
        // listobject.py:1142 EmptyListStrategy.getitems returns [].
        ListStrategy::Empty => Vec::new(),
        ListStrategy::Object => list.object_to_vec(),
        ListStrategy::Integer => list
            .int_items
            .as_slice()
            .iter()
            .map(|&v| w_int_new(v))
            .collect(),
        ListStrategy::Float => list
            .float_items
            .as_slice()
            .iter()
            .map(|&v| w_float_new(v))
            .collect(),
    }
}

fn normalize_insert_index(index: i64, len: usize) -> usize {
    if index < 0 {
        (index + len as i64).max(0) as usize
    } else {
        (index as usize).min(len)
    }
}

/// listobject.py:1712-1720 IntegerListStrategy.insert
/// Strategy-preserving: inserts on typed storage when type matches,
/// switches to Object only when incompatible.
pub unsafe fn w_list_insert(obj: PyObjectRef, index: i64, value: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // EmptyListStrategy doesn't override insert, so it falls through
        // ListStrategy.insert (listobject.py:983) → switches to typed strategy
        // via append. Mirror by switching first then re-dispatching.
        ListStrategy::Empty => {
            switch_to_correct_strategy(list, value);
            w_list_insert(obj, index, value);
        }
        ListStrategy::Integer => {
            if is_plain_int1(value) {
                let idx = normalize_insert_index(index, list.int_items.len());
                list.int_items.insert(idx, plain_int_w(value));
                return;
            }
            switch_to_object_strategy(list);
            w_list_insert(obj, index, value);
        }
        ListStrategy::Float => {
            if !value.is_null() && is_float(value) {
                let idx = normalize_insert_index(index, list.float_items.len());
                list.float_items.insert(idx, w_float_get_value(value));
                return;
            }
            switch_to_object_strategy(list);
            w_list_insert(obj, index, value);
        }
        ListStrategy::Object => {
            let idx = normalize_insert_index(index, list.length);
            list.object_insert(idx, value);
        }
    }
}

/// listobject.py:1850-1862 IntegerListStrategy.pop
/// Strategy-preserving: pops from typed storage, wraps result.
pub unsafe fn w_list_pop(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // listobject.py:1180 EmptyListStrategy.pop raises IndexError.
        ListStrategy::Empty => None,
        ListStrategy::Integer => {
            let len = list.int_items.len() as i64;
            if len == 0 {
                return None;
            }
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            let item = list.int_items.remove(idx as usize);
            Some(w_int_new(item))
        }
        ListStrategy::Float => {
            let len = list.float_items.len() as i64;
            if len == 0 {
                return None;
            }
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            let item = list.float_items.remove(idx as usize);
            Some(w_float_new(item))
        }
        ListStrategy::Object => {
            let len = list.length as i64;
            if len == 0 {
                return None;
            }
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(list.object_remove(idx as usize))
        }
    }
}

/// Remove and return last item. Returns `None` if empty.
pub unsafe fn w_list_pop_end(obj: PyObjectRef) -> Option<PyObjectRef> {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // listobject.py:1180 EmptyListStrategy.pop raises IndexError.
        ListStrategy::Empty => None,
        ListStrategy::Integer => {
            if list.int_items.len() == 0 {
                return None;
            }
            Some(w_int_new(list.int_items.pop()))
        }
        ListStrategy::Float => {
            if list.float_items.len() == 0 {
                return None;
            }
            Some(w_float_new(list.float_items.pop()))
        }
        ListStrategy::Object => {
            if list.length == 0 {
                return None;
            }
            Some(list.object_pop())
        }
    }
}

/// listobject.py:391 W_ListObject.clear — switches to EmptyListStrategy.
///
/// Drops any typed storage and resets the list to the EmptyListStrategy
/// state, exactly like PyPy. The next append will pick a fresh typed
/// strategy via switch_to_correct_strategy.
pub unsafe fn w_list_clear(obj: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    list.drop_object_items();
    list.int_items = IntArray::from_vec(Vec::new());
    list.int_items.fix_ptr();
    list.float_items = FloatArray::from_vec(Vec::new());
    list.float_items.fix_ptr();
    list.strategy = ListStrategy::Empty;
}

/// listobject.py:1873-1874 IntegerListStrategy.reverse
/// Strategy-preserving: reverses typed storage in place.
pub unsafe fn w_list_reverse(obj: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // Empty has nothing to reverse — falls through ListStrategy.reverse
        // (listobject.py defaults) which is a no-op for length 0.
        ListStrategy::Empty => {}
        ListStrategy::Integer => list.int_items.as_mut_slice().reverse(),
        ListStrategy::Float => list.float_items.as_mut_slice().reverse(),
        ListStrategy::Object => list.object_reverse(),
    }
}

/// listobject.py:1814-1844 deleteslice (step=1 simple case)
/// Strategy-preserving: drains from typed storage.
pub unsafe fn w_list_delslice(obj: PyObjectRef, start: usize, end: usize) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // listobject.py:1177 EmptyListStrategy.deleteslice is a no-op (pass).
        ListStrategy::Empty => {}
        ListStrategy::Integer => {
            let len = list.int_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.int_items.drain(s..e);
            }
        }
        ListStrategy::Float => {
            let len = list.float_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.float_items.drain(s..e);
            }
        }
        ListStrategy::Object => {
            let len = list.length;
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.object_drain(s..e);
            }
        }
    }
}

/// listobject.py:1613-1631 IntegerListStrategy._safe_find_or_count
/// Fast path for integer lists: unwrapped comparison.
unsafe fn int_find(items: &[i64], value: i64) -> Option<usize> {
    items.iter().position(|&v| v == value)
}

/// Python int/float cross-type equality: avoids false positives from
/// f64 precision loss (e.g. 2**53+1 != float(2**53)).
#[inline]
fn int_eq_float(ival: i64, fval: f64) -> bool {
    if !fval.is_finite() {
        return false;
    }
    let ival_f = ival as f64;
    if ival_f != fval {
        return false;
    }
    const I64_UPPER_F: f64 = (1u64 << 63) as f64;
    if fval >= I64_UPPER_F || fval < -I64_UPPER_F {
        return false;
    }
    fval as i64 == ival
}

/// listobject.py:2103-2125 FloatListStrategy._safe_find_or_count
/// Fast path for float lists: handles NaN via bit-pattern comparison.
unsafe fn float_find(items: &[f64], value: f64) -> Option<usize> {
    if !value.is_nan() {
        items.iter().position(|&v| v == value)
    } else {
        let bits = value.to_bits();
        items.iter().position(|&v| v.to_bits() == bits)
    }
}

/// Outcome of `W_ListObject.find_or_count` fast path. Mirrors the
/// short-circuit return in `IntegerListStrategy.find_or_count`
/// (listobject.py:1613) and `FloatListStrategy.find_or_count` — when the
/// strategy + needle type match, the typed pool is scanned in place.
/// Otherwise `NeedsGeneric` signals that the caller (pyre-interpreter)
/// must run `ListStrategy.find_or_count`'s generic `space.eq_w` loop.
pub enum ListFindFast {
    /// Fast path applicable, item found at this index (find mode).
    Found(i64),
    /// Fast path applicable, count matched this many times (count mode).
    Count(i64),
    /// Fast path applicable but item not present (find mode).
    NotFound,
    /// Strategy/item type mismatch; caller must run generic eq_w loop.
    NeedsGeneric,
}

/// Typed fast-path for `W_ListObject.find_or_count`. Handles
/// `IntegerListStrategy.find_or_count` (listobject.py:1613) and
/// `FloatListStrategy.find_or_count` (listobject.py:1928) fast paths
/// only. Callers must handle `NeedsGeneric` via the interpreter-level
/// `ListStrategy.find_or_count` which runs the `space.eq_w` loop.
pub unsafe fn w_list_find_or_count_fast(
    obj: PyObjectRef,
    w_item: PyObjectRef,
    start: i64,
    stop: i64,
    count: bool,
) -> ListFindFast {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // listobject.py:1126 EmptyListStrategy.find_or_count: returns
        // `0` in count mode and raises ValueError otherwise. Map the
        // ValueError to NotFound for the find case.
        ListStrategy::Empty => {
            if count {
                ListFindFast::Count(0)
            } else {
                ListFindFast::NotFound
            }
        }
        // listobject.py:1613 IntegerListStrategy.find_or_count: fast path
        // when `is_plain_int1(w_obj)`, else fall back to generic.
        ListStrategy::Integer if is_plain_int1(w_item) => {
            let target = if is_int(w_item) {
                w_int_get_value(w_item)
            } else {
                i64::try_from(w_long_get_value(w_item)).unwrap_or(0)
            };
            let items = list.int_items.as_slice();
            let stop = stop.min(items.len() as i64);
            let mut result: i64 = 0;
            let mut i = start.max(0);
            while i < stop {
                if items[i as usize] == target {
                    if count {
                        result += 1;
                    } else {
                        return ListFindFast::Found(i);
                    }
                }
                i += 1;
            }
            if count {
                ListFindFast::Count(result)
            } else {
                ListFindFast::NotFound
            }
        }
        // listobject.py:1928 FloatListStrategy.find_or_count → base.
        ListStrategy::Float if !w_item.is_null() && is_float(w_item) => {
            let target = w_float_get_value(w_item);
            let items = list.float_items.as_slice();
            let stop = stop.min(items.len() as i64);
            let mut result: i64 = 0;
            let mut i = start.max(0);
            while i < stop {
                if items[i as usize] == target {
                    if count {
                        result += 1;
                    } else {
                        return ListFindFast::Found(i);
                    }
                }
                i += 1;
            }
            if count {
                ListFindFast::Count(result)
            } else {
                ListFindFast::NotFound
            }
        }
        _ => ListFindFast::NeedsGeneric,
    }
}

/// listobject.py:1746-1758 setslice — strategy-preserving.
///
/// When replacement is a list with the same strategy, operates on typed
/// storage directly. Otherwise falls back to Object strategy.
/// `start` and `end` are already normalized (non-negative, clamped).
pub unsafe fn w_list_setslice(
    obj: PyObjectRef,
    start: usize,
    end: usize,
    w_other: PyObjectRef,
) -> Result<(), &'static str> {
    let list = &mut *(obj as *mut W_ListObject);
    if is_list(w_other) {
        let other = &*(w_other as *const W_ListObject);
        // listobject.py:1188 EmptyListStrategy.setslice: adopt donor's
        // strategy and storage wholesale. start/end are 0 because list
        // is empty, so this is just "become a copy of w_other".
        if list.strategy == ListStrategy::Empty {
            match other.strategy {
                ListStrategy::Empty => return Ok(()),
                ListStrategy::Integer => {
                    list.int_items = IntArray::from_vec(other.int_items.to_vec());
                    list.int_items.fix_ptr();
                    list.strategy = ListStrategy::Integer;
                    return Ok(());
                }
                ListStrategy::Float => {
                    list.float_items = FloatArray::from_vec(other.float_items.to_vec());
                    list.float_items.fix_ptr();
                    list.strategy = ListStrategy::Float;
                    return Ok(());
                }
                ListStrategy::Object => {
                    list.set_object_items_from_vec(other.object_to_vec());
                    list.strategy = ListStrategy::Object;
                    return Ok(());
                }
            }
        }
        // listobject.py:1752: not self.list_is_correct_type(w_other) and w_other.length() != 0
        // Only switch strategy when donor is non-empty AND has different type.
        // Empty donor → pure deletion, strategy preserved.
        let other_len = w_list_len(w_other);
        if list.strategy == other.strategy || other_len == 0 {
            match list.strategy {
                ListStrategy::Empty => unreachable!("handled above"),
                ListStrategy::Integer => {
                    let new_items = if list.strategy == other.strategy {
                        other.int_items.as_slice()
                    } else {
                        &[]
                    };
                    let s = start.min(list.int_items.len());
                    let e = end.min(list.int_items.len());
                    if obj == w_other {
                        let mut v = list.int_items.to_vec();
                        v.splice(s..e, new_items.iter().copied());
                        list.int_items = IntArray::from_vec(v);
                        list.int_items.fix_ptr();
                    } else {
                        // RPython AbstractUnwrappedStrategy.setslice mutates
                        // the unerased typed storage directly.
                        list.int_items.splice(s, e - s, new_items);
                    }
                    return Ok(());
                }
                ListStrategy::Float => {
                    let new_items = if list.strategy == other.strategy {
                        other.float_items.as_slice()
                    } else {
                        &[]
                    };
                    let s = start.min(list.float_items.len());
                    let e = end.min(list.float_items.len());
                    if obj == w_other {
                        let mut v = list.float_items.to_vec();
                        v.splice(s..e, new_items.iter().copied());
                        list.float_items = FloatArray::from_vec(v);
                        list.float_items.fix_ptr();
                    } else {
                        // RPython AbstractUnwrappedStrategy.setslice mutates
                        // the unerased typed storage directly.
                        list.float_items.splice(s, e - s, new_items);
                    }
                    return Ok(());
                }
                ListStrategy::Object => {}
            }
        }
    }
    // listobject.py:1751-1753: strategies differ and donor is non-empty →
    // switch to object strategy, then splice as objects.
    let new_items: Vec<PyObjectRef> = if is_list(w_other) {
        let other = &*(w_other as *const W_ListObject);
        temporarily_as_objects(other)
    } else {
        return Err("non-list iterable");
    };
    switch_to_object_strategy(list);
    let mut v = list.object_to_vec();
    let s = start.min(v.len());
    let e = end.min(v.len());
    v.splice(s..e, new_items);
    rebuild_object_items(list, v);
    Ok(())
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_list_append(list: i64, item: i64) -> i64 {
    unsafe { w_list_append(list as PyObjectRef, item as PyObjectRef) };
    0
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_list_getitem(list: i64, index: i64) -> i64 {
    unsafe {
        match w_list_getitem(list as PyObjectRef, index) {
            Some(value) => value as i64,
            None => panic!("list index out of range in JIT"),
        }
    }
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_list_setitem(list: i64, index: i64, value: i64) -> i64 {
    unsafe {
        if !w_list_setitem(list as PyObjectRef, index, value as PyObjectRef) {
            panic!("list assignment index out of range in JIT");
        }
    }
    0
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_list_reverse(list: i64) -> i64 {
    unsafe { w_list_reverse(list as PyObjectRef) };
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    #[test]
    fn test_list_create_and_access() {
        let items = vec![w_int_new(10), w_int_new(20), w_int_new(30)];
        let list = w_list_new(items);
        unsafe {
            assert!(is_list(list));
            assert_eq!(w_list_len(list), 3);
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 10);
            let item = w_list_getitem(list, 2).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 30);
        }
    }

    #[test]
    fn test_list_negative_index() {
        let items = vec![w_int_new(1), w_int_new(2), w_int_new(3)];
        let list = w_list_new(items);
        unsafe {
            let item = w_list_getitem(list, -1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 3);
        }
    }

    #[test]
    fn test_list_setitem() {
        let items = vec![w_int_new(1), w_int_new(2)];
        let list = w_list_new(items);
        unsafe {
            assert!(w_list_setitem(list, 0, w_int_new(99)));
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 99);
        }
    }

    #[test]
    fn test_list_append() {
        let list = w_list_new(vec![]);
        unsafe {
            w_list_append(list, w_int_new(42));
            assert_eq!(w_list_len(list), 1);
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 42);
        }
    }

    #[test]
    fn test_list_out_of_bounds() {
        let list = w_list_new(vec![w_int_new(1)]);
        unsafe {
            assert!(w_list_getitem(list, 5).is_none());
            assert!(w_list_getitem(list, -5).is_none());
            assert!(!w_list_setitem(list, 5, w_int_new(0)));
        }
    }

    #[test]
    fn test_jit_list_helpers_share_list_semantics() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        unsafe {
            assert_eq!(
                crate::intobject::w_int_get_value(jit_list_getitem(list as i64, 1) as PyObjectRef),
                2
            );
        }
        assert_eq!(jit_list_setitem(list as i64, 0, w_int_new(9) as i64), 0);
        assert_eq!(jit_list_append(list as i64, w_int_new(7) as i64), 0);
        unsafe {
            assert_eq!(w_list_len(list), 3);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                9
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 2).unwrap()),
                7
            );
        }
    }

    #[test]
    fn test_list_uses_integer_strategy_for_homogeneous_ints() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            assert!(!w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
        }
    }

    #[test]
    fn test_list_setitem_mixed_value_switches_to_object_strategy() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let float = crate::floatobject::w_float_new(3.5);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            assert!(w_list_setitem(list, 0, float));
            assert!(w_list_uses_object_storage(list));
            let value = w_list_getitem(list, 0).unwrap();
            assert!(crate::pyobject::is_float(value));
        }
    }

    #[test]
    fn test_list_append_mixed_value_switches_to_object_strategy() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let float = crate::floatobject::w_float_new(3.5);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_append(list, float);
            assert!(w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 2).unwrap();
            assert!(crate::pyobject::is_float(value));
        }
    }

    #[test]
    fn test_list_uses_float_strategy_for_homogeneous_floats() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.25),
            crate::floatobject::w_float_new(2.5),
            crate::floatobject::w_float_new(3.75),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            assert!(!w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 1).unwrap();
            assert!(crate::pyobject::is_float(value));
            assert_eq!(crate::floatobject::w_float_get_value(value), 2.5);
        }
    }

    #[test]
    fn test_list_setitem_mixed_on_float_strategy_switches_to_object_strategy() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            assert!(w_list_setitem(list, 0, w_int_new(7)));
            assert!(w_list_uses_object_storage(list));
            let value = w_list_getitem(list, 0).unwrap();
            assert!(crate::pyobject::is_int(value));
        }
    }

    #[test]
    fn test_list_append_mixed_on_float_strategy_switches_to_object_strategy() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            w_list_append(list, w_int_new(7));
            assert!(w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 2).unwrap();
            assert!(crate::pyobject::is_int(value));
        }
    }

    // ── per-strategy operation tests ─────────────────────────────────────────
    // These verify that pop/pop_end/insert/reverse/clear/delslice do NOT
    // switch to ObjectStrategy when the list is homogeneous (int or float).

    #[test]
    fn test_int_list_pop_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.pop (listobject.py:1855)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            let popped = w_list_pop(list, 1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(popped), 2);
            assert!(
                w_list_uses_int_storage(list),
                "pop must not switch strategy"
            );
            assert_eq!(w_list_len(list), 2);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                1
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()),
                3
            );
        }
    }

    #[test]
    fn test_int_list_pop_end_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.pop_end (listobject.py:1848)
        let list = w_list_new(vec![w_int_new(10), w_int_new(20)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            let popped = w_list_pop_end(list).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(popped), 20);
            assert!(
                w_list_uses_int_storage(list),
                "pop_end must not switch strategy"
            );
            assert_eq!(w_list_len(list), 1);
        }
    }

    #[test]
    fn test_int_list_insert_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.insert (listobject.py:1714)
        let list = w_list_new(vec![w_int_new(1), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_insert(list, 1, w_int_new(2));
            assert!(
                w_list_uses_int_storage(list),
                "insert int must not switch strategy"
            );
            assert_eq!(w_list_len(list), 3);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()),
                2
            );
        }
    }

    #[test]
    fn test_int_list_insert_float_switches_to_object() {
        // AbstractUnwrappedStrategy.switch_to_next_strategy (listobject.py:1720)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let fv = crate::floatobject::w_float_new(9.0);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_insert(list, 1, fv);
            assert!(w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
        }
    }

    #[test]
    fn test_int_list_reverse_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.reverse (listobject.py:1880)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_reverse(list);
            assert!(
                w_list_uses_int_storage(list),
                "reverse must not switch strategy"
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                3
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 2).unwrap()),
                1
            );
        }
    }

    #[test]
    fn test_new_empty_uses_empty_strategy() {
        // listobject.py:1092 fresh empty list uses EmptyListStrategy.
        let list = w_list_new(Vec::new());
        unsafe {
            assert!(w_list_uses_empty_storage(list));
            assert_eq!(w_list_len(list), 0);
        }
    }

    #[test]
    fn test_clear_resets_to_empty_strategy() {
        // listobject.py:391 W_ListObject.clear → EmptyListStrategy.
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_clear(list);
            assert!(
                w_list_uses_empty_storage(list),
                "clear must switch to EmptyListStrategy"
            );
            assert_eq!(w_list_len(list), 0);
        }
    }

    #[test]
    fn test_empty_first_int_append_switches_to_int_strategy() {
        // listobject.py:1170 EmptyListStrategy.append picks the typed strategy
        // matching the first item.
        let list = w_list_new(Vec::new());
        unsafe {
            assert!(w_list_uses_empty_storage(list));
            w_list_append(list, w_int_new(7));
            assert!(w_list_uses_int_storage(list));
            assert_eq!(w_list_len(list), 1);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                7
            );
        }
    }

    #[test]
    fn test_empty_first_float_append_switches_to_float_strategy() {
        let list = w_list_new(Vec::new());
        unsafe {
            assert!(w_list_uses_empty_storage(list));
            w_list_append(list, crate::floatobject::w_float_new(2.5));
            assert!(w_list_uses_float_storage(list));
            assert_eq!(w_list_len(list), 1);
        }
    }

    #[test]
    fn test_int_list_delslice_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.deleteslice (listobject.py:1815)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3), w_int_new(4)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_delslice(list, 1, 3);
            assert!(
                w_list_uses_int_storage(list),
                "delslice must not switch strategy"
            );
            assert_eq!(w_list_len(list), 2);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                1
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()),
                4
            );
        }
    }

    #[test]
    fn test_float_list_pop_stays_float_strategy() {
        // AbstractUnwrappedStrategy.pop (listobject.py:1855)
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
            crate::floatobject::w_float_new(3.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            let popped = w_list_pop(list, 0).unwrap();
            assert_eq!(crate::floatobject::w_float_get_value(popped), 1.0);
            assert!(
                w_list_uses_float_storage(list),
                "pop must not switch strategy"
            );
            assert_eq!(w_list_len(list), 2);
        }
    }

    #[test]
    fn test_float_list_reverse_stays_float_strategy() {
        // AbstractUnwrappedStrategy.reverse (listobject.py:1880)
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            w_list_reverse(list);
            assert!(
                w_list_uses_float_storage(list),
                "reverse must not switch strategy"
            );
            assert_eq!(
                crate::floatobject::w_float_get_value(w_list_getitem(list, 0).unwrap()),
                2.0
            );
        }
    }
}
