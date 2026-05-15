//! pypy/objspace/std/mapdict.py
//!
//! Mapdict provides per-instance dict and weakref slots for hasdict /
//! weakrefable types. PyPy stores these inside the mapdict map's "dict"
//! and "weakref" SPECIAL slots; pyre keeps thread-local side tables
//! keyed by object address because pyre has no mapdict.
//!
//! The names below mirror PyPy: `MapdictDictSupport.getdict` →
//! `_obj_getdict`, `MapdictWeakrefSupport.setweakref` →
//! `_mapdict_setweakref`, etc.

use crate::PyError;
use pyre_object::PyObjectRef;

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// objspace/std/mapdict.py:830 — MapdictDictSupport stores the
    /// instance dict in the "dict" SPECIAL slot of the mapdict map.
    /// pyre keeps a side table of address → W_DictObject because there
    /// is no mapdict; semantically this is the same backing store.
    pub static INSTANCE_DICT: RefCell<HashMap<usize, PyObjectRef>> =
        RefCell::new(HashMap::new());
}

thread_local! {
    /// objspace/std/mapdict.py:780-797 MapdictWeakrefSupport stores the
    /// lifeline in the "weakref" SPECIAL slot of the mapdict map. pyre
    /// keeps a side table because there is no mapdict; semantically
    /// this is the same per-instance lifeline storage.
    pub static WEAKREF_TABLE: RefCell<HashMap<usize, PyObjectRef>> =
        RefCell::new(HashMap::new());
}

// ── MapdictDictSupport ────────────────────────────────────────────────

/// objspace/std/mapdict.py:826-840 _obj_getdict.
///
/// ```python
/// @objectmodel.dont_inline
/// def _obj_getdict(self, space):
///     terminator = self._get_mapdict_map().terminator
///     assert isinstance(terminator, DictTerminator) or isinstance(terminator, DevolvedDictTerminator)
///     w_dict = self._get_mapdict_map().read(self, "dict", SPECIAL)
///     if w_dict is not None:
///         assert isinstance(w_dict, W_DictMultiObject)
///         return w_dict
///
///     strategy = space.fromcache(MapDictStrategy)
///     storage = strategy.erase(self)
///     w_dict = W_DictObject(space, strategy, storage)
///     flag = self._get_mapdict_map().write(self, "dict", SPECIAL, w_dict)
///     assert flag
///     return w_dict
/// ```
pub fn _obj_getdict(self_ref: PyObjectRef) -> PyObjectRef {
    let existing = INSTANCE_DICT.with(|table| table.borrow().get(&(self_ref as usize)).copied());
    if let Some(w_dict) = existing {
        return w_dict;
    }
    // PyPy stores this in the mapdict "dict" SPECIAL slot. pyre's temporary
    // mapdict adapter is an address-keyed side table; keep the holder
    // GC-managed so a user-held old __dict__ remains traceable after
    // _obj_setdict replaces the side-table entry.
    let w_dict = pyre_object::w_dict_new();
    INSTANCE_DICT.with(|table| {
        table.borrow_mut().insert(self_ref as usize, w_dict);
    });
    w_dict
}

fn current_owner_key(key: usize) -> usize {
    pyre_object::gc_hook::try_gc_current_object_address(key as *mut u8) as usize
}

/// Walk roots held by pyre's temporary mapdict side tables.
///
/// PyPy stores the instance dict and weakref lifeline in mapdict SPECIAL slots,
/// so the translated GC sees them as ordinary object fields. pyre keeps the
/// same logical data in address-keyed side tables until mapdict is ported into
/// the object layout; expose the value slots here so the backend GC can update
/// them when nursery objects move.
pub fn walk_mapdict_roots(mut visitor: impl FnMut(&mut PyObjectRef)) {
    let dict_values = INSTANCE_DICT.with(|table| {
        table
            .borrow()
            .iter()
            .map(|(&key, &dict)| (key, dict))
            .collect::<Vec<_>>()
    });
    // SAFETY: do not hold the RefCell borrow while invoking callbacks. The
    // visitor and w_dict_walk_entries_mut may re-enter mapdict/dict APIs.
    for (key, mut dict) in dict_values {
        visitor(&mut dict);
        let new_key = current_owner_key(key);
        INSTANCE_DICT.with(|table| {
            let mut table = table.borrow_mut();
            if new_key == key {
                if let Some(slot) = table.get_mut(&key) {
                    *slot = dict;
                }
            } else if table.remove(&key).is_some() {
                table.insert(new_key, dict);
            }
        });
        unsafe {
            pyre_object::w_dict_walk_entries_mut(dict, |slot| {
                visitor(slot);
            });
        }
    }
    let weakref_values = WEAKREF_TABLE.with(|table| {
        table
            .borrow()
            .iter()
            .map(|(&key, &value)| (key, value))
            .collect::<Vec<_>>()
    });
    for (key, mut value) in weakref_values {
        visitor(&mut value);
        let new_key = current_owner_key(key);
        WEAKREF_TABLE.with(|table| {
            let mut table = table.borrow_mut();
            if new_key == key {
                if let Some(slot) = table.get_mut(&key) {
                    *slot = value;
                }
            } else if table.remove(&key).is_some() {
                table.insert(new_key, value);
            }
        });
    }
}

/// objspace/std/mapdict.py:842-860 _obj_setdict.
///
/// ```python
/// @objectmodel.dont_inline
/// def _obj_setdict(self, space, w_dict):
///     from pypy.interpreter.error import oefmt
///     terminator = self._get_mapdict_map().terminator
///     assert isinstance(terminator, DictTerminator) or isinstance(terminator, DevolvedDictTerminator)
///     if not space.isinstance_w(w_dict, space.w_dict):
///         raise oefmt(space.w_TypeError, "setting dictionary to a non-dict")
///     assert isinstance(w_dict, W_DictMultiObject)
///     w_olddict = self.getdict(space)
///     ...
///     flag = self._get_mapdict_map().write(self, "dict", SPECIAL, w_dict)
///     assert flag
/// ```
pub fn _obj_setdict(self_ref: PyObjectRef, w_dict: PyObjectRef) -> Result<(), PyError> {
    if !unsafe { pyre_object::is_dict(w_dict) } {
        return Err(PyError::type_error(
            "setting dictionary to a non-dict".to_string(),
        ));
    }
    INSTANCE_DICT.with(|table| {
        table.borrow_mut().insert(self_ref as usize, w_dict);
    });
    Ok(())
}

// ── MapdictWeakrefSupport ─────────────────────────────────────────────

/// objspace/std/mapdict.py:780-787 MapdictWeakrefSupport.getweakref.
///
/// ```python
/// def getweakref(self):
///     from pypy.module._weakref.interp__weakref import WeakrefLifeline
///     lifeline = self._get_mapdict_map().read(self, "weakref", SPECIAL)
///     if lifeline is None:
///         return None
///     assert isinstance(lifeline, WeakrefLifeline)
///     return lifeline
/// ```
pub fn getweakref(self_ref: PyObjectRef) -> Option<PyObjectRef> {
    WEAKREF_TABLE.with(|table| table.borrow().get(&(self_ref as usize)).copied())
}

/// objspace/std/mapdict.py:789-793 MapdictWeakrefSupport.setweakref.
///
/// ```python
/// def setweakref(self, space, weakreflifeline):
///     from pypy.module._weakref.interp__weakref import WeakrefLifeline
///     assert isinstance(weakreflifeline, WeakrefLifeline)
///     self._get_mapdict_map().write(self, "weakref", SPECIAL, weakreflifeline)
/// ```
pub fn setweakref(self_ref: PyObjectRef, weakreflifeline: PyObjectRef) {
    WEAKREF_TABLE.with(|table| {
        table
            .borrow_mut()
            .insert(self_ref as usize, weakreflifeline);
    });
}

/// objspace/std/mapdict.py:795-797 MapdictWeakrefSupport.delweakref.
///
/// ```python
/// def delweakref(self):
///     self._get_mapdict_map().write(self, "weakref", SPECIAL, None)
/// ```
pub fn delweakref(self_ref: PyObjectRef) {
    WEAKREF_TABLE.with(|table| {
        table.borrow_mut().remove(&(self_ref as usize));
    });
}
