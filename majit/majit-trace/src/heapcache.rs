/// Heap cache for the tracing phase.
///
/// During tracing, the heap cache tracks field reads/writes to eliminate
/// redundant loads. If we read a field from an object and it was already
/// read or written in the same trace, we can reuse the cached value.
///
/// Translated from rpython/jit/metainterp/heapcache.py.
use std::marker::PhantomData;

use majit_ir::vec_set::VecSet;

// Vec<bool> helpers — RPython stores these as FrontendOp flags, not sets.
#[inline(always)]
fn vb_insert(v: &mut Vec<bool>, opref: OpRef) {
    if opref.is_constant() {
        return;
    }
    let i = opref.raw() as usize;
    if i >= v.len() {
        v.resize(i + 1, false);
    }
    v[i] = true;
}
#[inline(always)]
fn vb_remove(v: &mut Vec<bool>, opref: &OpRef) -> bool {
    if opref.is_constant() {
        return false;
    }
    let i = opref.raw() as usize;
    if i < v.len() && v[i] {
        v[i] = false;
        true
    } else {
        false
    }
}

use majit_ir::{EffectInfo, ExtraEffect, GcRef, OpCode, OpRef, Type};

/// Value-equality predicate over constant OpRefs.  Mirrors
/// `Const.same_constant` (history.py:204): two ConstInt/ConstFloat/
/// ConstPtr instances are equal when they share the same subclass and
/// underlying value, independent of Box identity.
///
/// The trait is defined here (not in `majit-ir`) because `majit-trace`
/// is the lowest crate that needs the predicate (for the
/// `_unique_const_heuristic` ConstPtr canonicalisation,
/// heapcache.py:96-104) and the implementation lives in `majit-metainterp`
/// (`history::ConstOprefOracle`).  `&dyn SameConstantOracle`
/// keeps the heapcache layer agnostic of the oracle's representation.
pub trait SameConstantOracle {
    fn same_constant(&self, a: OpRef, b: OpRef) -> bool;
}

// heapcache.py: HF_* flags stored per-box on RefFrontendOp.
// In majit these are tracked via separate HashSets (is_unescaped,
// seen_allocation, etc.), but we define the constants for reference.

/// heapcache.py: HF_LIKELY_VIRTUAL
pub const HF_LIKELY_VIRTUAL: u8 = 0x01;
/// heapcache.py: HF_KNOWN_CLASS
pub const HF_KNOWN_CLASS: u8 = 0x02;
/// heapcache.py: HF_KNOWN_NULLITY
pub const HF_KNOWN_NULLITY: u8 = 0x04;
/// heapcache.py: HF_SEEN_ALLOCATION
pub const HF_SEEN_ALLOCATION: u8 = 0x08;
/// heapcache.py: HF_IS_UNESCAPED
pub const HF_IS_UNESCAPED: u8 = 0x10;
/// heapcache.py: HF_NONSTD_VABLE
pub const HF_NONSTD_VABLE: u8 = 0x20;

/// heapcache.py helper aliases.
const HF_VERSION_INC: u32 = 0x40;
pub const HF_VERSION_MAX: u32 = 0xffff_ffff - HF_VERSION_INC;
const _HF_VERSION_INC: u32 = HF_VERSION_INC;
const _HF_VERSION_MAX: u32 = HF_VERSION_MAX;

// RPython `heapcache.py:27-41` defines module-level helpers
// `add_flags`, `remove_flags`, `test_flags` that mutate per-op storage
// (`ref_frontend_op._heapc_flags`). pyre routes the same logic through
// `HeapCache::_set_flag` / `_remove_flag` / `_check_flag` because pyre's
// `OpRef` is a bare index with no associated storage outside `HeapCache`'s
// own `heapc_flags: Vec<u32>` table. The standalone helpers therefore
// cannot exist as standalone functions and would be misleading stubs.

/// heapcache.py CacheEntry — per-descr cache of fieldbox values.
///
/// `cache_anything` / `cache_seen_allocation` store the cached
/// fieldbox as a bare [`OpRef`] — the Box identity itself.  RPython
/// `heapcache.py:60-95 cache_anything[box] = valuebox` stores a Box
/// object (carrying both identity and value); pyre carries the same
/// fact through `OpRef` + the frontend object's `value`
/// (`Op` / `InputArg` `value: Cell<Option<Value>>`).  Cache-hit sanity
/// checks (`pyjitpl.py:937 assert resvalue == upd.currfieldbox.
/// getint()`) read the cached OpRef's value via
/// `TraceCtx::box_value` — composing the const pool, standard-
/// virtualizable shadow, and the frontend object's `value` field in one
/// call.  No separate side table.
#[derive(Debug, Default)]
pub(crate) struct CacheEntry {
    cache_anything: vecset::VecMap<OpRef, OpRef>,
    cache_seen_allocation: vecset::VecMap<OpRef, OpRef>,
    quasiimmut_seen: Option<VecSet<OpRef>>,
    quasiimmut_seen_refs: Option<VecSet<usize>>,
    last_const_box: Option<OpRef>,
}

impl CacheEntry {
    pub fn new() -> Self {
        Self::default()
    }

    /// heapcache.py:53-58 _clear_cache_on_write
    pub fn _clear_cache_on_write(&mut self, seen_allocation_of_target: bool) {
        if !seen_allocation_of_target {
            self.cache_seen_allocation.clear();
        }
        self.cache_anything.clear();
        if let Some(seen) = &mut self.quasiimmut_seen {
            seen.clear();
        }
        if let Some(seen) = &mut self.quasiimmut_seen_refs {
            seen.clear();
        }
    }

    /// heapcache.py:79-82 _seen_alloc
    ///
    /// Pyre adapt: needs an explicit `cache: &HeapCache` parameter
    /// because `CacheEntry` is a separate struct from `HeapCache`
    /// (RPython attaches the heapcache reference to CacheEntry at
    /// __init__ time; in Rust we pass it through to avoid a back-
    /// reference + interior mutability dance).
    pub fn _seen_alloc(&self, ref_box: OpRef, cache: &HeapCache) -> bool {
        cache.saw_allocation(ref_box)
    }

    /// heapcache.py:84-88 _getdict
    pub fn _getdict(&self, seen_alloc: bool) -> &vecset::VecMap<OpRef, OpRef> {
        if seen_alloc {
            &self.cache_seen_allocation
        } else {
            &self.cache_anything
        }
    }

    /// Pyre adapt: Python doesn't need a separate `_mut` accessor;
    /// Rust's borrow checker does.  Mirrors `_getdict`'s body.
    pub fn _getdict_mut(&mut self, seen_alloc: bool) -> &mut vecset::VecMap<OpRef, OpRef> {
        if seen_alloc {
            &mut self.cache_seen_allocation
        } else {
            &mut self.cache_anything
        }
    }

    /// heapcache.py:90-94 do_write_with_aliasing
    pub fn do_write_with_aliasing(
        &mut self,
        ref_box: OpRef,
        fieldbox: OpRef,
        cache: &HeapCache,
        oracle: &dyn SameConstantOracle,
    ) {
        let ref_box = self._unique_const_heuristic(ref_box, oracle);
        let seen_alloc = self._seen_alloc(ref_box, cache);
        self._clear_cache_on_write(seen_alloc);
        self._getdict_mut(seen_alloc).insert(ref_box, fieldbox);
    }

    /// heapcache.py:96-104 _unique_const_heuristic.
    ///
    /// Only ConstPtr operands are canonicalised; non-constant OpRefs and
    /// non-Ref-typed constants pass through unchanged (matches the
    /// `isinstance(ref_box, ConstPtr)` guard on heapcache.py:99).
    /// `oracle.same_constant(last, ref_box)` is the value-aware
    /// comparison upstream uses (history.py:204 `Const.same_constant`).
    pub fn _unique_const_heuristic(
        &mut self,
        ref_box: OpRef,
        oracle: &dyn SameConstantOracle,
    ) -> OpRef {
        if !(ref_box.is_constant() && ref_box.ty() == Some(Type::Ref)) {
            return ref_box;
        }
        if let Some(last) = self.last_const_box {
            if oracle.same_constant(last, ref_box) {
                return last;
            }
        }
        self.last_const_box = Some(ref_box);
        ref_box
    }

    /// heapcache.py:106-114 read
    pub fn read(
        &mut self,
        ref_box: OpRef,
        cache: &HeapCache,
        oracle: &dyn SameConstantOracle,
    ) -> Option<OpRef> {
        let ref_box = self._unique_const_heuristic(ref_box, oracle);
        let seen_alloc = self._seen_alloc(ref_box, cache);
        self._getdict(seen_alloc)
            .get(&ref_box)
            .copied()
            .map(|opref| cache.maybe_replace_with_const(opref))
    }

    /// heapcache.py:116-119 read_now_known
    pub fn read_now_known(
        &mut self,
        ref_box: OpRef,
        fieldbox: OpRef,
        cache: &HeapCache,
        oracle: &dyn SameConstantOracle,
    ) {
        let ref_box = self._unique_const_heuristic(ref_box, oracle);
        let seen_alloc = self._seen_alloc(ref_box, cache);
        self._getdict_mut(seen_alloc).insert(ref_box, fieldbox);
    }

    /// heapcache.py:121-129 invalidate_unescaped — RPython makes this a
    /// public method (no underscore prefix) and `_invalidate_unescaped`
    /// is the helper that walks both caches.  pyre keeps the same
    /// public/private pair.
    ///
    /// `cache: &HeapCache` matches upstream's stored-back-reference
    /// `self.heapcache` (heapcache.py:51 `self.heapcache = heapcache`)
    /// so the per-entry filter calls the version-gated
    /// `HeapCache.is_unescaped(ref_box)` (heapcache.py:127-130 / 457-460)
    /// instead of any pre-snapshotted bit table.
    pub fn invalidate_unescaped(&mut self, cache: &HeapCache) {
        self._invalidate_unescaped(cache)
    }

    pub fn _invalidate_unescaped(&mut self, cache: &HeapCache) {
        self.cache_anything
            .retain(|&ref_box, _| cache.is_unescaped(ref_box));
        self.cache_seen_allocation
            .retain(|&ref_box, _| cache.is_unescaped(ref_box));
        if let Some(seen) = &mut self.quasiimmut_seen {
            seen.clear();
        }
        if let Some(seen) = &mut self.quasiimmut_seen_refs {
            seen.clear();
        }
    }
}

/// RPython heapcache.py: FieldUpdater helper struct.
///
/// In Rust, safe ownership makes this harder to express directly, so it stores
/// a raw pointer back to the cache for writeback.
pub struct FieldUpdater {
    ref_box: OpRef,
    currfieldbox: Option<OpRef>,
    cache: *mut HeapCache,
    descr: Option<u32>,
    _marker: PhantomData<HeapCache>,
}

impl FieldUpdater {
    pub fn new(ref_box: OpRef) -> Self {
        Self {
            ref_box,
            currfieldbox: None,
            cache: std::ptr::null_mut(),
            descr: None,
            _marker: PhantomData,
        }
    }

    pub fn with_cache(
        ref_box: OpRef,
        cache: &mut HeapCache,
        descr: u32,
        fieldbox: Option<OpRef>,
    ) -> Self {
        Self {
            ref_box,
            currfieldbox: fieldbox,
            cache: cache as *mut HeapCache,
            descr: Some(descr),
            _marker: PhantomData,
        }
    }

    /// heapcache.py:137 `self.currfieldbox` reader — exposes the
    /// in-flight Box the updater is wrapping.  Mirrors `pyjitpl.py:931
    /// upd.currfieldbox` direct attribute access.  Pyre carries the
    /// Box identity as an `OpRef`; downstream sanity readers look up
    /// the intrinsic value via `TraceCtx::box_value` (composing const
    /// pool, standard-virtualizable shadow, the frontend object's
    /// `value` field).
    pub fn currfieldbox(&self) -> Option<OpRef> {
        self.currfieldbox
    }

    /// heapcache.py:139-140
    ///
    /// ```text
    ///  def getfield_now_known(self, fieldbox):
    ///      self.cache.read_now_known(self.ref_box, fieldbox)
    /// ```
    pub fn getfield_now_known(&mut self, fieldbox: OpRef, oracle: &dyn SameConstantOracle) {
        let ref_box = self.ref_box;
        let (cache, descr_index) = match self.cache_and_descr() {
            Some(pair) => pair,
            None => return,
        };
        let mut entry = cache.heap_cache.remove(&descr_index).unwrap_or_default();
        entry.read_now_known(ref_box, fieldbox, cache, oracle);
        cache.heap_cache.insert(descr_index, entry);
    }

    /// heapcache.py:142-143
    ///
    /// ```text
    ///  def setfield(self, fieldbox):
    ///      self.cache.do_write_with_aliasing(self.ref_box, fieldbox)
    /// ```
    pub fn setfield(&mut self, fieldbox: OpRef, oracle: &dyn SameConstantOracle) {
        let ref_box = self.ref_box;
        let (cache, descr_index) = match self.cache_and_descr() {
            Some(pair) => pair,
            None => return,
        };
        let mut entry = cache.heap_cache.remove(&descr_index).unwrap_or_default();
        entry.do_write_with_aliasing(ref_box, fieldbox, cache, oracle);
        cache.heap_cache.insert(descr_index, entry);
    }

    fn cache_and_descr(&mut self) -> Option<(&mut HeapCache, u32)> {
        let descr_index = self.descr?;
        if self.cache.is_null() {
            return None;
        }
        // SAFETY: `cache` was supplied by `with_cache` which received
        // an `&mut HeapCache`; the FieldUpdater's lifetime must not
        // outlive that borrow (callers hold it stack-locally during
        // a single trace step, matching upstream's pattern at
        // pyjitpl.py:973-988 where `upd = heapcache.get_field_updater(...)`
        // is consumed before any other heapcache operation).
        let cache = unsafe { &mut *self.cache };
        Some((cache, descr_index))
    }
}

/// Heap cache for the tracing interpreter.
///
/// Tracks field values, known classes, and allocation status during
/// a single trace recording session.
pub struct HeapCache {
    /// heapcache.py:172 `self.heap_cache = {}` — maps descrs to
    /// `CacheEntry`.  Field reads/writes for a given descr land in the
    /// same `CacheEntry`, which owns the `cache_anything` /
    /// `cache_seen_allocation` dicts and the `last_const_box`
    /// `_unique_const_heuristic` LRU per heapcache.py:50-104.
    /// Backed by `vecset::VecMap` (sorted Vec + binary search) so the
    /// hot per-descr lookup is O(log n) instead of linear scan when the
    /// same descr is touched repeatedly across many frames.
    heap_cache: vecset::VecMap<u32, CacheEntry>,
    /// heapcache.py: `cached_arrayitems` — nested map descr → ConstInt-index → CacheEntry.
    /// heapcache.py:557 `cache.get(index, None)` — array cache keyed by
    /// the `ConstInt.getint()` value, not the index Box's identity. Two
    /// distinct ConstInt boxes carrying the same `i64` index land in the
    /// same slot, matching the upstream lookup semantics. `i64` indices
    /// can be negative, so `vecset::VecMap` (sorted Vec + binary search)
    /// is the natural no-HashMap substitute.
    heap_array_cache: vecset::VecMap<u32, vecset::VecMap<i64, CacheEntry>>,

    /// Known class map: object_ref -> class pointer. The class pointer is a
    /// `ConstInt` vtable address (model.py:199-201), an integer the GC never
    /// traces — not a ref. RPython: CacheEntry 내부. Vec indexed by OpRef.0.
    known_class: Vec<Option<i64>>,

    /// Quasi-immutable fields known in this trace.
    /// heapcache.py: `quasi_immut_known`.
    quasi_immut_known: VecSet<(OpRef, u32)>,

    /// RPython: FrontendOp flag. Vec<bool> indexed by OpRef.0.
    is_unescaped: Vec<bool>,

    /// RPython: FrontendOp flag. Vec<bool> indexed by OpRef.0.
    seen_allocation: Vec<bool>,

    /// RPython: FrontendOp flag. Vec<u8> indexed by OpRef.0.
    /// 0 = unknown, 1 = non-null, 2 = null.
    known_nullity: Vec<u8>,

    /// RPython: FrontendOp flag. Vec<bool> indexed by OpRef.0.
    likely_virtual: Vec<bool>,

    /// heapcache.py: loop-invariant call result cache.
    /// RPython stores exactly ONE result: (descr, arg0_int) → result.
    /// Subsequent calls overwrite the single entry.
    ///
    /// TODO: upstream's `result` is a Box that
    /// carries both the symbolic identity and the concrete value
    /// together; pyre splits these into the symbolic `OpRef` plus a
    /// concrete `i64` so `do_residual_call` can return the same
    /// `(opref, value)` tuple shape on cache hits as it does on
    /// freshly-executed calls.
    loopinvariant_descr: Option<u32>,
    loopinvariant_arg0: Option<i64>,
    loopinvariant_result: Option<OpRef>,
    loopinvariant_resvalue: Option<i64>,

    /// heapcache.py: per-box `_heapc_deps`.
    ///
    /// RPython stores either `None` or a list on each FrontendOp:
    /// `deps[0]` is the cached array length and `deps[1:]` are escape
    /// dependencies added by `_escape_from_write`. pyre's `OpRef` is a bare
    /// index, so the closest equivalent is a per-op side slot keyed by OpRef.
    heapc_deps: Vec<Option<Vec<Option<OpRef>>>>,

    /// heapcache.py: oldbox.set_replaced_with_const() in replace_box().
    ///
    /// RPython stores this on the box's `_forwarded`.  Pyre's `OpRef`
    /// carries both a position and a typed Box identity, so the key must be
    /// the full `OpRef`, not just `raw()`: `IntOp(n)` and `RefOp(n)` are
    /// different boxes upstream.
    replaced_with_const: Vec<(OpRef, OpRef)>,

    /// heapcache.py:176: need_guard_not_invalidated — set True on reset,
    /// consumed by quasi-immut field recording to decide whether to emit
    /// GUARD_NOT_INVALIDATED.
    need_guard_not_invalidated: bool,

    head_version: u32,
    likely_virtual_version: u32,
    /// RPython: FrontendOp flags. Vec<u32> indexed by OpRef.0.
    heapc_flags: Vec<u32>,
}

impl HeapCache {
    /// Create a new, empty heap cache.
    pub fn new() -> Self {
        HeapCache {
            heap_cache: vecset::VecMap::new(),
            heap_array_cache: vecset::VecMap::new(),
            known_class: Vec::new(),
            quasi_immut_known: VecSet::new(),
            is_unescaped: Vec::new(),
            seen_allocation: Vec::new(),
            known_nullity: Vec::new(),
            likely_virtual: Vec::new(),
            loopinvariant_descr: None,
            loopinvariant_arg0: None,
            loopinvariant_result: None,
            loopinvariant_resvalue: None,
            heapc_deps: Vec::new(),
            replaced_with_const: Vec::new(),
            need_guard_not_invalidated: true,
            head_version: 0,
            likely_virtual_version: 0,
            heapc_flags: Vec::new(),
        }
    }

    /// heapcache.py:43-47 `maybe_replace_with_const(box)`.
    fn maybe_replace_with_const(&self, opref: OpRef) -> OpRef {
        if opref.is_constant() {
            return opref;
        }
        self.replaced_with_const
            .iter()
            .rev()
            .find_map(|(old, new)| if *old == opref { Some(*new) } else { None })
            .unwrap_or(opref)
    }

    fn flags_for_ref(&self, opref: OpRef) -> u32 {
        if opref.is_constant() {
            return 0;
        }
        self.heapc_flags
            .get(opref.raw() as usize)
            .copied()
            .unwrap_or(0)
    }

    fn set_flags_for_ref(&mut self, opref: OpRef, flags: u32) {
        if opref.is_constant() {
            return;
        }
        let i = opref.raw() as usize;
        if i >= self.heapc_flags.len() {
            self.heapc_flags.resize(i + 1, 0);
        }
        self.heapc_flags[i] = flags;
    }

    fn versioned_or(self_flags: u32, op_version: u32) -> bool {
        self_flags >= op_version
    }

    /// RPython: test_head_version(ref_frontend_op)
    pub fn test_head_version(&self, opref: OpRef) -> bool {
        Self::versioned_or(self.flags_for_ref(opref), self.head_version)
    }

    /// RPython: test_likely_virtual_version(ref_frontend_op)
    pub fn test_likely_virtual_version(&self, opref: OpRef) -> bool {
        Self::versioned_or(self.flags_for_ref(opref), self.likely_virtual_version)
    }

    /// RPython: update_version(ref_frontend_op)
    /// heapcache.py:199-209
    ///
    /// ```text
    ///  def update_version(self, ref_frontend_op):
    ///      """Ensure the version of 'ref_frontend_op' is current. If not,
    ///      it will update 'ref_frontend_op' (removing most flags currently set).
    ///      """
    ///      if not self.test_head_version(ref_frontend_op):
    ///          f = self.head_version
    ///          if (self.test_likely_virtual_version(ref_frontend_op) and
    ///              test_flags(ref_frontend_op, HF_LIKELY_VIRTUAL)):
    ///              f |= HF_LIKELY_VIRTUAL
    ///          ref_frontend_op._set_heapc_flags(f)
    ///          ref_frontend_op._heapc_deps = None
    /// ```
    pub fn update_version(&mut self, opref: OpRef) {
        let old_flags = self.flags_for_ref(opref);
        if Self::versioned_or(old_flags, self.head_version) {
            return;
        }
        let mut flags = self.head_version;
        if Self::versioned_or(old_flags, self.likely_virtual_version)
            && (old_flags & u32::from(HF_LIKELY_VIRTUAL)) != 0
        {
            flags |= u32::from(HF_LIKELY_VIRTUAL);
        }
        self.set_flags_for_ref(opref, flags);
        // RPython: ref_frontend_op._heapc_deps = None
        self._remove_deps_for_box(opref);
    }

    /// RPython: _check_flag(box, flag)
    pub fn _check_flag(&self, opref: OpRef, flag: u8) -> bool {
        if !self.test_head_version(opref) {
            return false;
        }
        (self.flags_for_ref(opref) & u32::from(flag)) != 0
    }

    /// RPython: _set_flag(box, flag)
    pub fn _set_flag(&mut self, opref: OpRef, flag: u8) {
        if opref.is_constant() {
            return;
        }
        self.update_version(opref);
        let flags = self.flags_for_ref(opref) | u32::from(flag);
        self.set_flags_for_ref(opref, flags);
        // Keep mirrors: boolean flags used by this Rust implementation.
        match flag {
            HF_SEEN_ALLOCATION => {
                vb_insert(&mut self.seen_allocation, opref);
            }
            HF_KNOWN_CLASS => {
                let i = opref.raw() as usize;
                if i >= self.known_class.len() {
                    self.known_class.resize(i + 1, None);
                }
            }
            HF_KNOWN_NULLITY => {
                let i = opref.raw() as usize;
                if i >= self.known_nullity.len() {
                    self.known_nullity.resize(i + 1, 0);
                }
                if self.known_nullity[i] == 0 {
                    self.known_nullity[i] = 1;
                }
            }
            HF_IS_UNESCAPED => {
                vb_insert(&mut self.is_unescaped, opref);
            }
            HF_LIKELY_VIRTUAL => {
                vb_insert(&mut self.likely_virtual, opref);
            }
            // HF_NONSTD_VABLE has no mirror — heapc_flags is the source of truth.
            _ => {}
        }
    }

    fn _remove_flag(&mut self, opref: OpRef, flag: u8) {
        if opref.is_constant() {
            return;
        }
        let flags = self.flags_for_ref(opref);
        if flags == 0 {
            return;
        }
        let updated = flags & !u32::from(flag);
        self.set_flags_for_ref(opref, updated);
        match flag {
            HF_IS_UNESCAPED => {
                vb_remove(&mut self.is_unescaped, &opref);
            }
            HF_LIKELY_VIRTUAL => {
                vb_remove(&mut self.likely_virtual, &opref);
            }
            HF_SEEN_ALLOCATION => {
                vb_remove(&mut self.seen_allocation, &opref);
            }
            HF_KNOWN_NULLITY => {
                {
                    let _i = opref.raw() as usize;
                    if _i < self.known_nullity.len() {
                        self.known_nullity[_i] = 0;
                    }
                };
            }
            HF_KNOWN_CLASS => {
                {
                    let _i = opref.raw() as usize;
                    if _i < self.known_class.len() {
                        self.known_class[_i] = None;
                    }
                };
            }
            _ => {}
        }
    }

    /// RPython-compatible alias.
    pub fn _get_deps(&mut self, opref: OpRef) -> &mut Vec<Option<OpRef>> {
        self.update_version(opref);
        let i = opref.raw() as usize;
        if i >= self.heapc_deps.len() {
            self.heapc_deps.resize_with(i + 1, || None);
        }
        let deps = self.heapc_deps[i].get_or_insert_with(|| vec![None]);
        if deps.is_empty() {
            deps.push(None);
        }
        deps
    }

    /// heapcache.py:224-229
    ///
    /// ```text
    ///  def _escape_from_write(self, box, fieldbox):
    ///      if self.is_unescaped(box) and self.is_unescaped(fieldbox):
    ///          deps = self._get_deps(box)
    ///          deps.append(fieldbox)
    ///      elif fieldbox is not None:
    ///          self._escape_box(fieldbox)
    /// ```
    pub fn _escape_from_write(&mut self, r#box: OpRef, fieldbox: OpRef) {
        if self.is_unescaped(r#box) && self.is_unescaped(fieldbox) {
            let deps = self._get_deps(r#box);
            deps.push(Some(fieldbox));
        } else {
            // RPython's `elif fieldbox is not None` — pyre's OpRef is always
            // present (no None equivalent), so the branch always fires.
            self._escape_box(fieldbox);
        }
    }

    /// heapcache.py:295-309 `_escape_box(box)`.
    ///
    /// ```text
    ///  def _escape_box(self, box):
    ///      if isinstance(box, RefFrontendOp):
    ///          remove_flags(box, HF_LIKELY_VIRTUAL | HF_IS_UNESCAPED)
    ///          deps = box._heapc_deps
    ///          if deps is not None:
    ///              if not self.test_head_version(box):
    ///                  box._heapc_deps = None
    ///              else:
    ///                  # 'deps[0]' is abused to store the array length, keep it
    ///                  if deps[0] is None:
    ///                      box._heapc_deps = None
    ///                  else:
    ///                      box._heapc_deps = [deps[0]]
    ///                  for i in range(1, len(deps)):
    ///                      self._escape_box(deps[i])
    /// ```
    pub fn _escape_box(&mut self, opref: OpRef) {
        if opref.is_constant() {
            return;
        }
        if !vb_remove(&mut self.is_unescaped, &opref) {
            return;
        }
        // RPython remove_flags(box, HF_LIKELY_VIRTUAL | HF_IS_UNESCAPED).
        // _remove_flag updates heapc_flags AND mirrors HF_IS_UNESCAPED /
        // HF_LIKELY_VIRTUAL Vec<bool> back out, so the version-gated
        // _check_flag query stays consistent.
        self._remove_flag(opref, HF_LIKELY_VIRTUAL);
        self._remove_flag(opref, HF_IS_UNESCAPED);
        let i = opref.raw() as usize;
        let deps = self.heapc_deps.get_mut(i).and_then(Option::take);
        if let Some(deps) = deps {
            if self.test_head_version(opref) {
                let kept_len = deps.first().copied().flatten();
                if let Some(length) = kept_len {
                    self.heapc_deps[i] = Some(vec![Some(length)]);
                }
                for dep in deps.into_iter().skip(1).flatten() {
                    self._escape_box(dep);
                }
            }
        }
    }

    /// RPython: mark_escaped(opnum, descr, *argboxes) entrypoint.
    pub fn mark_escaped(&mut self, opnum: OpCode, _descr: Option<OpRef>, argboxes: &[OpRef]) {
        if opnum == OpCode::SetfieldGc {
            if argboxes.len() == 2 {
                self._escape_from_write(argboxes[0], argboxes[1]);
                return;
            }
        } else if opnum == OpCode::SetarrayitemGc {
            if argboxes.len() == 3 {
                self._escape_from_write(argboxes[0], argboxes[2]);
                return;
            }
        } else if !matches!(
            opnum,
            OpCode::GetfieldGcR
                | OpCode::GetfieldGcI
                | OpCode::GetfieldGcF
                | OpCode::PtrEq
                | OpCode::PtrNe
                | OpCode::InstancePtrEq
                | OpCode::InstancePtrNe
                | OpCode::AssertNotNone
        ) {
            self._escape_argboxes(argboxes);
        }
    }

    /// heapcache.py:259-293 mark_escaped_varargs.
    ///
    /// Upstream splits the two flavors:
    ///   * `mark_escaped` (line 232) handles SETFIELD_GC / SETARRAYITEM_GC
    ///     and asserts `opnum != CALL_N`.
    ///   * `mark_escaped_varargs` (line 259) handles CALL_N and special-cases
    ///     ARRAYCOPY / ARRAYMOVE with constant starts+length+single-descr to
    ///     skip arg escape entirely.
    ///
    /// `effectinfo` + `const_value` carry the upstream
    /// `descr.get_extra_info()` lookups; the closure returns the
    /// `ConstInt.getint()` value (heapcache.py:274-276 / :284-286).
    pub fn mark_escaped_varargs<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        const_value: F,
    ) {
        if opnum == OpCode::CallN {
            if let Some(ei) = effectinfo {
                if ei.single_write_descr_array.is_some() {
                    // heapcache.py:272-281: CALL_N + OS_ARRAYCOPY with all
                    // three index/length operands ConstInt → don't escape
                    // argboxes.
                    if ei.oopspecindex == majit_ir::OopSpecIndex::Arraycopy
                        && argboxes.len() >= 6
                        && const_value(argboxes[3]).is_some()
                        && const_value(argboxes[4]).is_some()
                        && const_value(argboxes[5]).is_some()
                    {
                        return;
                    }
                    // heapcache.py:282-290: CALL_N + OS_ARRAYMOVE with all
                    // three operands ConstInt → don't escape argboxes.
                    if ei.oopspecindex == majit_ir::OopSpecIndex::Arraymove
                        && argboxes.len() >= 5
                        && const_value(argboxes[2]).is_some()
                        && const_value(argboxes[3]).is_some()
                        && const_value(argboxes[4]).is_some()
                    {
                        return;
                    }
                }
            }
            // heapcache.py:291-293 fallback: escape all argboxes.
            self._escape_argboxes(argboxes);
            return;
        }
        self.mark_escaped(opnum, None, argboxes)
    }

    /// RPython: _escape_argboxes(*argboxes)
    pub fn _escape_argboxes(&mut self, args: &[OpRef]) {
        if args.is_empty() {
            return;
        }
        self._escape_box(args[0]);
        self._escape_argboxes(&args[1..]);
    }

    /// heapcache.py:518-522 `getfield(self, box, descr)`.
    ///
    /// ```text
    ///  def getfield(self, box, descr):
    ///      cache = self.heap_cache.get(descr, None)
    ///      if cache:
    ///          return cache.read(box)
    ///      return None
    /// ```
    ///
    /// `CacheEntry.read` (heapcache.py:106-114) handles the
    /// `_unique_const_heuristic` ConstPtr canonicalisation and the
    /// `maybe_replace_with_const` forwarding internally.  We take the
    /// entry out of `heap_cache` for the duration of the call so the
    /// borrow checker accepts `&mut entry` and `&self.heap_cache`'s
    /// neighbour fields simultaneously, then put it back.
    pub fn getfield_cached(
        &mut self,
        obj: OpRef,
        field_index: u32,
        oracle: &dyn SameConstantOracle,
    ) -> Option<OpRef> {
        let mut entry = self.heap_cache.remove(&field_index)?;
        let result = entry.read(obj, self, oracle);
        self.heap_cache.insert(field_index, entry);
        result
    }

    /// heapcache.py:538-540 `setfield(self, box, fieldbox, descr)`.
    ///
    /// ```text
    ///  def setfield(self, box, fieldbox, descr):
    ///      upd = self.get_field_updater(box, descr)
    ///      upd.setfield(fieldbox)
    /// ```
    ///
    /// The `upd.setfield` body is `cache.do_write_with_aliasing(ref_box,
    /// fieldbox)` (heapcache.py:142-143), which handles
    /// `_unique_const_heuristic`, `_clear_cache_on_write`, and the dict
    /// insertion in one step.  Aliasing semantics:
    /// `_clear_cache_on_write(seen_alloc)` clears `cache_anything` and,
    /// when `seen_alloc` is false (the target may alias anything else),
    /// also clears `cache_seen_allocation`, matching
    /// heapcache.py:70-77.
    pub fn setfield_cached(
        &mut self,
        obj: OpRef,
        field_index: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
    ) {
        let mut entry = self.heap_cache.remove(&field_index).unwrap_or_default();
        entry.do_write_with_aliasing(obj, value, self, oracle);
        self.heap_cache.insert(field_index, entry);
    }

    /// heapcache.py:534-536 `getfield_now_known(self, box, descr,
    /// fieldbox)`.
    ///
    /// ```text
    ///  def getfield_now_known(self, box, descr, fieldbox):
    ///      upd = self.get_field_updater(box, descr)
    ///      upd.getfield_now_known(fieldbox)
    /// ```
    ///
    /// `upd.getfield_now_known` delegates to
    /// `cache.read_now_known(ref_box, fieldbox)` (heapcache.py:116-119),
    /// which records the value without the aliasing-clear step.
    pub fn getfield_now_known(
        &mut self,
        obj: OpRef,
        field_index: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
    ) {
        let mut entry = self.heap_cache.remove(&field_index).unwrap_or_default();
        entry.read_now_known(obj, value, self, oracle);
        self.heap_cache.insert(field_index, entry);
    }

    /// heapcache.py: invalidate_unescaped — clear cached values for
    /// escaped objects only. Unescaped (newly allocated) objects cannot
    /// be affected by external calls, so their caches are preserved.
    pub fn invalidate_caches_for_escaped(&mut self) {
        // heapcache.py:362-365 — `for cache in self.heap_cache.itervalues():
        //                           cache.invalidate_unescaped()`.
        // Take/restore is the borrow-split equivalent of upstream's stored
        // back-reference (`CacheEntry.heapcache`): the entries are removed
        // so each `invalidate_unescaped` call receives a fresh `&HeapCache`
        // to run the version-gated `is_unescaped(ref_box)` check
        // (heapcache.py:127-130 / 457-460) without the borrow checker
        // tripping over `entry` and `self.heap_cache` simultaneously.
        let mut heap_cache = std::mem::take(&mut self.heap_cache);
        for entry in heap_cache.values_mut() {
            entry.invalidate_unescaped(self);
        }
        self.heap_cache = heap_cache;
        // heapcache.py:542-552: iterate cached_arrayitems and invalidate
        // per-CacheEntry entries whose box is no longer unescaped.
        let mut heap_array_cache = std::mem::take(&mut self.heap_array_cache);
        for caches in heap_array_cache.values_mut() {
            for cache in caches.values_mut() {
                cache.invalidate_unescaped(self);
            }
        }
        self.heap_array_cache = heap_array_cache;
    }

    /// heapcache.py:502-506
    ///
    /// ```text
    ///  def new(self, box):
    ///      assert isinstance(box, RefFrontendOp)
    ///      self.update_version(box)
    ///      add_flags(box, HF_LIKELY_VIRTUAL | HF_SEEN_ALLOCATION | HF_IS_UNESCAPED
    ///                     | HF_KNOWN_NULLITY)
    /// ```
    pub fn new_object(&mut self, opref: OpRef) {
        if opref.is_constant() {
            return;
        }
        self.update_version(opref);
        // RPython add_flags writes the bitwise OR of all four flags into the
        // versioned heapc_flags. We route through _set_flag so the Vec<bool>
        // mirrors stay in sync with heapc_flags.
        self._set_flag(opref, HF_LIKELY_VIRTUAL);
        self._set_flag(opref, HF_SEEN_ALLOCATION);
        self._set_flag(opref, HF_IS_UNESCAPED);
        self._set_flag(opref, HF_KNOWN_NULLITY);
    }

    /// heapcache.py:508-516 new_array
    ///
    /// ```text
    ///  def new_array(self, box, lengthbox):
    ///      assert isinstance(box, RefFrontendOp)
    ///      self.update_version(box)
    ///      flags = HF_SEEN_ALLOCATION | HF_KNOWN_NULLITY
    ///      if isinstance(lengthbox, Const):
    ///          # only constant-length arrays are virtuals
    ///          flags |= HF_LIKELY_VIRTUAL | HF_IS_UNESCAPED
    ///      add_flags(box, flags)
    ///      self.arraylen_now_known(box, lengthbox)
    /// ```
    pub fn new_array(&mut self, opref: OpRef, lengthbox: OpRef, length_is_const: bool) {
        if opref.is_constant() {
            return;
        }
        // RPython:
        //     self.update_version(box)
        //     flags = HF_SEEN_ALLOCATION | HF_KNOWN_NULLITY
        //     if isinstance(lengthbox, Const):
        //         flags |= HF_LIKELY_VIRTUAL | HF_IS_UNESCAPED
        //     add_flags(box, flags)
        //     self.arraylen_now_known(box, lengthbox)
        self.update_version(opref);
        self._set_flag(opref, HF_SEEN_ALLOCATION);
        // RPython adds HF_KNOWN_NULLITY directly via add_flags. Route through
        // nullity_now_known so the Vec<u8> value mirror also captures non-null.
        self.nullity_now_known(opref, true);
        if length_is_const {
            self._set_flag(opref, HF_LIKELY_VIRTUAL);
            self._set_flag(opref, HF_IS_UNESCAPED);
        }
        // heapcache.py:516: self.arraylen_now_known(box, lengthbox)
        self.arraylen_now_known(opref, lengthbox);
    }

    /// heapcache.py:485-486
    ///
    /// ```text
    ///  def is_known_nonstandard_virtualizable(self, box):
    ///      return self._check_flag(box, HF_NONSTD_VABLE) or self._check_flag(box, HF_SEEN_ALLOCATION)
    /// ```
    pub fn is_known_nonstandard_virtualizable(&self, opref: OpRef) -> bool {
        self._check_flag(opref, HF_NONSTD_VABLE) || self._check_flag(opref, HF_SEEN_ALLOCATION)
    }

    /// heapcache.py:488-491
    ///
    /// ```text
    ///  def nonstandard_virtualizables_now_known(self, box):
    ///      if isinstance(box, Const):
    ///          return
    ///      self._set_flag(box, HF_NONSTD_VABLE)
    /// ```
    pub fn nonstandard_virtualizables_now_known(&mut self, opref: OpRef) {
        if opref.is_constant() {
            return;
        }
        self._set_flag(opref, HF_NONSTD_VABLE);
    }

    /// heapcache.py:598-602 `replace_box(oldbox, newbox)`.
    ///
    /// ```text
    ///  def replace_box(self, oldbox, newbox):
    ///      # here, only for replacing a box with a const
    ///      if isinstance(oldbox, FrontendOp) and isinstance(newbox, Const):
    ///          assert newbox.same_constant(constant_from_op(oldbox))
    ///          oldbox.set_replaced_with_const()
    /// ```
    ///
    pub fn replace_box(&mut self, old: OpRef, new: OpRef) {
        if !old.is_constant() && new.is_constant() {
            if let Some((_, existing)) = self
                .replaced_with_const
                .iter_mut()
                .find(|(existing_old, _)| *existing_old == old)
            {
                *existing = new;
            } else {
                self.replaced_with_const.push((old, new));
            }
        }
    }

    /// heapcache.py:470-473
    ///
    /// ```text
    ///  def class_now_known(self, box):
    ///      if isinstance(box, Const):
    ///          return
    ///      self._set_flag(box, HF_KNOWN_CLASS | HF_KNOWN_NULLITY)
    /// ```
    ///
    /// pyre additionally remembers the concrete class pointer when this
    /// layer can decode it.
    pub fn class_now_known_maybe(&mut self, opref: OpRef, class: Option<i64>) {
        if opref.is_constant() {
            return;
        }
        let i = opref.raw() as usize;
        if i >= self.known_class.len() {
            self.known_class.resize(i + 1, None);
        }
        if let Some(class) = class {
            self.known_class[i] = Some(class);
        }
        // RPython _set_flag(box, HF_KNOWN_CLASS | HF_KNOWN_NULLITY).
        self._set_flag(opref, HF_KNOWN_CLASS);
        // RPython also writes HF_KNOWN_NULLITY in the same _set_flag call;
        // route through nullity_now_known so the Vec<u8> value mirror also
        // captures non-null.
        self.nullity_now_known(opref, true);
    }

    pub fn class_now_known(&mut self, opref: OpRef, class: i64) {
        self.class_now_known_maybe(opref, Some(class));
    }

    /// heapcache.py:467-468 is_class_known.
    ///   `return self._check_flag(box, HF_KNOWN_CLASS)`
    /// Version-gated through `_check_flag` so a `reset_keep_likely_virtuals`
    /// (which only bumps `head_version`) hides stale class info.
    pub fn is_class_known(&self, opref: OpRef) -> bool {
        if opref.is_constant() {
            return false;
        }
        self._check_flag(opref, HF_KNOWN_CLASS)
    }

    /// Get the known class of an object, if available.
    /// Mirrors heapcache.py:467 — only valid when the version is current,
    /// because the side `known_class` Vec may hold stale entries from
    /// before the last `reset_keep_likely_virtuals`.
    pub fn get_known_class(&self, opref: OpRef) -> Option<i64> {
        if opref.is_constant() {
            return None;
        }
        if !self._check_flag(opref, HF_KNOWN_CLASS) {
            return None;
        }
        self.known_class.get(opref.raw() as usize).and_then(|v| *v)
    }

    /// Forward every inline `OpRef::ConstPtr(GcRef)` cached as a
    /// *value* so it survives a moving minor collection. history.py:314
    /// `ConstPtr.value` is a gcref field the Python GC traces through the
    /// box object graph; pyre caches the inline gcref in plain `OpRef`
    /// slots and must forward them explicitly.
    ///
    /// Only value slots are forwarded — these are returned on cache hits
    /// and emitted into the op-graph (`replaced_with_const` /
    /// `loopinvariant_result` / `CacheEntry` field values), so a stale one
    /// is a use-after-move. The `cache_anything` / `cache_seen_allocation`
    /// / `quasi_immut_known` *keys* are deliberately left stale: a forwarded
    /// lookup key simply misses the stale-keyed entry and the cache
    /// repopulates (same contract as the `call_pure_results` cache), and an
    /// in-place key rewrite would break the sorted-`VecMap` ordering.
    pub fn walk_const_ptr_refs(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn forward(slot: &mut OpRef, visitor: &mut dyn FnMut(&mut GcRef)) {
            if let Some(gcref) = slot.as_const_ptr_mut() {
                visitor(gcref);
            }
        }
        fn forward_entry(entry: &mut CacheEntry, visitor: &mut dyn FnMut(&mut GcRef)) {
            for value in entry.cache_anything.values_mut() {
                forward(value, visitor);
            }
            for value in entry.cache_seen_allocation.values_mut() {
                forward(value, visitor);
            }
            if let Some(slot) = entry.last_const_box.as_mut() {
                forward(slot, visitor);
            }
        }
        for entry in self.heap_cache.values_mut() {
            forward_entry(entry, visitor);
        }
        for index_map in self.heap_array_cache.values_mut() {
            for entry in index_map.values_mut() {
                forward_entry(entry, visitor);
            }
        }
        if let Some(slot) = self.loopinvariant_result.as_mut() {
            forward(slot, visitor);
        }
        for (_, slot) in self.replaced_with_const.iter_mut() {
            forward(slot, visitor);
        }
        for deps in self.heapc_deps.iter_mut().flatten() {
            for slot in deps.iter_mut().flatten() {
                forward(slot, visitor);
            }
        }
    }

    /// heapcache.py:493-494 is_unescaped.
    ///   `return self._check_flag(box, HF_IS_UNESCAPED)`
    pub fn is_unescaped(&self, opref: OpRef) -> bool {
        self._check_flag(opref, HF_IS_UNESCAPED)
    }

    /// heapcache.py:79-82 `CacheEntry._seen_alloc(box)`:
    ///
    /// ```text
    ///  if not isinstance(ref_box, RefFrontendOp):
    ///      return False
    ///  return self.heapcache._check_flag(ref_box, HF_SEEN_ALLOCATION)
    /// ```
    pub fn saw_allocation(&self, opref: OpRef) -> bool {
        self._check_flag(opref, HF_SEEN_ALLOCATION)
    }

    /// Notify the cache about an operation, potentially invalidating entries.
    ///
    /// This should be called for every operation during tracing, so the cache
    /// can track which operations affect heap state.
    pub fn notify_op(&mut self, opcode: OpCode, args: &[OpRef], result: OpRef) {
        if opcode.is_malloc() {
            self.new_object(result);
            return;
        }
        // heapcache.py:234-241 `mark_escaped` routes SETFIELD_GC /
        // SETARRAYITEM_GC through the single `_escape_from_write(box,
        // fieldbox)` body. SETFIELD_GC: box=args[0], fieldbox=args[1];
        // SETARRAYITEM_GC: box=args[0], fieldbox=args[2]. The dependency
        // is recorded only when both are unescaped; in every other case
        // — including container unescaped but value already escaped —
        // the value escapes (heapcache.py:224-229 `elif fieldbox is not
        // None: self._escape_box(fieldbox)`).
        if opcode == OpCode::SetfieldGc && args.len() >= 2 {
            self._escape_from_write(args[0], args[1]);
        }
        if opcode == OpCode::SetarrayitemGc && args.len() >= 3 {
            self._escape_from_write(args[0], args[2]);
        }
        // heapcache.py: GUARD_VALUE → known constant + nonnull.
        if opcode == OpCode::GuardValue && args.len() >= 2 {
            self.nullity_now_known(args[0], true);
        }
        // heapcache.py:470-472 `class_now_known(box)` sets HF_KNOWN_CLASS
        // on args[0]. RPython stores only the flag; pyre additionally keeps
        // the concrete class pointer when it can decode the class operand.
        // Guard class operands are ConstInt vtable addresses upstream:
        // model.py:199-201 `cls_of_box()` returns ConstInt(ptr2int(typeptr))
        // and aarch64/regalloc.py:829 reads `op.getarg(1).getint()`.
        // Legacy pool-indexed class args still mark the class as known
        // without a concrete side value because this layer has no pool.
        if opcode == OpCode::GuardClass || opcode == OpCode::GuardNonnullClass {
            let class = args
                .get(1)
                .and_then(|class_val| class_val.const_int_value());
            self.class_now_known_maybe(args[0], class);
        }
        // heapcache.py: GUARD_NONNULL → known non-null.
        if opcode == OpCode::GuardNonnull && !args.is_empty() {
            self.nullity_now_known(args[0], true);
        }

        // heapcache.py:242-250: mark_escaped — escape arguments for
        // operations that are NOT in the whitelist.
        // GETFIELD_GC_*, PTR_EQ/NE, INSTANCE_PTR_EQ/NE, ASSERT_NOT_NONE
        // do NOT escape their arguments. SETFIELD_GC/SETARRAYITEM_GC are
        // handled above via _escape_from_write. Everything else escapes.
        let dont_escape = matches!(
            opcode,
            OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldGcPureI
                | OpCode::GetfieldGcPureR
                | OpCode::GetfieldGcPureF
                | OpCode::PtrEq
                | OpCode::PtrNe
                | OpCode::InstancePtrEq
                | OpCode::InstancePtrNe
                | OpCode::AssertNotNone
                | OpCode::SetfieldGc
                | OpCode::SetarrayitemGc
        ) || opcode.is_guard()
            || opcode.is_malloc()
            || opcode.has_no_side_effect();

        if !dont_escape {
            for &arg in args {
                self._escape_box(arg);
            }
        }
    }

    /// heapcache.py:211-216 invalidate_caches_varargs.
    ///
    /// `effectinfo` mirrors upstream `descr.get_extra_info()` consulted
    /// inside `clear_caches_varargs`; pyre threads the
    /// already-extracted EffectInfo through to avoid an extra
    /// `&dyn CallDescr` pass.
    pub fn invalidate_caches_varargs<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        self.mark_escaped_varargs(opnum, effectinfo, argboxes, &const_value);
        if Self::_clear_caches_not_necessary(opnum) {
            return;
        }
        self.clear_caches_varargs(opnum, effectinfo, argboxes, oracle, const_value);
    }

    /// heapcache.py:312-336
    ///
    /// ```text
    ///  def clear_caches_not_necessary(self, opnum, descr):
    ///      if (opnum == rop.SETFIELD_GC or
    ///          opnum == rop.SETARRAYITEM_GC or
    ///          opnum == rop.SETFIELD_RAW or
    ///          opnum == rop.SETARRAYITEM_RAW or
    ///          opnum == rop.SETINTERIORFIELD_GC or
    ///          opnum == rop.COPYSTRCONTENT or
    ///          opnum == rop.COPYUNICODECONTENT or
    ///          opnum == rop.STRSETITEM or
    ///          opnum == rop.UNICODESETITEM or
    ///          opnum == rop.SETFIELD_RAW or
    ///          opnum == rop.SETARRAYITEM_RAW or
    ///          opnum == rop.SETINTERIORFIELD_RAW or
    ///          opnum == rop.RECORD_EXACT_CLASS or
    ///          opnum == rop.RAW_STORE or
    ///          opnum == rop.ASSERT_NOT_NONE or
    ///          opnum == rop.RECORD_EXACT_CLASS or
    ///          opnum == rop.RECORD_EXACT_VALUE_I or
    ///          opnum == rop.RECORD_EXACT_VALUE_R):
    ///          return True
    ///      if (rop._OVF_FIRST <= opnum <= rop._OVF_LAST or
    ///          rop._NOSIDEEFFECT_FIRST <= opnum <= rop._NOSIDEEFFECT_LAST or
    ///          rop._GUARD_FIRST <= opnum <= rop._GUARD_LAST):
    ///          return True
    ///      return False
    /// ```
    ///
    /// CALL_* opcodes are deliberately NOT in this set — RPython invalidates
    /// caches whenever a residual call runs, since the callee could mutate
    /// fields the optimizer thinks are still cached.
    fn _clear_caches_not_necessary(opnum: OpCode) -> bool {
        matches!(
            opnum,
            OpCode::SetfieldGc
                | OpCode::SetarrayitemGc
                | OpCode::SetfieldRaw
                | OpCode::SetarrayitemRaw
                | OpCode::SetinteriorfieldGc
                | OpCode::SetinteriorfieldRaw
                | OpCode::Copystrcontent
                | OpCode::Copyunicodecontent
                | OpCode::Strsetitem
                | OpCode::Unicodesetitem
                | OpCode::RecordExactClass
                | OpCode::RecordExactValueR
                | OpCode::RecordExactValueI
                | OpCode::RawStore
                | OpCode::AssertNotNone
        ) || opnum.is_ovf()
            || opnum.has_no_side_effect()
            || opnum.is_guard()
    }

    /// RPython-compatible alias.
    pub fn clear_caches_not_necessary(&self, opnum: OpCode) -> bool {
        Self::_clear_caches_not_necessary(opnum)
    }

    /// RPython-compatible alias.
    pub fn clear_caches<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        self.clear_caches_varargs(opnum, effectinfo, argboxes, oracle, const_value)
    }

    /// heapcache.py:341-376 clear_caches_varargs.
    pub fn clear_caches_varargs<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        self.need_guard_not_invalidated = true;
        // RPython `heapcache.py:341-345`:
        //     if (OpHelpers.is_plain_call(opnum) or
        //         OpHelpers.is_call_loopinvariant(opnum) or
        //         OpHelpers.is_cond_call_value(opnum) or
        //         opnum == rop.COND_CALL):
        // `is_plain_call` matches `CALL_{I,R,F,N}` only — `CALL_PURE_*`,
        // `CALL_MAY_FORCE_*`, `CALL_ASSEMBLER_*`, `CALL_RELEASE_GIL_*`
        // all fall through to `reset_keep_likely_virtuals` (the
        // aggressive arm).  Pyre's `is_call()` is the broader
        // `_CALL_FIRST..=_CALL_LAST` range, so use the narrow
        // `is_plain_call()` predicate to mirror upstream's enumeration.
        if opnum.is_plain_call()
            || opnum.is_call_loopinvariant()
            || opnum.is_cond_call_value()
            || opnum == OpCode::CondCallN
        {
            if let Some(ei) = effectinfo {
                // heapcache.py:347-353 — elidable / loopinvariant calls
                // are pure (or already cached) and never invalidate the
                // heap.
                if matches!(
                    ei.extraeffect,
                    ExtraEffect::LoopInvariant
                        | ExtraEffect::ElidableCannotRaise
                        | ExtraEffect::ElidableOrMemoryError
                        | ExtraEffect::ElidableCanRaise,
                ) {
                    return;
                }
                // heapcache.py:355-361 — well-defined oopspec dispatch.
                let single_descr_idx = ei.single_write_descr_array.as_ref().map(|d| d.index());
                if ei.oopspecindex == majit_ir::OopSpecIndex::Arraycopy {
                    self._clear_caches_arraycopy(
                        opnum,
                        None,
                        argboxes,
                        single_descr_idx,
                        oracle,
                        const_value,
                    );
                    return;
                }
                if ei.oopspecindex == majit_ir::OopSpecIndex::Arraymove {
                    self._clear_caches_arraymove(
                        opnum,
                        None,
                        argboxes,
                        single_descr_idx,
                        oracle,
                        const_value,
                    );
                    return;
                }
            }
            // heapcache.py:362-369 — only invalidate things that escaped.
            // Take/restore mirrors `CacheEntry.heapcache` back-reference so
            // `invalidate_unescaped` calls the version-gated
            // `HeapCache.is_unescaped(ref_box)` per entry (heapcache.py:127-130 /
            // 457-460) rather than reading any pre-snapshotted bit table.
            let mut heap_cache = std::mem::take(&mut self.heap_cache);
            for cache in heap_cache.values_mut() {
                cache.invalidate_unescaped(self);
            }
            self.heap_cache = heap_cache;
            let mut heap_array_cache = std::mem::take(&mut self.heap_array_cache);
            for caches in heap_array_cache.values_mut() {
                for cache in caches.values_mut() {
                    cache.invalidate_unescaped(self);
                }
            }
            self.heap_array_cache = heap_array_cache;
            return;
        }
        // heapcache.py:372-376 — fallback: reset state for non-CALL ops
        // (release-GIL etc.) that we can't selectively invalidate.
        self.reset_keep_likely_virtuals();
    }

    /// Parity alias for RPython cache invalidation entrypoint.
    pub fn invalidate_caches<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        self.mark_escaped(opnum, None, argboxes);
        if Self::_clear_caches_not_necessary(opnum) {
            return;
        }
        // `invalidate_caches_varargs` will re-issue `mark_escaped_varargs`
        // (matching upstream's double-call shape at heapcache.py:215-216);
        // do NOT also call `mark_escaped` for non-CALL_N argboxes here —
        // upstream `invalidate_caches` (heapcache.py:212-216) ONLY does
        // the `mark_escaped` for the SETFIELD/SETARRAYITEM special cases
        // that `mark_escaped_varargs` would skip.  The 1:1 split is kept
        // by `mark_escaped`'s opnum filter.
        self.invalidate_caches_varargs(opnum, effectinfo, argboxes, oracle, const_value);
    }

    /// heapcache.py:378-381 _clear_caches_arraycopy
    ///
    /// ```text
    ///  def _clear_caches_arraycopy(self, opnum, descr, argboxes, effectinfo):
    ///      self._clear_caches_arrayop(argboxes[1], argboxes[2],
    ///                                 argboxes[3], argboxes[4], argboxes[5],
    ///                                 effectinfo)
    /// ```
    pub fn _clear_caches_arraycopy<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        _opnum: OpCode,
        _descr: Option<&EffectInfo>,
        argboxes: &[OpRef],
        single_write_descr_array: Option<u32>,
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        // argboxes layout from RPython oopspec ll_arraycopy:
        //   [func, src, dst, srcstart, dststart, length]
        if argboxes.len() < 6 {
            self.reset_keep_likely_virtuals();
            return;
        }
        self._clear_caches_arrayop(
            argboxes[1],
            argboxes[2],
            argboxes[3],
            argboxes[4],
            argboxes[5],
            single_write_descr_array,
            oracle,
            const_value,
        );
    }

    /// heapcache.py:383-386 _clear_caches_arraymove
    ///
    /// ```text
    ///  def _clear_caches_arraymove(self, opnum, descr, argboxes, effectinfo):
    ///      self._clear_caches_arrayop(argboxes[1], argboxes[1],
    ///                                 argboxes[2], argboxes[3], argboxes[4],
    ///                                 effectinfo)
    /// ```
    pub fn _clear_caches_arraymove<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        _opnum: OpCode,
        _descr: Option<&EffectInfo>,
        argboxes: &[OpRef],
        single_write_descr_array: Option<u32>,
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        // argboxes layout from RPython oopspec ll_arraymove:
        //   [func, arr, srcstart, dststart, length]
        if argboxes.len() < 5 {
            self.reset_keep_likely_virtuals();
            return;
        }
        self._clear_caches_arrayop(
            argboxes[1],
            argboxes[1],
            argboxes[2],
            argboxes[3],
            argboxes[4],
            single_write_descr_array,
            oracle,
            const_value,
        );
    }

    /// heapcache.py:388-447 _clear_caches_arrayop
    ///
    /// ```text
    ///  def _clear_caches_arrayop(self, source_box, dest_box,
    ///                            source_start_box, dest_start_box, length_box,
    ///                            effectinfo):
    ///      seen_allocation_of_target = self._check_flag(dest_box,
    ///                                                   HF_SEEN_ALLOCATION)
    ///      if (isinstance(source_start_box, ConstInt) and
    ///          isinstance(dest_start_box, ConstInt) and
    ///          isinstance(length_box, ConstInt) and
    ///          effectinfo.single_write_descr_array is not None):
    ///          ...per-index copy from source to dest...
    ///          return
    ///      elif effectinfo.single_write_descr_array is not None:
    ///          ...wholesale clear of dest descr submap...
    ///          return
    ///      self.reset_keep_likely_virtuals()
    /// ```
    ///
    /// `const_value` resolves a constant-namespace OpRef to its raw `i64`.
    /// RPython reads `box.getint()` directly from the ConstInt; majit needs
    /// a callback because HeapCache has no constant pool of its own.
    pub fn _clear_caches_arrayop_with_consts(
        &mut self,
        source_box: OpRef,
        dest_box: OpRef,
        source_start_box: OpRef,
        dest_start_box: OpRef,
        length_box: OpRef,
        single_write_descr_array: Option<u32>,
        const_value: impl Fn(OpRef) -> Option<i64>,
        oracle: &dyn SameConstantOracle,
    ) {
        let seen_allocation_of_target = self.saw_allocation(dest_box);
        let seen_allocation_of_source = self.saw_allocation(source_box);
        let srcstart = const_value(source_start_box);
        let dststart = const_value(dest_start_box);
        let length = const_value(length_box);
        if let (Some(srcstart), Some(dststart), Some(length), Some(descr)) =
            (srcstart, dststart, length, single_write_descr_array)
        {
            // heapcache.py:405-411: pick iteration direction.
            // ARRAYMOVE with srcstart < dststart needs reverse-order to
            // avoid clobbering values it still needs to read.
            let (mut index_current, index_delta, index_stop): (i64, i64, i64) =
                if srcstart < dststart {
                    (length - 1, -1, -1)
                } else {
                    (0, 1, length)
                };
            while index_current != index_stop {
                let i = index_current;
                index_current += index_delta;
                debug_assert!(i >= 0);
                // heapcache.py:418-422 — `indexcache.read(source_box)`.
                // The cache entry's `_unique_const_heuristic` canonicalises
                // the ConstPtr source so two distinct OpRefs for the same
                // gcref share the same dict slot.
                let raw_value = self
                    .heap_array_cache
                    .get_mut(&descr)
                    .and_then(|m| m.get_mut(&(srcstart + i)))
                    .and_then(|entry| {
                        let src = entry._unique_const_heuristic(source_box, oracle);
                        let dict = entry._getdict(seen_allocation_of_source);
                        dict.get(&src).copied()
                    });
                // heapcache.py:113 `return maybe_replace_with_const(res_box)`
                // — follow the FO_REPLACED_WITH_CONST forwarding so callers
                // see the canonical const replacement, not the stale Box.
                // The Box identity is the OpRef; its intrinsic `value`
                // travels with the frontend value slot, so the copy needs no
                // explicit payload handling.
                let value = raw_value.map(|opref| self.maybe_replace_with_const(opref));
                // heapcache.py:423-429: ...and write it to the dest cell.
                if let Some(value) = value {
                    let dst_index = dststart + i;
                    let entry = self
                        .heap_array_cache
                        .entry(descr)
                        .or_default()
                        .entry(dst_index)
                        .or_insert_with(CacheEntry::new);
                    // heapcache.py:90-94 `do_write_with_aliasing` —
                    // canonicalise dest, then `_clear_cache_on_write(seen_alloc)`
                    // BEFORE the insert so aliasing entries from prior
                    // writes get dropped (escaped target → wipe whole
                    // cache_anything; unescaped → only cache_anything).
                    let dst = entry._unique_const_heuristic(dest_box, oracle);
                    entry._clear_cache_on_write(seen_allocation_of_target);
                    entry
                        ._getdict_mut(seen_allocation_of_target)
                        .insert(dst, value);
                } else {
                    // heapcache.py:430-436: source had no cached value, so
                    // the dest's existing entry must be invalidated.
                    if let Some(idx_cache) = self
                        .heap_array_cache
                        .get_mut(&descr)
                        .and_then(|m| m.get_mut(&(dststart + i)))
                    {
                        idx_cache._clear_cache_on_write(seen_allocation_of_target);
                    }
                }
            }
            return;
        }
        // heapcache.py:438-446: known descr but non-constant indexes — clear
        // the entire dest descr submap.
        if let Some(descr) = single_write_descr_array {
            if let Some(submap) = self.heap_array_cache.get_mut(&descr) {
                for entry in submap.values_mut() {
                    entry._clear_cache_on_write(seen_allocation_of_target);
                }
            }
            return;
        }
        // heapcache.py:447: total fallback.
        self.reset_keep_likely_virtuals();
    }

    /// `_clear_caches_arrayop` accepts a const-resolution closure so
    /// production callers from `invalidate_caches_varargs` reach the
    /// per-index copy branch of `_clear_caches_arrayop_with_consts`
    /// (heapcache.py:393).  When the closure returns `None` for any
    /// index/length operand, the branch falls through to whole-descr
    /// clearing as upstream does (heapcache.py:438).
    pub fn _clear_caches_arrayop<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        source_box: OpRef,
        dest_box: OpRef,
        source_start_box: OpRef,
        dest_start_box: OpRef,
        length_box: OpRef,
        single_write_descr_array: Option<u32>,
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        self._clear_caches_arrayop_with_consts(
            source_box,
            dest_box,
            source_start_box,
            dest_start_box,
            length_box,
            single_write_descr_array,
            const_value,
            oracle,
        );
    }

    /// Alias kept for parity with older callsites.
    pub fn invalidate_caches_varargs_alias<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        self.invalidate_caches_varargs(opnum, effectinfo, argboxes, oracle, const_value)
    }

    /// heapcache.py:524-532 `get_field_updater(self, box, descr)`.
    ///
    /// ```text
    ///  def get_field_updater(self, box, descr):
    ///      cache = self.heap_cache.get(descr, None)
    ///      if cache is None:
    ///          cache = self.heap_cache[descr] = CacheEntry(self)
    ///          fieldbox = None
    ///      else:
    ///          fieldbox = cache.read(box)
    ///      return FieldUpdater(box, cache, fieldbox)
    /// ```
    ///
    /// `cache.read` (heapcache.py:106-114) handles the
    /// `_unique_const_heuristic` ConstPtr canonicalisation and the
    /// `maybe_replace_with_const` forwarding internally.  Need an
    /// `oracle` parameter because pyre's `same_constant` lives on the
    /// `ConstOprefOracle` (inline Const OpRef value compare) rather than
    /// on the OpRef itself.
    pub fn get_field_updater(
        &mut self,
        obj: OpRef,
        descr_index: u32,
        oracle: &dyn SameConstantOracle,
    ) -> FieldUpdater {
        let fieldbox = if let Some(mut entry) = self.heap_cache.remove(&descr_index) {
            let result = entry.read(obj, self, oracle);
            self.heap_cache.insert(descr_index, entry);
            result
        } else {
            // heapcache.py:528 `cache = self.heap_cache[descr] = CacheEntry(self)`.
            self.heap_cache.insert(descr_index, CacheEntry::new());
            None
        };
        FieldUpdater::with_cache(obj, self, descr_index, fieldbox)
    }

    // ── Array item caching (RPython heapcache.py cached_arrayitems) ──

    /// heapcache.py:542-553 `getarrayitem(self, box, indexbox, descr)`.
    /// The caller supplies the index as the raw `i64` value extracted
    /// via `ConstInt.getint()`; non-ConstInt inputs short-circuit to
    /// `None` at the caller boundary so the cache key is always the
    /// upstream-equivalent value, not the index Box's identity.
    ///
    /// `array` is routed through the indexcache's `_unique_const_heuristic`
    /// (heapcache.py:550 `indexcache.read(box)`) so two distinct
    /// ConstPtr OpRefs for the same gcref hit the same cache slot.
    pub fn getarrayitem_cache(
        &mut self,
        array: OpRef,
        index_value: i64,
        descr: u32,
        oracle: &dyn SameConstantOracle,
    ) -> Option<OpRef> {
        let entry = self
            .heap_array_cache
            .get_mut(&descr)?
            .get_mut(&index_value)?;
        let array = entry._unique_const_heuristic(array, oracle);
        let seen_alloc = self.saw_allocation(array);
        let entry = self.heap_array_cache.get(&descr)?.get(&index_value)?;
        let cached = entry._getdict(seen_alloc).get(&array).copied()?;
        Some(self.maybe_replace_with_const(cached))
    }

    /// heapcache.py:573-585 `setarrayitem`. Non-ConstInt index (`None`
    /// here) clears the whole descr submap; otherwise the cache entry
    /// for `(descr, index_value)` writes through
    /// `do_write_with_aliasing` which canonicalises `array` via
    /// `_unique_const_heuristic` before keying.
    pub fn setarrayitem_cache(
        &mut self,
        array: OpRef,
        index_value: Option<i64>,
        descr: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
    ) {
        let Some(index_value) = index_value else {
            if let Some(cache) = self.heap_array_cache.get_mut(&descr) {
                cache.clear();
            }
            return;
        };
        let seen_alloc = self.saw_allocation(array);
        let entry = self
            .heap_array_cache
            .entry(descr)
            .or_default()
            .entry(index_value)
            .or_insert_with(CacheEntry::new);
        // CacheEntry.do_write_with_aliasing internally canonicalises
        // ConstPtr operands via `_unique_const_heuristic`, replicating
        // heapcache.py:577 `indexcache.do_write_with_aliasing(box, ...)`.
        let array = entry._unique_const_heuristic(array, oracle);
        entry._clear_cache_on_write(seen_alloc);
        entry._getdict_mut(seen_alloc).insert(array, value);
    }

    /// heapcache.py:565-568 `getarrayitem_now_known`. Same canonical
    /// keying as `setarrayitem_cache` but without the alias clearing.
    pub fn getarrayitem_now_known(
        &mut self,
        array: OpRef,
        index_value: Option<i64>,
        descr: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
    ) {
        let Some(index_value) = index_value else {
            return;
        };
        let seen_alloc = self.saw_allocation(array);
        let entry = self
            .heap_array_cache
            .entry(descr)
            .or_default()
            .entry(index_value)
            .or_insert_with(CacheEntry::new);
        let array = entry._unique_const_heuristic(array, oracle);
        entry._getdict_mut(seen_alloc).insert(array, value);
    }

    /// Invalidate array caches for a specific array across every descr/index.
    pub fn invalidate_array_cache(&mut self, array: OpRef) {
        for cache in self.heap_array_cache.values_mut() {
            for entry in cache.values_mut() {
                entry.cache_anything.remove(&array);
                entry.cache_seen_allocation.remove(&array);
            }
        }
    }

    // ── Quasi-immutable tracking (RPython heapcache.py quasi_immut_known) ──

    /// Record that a quasi-immutable field is known.
    pub fn quasi_immut_now_known(&mut self, obj: OpRef, field_index: u32) {
        self.quasi_immut_known.insert((obj, field_index));
    }

    /// Check if a quasi-immutable field is already known.
    pub fn is_quasi_immut_known(&self, obj: OpRef, field_index: u32) -> bool {
        self.quasi_immut_known.contains(&(obj, field_index))
    }

    // ── Nullity tracking (heapcache.py nullity_now_known / is_nullity_known) ──

    /// heapcache.py:480-483
    ///
    /// ```text
    ///  def nullity_now_known(self, box):
    ///      if isinstance(box, Const):
    ///          return
    ///      self._set_flag(box, HF_KNOWN_NULLITY)
    /// ```
    ///
    /// pyre additionally tracks WHICH side of the nullity is known (1 =
    /// non-null, 2 = null) in the `known_nullity` Vec — RPython does not
    /// need this because callers re-read box.getref_base() at consume time.
    pub fn nullity_now_known(&mut self, opref: OpRef, is_nonnull: bool) {
        if opref.is_constant() {
            return;
        }
        let i = opref.raw() as usize;
        if i >= self.known_nullity.len() {
            self.known_nullity.resize(i + 1, 0);
        }
        self.known_nullity[i] = if is_nonnull { 1 } else { 2 };
        // RPython _set_flag(box, HF_KNOWN_NULLITY).
        self._set_flag(opref, HF_KNOWN_NULLITY);
    }

    /// Check if a value's nullity is known.
    /// heapcache.py:475-478: is_nullity_known(box)
    ///   if isinstance(box, Const): return bool(box.getref_base())
    ///
    /// `const_value` resolves a constant-namespace OpRef to its raw value.
    /// RPython reads `box.getref_base()` directly; Rust needs a lookup
    /// into the constant pool.
    pub fn is_nullity_known(
        &self,
        opref: OpRef,
        const_value: impl Fn(OpRef) -> Option<i64>,
    ) -> Option<bool> {
        if opref.is_constant() {
            // heapcache.py:477: return bool(box.getref_base())
            // A null ConstPtr (value 0) is known-null; non-zero is known-nonnull.
            return Some(const_value(opref).unwrap_or(0) != 0);
        }
        // heapcache.py:478: return self._check_flag(box, HF_KNOWN_NULLITY).
        // Version-gated so a stale `known_nullity` Vec entry from before
        // the last reset_keep_likely_virtuals does not leak through.
        if !self._check_flag(opref, HF_KNOWN_NULLITY) {
            return None;
        }
        self.known_nullity
            .get(opref.raw() as usize)
            .and_then(|v| if *v == 0 { None } else { Some(*v == 1) })
    }

    // ── Array length caching (heapcache.py arraylen_now_known / arraylen) ──

    /// heapcache.py:579-586 arraylen
    ///
    /// ```text
    ///  def arraylen(self, box):
    ///      if (isinstance(box, RefFrontendOp) and
    ///          self.test_head_version(box) and
    ///          box._heapc_deps is not None):
    ///          res_box = box._heapc_deps[0]
    ///          if res_box is not None:
    ///              return maybe_replace_with_const(res_box)
    ///      return None
    /// ```
    ///
    pub fn arraylen(&self, array: OpRef) -> Option<OpRef> {
        if array.is_constant() || !self.test_head_version(array) {
            return None;
        }
        self.heapc_deps
            .get(array.raw() as usize)
            .and_then(|deps| deps.as_ref())
            .and_then(|deps| deps.first().copied().flatten())
            .map(|opref| self.maybe_replace_with_const(opref))
    }

    /// heapcache.py:588-596 arraylen_now_known
    ///
    /// ```text
    ///  def arraylen_now_known(self, box, lengthbox):
    ///      # we store in '_heapc_deps' a list of boxes: the *first* box
    ///      # is the known length or None, and the remaining boxes are
    ///      # the regular dependencies.
    ///      if isinstance(box, Const):
    ///          return
    ///      deps = self._get_deps(box)
    ///      assert deps is not None
    ///      deps[0] = lengthbox
    /// ```
    ///
    /// `_get_deps` runs `update_version` as a side effect and ensures the
    /// `_heapc_deps` list exists with slot 0 reserved for the array length.
    pub fn arraylen_now_known(&mut self, array: OpRef, length: OpRef) {
        if array.is_constant() {
            return;
        }
        let deps = self._get_deps(array);
        deps[0] = Some(length);
    }

    // ── Likely virtual tracking (heapcache.py is_likely_virtual) ──

    /// Alias for `new_object` kept under the heapcache.py:502 name `new`.
    /// Used by `opimpl_virtual_ref` (pyjitpl.py:1807) which calls
    /// `self.metainterp.heapcache.new(resbox)` after recording VIRTUAL_REF.
    pub fn new_box(&mut self, opref: OpRef) {
        self.new_object(opref);
    }

    /// heapcache.py:496-500 is_likely_virtual.
    ///   `return (... self.test_likely_virtual_version(box) and
    ///            test_flags(box, HF_LIKELY_VIRTUAL))`
    ///
    /// Note: gates on `test_likely_virtual_version` (NOT
    /// `test_head_version`) so a `reset_keep_likely_virtuals` does not
    /// invalidate this flag — the older box is still trusted as likely
    /// virtual until the *next* version bump (line 184).
    pub fn is_likely_virtual(&self, opref: OpRef) -> bool {
        if !self.test_likely_virtual_version(opref) {
            return false;
        }
        let f = self.flags_for_ref(opref);
        (f as u8) & HF_LIKELY_VIRTUAL != 0
    }

    // ── Loop-invariant call result caching ──

    /// heapcache.py:629-634 call_loopinvariant_known_result
    ///
    /// ```text
    ///  def call_loopinvariant_known_result(self, allboxes, descr):
    ///      if self.loop_invariant_descr is not descr:
    ///          return None
    ///      if self.loop_invariant_arg0int != allboxes[0].getint():
    ///          return None
    ///      return self.loop_invariant_result
    /// ```
    ///
    /// Only ONE result is stored at a time. RPython matches by descr
    /// **identity** and the arg0 **integer value**; majit keys both
    /// values directly because the trace HeapCache deals in `descr.index()`
    /// + `i64` rather than Python objects.
    pub fn call_loopinvariant_known_result(
        &self,
        descr_index: u32,
        arg0_int: i64,
    ) -> Option<(OpRef, i64)> {
        if self.loopinvariant_descr != Some(descr_index) {
            return None;
        }
        if self.loopinvariant_arg0 != Some(arg0_int) {
            return None;
        }
        // Pair the cached symbolic OpRef with its cached concrete value
        // so the caller can return the same `(opref, value)` shape it
        // would emit for a fresh call.  See `loopinvariant_resvalue` for
        // the rationale.
        Some((
            self.loopinvariant_result?,
            self.loopinvariant_resvalue.unwrap_or(0),
        ))
    }

    /// heapcache.py:636-639 call_loopinvariant_now_known
    ///
    /// ```text
    ///  def call_loopinvariant_now_known(self, allboxes, descr, res):
    ///      self.loop_invariant_descr = descr
    ///      self.loop_invariant_arg0int = allboxes[0].getint()
    ///      self.loop_invariant_result = res
    /// ```
    pub fn call_loopinvariant_now_known(
        &mut self,
        descr_index: u32,
        arg0_int: i64,
        result: OpRef,
        resvalue: i64,
    ) {
        self.loopinvariant_descr = Some(descr_index);
        self.loopinvariant_arg0 = Some(arg0_int);
        self.loopinvariant_result = Some(result);
        self.loopinvariant_resvalue = Some(resvalue);
    }

    /// Void overload of `call_loopinvariant_now_known` — `pyjitpl.py:2109`
    /// invokes `heapcache.call_loopinvariant_now_known(allboxes, descr, res)`
    /// for `tp == 'v'` with `res = None` (`_record_helper_varargs` returns
    /// None for void).  Upstream stores `res = None` in the slot, evicting
    /// any prior typed result that shared the (descr, arg0) key.  The Rust
    /// split between symbolic `OpRef` and concrete `i64` requires a separate
    /// entry point; semantics match the upstream `res = None` store.
    pub fn call_loopinvariant_now_known_void(&mut self, descr_index: u32, arg0_int: i64) {
        self.loopinvariant_descr = Some(descr_index);
        self.loopinvariant_arg0 = Some(arg0_int);
        self.loopinvariant_result = None;
        self.loopinvariant_resvalue = None;
    }

    /// Internal alias retained for older callsites.
    pub fn call_loopinvariant_cache(
        &mut self,
        descr_index: u32,
        arg0_int: i64,
        result: OpRef,
        resvalue: i64,
    ) {
        self.call_loopinvariant_now_known(descr_index, arg0_int, result, resvalue);
    }

    /// Internal alias retained for older callsites.
    pub fn call_loopinvariant_lookup(
        &self,
        descr_index: u32,
        arg0_int: i64,
    ) -> Option<(OpRef, i64)> {
        self.call_loopinvariant_known_result(descr_index, arg0_int)
    }

    // ── Reset variants ──

    /// heapcache.py:163-181 reset
    ///
    /// ```text
    ///  def reset(self):
    ///      # Global reset of all flags. Update both version numbers so
    ///      # that any access to '_heapc_flags' will be marked as outdated.
    ///      assert self.head_version < _HF_VERSION_MAX
    ///      self.head_version += _HF_VERSION_INC
    ///      self.likely_virtual_version = self.head_version
    ///      #
    ///      # heap cache
    ///      self.heap_cache = {}
    ///      self.heap_array_cache = {}
    ///      self.need_guard_not_invalidated = True
    ///      #
    ///      # result of one loop invariant call
    ///      self.loop_invariant_result = None
    ///      self.loop_invariant_descr = None
    ///      self.loop_invariant_arg0int = -1
    /// ```
    ///
    /// majit also clears the standalone `Vec<bool>` flags
    /// (`is_unescaped`/`seen_allocation`/...) because those are NOT version-
    /// gated like RPython's `_heapc_flags` — version bump alone would not
    /// invalidate them.
    pub fn reset(&mut self) {
        // heapcache.py:166-168: bump head_version, sync likely_virtual_version.
        assert!(self.head_version < HF_VERSION_MAX);
        self.head_version += HF_VERSION_INC;
        self.likely_virtual_version = self.head_version;
        // heapcache.py:172-175: clear heap_cache + heap_array_cache.
        // Replacing `heap_cache = {}` drops every per-descr
        // `CacheEntry`, which in turn drops the per-descr
        // `last_const_box` `_unique_const_heuristic` LRU.
        self.heap_cache.clear();
        self.heap_array_cache.clear();
        // heapcache.py:176: need_guard_not_invalidated = True
        self.need_guard_not_invalidated = true;
        // heapcache.py:179-181: loop_invariant_result/descr/arg0int reset.
        self.loopinvariant_descr = None;
        self.loopinvariant_arg0 = None;
        self.loopinvariant_result = None;
        // majit-only: standalone Vec<bool> flags are not version-gated, so
        // a version bump cannot invalidate them. Clear them explicitly.
        self.known_class.clear();
        self.quasi_immut_known.clear();
        self.is_unescaped.clear();
        self.seen_allocation.clear();
        self.known_nullity.clear();
        self.likely_virtual.clear();
        self.heapc_deps.clear();
        // history.py:644-668 FO_REPLACED_WITH_CONST is stored on the
        // FrontendOp's `position_and_flags` field, so RPython's flag
        // dies with the FrontendOp at trace teardown.  pyre's
        // `replaced_with_const` is keyed by the full OpRef identity and OpRef
        // numbers are reused across traces, so clear it at the same trace
        // boundary that drops the FrontendOp objects in upstream.
        self.replaced_with_const.clear();
    }

    /// heapcache.py:176: check and consume need_guard_not_invalidated.
    /// Returns true the first time after reset (or after cache clearing).
    pub fn check_and_clear_guard_not_invalidated(&mut self) -> bool {
        let needed = self.need_guard_not_invalidated;
        self.need_guard_not_invalidated = false;
        needed
    }

    /// Whether GUARD_NOT_INVALIDATED is needed.
    pub fn need_guard_not_invalidated(&self) -> bool {
        self.need_guard_not_invalidated
    }

    /// heapcache.py:183-189 reset_keep_likely_virtuals
    ///
    /// ```text
    ///  def reset_keep_likely_virtuals(self):
    ///      # Update only 'head_version', but 'likely_virtual_version'
    ///      # remains at its older value.
    ///      assert self.head_version < _HF_VERSION_MAX
    ///      self.head_version += _HF_VERSION_INC
    ///      self.heap_cache = {}
    ///      self.heap_array_cache = {}
    /// ```
    ///
    /// `likely_virtual`, `loopinvariant_*`, and `_heapc_deps`
    /// `need_guard_not_invalidated` are intentionally preserved (a residual
    /// call that releases the GIL invalidates heap caches but the JIT can
    /// still trust prior allocation/likely-virtual hints).
    pub fn reset_keep_likely_virtuals(&mut self) {
        assert!(self.head_version < HF_VERSION_MAX);
        self.head_version += HF_VERSION_INC;
        self.heap_cache.clear();
        self.heap_array_cache.clear();
    }

    /// RPython-compatible alias kept for existing codepaths.
    pub fn _remove_deps_for_box(&mut self, opref: OpRef) {
        let i = opref.raw() as usize;
        if i < self.heapc_deps.len() {
            self.heapc_deps[i] = None;
        }
    }
}

impl Default for HeapCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test fixture for `_unique_const_heuristic`: two ConstPtr OpRefs
    /// `(typed-Ref, raw=10000)` and `(typed-Ref, raw=10001)` compare
    /// equal under the oracle iff their pre-registered indices are.
    struct FixedSameConstantOracle {
        same_pairs: Vec<(OpRef, OpRef)>,
    }

    impl SameConstantOracle for FixedSameConstantOracle {
        fn same_constant(&self, a: OpRef, b: OpRef) -> bool {
            if a == b {
                return true;
            }
            self.same_pairs
                .iter()
                .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
        }
    }

    /// Identity-only oracle for tests that exercise non-ConstPtr OpRefs.
    /// Same as `FixedSameConstantOracle { same_pairs: vec![] }` but
    /// shorter at the callsite.
    struct IdentitySameConstantOracle;

    impl SameConstantOracle for IdentitySameConstantOracle {
        fn same_constant(&self, a: OpRef, b: OpRef) -> bool {
            a == b
        }
    }

    const IDENTITY_ORACLE: &dyn SameConstantOracle = &IdentitySameConstantOracle;

    /// `_unique_const_heuristic` collapses consecutive equal ConstPtr
    /// arguments to the cached `last_const_box`, even when the two
    /// OpRefs are distinct (post-dedup-retirement shape).
    #[test]
    fn unique_const_heuristic_canonicalises_to_last_via_same_constant() {
        let mut entry = CacheEntry::new();
        let a = OpRef::const_ptr(majit_ir::GcRef(0xA000));
        let b = OpRef::const_ptr(majit_ir::GcRef(0xB000));
        let oracle = FixedSameConstantOracle {
            same_pairs: vec![(a, b)],
        };
        assert_eq!(entry._unique_const_heuristic(a, &oracle), a);
        assert_eq!(entry._unique_const_heuristic(b, &oracle), a);
    }

    /// Non-constant OpRefs bypass the heuristic unchanged
    /// (heapcache.py:99 `isinstance(ref_box, ConstPtr)` guard).
    #[test]
    fn unique_const_heuristic_skips_non_constant() {
        let mut entry = CacheEntry::new();
        let oracle = FixedSameConstantOracle { same_pairs: vec![] };
        let op = OpRef::ref_op(7);
        assert_eq!(entry._unique_const_heuristic(op, &oracle), op);
        assert!(entry.last_const_box.is_none());
    }

    /// Non-Ref-typed constants (ConstInt / ConstFloat) bypass the
    /// heuristic — upstream only canonicalises ConstPtr.
    #[test]
    fn unique_const_heuristic_skips_non_ref_constants() {
        let mut entry = CacheEntry::new();
        let oracle = FixedSameConstantOracle { same_pairs: vec![] };
        let ci = OpRef::const_int(42);
        assert_eq!(entry._unique_const_heuristic(ci, &oracle), ci);
        assert!(entry.last_const_box.is_none());
    }

    #[test]
    fn test_field_cache_basic() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(0);
        let field = 1;
        let val = OpRef::ref_op(2);

        assert_eq!(cache.getfield_cached(obj, field, IDENTITY_ORACLE), None);

        cache.getfield_now_known(obj, field, val, IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj, field, IDENTITY_ORACLE),
            Some(val)
        );
    }

    #[test]
    fn test_field_cache_overwrite() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(0);
        let field = 1;

        cache.getfield_now_known(obj, field, OpRef::ref_op(10), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(10))
        );

        cache.getfield_now_known(obj, field, OpRef::ref_op(20), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
    }

    #[test]
    fn test_setfield_aliasing() {
        let mut cache = HeapCache::new();
        let obj_a = OpRef::ref_op(0);
        let obj_b = OpRef::ref_op(1);
        let field = 5;

        // Both objects have a known field value
        cache.getfield_now_known(obj_a, field, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(obj_b, field, OpRef::ref_op(20), IDENTITY_ORACLE);

        // Writing to obj_a (which is NOT unescaped) should invalidate
        // obj_b's field cache for the same field (potential aliasing).
        cache.setfield_cached(obj_a, field, OpRef::ref_op(30), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj_a, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(30))
        );
        assert_eq!(cache.getfield_cached(obj_b, field, IDENTITY_ORACLE), None); // invalidated
    }

    /// heapcache.py:70-77 `_clear_cache_on_write(seen_alloc)`.  When the
    /// write target is seen-allocated, only `cache_anything` is cleared
    /// — entries for other seen-allocated boxes in
    /// `cache_seen_allocation` survive because distinct
    /// seen-allocation identities cannot alias each other.
    #[test]
    fn test_setfield_seen_alloc_preserves_other_seen_alloc_entries() {
        let mut cache = HeapCache::new();
        let obj_a = OpRef::ref_op(0);
        let obj_b = OpRef::ref_op(1);
        let field = 5;

        // Both targets have been observed allocating, so each lives in
        // the seen-allocation bucket and they don't alias each other.
        cache.new_object(obj_a);
        cache.new_object(obj_b);
        cache.getfield_now_known(obj_a, field, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(obj_b, field, OpRef::ref_op(20), IDENTITY_ORACLE);

        // Writing to obj_a leaves obj_b's seen-alloc entry intact.
        cache.setfield_cached(obj_a, field, OpRef::ref_op(30), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj_a, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(30))
        );
        assert_eq!(
            cache.getfield_cached(obj_b, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
    }

    /// heapcache.py:70-77 — when the write target is seen-allocated but
    /// some other cached box is not, the non-seen-alloc entry lives in
    /// `cache_anything` and is dropped by `_clear_cache_on_write` even
    /// though the target itself is in `cache_seen_allocation`.
    #[test]
    fn test_setfield_seen_alloc_clears_cache_anything() {
        let mut cache = HeapCache::new();
        let obj_a = OpRef::ref_op(0);
        let obj_b = OpRef::ref_op(1);
        let field = 5;

        cache.new_object(obj_a);
        // obj_b is NOT new_object'd → lives in cache_anything.
        cache.getfield_now_known(obj_a, field, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(obj_b, field, OpRef::ref_op(20), IDENTITY_ORACLE);

        cache.setfield_cached(obj_a, field, OpRef::ref_op(30), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj_a, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(30))
        );
        assert_eq!(cache.getfield_cached(obj_b, field, IDENTITY_ORACLE), None);
    }

    #[test]
    fn test_invalidate_caches() {
        let mut cache = HeapCache::new();
        cache.getfield_now_known(OpRef::ref_op(0), 1, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(OpRef::ref_op(1), 2, OpRef::ref_op(20), IDENTITY_ORACLE);

        cache.reset_keep_likely_virtuals();
        assert_eq!(
            cache.getfield_cached(OpRef::ref_op(0), 1, IDENTITY_ORACLE),
            None
        );
        assert_eq!(
            cache.getfield_cached(OpRef::ref_op(1), 2, IDENTITY_ORACLE),
            None
        );
    }

    #[test]
    fn test_invalidate_caches_for_escaped() {
        let mut cache = HeapCache::new();
        let escaped_obj = OpRef::ref_op(0);
        let unescaped_obj = OpRef::ref_op(1);

        cache.new_object(unescaped_obj);
        cache.getfield_now_known(escaped_obj, 1, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(unescaped_obj, 1, OpRef::ref_op(20), IDENTITY_ORACLE);

        cache.invalidate_caches_for_escaped();
        assert_eq!(cache.getfield_cached(escaped_obj, 1, IDENTITY_ORACLE), None);
        assert_eq!(
            cache.getfield_cached(unescaped_obj, 1, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
    }

    #[test]
    fn test_new_object() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(5);

        assert!(!cache.is_unescaped(obj));
        assert!(!cache.saw_allocation(obj));

        cache.new_object(obj);
        assert!(cache.is_unescaped(obj));
        assert!(cache.saw_allocation(obj));
    }

    #[test]
    fn test_mark_escaped() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(5);

        cache.new_object(obj);
        assert!(cache.is_unescaped(obj));

        cache._escape_box(obj);
        assert!(!cache.is_unescaped(obj));
        // saw_allocation is permanent
        assert!(cache.saw_allocation(obj));
    }

    #[test]
    fn test_known_class() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(0);
        let cls = 0x1000_i64;

        assert!(!cache.is_class_known(obj));
        assert_eq!(cache.get_known_class(obj), None);

        cache.class_now_known(obj, cls);
        assert!(cache.is_class_known(obj));
        assert_eq!(cache.get_known_class(obj), Some(cls));
    }

    #[test]
    fn test_notify_guard_class_preserves_inline_class_value() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(3);
        let cls_val = 0xCAFE_i64;
        let cls = OpRef::const_int(cls_val);

        cache.notify_op(OpCode::GuardClass, &[obj, cls], OpRef::NONE);

        assert!(cache.is_class_known(obj));
        assert_eq!(cache.get_known_class(obj), Some(cls_val));
    }

    #[test]
    fn test_walk_const_ptr_refs_forwards_replaced_with_const() {
        // history.py:314 parity: a ConstPtr cached as a replacement
        // value must survive a moving minor collection. Forward it and read
        // back through maybe_replace_with_const.
        let mut cache = HeapCache::new();
        let old = OpRef::ref_op(3);
        let new = OpRef::const_ptr(GcRef(0x1000));
        cache.replace_box(old, new);
        assert_eq!(cache.maybe_replace_with_const(old), new);

        cache.walk_const_ptr_refs(&mut |gcref: &mut GcRef| {
            gcref.0 = gcref.0.wrapping_add(0x1_0000);
        });

        assert_eq!(
            cache.maybe_replace_with_const(old),
            OpRef::const_ptr(GcRef(0x1_1000))
        );
    }

    #[test]
    fn test_walk_const_ptr_refs_leaves_cache_keys_stale() {
        // Cache *keys* are intentionally not forwarded: an in-place key
        // rewrite would break the sorted-VecMap ordering, and a stale key
        // simply misses + repopulates (the live lookup arrives already
        // forwarded). A `ConstPtr` object used as a `cache_anything`
        // key must therefore stay at its pre-collection address.
        let mut cache = HeapCache::new();
        let const_obj = OpRef::const_ptr(GcRef(0x2000));
        let field = 7;
        // A non-const cached value so the walk leaves the value slot alone
        // and the assertions isolate the key's address.
        cache.getfield_now_known(const_obj, field, OpRef::ref_op(20), IDENTITY_ORACLE);

        cache.walk_const_ptr_refs(&mut |gcref: &mut GcRef| {
            gcref.0 = gcref.0.wrapping_add(0x1_0000);
        });

        // The entry is still keyed by the original (pre-move) address...
        assert_eq!(
            cache.getfield_cached(const_obj, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
        // ...and was NOT re-keyed to the forwarded address.
        assert_eq!(
            cache.getfield_cached(OpRef::const_ptr(GcRef(0x1_2000)), field, IDENTITY_ORACLE),
            None
        );
    }

    #[test]
    fn test_notify_op_malloc() {
        let mut cache = HeapCache::new();
        let result = OpRef::ref_op(3);

        cache.notify_op(OpCode::New, &[], result);
        assert!(cache.is_unescaped(result));
        assert!(cache.saw_allocation(result));
    }

    #[test]
    fn test_reset() {
        let mut cache = HeapCache::new();
        cache.new_object(OpRef::ref_op(0));
        cache.class_now_known(OpRef::ref_op(0), 0x1000);
        cache.getfield_now_known(OpRef::ref_op(0), 1, OpRef::ref_op(10), IDENTITY_ORACLE);

        cache.reset();
        assert!(!cache.is_unescaped(OpRef::ref_op(0)));
        assert!(!cache.is_class_known(OpRef::ref_op(0)));
        assert_eq!(
            cache.getfield_cached(OpRef::ref_op(0), 1, IDENTITY_ORACLE),
            None
        );
    }

    #[test]
    fn test_different_fields_independent() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(0);

        cache.getfield_now_known(obj, 1, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(obj, 2, OpRef::ref_op(20), IDENTITY_ORACLE);

        // Writing field 1 should not affect field 2
        cache.setfield_cached(obj, 1, OpRef::ref_op(30), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj, 1, IDENTITY_ORACLE),
            Some(OpRef::ref_op(30))
        );
        assert_eq!(
            cache.getfield_cached(obj, 2, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
    }

    #[test]
    fn test_recursive_escape() {
        let mut cache = HeapCache::new();
        let container = OpRef::ref_op(0);
        let value = OpRef::ref_op(1);
        let inner = OpRef::ref_op(2);

        cache.new_object(container);
        cache.new_object(value);
        cache.new_object(inner);

        // SETFIELD_GC(container, value): value stored in container
        cache.notify_op(OpCode::SetfieldGc, &[container, value], OpRef::NONE);
        // SETFIELD_GC(value, inner): inner stored in value
        cache.notify_op(OpCode::SetfieldGc, &[value, inner], OpRef::NONE);

        // Container is still unescaped
        assert!(cache.is_unescaped(container));
        // Value is still unescaped (container is unescaped)
        assert!(cache.is_unescaped(value));

        // Now mark container as escaped
        cache._escape_box(container);
        assert!(!cache.is_unescaped(container));
        // value should also be escaped (stored in container)
        assert!(!cache.is_unescaped(value));
    }

    #[test]
    fn test_nullity_tracking() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(10);

        assert_eq!(cache.is_nullity_known(obj, |_| None), None);
        cache.nullity_now_known(obj, true);
        assert_eq!(cache.is_nullity_known(obj, |_| None), Some(true));

        cache.nullity_now_known(obj, false);
        assert_eq!(cache.is_nullity_known(obj, |_| None), Some(false));
    }

    #[test]
    fn test_arraylen_caching() {
        let mut cache = HeapCache::new();
        let arr = OpRef::ref_op(5);

        assert_eq!(cache.arraylen(arr), None);
        cache.arraylen_now_known(arr, OpRef::int_op(100));
        assert_eq!(cache.arraylen(arr), Some(OpRef::int_op(100)));
    }

    #[test]
    fn test_arraylen_reset_keep_likely_virtuals_invalidates_length() {
        let mut cache = HeapCache::new();
        let arr = OpRef::ref_op(5);

        cache.arraylen_now_known(arr, OpRef::int_op(100));
        assert_eq!(cache.arraylen(arr), Some(OpRef::int_op(100)));

        cache.reset_keep_likely_virtuals();
        assert_eq!(cache.arraylen(arr), None);
    }

    #[test]
    fn test_replace_box_marks_old_as_const() {
        let mut cache = HeapCache::new();
        let old = OpRef::ref_op(5);
        let new = OpRef::const_ptr(majit_ir::GcRef(0xDEAD));

        cache.arraylen_now_known(old, old);
        cache.replace_box(old, new);

        assert_eq!(cache.arraylen(old), Some(new));
    }

    #[test]
    fn test_replace_box_keeps_typed_opref_identity() {
        let mut cache = HeapCache::new();
        let old_ref = OpRef::ref_op(5);
        let same_raw_int = OpRef::int_op(5);
        let new_ref = OpRef::const_ptr(GcRef(0));

        cache.arraylen_now_known(old_ref, old_ref);
        cache.arraylen_now_known(OpRef::ref_op(6), same_raw_int);
        cache.replace_box(old_ref, new_ref);

        assert_eq!(cache.arraylen(old_ref), Some(new_ref));
        assert_eq!(cache.arraylen(OpRef::ref_op(6)), Some(same_raw_int));
    }

    #[test]
    fn test_likely_virtual() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(3);

        assert!(!cache.is_likely_virtual(obj));
        cache.new_object(obj);
        assert!(cache.is_likely_virtual(obj));

        // reset keeps likely_virtual
        cache.reset_keep_likely_virtuals();
        assert!(cache.is_likely_virtual(obj));

        // full reset clears it
        cache.reset();
        assert!(!cache.is_likely_virtual(obj));
    }

    #[test]
    fn test_guard_tracking_in_notify_op() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(10);

        // GUARD_NONNULL makes nullity known
        cache.notify_op(OpCode::GuardNonnull, &[obj], OpRef::NONE);
        assert_eq!(cache.is_nullity_known(obj, |_| None), Some(true));
    }

    #[test]
    fn test_loopinvariant_void_evicts_typed_slot() {
        let mut cache = HeapCache::new();
        let descr_index: u32 = 7;
        let arg0_int: i64 = 0xC0FFEE;

        // Prime with a typed entry — pyjitpl.py:2087-2110 tp == 'i' branch.
        let typed_result = OpRef::ref_op(42);
        cache.call_loopinvariant_now_known(descr_index, arg0_int, typed_result, 99);
        assert_eq!(
            cache.call_loopinvariant_known_result(descr_index, arg0_int),
            Some((typed_result, 99))
        );

        // pyjitpl.py:2103-2109 tp == 'v' branch: res = None,
        // call_loopinvariant_now_known(allboxes, descr, None).
        cache.call_loopinvariant_now_known_void(descr_index, arg0_int);

        // heapcache.py:629-634: subsequent lookup returns None — the
        // (descr, arg0) slot is still owned but its result is None,
        // so `if res is not None: return res` short-circuit misses.
        assert_eq!(
            cache.call_loopinvariant_known_result(descr_index, arg0_int),
            None
        );
    }
}
