/// Heap cache for the tracing phase.
///
/// During tracing, the heap cache tracks field reads/writes to eliminate
/// redundant loads. If we read a field from an object and it was already
/// read or written in the same trace, we can reuse the cached value.
///
/// Translated from rpython/jit/metainterp/heapcache.py.
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

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

use majit_ir::{EffectInfo, ExtraEffect, GcRef, OpCode, OpRef};

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

#[derive(Debug, Default)]
pub(crate) struct CacheEntry {
    cache_anything: HashMap<OpRef, OpRef>,
    cache_seen_allocation: HashMap<OpRef, OpRef>,
    quasiimmut_seen: Option<HashSet<OpRef>>,
    quasiimmut_seen_refs: Option<HashSet<usize>>,
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
    pub fn _getdict(&self, seen_alloc: bool) -> &HashMap<OpRef, OpRef> {
        if seen_alloc {
            &self.cache_seen_allocation
        } else {
            &self.cache_anything
        }
    }

    /// Pyre adapt: Python doesn't need a separate `_mut` accessor;
    /// Rust's borrow checker does.  Mirrors `_getdict`'s body.
    pub fn _getdict_mut(&mut self, seen_alloc: bool) -> &mut HashMap<OpRef, OpRef> {
        if seen_alloc {
            &mut self.cache_seen_allocation
        } else {
            &mut self.cache_anything
        }
    }

    /// heapcache.py:90-94 do_write_with_aliasing
    #[allow(dead_code)]
    pub fn do_write_with_aliasing(&mut self, ref_box: OpRef, fieldbox: OpRef, cache: &HeapCache) {
        let ref_box = self._unique_const_heuristic(ref_box);
        let seen_alloc = self._seen_alloc(ref_box, cache);
        self._clear_cache_on_write(seen_alloc);
        self._getdict_mut(seen_alloc).insert(ref_box, fieldbox);
    }

    /// heapcache.py:96-104 _unique_const_heuristic
    pub fn _unique_const_heuristic(&mut self, ref_box: OpRef) -> OpRef {
        if let Some(last) = self.last_const_box {
            if last == ref_box {
                return last;
            }
        }
        self.last_const_box = Some(ref_box);
        ref_box
    }

    /// heapcache.py:106-114 read
    #[allow(dead_code)]
    pub fn read(&mut self, ref_box: OpRef, cache: &HeapCache) -> Option<OpRef> {
        let ref_box = self._unique_const_heuristic(ref_box);
        let seen_alloc = self._seen_alloc(ref_box, cache);
        self._getdict(seen_alloc)
            .get(&ref_box)
            .copied()
            .map(|opref| cache.maybe_replace_with_const(opref))
    }

    /// heapcache.py:116-119 read_now_known
    #[allow(dead_code)]
    pub fn read_now_known(&mut self, ref_box: OpRef, fieldbox: OpRef, cache: &HeapCache) {
        let ref_box = self._unique_const_heuristic(ref_box);
        let seen_alloc = self._seen_alloc(ref_box, cache);
        self._getdict_mut(seen_alloc).insert(ref_box, fieldbox);
    }

    /// heapcache.py:121-129 invalidate_unescaped — RPython makes this a
    /// public method (no underscore prefix) and `_invalidate_unescaped`
    /// is the helper that walks both caches.  pyre keeps the same
    /// public/private pair.
    pub fn invalidate_unescaped(&mut self, unescaped: &[bool]) {
        self._invalidate_unescaped(unescaped)
    }

    pub fn _invalidate_unescaped(&mut self, unescaped: &[bool]) {
        self.cache_anything.retain(|box_ref, _| {
            unescaped
                .get(box_ref.raw() as usize)
                .copied()
                .unwrap_or(false)
        });
        self.cache_seen_allocation.retain(|box_ref, _| {
            unescaped
                .get(box_ref.raw() as usize)
                .copied()
                .unwrap_or(false)
        });
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

    pub fn getfield_now_known(&mut self, fieldbox: OpRef) {
        self.currfieldbox = Some(fieldbox);
    }

    pub fn setfield(&mut self, _fieldbox: OpRef) {
        self.currfieldbox = Some(_fieldbox);
        let Some(descr) = self.descr else {
            return;
        };
        // RPython writes through the cache updater; we keep the descriptor
        // and target object in the updater.
        if !self.cache.is_null() {
            // SAFETY: `FieldUpdater` is only created by `HeapCache::get_field_updater`
            // and tied to an active mutable borrow of that cache.
            unsafe {
                let cache = &mut *self.cache;
                cache.setfield(self.ref_box, descr, _fieldbox);
            }
        }
    }
}

/// Heap cache for the tracing interpreter.
///
/// Tracks field values, known classes, and allocation status during
/// a single trace recording session.
pub struct HeapCache {
    /// Field cache: (object_ref, field_descr_index) -> cached value.
    field_cache: HashMap<(OpRef, u32), OpRef>,

    heap_cache: HashMap<u32, CacheEntry>,
    /// heapcache.py: `cached_arrayitems` — nested map descr → ConstInt-index → CacheEntry.
    heap_array_cache: HashMap<u32, HashMap<u32, CacheEntry>>,

    /// Known class map: object_ref -> class pointer.
    /// RPython: stored inside `CacheEntry`. Vec indexed by OpRef.0.
    known_class: Vec<Option<GcRef>>,

    /// Quasi-immutable fields known in this trace.
    /// heapcache.py: `quasi_immut_known`.
    quasi_immut_known: HashSet<(OpRef, u32)>,

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
    /// PRE-EXISTING-ADAPTATION: upstream's `result` is a Box that
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
    /// RPython stores this on the box's `_forwarded`; pyre stores the same
    /// bit of heapcache-visible state in a per-op slot keyed by OpRef.
    replaced_with_const: Vec<Option<OpRef>>,

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
            field_cache: HashMap::new(),
            heap_cache: HashMap::new(),
            heap_array_cache: HashMap::new(),
            known_class: Vec::new(),
            quasi_immut_known: HashSet::new(),
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
            .get(opref.raw() as usize)
            .and_then(|v| *v)
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
                if self.known_class[i].is_none() {
                    self.known_class[i] = Some(GcRef(0));
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

    /// heapcache.py:259-281 mark_escaped_varargs.
    /// RPython has both mark_escaped (specialize.arg(1) for opnum) and
    /// mark_escaped_varargs.  pyre routes both through the same impl
    /// since Rust does not need the per-opnum specialization.
    pub fn mark_escaped_varargs(
        &mut self,
        opnum: OpCode,
        descr: Option<OpRef>,
        argboxes: &[OpRef],
    ) {
        self.mark_escaped(opnum, descr, argboxes)
    }

    /// RPython: _escape_argboxes(*argboxes)
    pub fn _escape_argboxes(&mut self, args: &[OpRef]) {
        if args.is_empty() {
            return;
        }
        self._escape_box(args[0]);
        self._escape_argboxes(&args[1..]);
    }

    /// Look up a cached field value.
    ///
    /// Returns the OpRef that holds the value of `(obj, field_index)` if
    /// it was previously read or written in this trace, or None.
    pub fn getfield(&self, obj: OpRef, field_index: u32) -> Option<OpRef> {
        self.getfield_cached(obj, field_index)
    }

    pub fn getfield_cached(&self, obj: OpRef, field_index: u32) -> Option<OpRef> {
        self.field_cache.get(&(obj, field_index)).copied()
    }

    /// Record a field value (either from a GETFIELD result or a SETFIELD value).
    ///
    /// After `setfield_cached(obj, field_index, value)`, any subsequent
    /// `getfield_cached(obj, field_index)` will return `Some(value)`.
    ///
    /// When the object is not unescaped, this also invalidates entries for
    /// the same field on other objects (aliasing).
    pub fn setfield(&mut self, obj: OpRef, field_index: u32, value: OpRef) {
        self.setfield_cached(obj, field_index, value);
    }

    pub fn setfield_cached(&mut self, obj: OpRef, field_index: u32, value: OpRef) {
        // heapcache.py:493 is_unescaped — version-gated through _check_flag.
        let obj_is_unescaped = self.is_unescaped(obj);
        if !obj_is_unescaped {
            // Potential aliasing: clear all cached values for this field
            // from objects that are not known-unescaped.
            // Snapshot version + flags so the closure can re-check without
            // borrowing self mutably.
            let head_version = self.head_version;
            let flags_snapshot: Vec<u32> = self.heapc_flags.clone();
            self.field_cache.retain(|&(cached_obj, cached_field), _| {
                if cached_field != field_index {
                    return true;
                }
                // Keep entries for unescaped objects (no aliasing possible)
                let i = cached_obj.raw() as usize;
                let f = flags_snapshot.get(i).copied().unwrap_or(0);
                Self::versioned_or(f, head_version) && (f as u8) & HF_IS_UNESCAPED != 0
            });
        }
        self.field_cache.insert((obj, field_index), value);
    }

    pub fn getfield_now_known(&mut self, obj: OpRef, field_index: u32, value: OpRef) {
        self.field_cache.insert((obj, field_index), value);
    }

    /// heapcache.py: invalidate_unescaped — clear cached values for
    /// escaped objects only. Unescaped (newly allocated) objects cannot
    /// be affected by external calls, so their caches are preserved.
    pub fn invalidate_caches_for_escaped(&mut self) {
        // heapcache.py:493 — version-gated is_unescaped check via the
        // shared (head_version, heapc_flags) snapshot pattern so the
        // closure does not need to borrow self.
        let head_version = self.head_version;
        let flags_snapshot: Vec<u32> = self.heapc_flags.clone();
        let still_unescaped = |obj: OpRef| -> bool {
            let i = obj.raw() as usize;
            let f = flags_snapshot.get(i).copied().unwrap_or(0);
            Self::versioned_or(f, head_version) && (f as u8) & HF_IS_UNESCAPED != 0
        };
        self.field_cache.retain(|&(obj, _), _| still_unescaped(obj));
        // heapcache.py:542-552: iterate cached_arrayitems and invalidate
        // per-CacheEntry entries whose box is no longer unescaped.
        let unescaped: Vec<bool> = (0..flags_snapshot.len())
            .map(|i| still_unescaped(OpRef::ref_op(i as u32)))
            .collect();
        for caches in self.heap_array_cache.values_mut() {
            for cache in caches.values_mut() {
                cache.invalidate_unescaped(&unescaped);
            }
        }
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
            let i = old.raw() as usize;
            if i >= self.replaced_with_const.len() {
                self.replaced_with_const.resize(i + 1, None);
            }
            self.replaced_with_const[i] = Some(new);
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
    /// pyre additionally remembers the concrete class pointer in the
    /// `known_class` Vec because OpRef carries no static type token —
    /// RPython retrieves the class via box.getref_base().
    pub fn class_now_known(&mut self, opref: OpRef, class: GcRef) {
        if opref.is_constant() {
            return;
        }
        let i = opref.raw() as usize;
        if i >= self.known_class.len() {
            self.known_class.resize(i + 1, None);
        }
        self.known_class[i] = Some(class);
        // RPython _set_flag(box, HF_KNOWN_CLASS | HF_KNOWN_NULLITY).
        self._set_flag(opref, HF_KNOWN_CLASS);
        // RPython also writes HF_KNOWN_NULLITY in the same _set_flag call;
        // route through nullity_now_known so the Vec<u8> value mirror also
        // captures non-null.
        self.nullity_now_known(opref, true);
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
    pub fn get_known_class(&self, opref: OpRef) -> Option<GcRef> {
        if opref.is_constant() {
            return None;
        }
        if !self._check_flag(opref, HF_KNOWN_CLASS) {
            return None;
        }
        self.known_class.get(opref.raw() as usize).and_then(|v| *v)
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
        // heapcache.py: SETFIELD_GC tracking.
        // The written value becomes reachable from the container.
        // If the container later escapes, the value also escapes.
        // heapcache.py: _escape_from_write — only record dependency if
        // BOTH container and value are unescaped. If container is unescaped
        // but value is not, escape the value immediately.
        if opcode == OpCode::SetfieldGc && args.len() >= 2 {
            let container = args[0];
            let value = args[1];
            // heapcache.py:493 is_unescaped — version-gated.
            let container_unescaped = self.is_unescaped(container);
            let value_unescaped = self.is_unescaped(value);
            if container_unescaped && value_unescaped {
                self._get_deps(container).push(Some(value));
            } else if container_unescaped {
                // Container unescaped, value already escaped — no-op
            } else {
                self._escape_box(value);
            }
        }
        if opcode == OpCode::SetarrayitemGc && args.len() >= 3 {
            let container = args[0];
            let value = args[2];
            let container_unescaped = self.is_unescaped(container);
            let value_unescaped = self.is_unescaped(value);
            if container_unescaped && value_unescaped {
                self._get_deps(container).push(Some(value));
            } else if container_unescaped {
                // Container unescaped, value already escaped — no-op
            } else {
                self._escape_box(value);
            }
        }
        // heapcache.py: GUARD_VALUE → known constant + nonnull.
        if opcode == OpCode::GuardValue && args.len() >= 2 {
            self.nullity_now_known(args[0], true);
        }
        // heapcache.py: GUARD_CLASS/GUARD_NONNULL_CLASS → known class.
        if opcode == OpCode::GuardClass || opcode == OpCode::GuardNonnullClass {
            if args.len() >= 2 {
                if let Some(class_val) = args.get(1) {
                    self.class_now_known(args[0], GcRef(class_val.raw() as usize));
                }
            }
            self.nullity_now_known(args[0], true);
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
    pub fn invalidate_caches_varargs(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
    ) {
        self.mark_escaped_varargs(opnum, None, argboxes);
        if Self::_clear_caches_not_necessary(opnum) {
            return;
        }
        self.clear_caches_varargs(opnum, effectinfo, argboxes);
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
    pub fn clear_caches(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
    ) {
        self.clear_caches_varargs(opnum, effectinfo, argboxes)
    }

    /// heapcache.py:341-376 clear_caches_varargs.
    pub fn clear_caches_varargs(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
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
                    self._clear_caches_arraycopy(opnum, None, argboxes, single_descr_idx);
                    return;
                }
                if ei.oopspecindex == majit_ir::OopSpecIndex::Arraymove {
                    self._clear_caches_arraymove(opnum, None, argboxes, single_descr_idx);
                    return;
                }
            }
            // heapcache.py:362-369 — only invalidate things that escaped.
            let unescaped = self.is_unescaped.clone();
            for cache in self.heap_cache.values_mut() {
                cache.invalidate_unescaped(&unescaped);
            }
            for caches in self.heap_array_cache.values_mut() {
                for cache in caches.values_mut() {
                    cache.invalidate_unescaped(&unescaped);
                }
            }
            return;
        }
        // heapcache.py:372-376 — fallback: reset state for non-CALL ops
        // (release-GIL etc.) that we can't selectively invalidate.
        self.reset_keep_likely_virtuals();
    }

    /// Parity alias for RPython cache invalidation entrypoint.
    pub fn invalidate_caches(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
    ) {
        self.mark_escaped(opnum, None, argboxes);
        if Self::_clear_caches_not_necessary(opnum) {
            return;
        }
        self.invalidate_caches_varargs(opnum, effectinfo, argboxes);
    }

    /// heapcache.py:378-381 _clear_caches_arraycopy
    ///
    /// ```text
    ///  def _clear_caches_arraycopy(self, opnum, descr, argboxes, effectinfo):
    ///      self._clear_caches_arrayop(argboxes[1], argboxes[2],
    ///                                 argboxes[3], argboxes[4], argboxes[5],
    ///                                 effectinfo)
    /// ```
    pub fn _clear_caches_arraycopy(
        &mut self,
        _opnum: OpCode,
        _descr: Option<&EffectInfo>,
        argboxes: &[OpRef],
        single_write_descr_array: Option<u32>,
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
    pub fn _clear_caches_arraymove(
        &mut self,
        _opnum: OpCode,
        _descr: Option<&EffectInfo>,
        argboxes: &[OpRef],
        single_write_descr_array: Option<u32>,
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
    ) {
        let seen_allocation_of_target = self.saw_allocation(dest_box);
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
                // heapcache.py:418-422: read the source value...
                let value = self
                    .heap_array_cache
                    .get(&descr)
                    .and_then(|m| m.get(&((srcstart + i) as u32)))
                    .and_then(|entry| {
                        // RPython: indexcache.read(box) — looks up `box`
                        // in cache_anything / cache_seen_allocation.
                        let saw = self.saw_allocation(source_box);
                        if saw {
                            entry.cache_seen_allocation.get(&source_box).copied()
                        } else {
                            entry.cache_anything.get(&source_box).copied()
                        }
                    });
                // heapcache.py:423-429: ...and write it to the dest cell.
                if let Some(value) = value {
                    let dst_index = (dststart + i) as u32;
                    let entry = self
                        .heap_array_cache
                        .entry(descr)
                        .or_default()
                        .entry(dst_index)
                        .or_insert_with(CacheEntry::new);
                    // RPython setarrayitem -> indexcache.do_write_with_aliasing.
                    // We approximate by writing into the appropriate sub-map.
                    if seen_allocation_of_target {
                        entry.cache_seen_allocation.insert(dest_box, value);
                    } else {
                        entry.cache_anything.insert(dest_box, value);
                    }
                } else {
                    // heapcache.py:430-436: source had no cached value, so
                    // the dest's existing entry must be invalidated.
                    if let Some(idx_cache) = self
                        .heap_array_cache
                        .get_mut(&descr)
                        .and_then(|m| m.get_mut(&((dststart + i) as u32)))
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

    /// Convenience entrypoint with no constant resolution — falls through
    /// to `reset_keep_likely_virtuals` whenever indices cannot be evaluated.
    pub fn _clear_caches_arrayop(
        &mut self,
        source_box: OpRef,
        dest_box: OpRef,
        source_start_box: OpRef,
        dest_start_box: OpRef,
        length_box: OpRef,
        single_write_descr_array: Option<u32>,
    ) {
        self._clear_caches_arrayop_with_consts(
            source_box,
            dest_box,
            source_start_box,
            dest_start_box,
            length_box,
            single_write_descr_array,
            |_| None,
        );
    }

    /// Alias kept for parity with older callsites.
    pub fn invalidate_caches_varargs_alias(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
    ) {
        self.invalidate_caches_varargs(opnum, effectinfo, argboxes)
    }

    /// RPython: get_field_updater(box, descr)
    pub fn get_field_updater(&mut self, obj: OpRef, descr_index: u32) -> FieldUpdater {
        let seen_allocation = self.saw_allocation(obj);
        let fieldbox = self.heap_cache.get(&descr_index).and_then(|cache| {
            if seen_allocation {
                cache.cache_seen_allocation.get(&obj).copied()
            } else {
                cache.cache_anything.get(&obj).copied()
            }
        });
        let updater = self
            .heap_cache
            .entry(descr_index)
            .or_insert_with(CacheEntry::new);
        let curr = if let Some(fieldbox) = fieldbox {
            fieldbox
        } else if seen_allocation {
            updater
                .cache_seen_allocation
                .get(&obj)
                .copied()
                .unwrap_or(OpRef::NONE)
        } else {
            updater
                .cache_anything
                .get(&obj)
                .copied()
                .unwrap_or(OpRef::NONE)
        };
        FieldUpdater::with_cache(obj, self, descr_index, Some(curr))
    }

    /// RPython alias: allocate a per-descr/index array cache entry.
    pub(crate) fn _get_or_make_array_cache_entry(
        &mut self,
        index: OpRef,
        descr_index: u32,
    ) -> Option<&mut CacheEntry> {
        let index = index.raw();
        Some(
            self.heap_array_cache
                .entry(descr_index)
                .or_default()
                .entry(index)
                .or_insert_with(CacheEntry::new),
        )
    }

    // ── Array item caching (RPython heapcache.py cached_arrayitems) ──

    /// Look up a cached array item value.
    pub fn getarrayitem(&self, array: OpRef, index: OpRef, descr: u32) -> Option<OpRef> {
        self.getarrayitem_cache(array, index, descr)
    }

    /// heapcache.py:542-553 getarrayitem: ConstInt-only lookup into
    /// heap_array_cache[descr][index].read(box).
    pub fn getarrayitem_cache(&self, array: OpRef, index: OpRef, descr: u32) -> Option<OpRef> {
        if !index.is_constant() {
            return None;
        }
        let entry = self.heap_array_cache.get(&descr)?.get(&index.raw())?;
        let seen_alloc = self.saw_allocation(array);
        let opref = entry._getdict(seen_alloc).get(&array).copied()?;
        Some(self.maybe_replace_with_const(opref))
    }

    /// Record an array item write.
    pub fn setarrayitem(&mut self, array: OpRef, index: OpRef, descr: u32, value: OpRef) {
        self.setarrayitem_cache(array, index, descr, value)
    }

    /// heapcache.py:573-585 setarrayitem: non-ConstInt clears the whole
    /// descr submap; ConstInt writes via indexcache.do_write_with_aliasing.
    pub fn setarrayitem_cache(&mut self, array: OpRef, index: OpRef, descr: u32, value: OpRef) {
        if !index.is_constant() {
            if let Some(cache) = self.heap_array_cache.get_mut(&descr) {
                cache.clear();
            }
            return;
        }
        let seen_alloc = self.saw_allocation(array);
        let entry = self
            .heap_array_cache
            .entry(descr)
            .or_default()
            .entry(index.raw())
            .or_insert_with(CacheEntry::new);
        entry._clear_cache_on_write(seen_alloc);
        entry._getdict_mut(seen_alloc).insert(array, value);
    }

    /// Record an array item read.
    pub fn getarrayitem_now_known_alias(
        &mut self,
        array: OpRef,
        index: OpRef,
        descr: u32,
        value: OpRef,
    ) {
        self.getarrayitem_now_known(array, index, descr, value);
    }

    /// heapcache.py:565-568 getarrayitem_now_known: ConstInt-only
    /// read_now_known into indexcache.
    pub fn getarrayitem_now_known(&mut self, array: OpRef, index: OpRef, descr: u32, value: OpRef) {
        if !index.is_constant() {
            return;
        }
        let seen_alloc = self.saw_allocation(array);
        let entry = self
            .heap_array_cache
            .entry(descr)
            .or_default()
            .entry(index.raw())
            .or_insert_with(CacheEntry::new);
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
        self.field_cache.clear();
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
        // `replaced_with_const` Vec is keyed by OpRef.0 (a position
        // index) and OpRef numbers are reused across traces, so we
        // clear it at the same trace boundary that drops the
        // FrontendOp objects in upstream.  Without this the next
        // trace can pick up a stale substitution from the previous
        // trace's `replace_box`.
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
        self.field_cache.clear();
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

    #[test]
    fn test_field_cache_basic() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(0);
        let field = 1;
        let val = OpRef::ref_op(2);

        assert_eq!(cache.getfield_cached(obj, field), None);

        cache.getfield_now_known(obj, field, val);
        assert_eq!(cache.getfield_cached(obj, field), Some(val));
    }

    #[test]
    fn test_field_cache_overwrite() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(0);
        let field = 1;

        cache.getfield_now_known(obj, field, OpRef::ref_op(10));
        assert_eq!(cache.getfield_cached(obj, field), Some(OpRef::ref_op(10)));

        cache.getfield_now_known(obj, field, OpRef::ref_op(20));
        assert_eq!(cache.getfield_cached(obj, field), Some(OpRef::ref_op(20)));
    }

    #[test]
    fn test_setfield_aliasing() {
        let mut cache = HeapCache::new();
        let obj_a = OpRef::ref_op(0);
        let obj_b = OpRef::ref_op(1);
        let field = 5;

        // Both objects have a known field value
        cache.getfield_now_known(obj_a, field, OpRef::ref_op(10));
        cache.getfield_now_known(obj_b, field, OpRef::ref_op(20));

        // Writing to obj_a (which is NOT unescaped) should invalidate
        // obj_b's field cache for the same field (potential aliasing).
        cache.setfield_cached(obj_a, field, OpRef::ref_op(30));
        assert_eq!(cache.getfield_cached(obj_a, field), Some(OpRef::ref_op(30)));
        assert_eq!(cache.getfield_cached(obj_b, field), None); // invalidated
    }

    #[test]
    fn test_setfield_no_aliasing_for_unescaped() {
        let mut cache = HeapCache::new();
        let obj_a = OpRef::ref_op(0);
        let obj_b = OpRef::ref_op(1);
        let field = 5;

        // obj_a is a newly allocated object
        cache.new_object(obj_a);
        cache.getfield_now_known(obj_a, field, OpRef::ref_op(10));
        cache.getfield_now_known(obj_b, field, OpRef::ref_op(20));

        // Writing to obj_a (unescaped) does NOT cause aliasing invalidation
        cache.setfield_cached(obj_a, field, OpRef::ref_op(30));
        assert_eq!(cache.getfield_cached(obj_a, field), Some(OpRef::ref_op(30)));
        assert_eq!(cache.getfield_cached(obj_b, field), Some(OpRef::ref_op(20))); // preserved
    }

    #[test]
    fn test_invalidate_caches() {
        let mut cache = HeapCache::new();
        cache.getfield_now_known(OpRef::ref_op(0), 1, OpRef::ref_op(10));
        cache.getfield_now_known(OpRef::ref_op(1), 2, OpRef::ref_op(20));

        cache.reset_keep_likely_virtuals();
        assert_eq!(cache.getfield_cached(OpRef::ref_op(0), 1), None);
        assert_eq!(cache.getfield_cached(OpRef::ref_op(1), 2), None);
    }

    #[test]
    fn test_invalidate_caches_for_escaped() {
        let mut cache = HeapCache::new();
        let escaped_obj = OpRef::ref_op(0);
        let unescaped_obj = OpRef::ref_op(1);

        cache.new_object(unescaped_obj);
        cache.getfield_now_known(escaped_obj, 1, OpRef::ref_op(10));
        cache.getfield_now_known(unescaped_obj, 1, OpRef::ref_op(20));

        cache.invalidate_caches_for_escaped();
        assert_eq!(cache.getfield_cached(escaped_obj, 1), None);
        assert_eq!(
            cache.getfield_cached(unescaped_obj, 1),
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
        let cls = GcRef(0x1000);

        assert!(!cache.is_class_known(obj));
        assert_eq!(cache.get_known_class(obj), None);

        cache.class_now_known(obj, cls);
        assert!(cache.is_class_known(obj));
        assert_eq!(cache.get_known_class(obj), Some(cls));
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
        cache.class_now_known(OpRef::ref_op(0), GcRef(0x1000));
        cache.getfield_now_known(OpRef::ref_op(0), 1, OpRef::ref_op(10));

        cache.reset();
        assert!(!cache.is_unescaped(OpRef::ref_op(0)));
        assert!(!cache.is_class_known(OpRef::ref_op(0)));
        assert_eq!(cache.getfield_cached(OpRef::ref_op(0), 1), None);
    }

    #[test]
    fn test_different_fields_independent() {
        let mut cache = HeapCache::new();
        let obj = OpRef::ref_op(0);

        cache.getfield_now_known(obj, 1, OpRef::ref_op(10));
        cache.getfield_now_known(obj, 2, OpRef::ref_op(20));

        // Writing field 1 should not affect field 2
        cache.setfield_cached(obj, 1, OpRef::ref_op(30));
        assert_eq!(cache.getfield_cached(obj, 1), Some(OpRef::ref_op(30)));
        assert_eq!(cache.getfield_cached(obj, 2), Some(OpRef::ref_op(20)));
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
        let new = OpRef::from_const(0);

        cache.arraylen_now_known(old, old);
        cache.replace_box(old, new);

        assert_eq!(cache.arraylen(old), Some(new));
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
