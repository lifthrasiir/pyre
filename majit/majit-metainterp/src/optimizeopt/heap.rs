/// Heap optimization pass: caches field/array reads and eliminates redundant loads/stores.
///
/// Translated from rpython/jit/metainterp/optimizeopt/heap.py.
///
/// Optimizations performed:
/// - Read-after-write elimination: SETFIELD then GETFIELD on same obj/field -> use cached value
/// - Write-after-write elimination: two SETFIELDs on same obj/field -> only keep the last
/// - Read-after-read elimination: two GETFIELDs on same obj/field -> reuse first result
/// - Same for array items (SETARRAYITEM_GC / GETARRAYITEM_GC) with constant index
/// - Cache invalidation on calls and side-effecting operations
/// - Lazy set emission: SETFIELD_GC is delayed until a guard or side-effecting op forces it
/// - GUARD_NOT_INVALIDATED deduplication
use std::collections::HashMap;

#[inline(always)]
fn vb_set(v: &mut Vec<bool>, i: u32) {
    let i = i as usize;
    if i >= v.len() {
        v.resize(i + 1, false);
    }
    v[i] = true;
}
#[inline(always)]
fn vb_get(v: &[bool], i: u32) -> bool {
    v.get(i as usize).copied().unwrap_or(false)
}
#[inline(always)]
fn vb_unset(v: &mut Vec<bool>, i: u32) {
    if let Some(s) = v.get_mut(i as usize) {
        *s = false;
    }
}

#[inline(always)]
fn use_untranslated_heap_ordering() -> bool {
    cfg!(debug_assertions)
}

#[inline]
fn sort_descr_entries_untranslated<T>(entries: &mut [(u32, DescrRef, T)]) {
    if use_untranslated_heap_ordering() {
        entries.sort_by(|a, b| b.1.repr().cmp(&a.1.repr()));
    }
}

#[inline]
fn sort_descr_item_refs_untranslated<T>(entries: &mut [(&DescrRef, T)]) {
    if use_untranslated_heap_ordering() {
        entries.sort_by(|a, b| b.0.repr().cmp(&a.0.repr()));
    }
}

#[inline]
fn sort_array_index_entries_untranslated<T>(entries: &mut [(i64, T)]) {
    if use_untranslated_heap_ordering() {
        entries.sort_by_key(|(index, _)| *index);
    }
}

use majit_ir::{DescrRef, OopSpecIndex, Op, OpCode, OpRef, descr::descr_identity};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// Cache key for a field access: (struct OpRef, field descriptor identity).
///
/// RPython `heap.py` keys `cached_fields` and green/quasi field caches by the
/// descriptor object itself.  `FieldDescr.get_index()` is a different domain:
/// it indexes `PtrInfo._fields`.
type FieldKey = (OpRef, usize);

/// heap.py:20-165 AbstractCachedEntry
///
/// PyPy uses Python inheritance to share `do_setfield`,
/// `force_lazy_set`, `getfield_from_cache`, `possible_aliasing` and
/// `possible_aliasing_two_infos` between `CachedField` and
/// `ArrayCachedItem`. The Rust port keeps the same per-method
/// signatures as inherent methods on each struct (so call sites read
/// like the PyPy source). Shared bodies are kept in
/// `abstract_cached_entry::*` free helpers and invoked from each
/// inherent method.
///
/// Rust-specific naming notes that mirror PyPy's contract:
/// - PyPy `cached_infos: [PtrInfo]` is replaced by `cached_structs:
///   Vec<OpRef>` because Rust's borrow checker forbids holding
///   parallel `&mut PtrInfo` references; the PtrInfo itself is read
///   on-demand from `ctx.forwarded[opref]` / `ctx.const_infos[gcref]`.
/// - PyPy `descr` parameters are carried as descriptor references for
///   field-cache identity, with a separate `field_idx` / `descr_idx`
///   (u32) only where the RPython source indexes `PtrInfo` slots or
///   EffectInfo bitsets.
/// heap.py:168-226 CachedField(AbstractCachedEntry)
struct CachedField {
    /// heap.py:39 cached_structs — struct OpRefs with a cached value
    /// for this descr. Replaces RPython's parallel `cached_infos`;
    /// the PtrInfo itself is read on-demand from
    /// `ctx.get_ptr_info(opref)` / `ctx.get_const_info(opref)`.
    cached_structs: Vec<OpRef>,
    /// heap.py:40 _lazy_set — at most one pending SetfieldGc per descr.
    /// The leading OpRef caches `get_box_replacement(op.getarg(0))`
    /// so `possible_aliasing`/`getfield_from_cache` do not have to
    /// re-resolve it on every probe (RPython relies on Python object
    /// identity here, which is implicit).
    lazy_set: Option<(OpRef, Op)>,
}

impl CachedField {
    fn new() -> Self {
        CachedField {
            cached_structs: Vec::new(),
            lazy_set: None,
        }
    }

    /// heap.py:42-49 AbstractCachedEntry.register_info(structop, info)
    ///
    /// Tracks `struct_opref` so subsequent `invalidate(descr)` knows
    /// to clear `opinfo._fields[descr_idx]`. RPython appends to both
    /// `cached_structs` and `cached_infos`; the Rust port skips
    /// `cached_infos` and reads PtrInfo on-demand.
    fn register_info(&mut self, struct_opref: OpRef) {
        if !self.cached_structs.contains(&struct_opref) {
            self.cached_structs.push(struct_opref);
        }
    }

    /// heap.py:59-65 AbstractCachedEntry.possible_aliasing
    fn possible_aliasing(&self, struct_opref: OpRef) -> bool {
        match &self.lazy_set {
            Some((lazy_obj, _)) => *lazy_obj != struct_opref,
            None => false,
        }
    }

    /// heap.py:169-170 CachedField._get_rhs_from_set_op
    fn _get_rhs_from_set_op(op: &Op) -> OpRef {
        op.arg(1)
    }

    /// heap.py:189-196 CachedField.invalidate(descr)
    ///
    /// PyPy iterates `cached_infos` and writes
    /// `opinfo._fields[descr.get_index()] = None`. The Rust port walks
    /// `cached_structs`, resolves each opref through `ctx.forwarded`
    /// OR `ctx.const_infos` (the latter mirrors `info.py:715-726
    /// ConstPtrInfo._get_info` which routes constant bases through
    /// `optheap.const_infos[gcref]`), and calls
    /// `info.clear_field(descr_idx)`.
    fn invalidate(&mut self, descr_idx: u32, ctx: &mut OptContext) {
        // heap.py:190-191: descr.is_always_pure() short-circuit is
        // performed by the caller; the CachedField does not carry the
        // descriptor object.
        for &obj in &self.cached_structs {
            let resolved = ctx.get_box_replacement(obj);
            if let Some(info) = ctx.get_ptr_info_mut(resolved) {
                info.clear_field(descr_idx);
            }
            // Clear existing const_infos slot if present; do NOT create.
            if let Some(info) = ctx.get_const_info_mut_if_exists(resolved) {
                info.clear_field(descr_idx);
            }
        }
        self.cached_structs.clear();
    }

    /// heap.py:177-187 CachedField._getfield(opinfo, descr, optheap, true_force=True)
    ///
    /// Returns the raw FieldEntry from _fields — caller checks Preamble vs Value.
    /// RPython: `res = opinfo.getfield(descr, optheap)`
    ///          `if isinstance(res, PreambleOp): ...`
    fn _getfield(
        &self,
        struct_opref: OpRef,
        descr: &DescrRef,
        field_idx: u32,
        ctx: &mut OptContext,
    ) -> Option<crate::optimizeopt::info::FieldEntry> {
        // info.py:212-214: return self._fields[fielddescr.get_index()]
        if let Some(info) = ctx.get_ptr_info(struct_opref) {
            if let Some(entry) = info.getfield(field_idx) {
                return Some(entry);
            }
        }
        // info.py:738-743 ConstPtrInfo.getfield → _get_info(parent_descr, optheap)
        let parent_descr = descr.as_field_descr().and_then(|fd| fd.get_parent_descr());
        if let Some(info) = ctx.get_const_info_mut(struct_opref, parent_descr) {
            if let Some(entry) = info.getfield(field_idx) {
                return Some(entry);
            }
        }
        None
    }

    /// heap.py:103-120 AbstractCachedEntry.getfield_from_cache
    /// heap.py:103-120 AbstractCachedEntry.getfield_from_cache
    fn getfield_from_cache(
        &self,
        struct_opref: OpRef,
        descr: &DescrRef,
        field_idx: u32,
        ctx: &mut OptContext,
    ) -> Option<crate::optimizeopt::info::FieldEntry> {
        if let Some((lazy_obj, lazy_op)) = &self.lazy_set {
            if *lazy_obj == struct_opref {
                return Some(crate::optimizeopt::info::FieldEntry::Value(
                    Self::_get_rhs_from_set_op(lazy_op),
                ));
            }
        }
        self._getfield(struct_opref, descr, field_idx, ctx)
    }

    /// heap.py:198-204 CachedField._cannot_alias_via_classes_or_lengths
    fn _cannot_alias_via_classes_or_lengths(
        opref1: OpRef,
        opref2: OpRef,
        ctx: &mut OptContext,
    ) -> bool {
        // info.py:880 get_known_class. PyPy: opinfo1.get_known_class(cpu)
        // / opinfo2.get_known_class(cpu); CANNOT_ALIAS iff both are
        // known and not the same constant.
        let class1 = ctx.getptrinfo(opref1).and_then(|i| i.get_known_class());
        let class2 = ctx.getptrinfo(opref2).and_then(|i| i.get_known_class());
        matches!((class1, class2), (Some(c1), Some(c2)) if c1 != c2)
    }

    /// heap.py:206-226 CachedField._cannot_alias_via_content
    fn _cannot_alias_via_content(opref1: OpRef, opref2: OpRef, ctx: &mut OptContext) -> bool {
        // heap.py:207-210: both must be AbstractStructPtrInfo
        let (Some(info1), Some(info2)) = (ctx.get_ptr_info(opref1), ctx.get_ptr_info(opref2))
        else {
            return false;
        };
        // heap.py:211-216: all_items() may be None
        let f1 = info1.all_items();
        let f2 = info2.all_items();
        if f1.is_empty() || f2.is_empty() {
            return false;
        }
        // heap.py:217-225: shared field with two different constants
        // → CANNOT_ALIAS. RPython iterates positionally; the Rust port
        // matches by field_idx (equivalent for the same descriptor layout).
        for (idx1, e1) in &f1 {
            for (idx2, e2) in &f2 {
                if idx1 != idx2 {
                    continue;
                }
                // Only compare concrete values; Preamble entries have no
                // known constant value.
                if let (Some(v1), Some(v2)) = (e1.as_opref(), e2.as_opref()) {
                    let v1r = ctx.get_box_replacement(v1);
                    let v2r = ctx.get_box_replacement(v2);
                    if ctx.is_constant(v1r) && ctx.is_constant(v2r) && v1r != v2r {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// heapcache.py:119-130 invalidate_unescaped — drop entries whose
    /// struct opref has escaped, propagating the clear into PtrInfo
    /// just like `invalidate(descr)` does for the full set.
    fn invalidate_unescaped(&mut self, unescaped: &[bool], descr_idx: u32, ctx: &mut OptContext) {
        let mut i = 0;
        while i < self.cached_structs.len() {
            let obj = self.cached_structs[i];
            if vb_get(unescaped, obj.0) {
                i += 1;
            } else {
                self.cached_structs.swap_remove(i);
                let resolved = ctx.get_box_replacement(obj);
                if let Some(info) = ctx.get_ptr_info_mut(resolved) {
                    info.clear_field(descr_idx);
                }
                if let Some(info) = ctx.get_const_info_mut_if_exists(resolved) {
                    info.clear_field(descr_idx);
                }
            }
        }
    }

    /// heap.py:172-175 CachedField.put_field_back_to_info
    fn put_field_back_to_info(&mut self, op: &Op, ctx: &mut OptContext) {
        // info.py:203-211 opinfo.setfield(descr, struct, op, optheap, cf=self)
        // PyPy: `setfield(..., cf=cf)` calls `cf.register_info(struct, self)`
        // (info.py:209-210). The Rust port performs both halves here.
        let descr_idx = op
            .descr
            .as_ref()
            .map(OptHeap::field_slot_index)
            .unwrap_or(0);
        let arg = ctx.get_box_replacement(op.arg(1));
        let struct_opref = ctx.get_box_replacement(op.arg(0));
        self.register_info(struct_opref);
        ctx.structinfo_setfield(op, descr_idx, arg);
    }

    /// heap.py:51-57 AbstractCachedEntry.produce_potential_short_preamble_ops
    ///
    /// Iterates `cached_structs` and emits a getfield op for each
    /// cached entry that still has a non-None `opinfo._fields[descr_idx]`.
    /// PyPy's method calls `info.produce_short_preamble_ops(...)` on
    /// each cached_info, which itself emits a `GETFIELD_GC` /
    /// `GETARRAYITEM_GC` to the short preamble; the Rust port still
    /// inlines the emission here even though
    /// [`crate::optimizeopt::info::produce_short_preamble_ops`] is now
    /// ported. Migration is a follow-up cleanup — the inline path
    /// stays for callers that lack the `OptContext` plumbing the info
    /// helper expects.
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        descr: &DescrRef,
        descr_idx: u32,
        ctx: &mut OptContext,
    ) {
        debug_assert!(self.lazy_set.is_none());
        for &cached in &self.cached_structs {
            let structbox = ctx.get_box_replacement(cached);
            if structbox.is_none() {
                continue;
            }
            let cached_val = match ctx
                .get_ptr_info(structbox)
                .and_then(|info| info.getfield(descr_idx))
                .map(|entry| entry.as_seen_opref())
                .or_else(|| {
                    let parent_descr = descr.as_field_descr().and_then(|fd| fd.get_parent_descr());
                    ctx.get_const_info_mut(structbox, parent_descr)
                        .and_then(|info| info.getfield(descr_idx))
                        .map(|entry| entry.as_seen_opref())
                }) {
                Some(v) if !v.is_none() => v,
                _ => continue,
            };
            let opcode = descr
                .as_field_descr()
                .map(|fd| OpCode::getfield_for_type(fd.field_type()))
                .unwrap_or(OpCode::GetfieldGcI);
            let mut op = Op::with_descr(opcode, &[structbox], descr.clone());
            op.pos = cached_val;
            sb.add_heap_op(op);
        }
    }
}

/// Cache key for an array item access: (array OpRef, descriptor index, constant array index).
type ArrayItemKey = (OpRef, u32, i64);

/// heap.py:228-298 ArrayCachedItem(AbstractCachedEntry)
struct ArrayCachedItem {
    /// heap.py:229-230 self.index — constant array index this entry
    /// is keyed by. RPython stores it as part of `ArrayCachedItem.__init__`.
    index: i64,
    /// heap.py:39 cached_structs — array OpRefs whose `_items[index]`
    /// slot holds a cached value. Replaces RPython's `cached_infos`.
    cached_structs: Vec<OpRef>,
    /// heap.py:40 _lazy_set — at most one pending SetarrayitemGc.
    lazy_set: Option<(OpRef, Op)>,
}

impl ArrayCachedItem {
    fn new(index: i64) -> Self {
        // heap.py:230 assert index >= 0; self.index = index
        debug_assert!(index >= 0);
        ArrayCachedItem {
            index,
            cached_structs: Vec::new(),
            lazy_set: None,
        }
    }

    /// heap.py:42-49 AbstractCachedEntry.register_info(structop, info)
    fn register_info(&mut self, array_opref: OpRef) {
        if !self.cached_structs.contains(&array_opref) {
            self.cached_structs.push(array_opref);
        }
    }

    /// heap.py:59-65 AbstractCachedEntry.possible_aliasing
    fn possible_aliasing(&self, array_opref: OpRef) -> bool {
        match &self.lazy_set {
            Some((lazy_obj, _)) => *lazy_obj != array_opref,
            None => false,
        }
    }

    /// heap.py:235-236 ArrayCachedItem._get_rhs_from_set_op
    fn _get_rhs_from_set_op(op: &Op) -> OpRef {
        op.arg(2)
    }

    /// heap.py:268-276 ArrayCachedItem._cannot_alias_via_classes_or_lengths
    fn _cannot_alias_via_classes_or_lengths(
        opref1: OpRef,
        opref2: OpRef,
        ctx: &mut OptContext,
    ) -> bool {
        use crate::optimizeopt::info::PtrInfo;
        // heap.py:269-274: both must be ArrayPtrInfo with known_ne lenbounds
        let len1 = match ctx.get_ptr_info(opref1) {
            Some(PtrInfo::Array(v)) => v.lenbound.clone(),
            _ => return false,
        };
        let len2 = match ctx.get_ptr_info(opref2) {
            Some(PtrInfo::Array(v)) => v.lenbound.clone(),
            _ => return false,
        };
        len1.known_ne(&len2)
    }

    /// heap.py:278-298 ArrayCachedItem._cannot_alias_via_content
    fn _cannot_alias_via_content(opref1: OpRef, opref2: OpRef, ctx: &mut OptContext) -> bool {
        use crate::optimizeopt::info::{FieldEntry, PtrInfo};
        // heap.py:279-282: isinstance(opinfo, ArrayPtrInfo)
        // info.py:530 all_items() returns _items (the dense list, None slots included).
        // Clone to avoid borrow conflict with ctx below.
        let items1: Vec<FieldEntry> = match ctx.get_ptr_info(opref1) {
            Some(PtrInfo::Array(a)) => a.items.clone(),
            _ => return false,
        };
        let items2: Vec<FieldEntry> = match ctx.get_ptr_info(opref2) {
            Some(PtrInfo::Array(a)) => a.items.clone(),
            _ => return false,
        };
        // heap.py:288-298: slot-by-slot comparison preserving index alignment.
        // None/Preamble slots are kept at their original positions.
        let len = items1.len().min(items2.len());
        for i in 0..len {
            // heap.py:289-290: value = get_box_replacement(content[index])
            let v1 = items1[i].as_opref().map(|r| ctx.get_box_replacement(r));
            let v2 = items2[i].as_opref().map(|r| ctx.get_box_replacement(r));
            // heap.py:291-292: if value is None: continue
            let (Some(v1), Some(v2)) = (v1, v2) else {
                continue;
            };
            if v1.is_none() || v2.is_none() {
                continue;
            }
            // heap.py:293-294: if not value.is_constant(): continue
            let (Some(c1), Some(c2)) = (ctx.get_constant(v1), ctx.get_constant(v2)) else {
                continue;
            };
            // heap.py:296: if not value1.same_constant(value2): return CANNOT_ALIAS
            if c1 != c2 {
                return true;
            }
        }
        false
    }

    /// heap.py:257-266 ArrayCachedItem.invalidate(descr)
    ///
    /// PyPy iterates `cached_infos` and writes
    /// `opinfo._items[self.index] = None`. The Rust port walks
    /// `cached_structs` and routes through `ctx.forwarded` /
    /// `ctx.const_infos`. The `self.parent.clear_varindex()` half is
    /// performed by the caller (`ArrayCacheSubMap::invalidate_index`)
    /// because Rust forbids the back-pointer.
    fn invalidate(&mut self, ctx: &mut OptContext) {
        for &obj in &self.cached_structs {
            let resolved = ctx.get_box_replacement(obj);
            if let Some(info) = ctx.get_ptr_info_mut(resolved) {
                info.clear_item(self.index as usize);
            }
            // info.py:728 ConstPtrInfo._get_array_info — only clear
            // an existing ArrayPtrInfo slot; do NOT create on miss.
            if let Some(info) = ctx.get_const_info_mut_if_exists(resolved) {
                info.clear_item(self.index as usize);
            }
        }
        self.cached_structs.clear();
    }

    /// heap.py:238-250 ArrayCachedItem._getfield(opinfo, descr, optheap)
    ///
    /// Takes `descr` so the constant-base path can route through
    /// `ConstPtrInfo._get_array_info(descr, optheap)` (info.py:728-735)
    /// which creates an `ArrayPtrInfo` on miss.
    /// heap.py:238-250 ArrayCachedItem._getfield — returns raw FieldEntry.
    fn _getfield(
        &self,
        array_opref: OpRef,
        descr: &DescrRef,
        ctx: &mut OptContext,
    ) -> Option<crate::optimizeopt::info::FieldEntry> {
        if self.index < 0 {
            return None;
        }
        let idx = self.index as usize;
        if let Some(info) = ctx.get_ptr_info(array_opref) {
            if let Some(entry) = info.getitem(idx) {
                return Some(entry);
            }
        }
        // info.py:746-748 ConstPtrInfo.getitem → _get_array_info(descr, optheap)
        if let Some(info) = ctx.get_const_info_array_mut(array_opref, descr.clone()) {
            if let Some(entry) = info.getitem(idx) {
                return Some(entry);
            }
        }
        None
    }

    /// heap.py:103-120 AbstractCachedEntry.getfield_from_cache
    fn getfield_from_cache(
        &self,
        array_opref: OpRef,
        descr: &DescrRef,
        ctx: &mut OptContext,
    ) -> Option<crate::optimizeopt::info::FieldEntry> {
        if let Some((lazy_obj, lazy_op)) = &self.lazy_set {
            if *lazy_obj == array_opref {
                return Some(crate::optimizeopt::info::FieldEntry::Value(
                    Self::_get_rhs_from_set_op(lazy_op),
                ));
            }
        }
        self._getfield(array_opref, descr, ctx)
    }

    /// heap.py:252-255 ArrayCachedItem.put_field_back_to_info
    fn put_field_back_to_info(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg = ctx.get_box_replacement(op.arg(2));
        let struct_opref = ctx.get_box_replacement(op.arg(0));
        self.register_info(struct_opref);
        ctx.arrayinfo_setitem(op, self.index as usize, arg);
    }

    /// heap.py:51-57 AbstractCachedEntry.produce_potential_short_preamble_ops
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        descr: &DescrRef,
        ctx: &mut OptContext,
    ) {
        debug_assert!(self.lazy_set.is_none());
        for &cached in &self.cached_structs {
            let arraybox = ctx.get_box_replacement(cached);
            if arraybox.is_none() {
                continue;
            }
            let cached_val = match ctx
                .get_ptr_info(arraybox)
                .and_then(|info| info.getitem(self.index as usize))
                .map(|entry| entry.as_seen_opref())
                .or_else(|| {
                    ctx.get_const_info_array_mut(arraybox, descr.clone())
                        .and_then(|info| info.getitem(self.index as usize))
                        .map(|entry| entry.as_seen_opref())
                }) {
                Some(v) if !v.is_none() => v,
                _ => continue,
            };
            // compile.py:451 ResOperation(... [arrayop, ConstInt(index)] ...)
            let idx_ref = ctx.make_constant_int(self.index as i64);
            let opcode = descr
                .as_array_descr()
                .map(|array_descr| OpCode::getarrayitem_for_type(array_descr.item_type()))
                .unwrap_or(OpCode::GetarrayitemGcI);
            let mut op = Op::with_descr(opcode, &[arraybox, idx_ref], descr.clone());
            op.pos = cached_val;
            sb.add_heap_op(op);
        }
    }

    /// heapcache.py:119-130 invalidate_unescaped — drop array entries
    /// whose array opref has escaped, propagating the clear into the
    /// `arrayinfo._items[index]` slot.
    fn invalidate_unescaped(&mut self, unescaped: &[bool], ctx: &mut OptContext) {
        if self.index < 0 {
            return;
        }
        let idx = self.index as usize;
        let mut i = 0;
        while i < self.cached_structs.len() {
            let obj = self.cached_structs[i];
            if vb_get(unescaped, obj.0) {
                i += 1;
            } else {
                self.cached_structs.swap_remove(i);
                let resolved = ctx.get_box_replacement(obj);
                if let Some(info) = ctx.get_ptr_info_mut(resolved) {
                    info.clear_item(idx);
                }
                if let Some(info) = ctx.get_const_info_mut_if_exists(resolved) {
                    info.clear_item(idx);
                }
            }
        }
    }
}

/// heap.py:300-324 ArrayCacheSubMap
///
/// Per-arraydescr container holding both constant-index entries and a
/// variable-index triples list. Mirrors RPython's `ArrayCacheSubMap`
/// 1:1.
struct ArrayCacheSubMap {
    /// heap.py:302: const_indexes = {} (int -> ArrayCachedItem)
    const_indexes: HashMap<i64, ArrayCachedItem>,
    /// heap.py:305-306: cached_varindex_triples = None
    /// List of (arrayinfo, indexbox, resbox). RPython uses Python object
    /// identity for arrayinfo; majit uses the canonical array OpRef.
    cached_varindex_triples: Option<Vec<(OpRef, OpRef, OpRef)>>,
}

impl ArrayCacheSubMap {
    fn new() -> Self {
        ArrayCacheSubMap {
            const_indexes: HashMap::new(),
            cached_varindex_triples: None,
        }
    }

    /// heap.py:305-306 clear_varindex
    fn clear_varindex(&mut self) {
        self.cached_varindex_triples = None;
    }

    /// heap.py:308-314 cache_varindex_read
    fn cache_varindex_read(&mut self, arrayinfo: OpRef, indexbox: OpRef, resbox: OpRef) {
        let entry = (arrayinfo, indexbox, resbox);
        if self.cached_varindex_triples.is_none() {
            self.cached_varindex_triples = Some(vec![entry]);
            return;
        }
        self.cached_varindex_triples.as_mut().unwrap().push(entry);
    }

    /// heap.py:316-317 cache_varindex_write
    fn cache_varindex_write(&mut self, arrayinfo: OpRef, indexbox: OpRef, resbox: OpRef) {
        self.cached_varindex_triples = Some(vec![(arrayinfo, indexbox, resbox)]);
    }

    /// heap.py:319-324 lookup_cached
    fn lookup_cached(
        &self,
        arrayinfo: OpRef,
        indexbox: OpRef,
        ctx: &mut OptContext,
    ) -> Option<OpRef> {
        if let Some(triples) = &self.cached_varindex_triples {
            for &(cached_arrayinfo, cached_index, cached_result) in triples {
                if cached_arrayinfo == arrayinfo
                    && ctx.get_box_replacement(cached_index) == indexbox
                {
                    return Some(ctx.get_box_replacement(cached_result));
                }
            }
        }
        None
    }

    /// heap.py:257-266 ArrayCachedItem.invalidate (parent step inlined)
    ///
    /// Clears the cached entries at `index` (also clearing
    /// `arrayinfo._items[index]` for each cached_struct via
    /// `cai.invalidate(ctx)`) AND calls `self.parent.clear_varindex()`.
    /// The parent step is inlined here because Rust forbids the
    /// back-pointer that PyPy uses on `ArrayCachedItem.parent`.
    fn invalidate_index(&mut self, index: i64, ctx: &mut OptContext) {
        if let Some(cai) = self.const_indexes.get_mut(&index) {
            cai.invalidate(ctx);
        }
        self.clear_varindex();
    }
}

/// Heap optimization pass.
///
/// Caches field and array item values to eliminate redundant loads, and delays
/// store emission (lazy sets) to enable write-after-write elimination.
///
/// Green field optimization: immutable field caches survive cache invalidation
/// by calls and side-effecting operations. When an immutable field is read from
/// a constant object, the result is also a constant (green field folding).
///
/// Aliasing analysis: objects allocated during the trace (NEW, NEW_WITH_VTABLE,
/// NEW_ARRAY, etc.) cannot alias each other or pre-existing objects. Their field
/// caches survive writes to other objects. Objects that haven't escaped (not
/// passed to calls or stored into the heap) keep their caches across calls.
pub struct OptHeap {
    /// Per-descr field cache: field_idx → CachedField.
    /// RPython heap.py: cached_fields OrderedDict keyed by descr.
    cached_fields: Vec<(u32, DescrRef, CachedField)>,
    /// Immutable (pure) field cache — separate to survive all invalidation.
    /// RPython heap.py: is_always_pure() fields are never invalidated.
    immutable_cached_fields: HashMap<FieldKey, OpRef>,
    /// heap.py:332: cached_arrayitems OrderedDict keyed by array descr.
    cached_arrayitems: Vec<(u32, DescrRef, ArrayCacheSubMap)>,
    /// Whether we've already emitted a GUARD_NOT_INVALIDATED.
    seen_guard_not_invalidated: bool,
    /// Postponed operation: held back until the next GUARD_NO_EXCEPTION.
    /// RPython heap.py: `postponed_op` — delays emission of operations
    /// that may raise (CALL_MAY_FORCE, comparison ops) until we see
    /// a GUARD_NO_EXCEPTION, ensuring correct exception semantics.
    postponed_op: Option<Op>,
    /// Array descriptor indices known to be immutable. RPython: descr.is_always_pure().
    immutable_array_descrs: Vec<bool>,

    // ── Aliasing analysis state — RPython: PtrInfo flags ──
    seen_allocation: Vec<bool>,
    unescaped: Vec<bool>,
    known_nonnull: Vec<bool>,
    /// heapcache.py: _heapc_deps — per-box dependency list.
    /// When an unescaped value is stored into an unescaped container,
    /// the value is recorded as a dependency of the container instead
    /// of being immediately escaped. When the container escapes later,
    /// all its dependencies are transitively escaped.
    heapc_deps: HashMap<u32, Vec<OpRef>>,

    // heap.py:27 OptHeap inherits Optimization.last_emitted_operation,
    // which is set to REMOVED by `_optimize_CALL_DICT_LOOKUP`
    // (heap.py:527) when a folded CALL_PURE collapses into its cached
    // result. `optimize_GUARD_NO_EXCEPTION` (heap.py:530-533) reads the
    // flag and skips emitting the trailing guard.
    //
    // _optimize_CALL_DICT_LOOKUP is not yet ported — it depends on
    // `extradescrs` on EffectInfo (rpython rordereddict descriptor
    // pairing) which majit-ir does not carry. Until that lands the
    // OptHeap REMOVED setter does not exist, so the corresponding
    // reader is omitted in optimize_GUARD_NO_EXCEPTION rather than
    // installed as dead code.
    /// Fields known to be quasi-immutable: (obj, field_idx) -> cached value OpRef.
    /// Populated by QUASIIMMUT_FIELD, consumed by subsequent GETFIELD_GC_*.
    /// Survives calls (guarded by GUARD_NOT_INVALIDATED).
    quasi_immut_cache: HashMap<FieldKey, OpRef>,
}

impl OptHeap {
    pub fn new() -> Self {
        OptHeap {
            cached_fields: Vec::new(),
            immutable_cached_fields: HashMap::new(),
            cached_arrayitems: Vec::new(),
            seen_guard_not_invalidated: false,
            postponed_op: None,
            immutable_array_descrs: Vec::new(),
            seen_allocation: Vec::new(),
            unescaped: Vec::new(),
            known_nonnull: Vec::new(),
            heapc_deps: HashMap::new(),
            quasi_immut_cache: HashMap::new(),
        }
    }

    fn cached_field_pos_for_descr(&self, descr: &DescrRef) -> Option<usize> {
        let identity = descr_identity(descr);
        self.cached_fields
            .iter()
            .position(|(_, cached_descr, _)| descr_identity(cached_descr) == identity)
    }

    fn cached_array_pos_for_descr(&self, descr: &DescrRef) -> Option<usize> {
        let identity = descr_identity(descr);
        self.cached_arrayitems
            .iter()
            .position(|(_, cached_descr, _)| descr_identity(cached_descr) == identity)
    }

    fn cached_array_pos_for_index(&self, descr_idx: u32) -> Option<usize> {
        self.cached_arrayitems
            .iter()
            .position(|(idx, _, _)| *idx == descr_idx)
    }

    /// RPython field descriptor identity for heap cache keys.
    ///
    /// This is the Rust equivalent of using `descr` as a Python dict key in
    /// `heap.py:392-397`.
    fn field_cache_identity(descr: &DescrRef) -> usize {
        descr_identity(descr)
    }

    /// Compute the `PtrInfo._fields` slot for a field descriptor.
    ///
    /// RPython uses `descr.get_index()` only for `info._fields[index]`
    /// (`info.py:203-214`).  In majit this is `FieldDescr::index_in_parent`
    /// when a parent SizeDescr is available; older/simple descriptors fall
    /// back to their `Descr::index()`.
    fn field_slot_index(descr: &DescrRef) -> u32 {
        let descr_idx = descr.index();
        let Some(field_descr) = descr.as_field_descr() else {
            return descr_idx;
        };
        if field_descr.get_parent_descr().is_some() {
            field_descr.index_in_parent() as u32
        } else {
            descr_idx
        }
    }

    /// Existing majit `EffectInfo` bitsets are indexed by the descriptor's
    /// effect/global index domain, not by `PtrInfo._fields` slot.
    fn field_effect_index(descr: &DescrRef) -> u32 {
        descr.index()
    }

    /// heapcache.py:295-309 _escape_box: escape a box and transitively
    /// escape all its dependencies.
    fn escape_box(&mut self, opref: OpRef) {
        vb_unset(&mut self.unescaped, opref.0);
        if let Some(deps) = self.heapc_deps.remove(&opref.0) {
            for dep in deps {
                self.escape_box(dep);
            }
        }
    }

    /// heapcache.py:224-230 _escape_from_write: when storing a value into
    /// a container, record dependency if both are unescaped; otherwise
    /// escape the value immediately.
    fn escape_from_write(&mut self, container: OpRef, value: OpRef) {
        if vb_get(&self.unescaped, container.0) && vb_get(&self.unescaped, value.0) {
            self.heapc_deps.entry(container.0).or_default().push(value);
        } else if !value.is_none() {
            self.escape_box(value);
        }
    }

    /// Build the field cache key from a GETFIELD or SETFIELD op.
    ///
    /// For GETFIELD_GC_I/R/F: args = [obj], descr = field descriptor.
    /// For SETFIELD_GC: args = [obj, value], descr = field descriptor.
    fn field_key(op: &Op) -> Option<FieldKey> {
        let descr = op.descr.as_ref()?;
        let obj = op.arg(0);
        Some((obj, Self::field_cache_identity(descr)))
    }

    /// heap.py:409-415 arrayitem_cache: constant-index array cache key.
    /// Canonicalizes array and index through get_box_replacement.
    fn arrayitem_key(op: &Op, ctx: &mut OptContext) -> Option<ArrayItemKey> {
        let descr = op.descr.as_ref()?;
        let array = ctx.get_box_replacement(op.arg(0));
        let index_ref = ctx.get_box_replacement(op.arg(1));
        let index_val = ctx.get_constant_int(index_ref)?;
        Some((array, descr.index(), index_val))
    }

    /// Register a struct opref in the per-descr CachedField.
    ///
    fn cache_field(&mut self, obj: OpRef, descr: &DescrRef) {
        let field_idx = Self::field_slot_index(descr);
        let pos = match self.cached_field_pos_for_descr(descr) {
            Some(pos) => pos,
            None => {
                self.cached_fields
                    .push((field_idx, descr.clone(), CachedField::new()));
                self.cached_fields.len() - 1
            }
        };
        let cf = &mut self.cached_fields[pos].2;
        cf.register_info(obj);
    }

    /// heap.py:392: field_cache (read-only borrow variant).
    fn get_cached_field(&self, descr: &DescrRef) -> Option<&CachedField> {
        self.cached_field_pos_for_descr(descr)
            .map(|pos| &self.cached_fields[pos].2)
    }

    /// heap.py:392-397 field_cache — get or create CachedField for a descr.
    fn field_cache(&mut self, descr: &DescrRef) -> &mut CachedField {
        let field_idx = Self::field_slot_index(descr);
        let pos = match self.cached_field_pos_for_descr(descr) {
            Some(pos) => pos,
            None => {
                self.cached_fields
                    .push((field_idx, descr.clone(), CachedField::new()));
                self.cached_fields.len() - 1
            }
        };
        &mut self.cached_fields[pos].2
    }

    fn get_cached_field_mut(&mut self, descr: &DescrRef) -> Option<&mut CachedField> {
        let pos = self.cached_field_pos_for_descr(descr)?;
        Some(&mut self.cached_fields[pos].2)
    }

    /// heap.py:399-407 arrayitem_submap(descr, create_if_nonexistant=True)
    fn arrayitem_submap(&mut self, descr: &DescrRef) -> &mut ArrayCacheSubMap {
        let descr_idx = descr.index();
        let pos = match self.cached_array_pos_for_descr(descr) {
            Some(pos) => pos,
            None => {
                self.cached_arrayitems
                    .push((descr_idx, descr.clone(), ArrayCacheSubMap::new()));
                self.cached_arrayitems.len() - 1
            }
        };
        &mut self.cached_arrayitems[pos].2
    }

    /// heap.py:409-415 arrayitem_cache(descr, index)
    /// → submap[descr].const_indexes[index] (or insert).
    fn arrayitem_cache(&mut self, descr: &DescrRef, index: i64) -> &mut ArrayCachedItem {
        self.arrayitem_submap(descr)
            .const_indexes
            .entry(index)
            .or_insert_with(|| ArrayCachedItem::new(index))
    }

    fn get_cached_array_submap(&self, descr_idx: u32) -> Option<&ArrayCacheSubMap> {
        self.cached_array_pos_for_index(descr_idx)
            .map(|pos| &self.cached_arrayitems[pos].2)
    }

    fn get_cached_array_submap_mut(&mut self, descr_idx: u32) -> Option<&mut ArrayCacheSubMap> {
        let pos = self.cached_array_pos_for_index(descr_idx)?;
        Some(&mut self.cached_arrayitems[pos].2)
    }

    fn get_cached_array_descr(&self, descr_idx: u32) -> Option<DescrRef> {
        self.cached_array_pos_for_index(descr_idx)
            .map(|pos| self.cached_arrayitems[pos].1.clone())
    }

    fn cache_arrayitem(
        &mut self,
        array: OpRef,
        descr_idx: u32,
        index: i64,
        descr: Option<&DescrRef>,
    ) {
        let Some(descr) = descr else {
            if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                submap
                    .const_indexes
                    .entry(index)
                    .or_insert_with(|| ArrayCachedItem::new(index))
                    .register_info(array);
            }
            return;
        };
        let cai = self.arrayitem_cache(descr, index);
        cai.register_info(array);
    }

    /// heap.py: force_lazy_set — emit lazy setfields.
    /// If any lazy setfield argument references the postponed_op,
    /// emit the postponed_op first (RPython heap.py exact logic).
    /// Emit a lazy setfield after resolving forwarding and forcing a virtual
    /// rhs if needed.
    ///
    /// RPython heap.py: force_lazy_set → emit_extra(op, emit=False).
    ///
    /// RPython parity: if the RHS value is virtual, do NOT emit the SetfieldGc.
    /// Instead, add it to pendingfields for the guard's resume data
    /// (heap.py:618-620). The virtual will be materialized at guard failure
    /// time via rd_virtuals, and the pending setfield replayed after.
    ///
    /// `allow_force`: true for call-triggered force, false for JUMP/flush.
    /// heap.py: cf.force_lazy_set(optheap, descr)
    ///
    /// Emit a lazy SetfieldGc. If the value is virtual, return false (the
    /// caller should handle it via pendingfields / rd_pendingfields).
    fn emit_lazy_setfield(op: &mut Op, ctx: &mut OptContext, _allow_force: bool) -> bool {
        let orig_val = ctx.get_box_replacement(op.arg(1));

        // heap.py:136: emit_extra(op, emit=False) re-processes through passes.
        // Virtual values are skipped — handled by rd_pendingfields at guard
        // time or dropped at JUMP.
        if let Some(info) = ctx.get_ptr_info(orig_val) {
            if info.is_virtual() {
                return false;
            }
        }

        // Non-virtual path: resolve forwarding and route after heap
        for arg in op.args.iter_mut() {
            *arg = ctx.get_box_replacement(*arg);
        }
        // heap.py:136: emit_extra(op, emit=False) → next_optimization
        ctx.emit_extra(ctx.current_pass_idx, op.clone());
        true
    }

    /// heap.py:122-145: force_lazy_set → emit_extra(op, emit=False)
    ///
    /// For each CachedField with a pending lazy set:
    /// 1. invalidate(descr) — clear conflicting cache entries
    /// 2. emit_extra(op, emit=False) — route through passes AFTER heap
    /// 3. put_field_back_to_info — restore this specific cache entry
    ///
    /// `heap_pass_idx`: this pass's own index. RPython uses
    /// `self.next_optimization` which always starts AFTER heap.
    fn force_all_lazy_setfields(&mut self, heap_pass_idx: usize, ctx: &mut OptContext) {
        if let Some(ref postponed) = self.postponed_op {
            let postponed_pos = postponed.pos;
            let needs_postponed = self.cached_fields.iter().any(|(_, _, cf)| {
                cf.lazy_set
                    .as_ref()
                    .map_or(false, |(_, op)| op.args.iter().any(|a| *a == postponed_pos))
            });
            if needs_postponed {
                // RPython emit_postponed_op: route through next_optimization
                if let Some(p) = self.postponed_op.take() {
                    ctx.emit_extra(heap_pass_idx, p);
                }
            }
        }
        // Collect all lazy sets from all CachedFields.
        let mut field_entries: Vec<_> = self
            .cached_fields
            .iter_mut()
            .map(|(field_idx, descr, cf)| (*field_idx, descr.clone(), cf))
            .collect();
        sort_descr_entries_untranslated(&mut field_entries);
        let pending: Vec<(u32, DescrRef, OpRef, Op)> = field_entries
            .into_iter()
            .filter_map(|(field_idx, descr, cf)| {
                cf.lazy_set
                    .take()
                    .map(|(obj, op)| (field_idx, descr, obj, op))
            })
            .collect();
        for (field_idx, descr, obj, mut op) in pending {
            // heap.py:129: invalidate(descr) — skip if is_always_pure
            if !descr.is_always_pure() {
                if let Some(cf) = self.get_cached_field_mut(&descr) {
                    cf.invalidate(field_idx, ctx);
                }
            }
            // Resolve args after invalidation.
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(1);
            // heap.py:142-143 put_field_back_to_info needs the lazy_set Op
            // AFTER it's been emitted by emit_extra. Clone it so we can
            // route the structinfo write through `structinfo_setfield`
            // without restoring or reconstructing the op.
            let put_back_op = op.clone();
            // heap.py:136: emit_extra(op, emit=False) — route after heap
            ctx.emit_extra(heap_pass_idx, op);
            // heap.py:142-143: put_field_back_to_info — restore cache +
            // structinfo. The orthodox helper handles both the Forwarded
            // path (regular box) and the constant path (const_infos).
            self.cache_field(obj, &descr);
            ctx.structinfo_setfield(&put_back_op, field_idx, final_value);
        }
    }

    fn force_all_lazy_setarrayitems(&mut self, heap_pass_idx: usize, ctx: &mut OptContext) {
        // heap.py:600-606 force_all_lazy_sets array half:
        //   for submap in self.cached_arrayitems.itervalues():
        //       items = submap.const_indexes.items() ...
        //       for index, cf in items: cf.force_lazy_set(self, None)
        let mut pending = Vec::new();
        for (descr_idx, _, submap) in &mut self.cached_arrayitems {
            let mut index_entries: Vec<_> = submap
                .const_indexes
                .iter_mut()
                .map(|(&index, cai)| (index, cai))
                .collect();
            sort_array_index_entries_untranslated(&mut index_entries);
            for (index, cai) in index_entries {
                if let Some((obj, op)) = cai.lazy_set.take() {
                    pending.push((*descr_idx, index, obj, op));
                }
            }
        }
        for (descr_idx, index, _obj, mut op) in pending {
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(2);
            let array_ref = op.arg(0);
            let descr = op.descr.clone();
            let put_back_op = op.clone();
            // emit_extra(op, emit=False): route through passes after heap
            ctx.emit_extra(heap_pass_idx, op);
            self.cache_arrayitem(array_ref, descr_idx, index, descr.as_ref());
            // info.py: ArrayPtrInfo.setitem — keep PtrInfo in sync.
            ctx.arrayinfo_setitem(&put_back_op, index as usize, final_value);
        }
    }

    /// Force all pending lazy stores (both fields and array items).
    /// `heap_pass_idx`: this pass's own pipeline index for emit routing.
    fn force_all_lazy_sets(&mut self, heap_pass_idx: usize, ctx: &mut OptContext) {
        self.force_all_lazy_setfields(heap_pass_idx, ctx);
        self.force_all_lazy_setarrayitems(heap_pass_idx, ctx);
    }

    /// heap.py:608-637 force_lazy_sets_for_guard()
    ///
    /// Returns pendingfields: SetfieldGc/SetarrayitemGc ops where the stored
    /// VALUE is virtual. These go into rd_pendingfields on the guard's resume
    /// data (emitting_operation stores them in ctx.pending_for_guard →
    /// optimizer.rs encodes as op.rd_pendingfields).
    /// Non-virtual lazy sets are emitted (forced) immediately.
    fn force_lazy_sets_for_guard(&mut self, self_pass_idx: usize, ctx: &mut OptContext) -> Vec<Op> {
        let mut pendingfields = Vec::new();

        // heap.py:610-621: iterate cached fields
        // Collect all lazy sets from CachedFields.
        let mut ordered_entries: Vec<_> = self
            .cached_fields
            .iter_mut()
            .map(|(field_idx, descr, cf)| (*field_idx, descr.clone(), cf))
            .collect();
        sort_descr_entries_untranslated(&mut ordered_entries);
        let field_entries: Vec<(u32, DescrRef, OpRef, Op)> = ordered_entries
            .into_iter()
            .filter_map(|(field_idx, descr, cf)| {
                cf.lazy_set
                    .take()
                    .map(|(obj, op)| (field_idx, descr, obj, op))
            })
            .collect();
        for (field_idx, descr, obj, mut op) in field_entries {
            // heap.py:617-618: val = op.getarg(1); if is_virtual(val)
            let value_ref = ctx.get_box_replacement(op.arg(1));
            let is_virtual = matches!(
                ctx.get_ptr_info(value_ref),
                Some(info) if info.is_virtual()
            );
            if is_virtual {
                // heap.py:618-619: virtual value → pendingfields
                pendingfields.push(op);
                continue;
            }
            // heap.py:621: cf.force_lazy_set(self, descr) →
            // invalidate first, then emit_extra(op, emit=False),
            // then put_field_back_to_info restores the cache.
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(1);
            // heap.py:129,189-191: invalidate(descr) — skip if is_always_pure
            if !descr.is_always_pure() {
                if let Some(cf) = self.get_cached_field_mut(&descr) {
                    cf.invalidate(field_idx, ctx);
                }
            }
            // heap.py:142-143 put_field_back_to_info needs the lazy_set Op
            // AFTER it's been emitted by emit_extra. Clone it so the
            // structinfo write goes through `structinfo_setfield` (which
            // also handles the constant arg0 → const_infos route).
            let put_back_op = op.clone();
            // emit_extra(op, emit=False): route through passes after heap.
            // RPython: self.next_optimization — always starts AFTER heap,
            // regardless of which pass emitted the guard that triggered this.
            ctx.emit_extra(self_pass_idx, op);
            // heap.py:142-143: put_field_back_to_info — restore cache + PtrInfo
            self.cache_field(obj, &descr);
            ctx.structinfo_setfield(&put_back_op, field_idx, final_value);
        }

        // heap.py:622-636: iterate cached array items
        //   for descr, submap in self.cached_arrayitems.iteritems():
        //       for index, cf in submap.const_indexes.iteritems():
        let array_entries: Vec<(u32, i64, OpRef, Op)> = self
            .cached_arrayitems
            .iter_mut()
            .flat_map(|(descr_idx, _, submap)| {
                submap
                    .const_indexes
                    .iter_mut()
                    .filter_map(move |(&index, cai)| {
                        cai.lazy_set
                            .take()
                            .map(|(obj, op)| (*descr_idx, index, obj, op))
                    })
            })
            .collect();
        for (descr_idx, index, _obj, mut op) in array_entries {
            // heap.py:631-633: assert container not virtual; check value virtual
            let value_ref = ctx.get_box_replacement(op.arg(2));
            let is_virtual = matches!(
                ctx.get_ptr_info(value_ref),
                Some(info) if info.is_virtual()
            );
            if is_virtual {
                // heap.py:634: pendingfields.append(op)
                pendingfields.push(op);
                continue;
            }

            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(2);
            let array_ref = op.arg(0);
            let descr = op.descr.clone();
            let put_back_op = op.clone();
            // emit_extra(op, emit=False): route through passes after heap.
            ctx.emit_extra(self_pass_idx, op);
            self.cache_arrayitem(array_ref, descr_idx, index, descr.as_ref());
            // info.py: ArrayPtrInfo.setitem — keep PtrInfo in sync.
            ctx.arrayinfo_setitem(&put_back_op, index as usize, final_value);
        }

        pendingfields
    }

    /// Invalidate caches on calls and other side-effecting operations.
    ///
    /// Caches that survive:
    /// - Immutable (green) field caches: values never change.
    /// - Unescaped object caches: calls cannot access objects that haven't
    ///   been passed to a call or stored into the heap.
    /// heap.py:379-391: invalidate non-pure field/array caches.
    /// Only `is_always_pure` (immutable) fields survive.
    ///
    /// heap.py:189-196 `CachedField.invalidate(descr)` clears
    /// `opinfo._fields[idx]` for every cached_info BEFORE clearing the
    /// `cached_infos`/`cached_structs` lists. The Rust port routes that
    /// PtrInfo cleanup through `invalidate_with_ctx` so the per-pass
    /// "single source of truth" stays in sync after a clean.
    fn clean_caches(&mut self, ctx: &mut OptContext) {
        let mut field_entries: Vec<_> = self
            .cached_fields
            .iter_mut()
            .map(|(field_idx, descr, cf)| (*field_idx, descr.clone(), cf))
            .collect();
        sort_descr_entries_untranslated(&mut field_entries);
        for (field_idx, descr, cf) in field_entries {
            // heap.py:384: if not descr.is_always_pure(): cf.invalidate()
            if !descr.is_always_pure() {
                cf.invalidate(field_idx, ctx);
            }
        }
        // heap.py:386-389:
        //   for descr, submap in self.cached_arrayitems.iteritems():
        //       if not descr.is_always_pure():
        //           for index, cf in submap.const_indexes.iteritems():
        //               cf.invalidate(None)
        //
        // RPython's `cf.invalidate(None)` clears `cached_infos` items AND
        // calls `self.parent.clear_varindex()` (heap.py:266). The Rust port
        // walks `cached_arrayitems` directly so each `cai.invalidate(ctx)`
        // can drop the matching `arrayinfo._items[index]` slot through
        // `ctx.get_ptr_info_mut` / `ctx.get_const_info_mut`.
        let immutable_arrays = self.immutable_array_descrs.clone();
        let descr_idxs: Vec<u32> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, _, _)| *idx)
            .collect();
        for descr_idx in descr_idxs {
            if vb_get(&immutable_arrays, descr_idx) {
                continue;
            }
            let indexes: Vec<i64> = match self.get_cached_array_submap(descr_idx) {
                Some(submap) => submap.const_indexes.keys().copied().collect(),
                None => continue,
            };
            for index in indexes {
                if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    if let Some(cai) = submap.const_indexes.get_mut(&index) {
                        cai.invalidate(ctx);
                    }
                    // heap.py:266 self.parent.clear_varindex()
                    submap.clear_varindex();
                }
            }
        }
        // heap.py:390: self.cached_dict_reads.clear()
        // (no dict_reads cache in majit)
    }

    /// heapcache.py:363-369 invalidate_unescaped / clear_caches_varargs
    ///
    /// Preserve cached values for unescaped allocations across residual calls.
    /// Only entries attached to boxes that have escaped are invalidated.
    /// Removed entries also have their `opinfo._fields[idx]` /
    /// `opinfo._items[index]` slot cleared so reads via `_getfield`
    /// cannot resurrect them.
    fn invalidate_caches_for_escaped(&mut self, ctx: &mut OptContext) {
        let immutable_arrays = self.immutable_array_descrs.clone();
        let unescaped_snapshot = self.unescaped.clone();
        let field_positions: Vec<usize> = (0..self.cached_fields.len()).collect();
        for pos in field_positions {
            let (field_idx, descr) = {
                let (field_idx, descr, _) = &self.cached_fields[pos];
                (*field_idx, descr.clone())
            };
            if descr.is_always_pure() {
                continue;
            }
            // Snapshot escaped opref list to avoid the borrow conflict
            // between iterating cached_structs and mutating PtrInfo.
            let escaped_opref: Vec<OpRef> = self.cached_fields[pos]
                .2
                .cached_structs
                .iter()
                .copied()
                .filter(|obj| !vb_get(&unescaped_snapshot, obj.0))
                .collect();
            // Drop the escaped entries from cached_structs and clear the
            // matching PtrInfo / const_infos slot.
            self.cached_fields[pos]
                .2
                .cached_structs
                .retain(|obj| vb_get(&unescaped_snapshot, obj.0));
            for obj in escaped_opref {
                let resolved = ctx.get_box_replacement(obj);
                if let Some(info) = ctx.get_ptr_info_mut(resolved) {
                    info.clear_field(field_idx);
                }
                if let Some(info) = ctx.get_const_info_mut_if_exists(resolved) {
                    info.clear_field(field_idx);
                }
            }
        }
        let descr_idxs: Vec<u32> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, _, _)| *idx)
            .collect();
        for descr_idx in descr_idxs {
            if vb_get(&immutable_arrays, descr_idx) {
                continue;
            }
            let indexes: Vec<i64> = match self.get_cached_array_submap(descr_idx) {
                Some(submap) => submap.const_indexes.keys().copied().collect(),
                None => continue,
            };
            for index in indexes {
                let escaped_opref: Vec<OpRef> = self
                    .get_cached_array_submap(descr_idx)
                    .and_then(|s| s.const_indexes.get(&index))
                    .map(|cai| {
                        cai.cached_structs
                            .iter()
                            .copied()
                            .filter(|obj| !vb_get(&unescaped_snapshot, obj.0))
                            .collect()
                    })
                    .unwrap_or_default();
                if let Some(cai) = self
                    .get_cached_array_submap_mut(descr_idx)
                    .and_then(|s| s.const_indexes.get_mut(&index))
                {
                    cai.cached_structs
                        .retain(|obj| vb_get(&unescaped_snapshot, obj.0));
                }
                let idx = if index >= 0 {
                    Some(index as usize)
                } else {
                    None
                };
                if let Some(idx) = idx {
                    for obj in escaped_opref {
                        let resolved = ctx.get_box_replacement(obj);
                        if let Some(info) = ctx.get_ptr_info_mut(resolved) {
                            info.clear_item(idx);
                        }
                        if let Some(info) = ctx.get_const_info_mut_if_exists(resolved) {
                            info.clear_item(idx);
                        }
                    }
                }
            }
            // varindex triples reference array boxes by OpRef; drop any
            // triple whose arrayinfo (first slot) has escaped.
            if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                if let Some(triples) = submap.cached_varindex_triples.as_mut() {
                    triples.retain(|&(arrayinfo, _, _)| vb_get(&unescaped_snapshot, arrayinfo.0));
                    if triples.is_empty() {
                        submap.cached_varindex_triples = None;
                    }
                }
            }
        }
    }

    /// Invalidate only array cache entries affected by an ARRAYCOPY/ARRAYMOVE.
    ///
    /// Instead of clearing all array caches, only remove entries for the
    /// destination array within the copied index range. Entries for other
    /// arrays, or entries outside the range, are kept. Removed entries
    /// also have their PtrInfo `_items[index]` slot cleared so reads via
    /// `read_item_via_info` cannot resurrect them.
    fn invalidate_array_caches_for_copy(
        &mut self,
        ctx: &mut OptContext,
        dest_ref: OpRef,
        dest_start: Option<i64>,
        length: Option<i64>,
    ) {
        let dest_resolved = ctx.get_box_replacement(dest_ref);
        let descr_idxs: Vec<u32> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, _, _)| *idx)
            .collect();
        for descr_idx in descr_idxs {
            let indexes: Vec<i64> = match self.get_cached_array_submap(descr_idx) {
                Some(submap) => submap.const_indexes.keys().copied().collect(),
                None => continue,
            };
            for index in indexes {
                let in_range = match (dest_start, length) {
                    (Some(start), Some(len)) => index >= start && index < start + len,
                    _ => true,
                };
                if !in_range {
                    continue;
                }
                if let Some(cai) = self
                    .get_cached_array_submap_mut(descr_idx)
                    .and_then(|s| s.const_indexes.get_mut(&index))
                {
                    cai.cached_structs.retain(|obj| *obj != dest_ref);
                }
                if index >= 0 {
                    if let Some(info) = ctx.get_ptr_info_mut(dest_resolved) {
                        info.clear_item(index as usize);
                    }
                    if let Some(info) = ctx.get_const_info_mut_if_exists(dest_resolved) {
                        info.clear_item(index as usize);
                    }
                }
            }
            // Variable-index triples touching `dest_ref` are invalidated.
            if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                if let Some(triples) = submap.cached_varindex_triples.as_mut() {
                    triples.retain(|&(arrayinfo, _, _)| arrayinfo != dest_ref);
                    if triples.is_empty() {
                        submap.cached_varindex_triples = None;
                    }
                }
            }
        }
    }

    /// Extract OopSpecIndex from a call op's descriptor, if available.
    fn get_oopspecindex(op: &Op) -> OopSpecIndex {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.get_extra_info().oopspecindex)
            .unwrap_or(OopSpecIndex::None)
    }

    /// Mark call arguments as escaped.
    /// heapcache.py:259-293 mark_escaped_varargs parity.
    /// ARRAYCOPY/ARRAYMOVE with constant indices and known array descriptor
    /// do NOT escape arguments.
    fn mark_escaped_varargs(&mut self, op: &Op, ctx: &mut OptContext) {
        let oopspec = Self::get_oopspecindex(op);
        // heapcache.py:275: single_write_descr_array is not None.
        // RPython: exactly one array write descriptor is known.
        // Bitstring: is_power_of_two means exactly one bit set.
        let has_single_write_descr = op
            .descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| {
                let w = cd.get_extra_info().write_descrs_arrays;
                w != 0 && w.is_power_of_two()
            })
            .unwrap_or(false);
        if oopspec == OopSpecIndex::Arraycopy
            && has_single_write_descr
            && op.args.len() >= 6
            && ctx.is_constant(op.args[3])
            && ctx.is_constant(op.args[4])
            && ctx.is_constant(op.args[5])
        {
            return;
        }
        if oopspec == OopSpecIndex::Arraymove
            && has_single_write_descr
            && op.args.len() >= 5
            && ctx.is_constant(op.args[2])
            && ctx.is_constant(op.args[3])
            && ctx.is_constant(op.args[4])
        {
            return;
        }
        for &arg in &op.args {
            self.escape_box(arg);
        }
    }

    /// heap.py: check if a call has random effects (EffectInfo).
    /// Calls with HAS_RANDOM_EFFECTS invalidate all caches.
    /// Calls without it only invalidate non-immutable/non-unescaped entries.
    fn call_has_random_effects(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.get_extra_info().has_random_effects())
            .unwrap_or(true) // conservative: assume random effects if unknown
    }

    /// heap.py: check if a call can invalidate quasi-immutable fields.
    fn call_can_invalidate(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.get_extra_info().check_can_invalidate())
            .unwrap_or(true)
    }

    /// heap.py: check if a call forces virtual/virtualizable objects.
    fn call_forces_virtual(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.get_extra_info().check_forces_virtual_or_virtualizable())
            .unwrap_or(false)
    }

    /// heap.py: force_from_effectinfo(effectinfo)
    ///
    /// Selective cache invalidation based on EffectInfo bitstrings.
    /// Instead of invalidating all caches, only force/invalidate
    /// fields and arrays that the call may read or write.
    ///
    /// heapcache.py:341 parity: mark_escaped_varargs runs BEFORE
    /// invalidate_unescaped so that call arguments are already escaped
    /// when cache invalidation checks unescaped status.
    fn force_from_effectinfo(&mut self, op: &Op, ctx: &mut OptContext) {
        // heapcache.py:259-293: escape call arguments first
        self.mark_escaped_varargs(op, ctx);

        let ei = match op.descr.as_ref().and_then(|d| d.as_call_descr()) {
            Some(cd) => cd.get_extra_info().clone(),
            None => {
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                self.clean_caches(ctx);
                return;
            }
        };

        // RPython effectinfo.py: zero bitstrings mean the call touches NO
        // tracked heap fields (e.g., I/O). Only fall back to conservative
        // invalidation for calls with ForcesVirtual/RandomEffects.
        // heap.py:567-571: forces_virtual_or_virtualizable → force virtualref field
        // (In RPython this forces vrefinfo.descr_forced; majit has no virtualref
        // field tracking yet, so this is a no-op placeholder.)
        //
        // Note: has_random_effects() calls are filtered BEFORE reaching this
        // function (emitting_operation line 460: !has_random_effects → return).
        // No special handling needed here.

        // heap.py:542-558 force_from_effectinfo field/array loops.
        // Each descr is checked for readonly and write effects.
        // cf.force_lazy_set(optheap, descr, can_cache) is the core:
        //   can_cache=True  (readonly): invalidate → emit → put_field_back_to_info
        //   can_cache=False (write):    invalidate → emit → return (no put_back)
        //                               if no lazy_set: just invalidate
        // heap.py:189-191: invalidate checks descr.is_always_pure()

        // heap.py:542-552: for fielddescr, cf in self.cached_fields.items()
        let field_entries: Vec<(u32, DescrRef)> = self
            .cached_fields
            .iter()
            .map(|(idx, descr, _)| (*idx, descr.clone()))
            .collect();
        for (field_idx, descr) in field_entries {
            let effect_idx = Self::field_effect_index(&descr);
            if ei.check_readonly_descr_field(effect_idx) {
                // heap.py:543-544: cf.force_lazy_set(self, fielddescr) [can_cache=True]
                let lazy_info = self
                    .get_cached_field_mut(&descr)
                    .and_then(|cf| cf.lazy_set.take());
                if let Some((obj, mut lazy_op)) = lazy_info {
                    // heap.py:129: self.invalidate(descr)
                    if !descr.is_always_pure() {
                        if let Some(cf) = self.get_cached_field_mut(&descr) {
                            cf.invalidate(field_idx, ctx);
                        }
                    }
                    // heap.py:131-135: emit postponed if referenced
                    let needs_postponed = self
                        .postponed_op
                        .as_ref()
                        .map_or(false, |p| lazy_op.args.iter().any(|a| *a == p.pos));
                    if needs_postponed {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                    // heap.py:136: emit_extra(op, emit=False)
                    for arg in lazy_op.args.iter_mut() {
                        *arg = ctx.get_box_replacement(*arg);
                    }
                    let final_value = lazy_op.arg(1);
                    let put_back_op = lazy_op.clone();
                    Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                    // heap.py:142-143: put_field_back_to_info
                    self.cache_field(obj, &descr);
                    ctx.structinfo_setfield(&put_back_op, field_idx, final_value);
                }
            }
            if ei.check_write_descr_field(effect_idx) {
                // heap.py:545-546: cf.force_lazy_set(self, fielddescr, can_cache=False)
                let lazy_info = self
                    .get_cached_field_mut(&descr)
                    .and_then(|cf| cf.lazy_set.take());
                if let Some((_obj, mut lazy_op)) = lazy_info {
                    // heap.py:129: self.invalidate(descr)
                    if !descr.is_always_pure() {
                        if let Some(cf) = self.get_cached_field_mut(&descr) {
                            cf.invalidate(field_idx, ctx);
                        }
                    }
                    // heap.py:131-135: emit postponed if referenced
                    let needs_postponed = self
                        .postponed_op
                        .as_ref()
                        .map_or(false, |p| lazy_op.args.iter().any(|a| *a == p.pos));
                    if needs_postponed {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                    // heap.py:136: emit_extra(op, emit=False)
                    Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                    // heap.py:137-138: can_cache=False → no put_field_back_to_info
                } else {
                    // heap.py:144-145: elif not can_cache: self.invalidate(descr)
                    if !descr.is_always_pure() {
                        if let Some(cf) = self.get_cached_field_mut(&descr) {
                            cf.invalidate(field_idx, ctx);
                        }
                    }
                }
            }
        }

        // heap.py:554-558: for arraydescr, submap in self.cached_arrayitems.items()
        let array_descrs: Vec<u32> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, _, _)| *idx)
            .collect();
        for descr_idx in array_descrs {
            let read = ei.check_readonly_descr_array(descr_idx);
            let write = ei.check_write_descr_array(descr_idx);
            if !read && !write {
                continue;
            }
            let indexes: Vec<i64> = match self.get_cached_array_submap(descr_idx) {
                Some(submap) => submap.const_indexes.keys().copied().collect(),
                None => continue,
            };
            for index in indexes {
                if read {
                    // heap.py:555-556: force_lazy_setarrayitem_submap(submap) [can_cache=True]
                    // → cf.force_lazy_set(self, None, can_cache=True)
                    let lazy_info = self
                        .get_cached_array_submap_mut(descr_idx)
                        .and_then(|s| s.const_indexes.get_mut(&index))
                        .and_then(|cai| cai.lazy_set.take());
                    if let Some((arr, mut lazy_op)) = lazy_info {
                        // heap.py:129: self.invalidate(descr)
                        if let Some(cai) = self
                            .get_cached_array_submap_mut(descr_idx)
                            .and_then(|s| s.const_indexes.get_mut(&index))
                        {
                            cai.invalidate(ctx);
                        }
                        // heap.py:131-135: emit postponed if referenced
                        let needs_postponed = self
                            .postponed_op
                            .as_ref()
                            .map_or(false, |p| lazy_op.args.iter().any(|a| *a == p.pos));
                        if needs_postponed {
                            if let Some(p) = self.postponed_op.take() {
                                ctx.emit_extra(ctx.current_pass_idx, p);
                            }
                        }
                        // heap.py:136: emit_extra(op, emit=False)
                        for arg in lazy_op.args.iter_mut() {
                            *arg = ctx.get_box_replacement(*arg);
                        }
                        let final_value = lazy_op.arg(2);
                        let put_back_op = lazy_op.clone();
                        Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                        // heap.py:142-143: put_field_back_to_info
                        self.cache_arrayitem(arr, descr_idx, index, put_back_op.descr.as_ref());
                        ctx.arrayinfo_setitem(&put_back_op, index as usize, final_value);
                    }
                }
                if write {
                    // heap.py:557-558: force_lazy_setarrayitem_submap(submap, can_cache=False)
                    // → cf.force_lazy_set(self, None, can_cache=False)
                    let lazy_info = self
                        .get_cached_array_submap_mut(descr_idx)
                        .and_then(|s| s.const_indexes.get_mut(&index))
                        .and_then(|cai| cai.lazy_set.take());
                    if let Some((_arr, mut lazy_op)) = lazy_info {
                        // heap.py:129: self.invalidate(descr)
                        if let Some(cai) = self
                            .get_cached_array_submap_mut(descr_idx)
                            .and_then(|s| s.const_indexes.get_mut(&index))
                        {
                            cai.invalidate(ctx);
                        }
                        // heap.py:131-135: emit postponed if referenced
                        let needs_postponed = self
                            .postponed_op
                            .as_ref()
                            .map_or(false, |p| lazy_op.args.iter().any(|a| *a == p.pos));
                        if needs_postponed {
                            if let Some(p) = self.postponed_op.take() {
                                ctx.emit_extra(ctx.current_pass_idx, p);
                            }
                        }
                        // heap.py:136: emit_extra(op, emit=False)
                        Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                        // can_cache=False → no put_field_back_to_info
                    } else {
                        // heap.py:144-145: elif not can_cache: self.invalidate(descr)
                        if let Some(cai) = self
                            .get_cached_array_submap_mut(descr_idx)
                            .and_then(|s| s.const_indexes.get_mut(&index))
                        {
                            cai.invalidate(ctx);
                        }
                    }
                }
            }
            if write {
                // heap.py:266 ArrayCachedItem.invalidate → parent.clear_varindex()
                if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    submap.clear_varindex();
                }
            }
        }
    }

    // ── Handlers for specific opcodes ──

    fn optimize_getfield(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return OptimizationResult::Emit(op.clone()),
        };
        let descr = op.descr.as_ref().unwrap();
        let field_idx = Self::field_slot_index(descr);

        // heap.py:640-643: constant_fold — pure getfield on constant object.
        //   if descr.is_always_pure() and self.get_constant_box(arg0):
        //       resbox = self.optimizer.constant_fold(op)
        //       self.optimizer.make_constant(op, resbox)
        if descr.is_always_pure() {
            if ctx.get_constant_box(op.arg(0)).is_some() {
                if let Some(value) = ctx.constant_fold(&op) {
                    let const_pos = ctx.alloc_op_position();
                    ctx.make_constant(const_pos, value);
                    ctx.replace_op(op.pos, const_pos);
                    return OptimizationResult::Remove;
                }
            }
        }

        let _struct_ref = ctx.ensure_ptr_info_arg0(op);

        // heap.py:103-120: getfield_from_cache — 3-way aliasing check.
        let (raw_obj, _) = key;
        // heap.py:645
        //     structinfo = self.ensure_ptr_info_arg0(op)
        //
        // PyPy passes `structinfo` directly to `cf.getfield_from_cache`,
        // which uses Python object identity. The Rust port's
        // `cached_fields` map is keyed by `OpRef` instead, so we resolve
        // arg0 once and then call `ensure_ptr_info_arg0` purely for its
        // side-effect of installing a `PtrInfo` slot on `box._forwarded`.
        // Subsequent passes (intbounds, virtualstate) and the local
        // `setfield` mutation point all read/write that slot via the
        // canonical OpRef.
        let obj = ctx.get_box_replacement(raw_obj);
        let _ = ctx.ensure_ptr_info_arg0(op);
        let mut force_lazy = false;
        if let Some(cf) = self.get_cached_field(descr) {
            if let Some((lazy_obj, lazy_op)) = &cf.lazy_set {
                if *lazy_obj == obj {
                    // MUST_ALIAS: lazy_set targets the same struct → return rhs
                    let cached = lazy_op.arg(1);
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
                // heap.py:67-75 possible_aliasing_two_infos:
                //     if opinfo1.same_info(opinfo2): return MUST_ALIAS
                //     if cf._cannot_alias_via_classes_or_lengths(...): return CANNOT_ALIAS
                //     if cf._cannot_alias_via_content(...): return CANNOT_ALIAS
                //     return UNKNOWN_ALIAS
                let lazy_obj_resolved = ctx.get_box_replacement(*lazy_obj);
                let cannot_alias =
                    CachedField::_cannot_alias_via_classes_or_lengths(lazy_obj_resolved, obj, ctx)
                        || CachedField::_cannot_alias_via_content(lazy_obj_resolved, obj, ctx);
                if !cannot_alias {
                    // UNKNOWN_ALIAS → force_lazy_set, return None (cache miss)
                    force_lazy = true;
                }
                // CANNOT_ALIAS: fall through to _getfield below (heap.py:117)
            }
            // heap.py:117-120: always check cache entries after alias analysis.
            // RPython falls through here even when lazy_set exists (CANNOT_ALIAS).
            if !force_lazy {
                if let Some(entry) = cf._getfield(obj, descr, field_idx, ctx) {
                    // heap.py:182-186: isinstance(res, PreambleOp)
                    match entry {
                        crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                            // heap.py:185-186: force_op_from_preamble, setfield
                            let cached = pop.resolved;
                            ctx.force_op_from_preamble_op(&pop);
                            ctx.structinfo_setfield(op, field_idx, cached);
                            self.field_cache(descr).register_info(obj);
                            let cached = ctx.get_box_replacement(cached);
                            ctx.replace_op(op.pos, cached);
                            return OptimizationResult::Remove;
                        }
                        crate::optimizeopt::info::FieldEntry::Value(cached) => {
                            if !cached.is_none() {
                                let cached = ctx.get_box_replacement(cached);
                                ctx.replace_op(op.pos, cached);
                                return OptimizationResult::Remove;
                            }
                        }
                    }
                }
            }
        }
        // heap.py:109-111: UNKNOWN_ALIAS → force lazy_set and return cache miss
        // heap.py:122: force_lazy_set(can_cache=True) — reads don't destroy cache,
        // so put_field_back_to_info restores the lazy value into the cache.
        // (Contrast with write-descr force in force_from_effectinfo which uses
        // can_cache=False and does NOT restore the value.)
        if force_lazy {
            let lazy_data = self
                .get_cached_field_mut(descr)
                .and_then(|cf| cf.lazy_set.take());
            if let Some((lazy_obj, mut lazy_op)) = lazy_data {
                if !descr.is_always_pure() {
                    if let Some(cf) = self.get_cached_field_mut(descr) {
                        cf.invalidate(field_idx, ctx);
                    }
                }
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos;
                    if lazy_op.args.iter().any(|a| *a == ppos) {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                }
                Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                // can_cache=True: put_field_back_to_info
                let final_value = lazy_op.arg(1);
                let lazy_descr = lazy_op.descr.as_ref().unwrap().clone();
                let lazy_field_idx = Self::field_slot_index(&lazy_descr);
                self.field_cache(&lazy_descr).register_info(lazy_obj);
                // heap.py:122 (force_lazy_set → put_field_back_to_info):
                //     opinfo.setfield(...) on the structinfo of lazy_obj.
                // Routes constants through `const_infos` per
                // `info.py:750-752 ConstPtrInfo.setfield`.
                ctx.structinfo_setfield(&lazy_op, lazy_field_idx, final_value);
            }
            // Cache miss — fall through to emit the getfield
        }

        // Virtualizable fields are loop-variant; skip caching/import.
        let is_vable_field = descr.is_virtualizable();

        // RPython parity: PreambleOp detection is now inline in _getfield →
        // FieldEntry::Preamble (heap.py:182-186). The getfield_from_cache path
        // above handles both Value and Preamble entries uniformly.

        // Check immutable field cache first — these survive all invalidation.
        if let Some(&cached) = self.immutable_cached_fields.get(&key) {
            let cached = ctx.get_box_replacement(cached);
            ctx.replace_op(op.pos, cached);
            return OptimizationResult::Remove;
        }

        // Check read cache (after import).
        if let Some(cf) = self.get_cached_field(descr) {
            if let Some(entry) = cf._getfield(obj, descr, field_idx, ctx) {
                match entry {
                    crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                        let cached = pop.resolved;
                        ctx.force_op_from_preamble_op(&pop);
                        ctx.structinfo_setfield(op, field_idx, cached);
                        self.field_cache(descr).register_info(obj);
                        let cached = ctx.get_box_replacement(cached);
                        ctx.replace_op(op.pos, cached);
                        return OptimizationResult::Remove;
                    }
                    crate::optimizeopt::info::FieldEntry::Value(cached) => {
                        if !cached.is_none() {
                            let cached = ctx.get_box_replacement(cached);
                            ctx.replace_op(op.pos, cached);
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
        }

        // Check quasi-immutable cache: if this field was marked by
        // QUASIIMMUT_FIELD, the value is stable (guarded by GUARD_NOT_INVALIDATED).
        if let Some(&qi_cached) = self.quasi_immut_cache.get(&key) {
            if !qi_cached.is_none() {
                // Subsequent read: reuse the cached value.
                let qi_cached = ctx.get_box_replacement(qi_cached);
                ctx.replace_op(op.pos, qi_cached);
                return OptimizationResult::Remove;
            }
            // First read after QUASIIMMUT_FIELD: emit the load, then cache
            // the result so it survives calls (unlike normal mutable fields).
            self.quasi_immut_cache.insert(key, op.pos);
            self.cache_field(obj, descr);
            ctx.structinfo_setfield(op, field_idx, op.pos);
            return OptimizationResult::Emit(op.clone());
        }

        // Cache miss: emit the load and cache the result.
        // heap.py line 652: make_nonnull(op.getarg(0))
        // optimizer.py:437-448: only set NonNull if no existing PtrInfo.
        // heap.py postprocess_GETFIELD_GC_I:
        //     structinfo = self.ensure_ptr_info_arg0(op)
        //     structinfo.setfield(descr, op.getarg(0), op, ...)
        //
        // The PyPy primitive returns the same structinfo regardless of
        // whether it was newly installed or already present, so the
        // line-by-line port reuses the EnsuredPtrInfo handle for both
        // the nonnull bookkeeping and the field write.
        let struct_ref = ctx.get_box_replacement(op.arg(0));
        vb_set(&mut self.known_nonnull, struct_ref.0);
        self.cache_field(obj, descr);
        // Save immutable fields in the permanent cache — they survive all
        // invalidation because the value never changes.
        if descr.is_always_pure() {
            self.immutable_cached_fields.insert(key, op.pos);
        }
        // heap.py postprocess_GETFIELD_GC_I: structinfo.setfield(descr, op)
        //
        // PyPy info.py:750-752 routes ConstPtrInfo.setfield through
        // optheap.const_infos via `_get_info(parent_descr, optheap)`, so
        // a constant struct base ALSO gets its field cached. The Rust
        // port mirrors that via `OptContext::structinfo_setfield`,
        // which dispatches by `arg0.is_constant()` to either
        // `const_infos[gcref]` (constant) or
        // `ensure_ptr_info_arg0(op).as_mut()` (regular).
        if !is_vable_field {
            ctx.structinfo_setfield(op, field_idx, op.pos);
        }
        // Virtualizable Ref fields (linked list head) need a null guard.
        let is_vable_ref =
            is_vable_field && matches!(op.opcode, OpCode::GetfieldGcR | OpCode::GetfieldGcPureR);
        if is_vable_ref {
            ctx.emit(op.clone());
            let zero_ref = ctx.make_constant_int(0);
            let cmp_pos = ctx.alloc_op_position();
            let mut cmp_op = Op::new(OpCode::IntNe, &[op.pos, zero_ref]);
            cmp_op.pos = cmp_pos;
            ctx.emit(cmp_op);
            // unroll.py:409 parity: synthetic guards inherit
            // rd_resume_position from patchguardop (the optimizer's
            // running GUARD_FUTURE_CONDITION). Without this, the guard
            // arrives at store_final_boxes_in_guard with -1 and would
            // be silently dropped under the patchguardop-only fallback.
            let mut guard_op = Op::new(OpCode::GuardTrue, &[cmp_pos]);
            if let Some(ref patch) = ctx.patchguardop {
                guard_op.rd_resume_position = patch.rd_resume_position;
            }
            ctx.emit(guard_op);
            return OptimizationResult::Remove;
        }
        OptimizationResult::Emit(op.clone())
    }

    fn optimize_setfield(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return OptimizationResult::Emit(op.clone()),
        };
        let descr = op.descr.as_ref().unwrap();
        let field_idx = Self::field_slot_index(descr);
        let (raw_obj, _) = key;
        // heap.py:78 (CachedField.do_setfield via optheap.ensure_ptr_info_arg0):
        //     structinfo = optheap.ensure_ptr_info_arg0(op)
        //
        // Same shape as optimize_getfield: pull the canonical OpRef for
        // the cache key, install the structinfo as a side effect.
        let obj = ctx.get_box_replacement(raw_obj);
        let _ = ctx.ensure_ptr_info_arg0(op);
        let new_value = op.arg(1);

        // heapcache.py:224-230 _escape_from_write parity:
        // record dependency if both container and value are unescaped;
        // otherwise escape the value immediately.
        self.escape_from_write(obj, new_value);

        // heap.py:77-101: do_setfield — check write-after-write, aliasing, lazy set
        // Check write-after-write first (before possible_aliasing).
        {
            let cf = self.field_cache(descr);
            if let Some((lazy_obj, lazy_op)) = &cf.lazy_set {
                if *lazy_obj == obj && lazy_op.arg(1) == new_value {
                    return OptimizationResult::Remove;
                }
            } else if let Some(entry) = cf._getfield(obj, descr, field_idx, ctx) {
                match entry {
                    crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                        let cached_seen = ctx.get_box_replacement(pop.op);
                        if cached_seen == new_value {
                            let cached = pop.resolved;
                            ctx.force_op_from_preamble_op(&pop);
                            ctx.structinfo_setfield(op, field_idx, cached);
                            self.field_cache(descr).register_info(obj);
                            return OptimizationResult::Remove;
                        }
                    }
                    crate::optimizeopt::info::FieldEntry::Value(cached) => {
                        let cached_resolved = ctx.get_box_replacement(cached);
                        if cached_resolved == new_value {
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
        }

        // heap.py:81-83: possible_aliasing → force_lazy_set.
        // RPython: possible_aliasing checks only whether lazy_set targets
        // a DIFFERENT object. If yes, force unconditionally — the
        // _cannot_alias_via_classes optimization applies to cache
        // invalidation (possible_aliasing_two_infos), not to force_lazy_set.
        let needs_force = self
            .get_cached_field(descr)
            .map_or(false, |cf| cf.possible_aliasing(obj));
        if needs_force {
            let lazy_data = self
                .get_cached_field_mut(descr)
                .and_then(|cf| cf.lazy_set.take());
            if let Some((lazy_obj, mut lazy_op)) = lazy_data {
                // heap.py:122-143: force_lazy_set
                // 1. invalidate (skip pure)
                if !descr.is_always_pure() {
                    if let Some(cf) = self.get_cached_field_mut(descr) {
                        cf.invalidate(field_idx, ctx);
                    }
                }
                // 2. emit postponed_op if referenced
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos;
                    if lazy_op.args.iter().any(|a| *a == ppos) {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                }
                // 3. emit the setfield
                Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                // 4. put_field_back_to_info: heap.py:122 calls
                //    `opinfo.setfield(...)` on the structinfo of lazy_obj.
                //    Constant struct bases route through `const_infos`
                //    via `info.py:750-752 ConstPtrInfo.setfield`.
                let final_value = lazy_op.arg(1);
                let lazy_descr = lazy_op.descr.as_ref().unwrap().clone();
                let lazy_field_idx = Self::field_slot_index(&lazy_descr);
                self.cache_field(lazy_obj, &lazy_descr);
                ctx.structinfo_setfield(&lazy_op, lazy_field_idx, final_value);
            }
        }

        // heap.py:84-101: after force, recheck cached value
        {
            if let Some(entry) = self
                .field_cache(descr)
                ._getfield(obj, descr, field_idx, ctx)
            {
                match entry {
                    crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                        let cached_seen = ctx.get_box_replacement(pop.op);
                        if cached_seen == new_value {
                            let cached = pop.resolved;
                            ctx.force_op_from_preamble_op(&pop);
                            ctx.structinfo_setfield(op, field_idx, cached);
                            self.field_cache(descr).register_info(obj);
                            self.field_cache(descr).lazy_set = None;
                            return OptimizationResult::Remove;
                        }
                    }
                    crate::optimizeopt::info::FieldEntry::Value(cached) => {
                        let cached_resolved = ctx.get_box_replacement(cached);
                        if cached_resolved == new_value {
                            self.field_cache(descr).lazy_set = None;
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
        }

        // RPython do_setfield: no separate invalidation here.
        // Aliasing is handled by force_lazy_set → invalidate inside the
        // possible_aliasing path above. Getfield handles read-after-write
        // aliasing via its own force path.

        // heap.py:89-91 do_setfield common case:
        //     self._lazy_set = op
        // PyPy do_setfield does NOT touch opinfo._fields here. The
        // PtrInfo write is deferred until force_lazy_set → put_field_back_to_info
        // (heap.py:142-143), which is the only path that calls
        // `opinfo.setfield(..., cf=cf)` and registers the cached_info link.
        // getfield_from_cache checks `_lazy_set` first, so reads still see
        // the pending value before it gets committed.
        let cf = self.field_cache(descr);
        cf.lazy_set = Some((obj, op.clone()));
        OptimizationResult::Remove
    }

    fn optimize_getarrayitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Install ArrayPtrInfo via ensure_ptr_info_arg0 (return value
        // unused — we re-borrow further down via a fresh call so the
        // intermediate cache mutations can take &mut ctx without
        // tripping the borrow checker).
        let _ = ctx.ensure_ptr_info_arg0(op);
        let array_ref = ctx.get_box_replacement(op.arg(0));

        // Try constant-index cache first.
        if let Some(key) = Self::arrayitem_key(op, ctx) {
            let (array, descr_idx, const_index) = key;
            let descr = op.descr.as_ref().unwrap();
            // heap.py:103-120 getfield_from_cache — 3-way aliasing check.
            // PyPy's shared AbstractCachedEntry method on ArrayCachedItem
            // calls possible_aliasing_two_infos which can force_lazy_set
            // on UNKNOWN_ALIAS. The Rust port inlines this at the call
            // site because force_lazy_set needs &mut OptHeap + &mut OptContext.
            let mut force_lazy_arr = false;
            if let Some(cai) = self
                .get_cached_array_submap(descr_idx)
                .and_then(|s| s.const_indexes.get(&const_index))
            {
                if let Some((lazy_obj, lazy_op)) = &cai.lazy_set {
                    if *lazy_obj == array {
                        // MUST_ALIAS: lazy_set targets the same array → return rhs
                        let cached = lazy_op.arg(2);
                        ctx.replace_op(op.pos, cached);
                        return OptimizationResult::Remove;
                    }
                    // heap.py:108 possible_aliasing_two_infos
                    let lazy_obj_resolved = ctx.get_box_replacement(*lazy_obj);
                    let cannot_alias = ArrayCachedItem::_cannot_alias_via_classes_or_lengths(
                        lazy_obj_resolved,
                        array,
                        ctx,
                    ) || ArrayCachedItem::_cannot_alias_via_content(
                        lazy_obj_resolved,
                        array,
                        ctx,
                    );
                    if !cannot_alias {
                        // UNKNOWN_ALIAS → force_lazy_set
                        force_lazy_arr = true;
                    }
                    // CANNOT_ALIAS: fall through to _getfield
                }
                if !force_lazy_arr {
                    if let Some(entry) = cai._getfield(array, op.descr.as_ref().unwrap(), ctx) {
                        match entry {
                            crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                                let cached = pop.resolved;
                                ctx.force_op_from_preamble_op(&pop);
                                self.arrayitem_cache(descr, const_index)
                                    .register_info(array);
                                ctx.arrayinfo_setitem(op, const_index as usize, cached);
                                let cached = ctx.get_box_replacement(cached);
                                ctx.replace_op(op.pos, cached);
                                return OptimizationResult::Remove;
                            }
                            crate::optimizeopt::info::FieldEntry::Value(cached) => {
                                if !cached.is_none() {
                                    let cached = ctx.get_box_replacement(cached);
                                    ctx.replace_op(op.pos, cached);
                                    return OptimizationResult::Remove;
                                }
                            }
                        }
                    }
                }
            }
            // heap.py:109-111: UNKNOWN_ALIAS → force lazy_set (can_cache=True)
            if force_lazy_arr {
                let lazy_data = self
                    .get_cached_array_submap_mut(descr_idx)
                    .and_then(|s| s.const_indexes.get_mut(&const_index))
                    .and_then(|cai| cai.lazy_set.take());
                if let Some((_lazy_obj, mut lazy_op)) = lazy_data {
                    if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                        submap.invalidate_index(const_index, ctx);
                    }
                    if let Some(ref postponed) = self.postponed_op {
                        let ppos = postponed.pos;
                        if lazy_op.args.iter().any(|a| *a == ppos) {
                            if let Some(p) = self.postponed_op.take() {
                                ctx.emit_extra(ctx.current_pass_idx, p);
                            }
                        }
                    }
                    Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                    // can_cache=True: put_field_back_to_info
                    let final_value = lazy_op.arg(2);
                    let descr = lazy_op.descr.clone();
                    let lazy_obj = ctx.get_box_replacement(lazy_op.arg(0));
                    self.cache_arrayitem(lazy_obj, descr_idx, const_index, descr.as_ref());
                    ctx.arrayinfo_setitem(&lazy_op, const_index as usize, final_value);
                }
                // Cache miss — fall through to emit the getarrayitem
            }
            // Consume the imported short arrayitem: remove it so that if a later
            // setarrayitem/call invalidates cached_arrayitems, the stale preamble
            // value cannot re-populate the cache on a subsequent getarrayitem.
            let pop = ctx
                .get_ptr_info_mut(array)
                .and_then(|info| info.take_preamble_item(const_index as usize))
                .or_else(|| {
                    ctx.get_const_info_mut_if_exists(array)
                        .and_then(|info| info.take_preamble_item(const_index as usize))
                });
            if let Some(pop) = pop {
                let cached = pop.resolved;
                ctx.force_op_from_preamble_op(&pop);
                self.arrayitem_cache(descr, const_index)
                    .register_info(array);
                ctx.arrayinfo_setitem(op, const_index as usize, cached);
                let cached = ctx.get_box_replacement(cached);
                ctx.replace_op(op.pos, cached);
                return OptimizationResult::Remove;
            }
            if let Some(cai) = self
                .get_cached_array_submap(descr_idx)
                .and_then(|s| s.const_indexes.get(&const_index))
            {
                if let Some(entry) = cai._getfield(array, op.descr.as_ref().unwrap(), ctx) {
                    match entry {
                        crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                            let cached = pop.resolved;
                            ctx.force_op_from_preamble_op(&pop);
                            self.arrayitem_cache(descr, const_index)
                                .register_info(array);
                            ctx.arrayinfo_setitem(op, const_index as usize, cached);
                            let cached = ctx.get_box_replacement(cached);
                            ctx.replace_op(op.pos, cached);
                            return OptimizationResult::Remove;
                        }
                        crate::optimizeopt::info::FieldEntry::Value(cached) => {
                            if !cached.is_none() {
                                let cached = ctx.get_box_replacement(cached);
                                ctx.replace_op(op.pos, cached);
                                return OptimizationResult::Remove;
                            }
                        }
                    }
                }
            }
            self.cache_arrayitem(array, descr_idx, const_index, op.descr.as_ref());
            // Track immutable array descriptors so clean_caches() preserves them.
            // Field descriptors use `descr.is_always_pure()` directly because
            // their cache is keyed by descriptor identity.
            if let Some(descr) = &op.descr {
                if descr.is_always_pure() {
                    vb_set(&mut self.immutable_array_descrs, descr_idx);
                }
            }
            // heap.py:676-681:
            //     arrayinfo = self.ensure_ptr_info_arg0(op)
            //     ...
            //     arrayinfo.getlenbound(None).make_gt_const(index)
            //
            // PyPy then `arrayinfo.setitem(...)` records the cached element.
            // The Rust port:
            //   1) `make_nonnull(op.getarg(0))` (heap.py:701) on the box itself
            //   2) for non-constant arg0, narrow the lenbound on the
            //      Forwarded::Info(ArrayPtrInfo) slot via `ensure_ptr_info_arg0`
            //   3) `arrayinfo.setitem(...)` via `arrayinfo_setitem` which
            //      routes constant arg0 through `_get_array_info` /
            //      `const_infos[gcref]` and regular arg0 through
            //      `ensure_ptr_info_arg0(op).as_mut().setitem(...)`.
            let array_ref = ctx.get_box_replacement(op.arg(0));
            vb_set(&mut self.known_nonnull, array_ref.0);
            if const_index >= 0 {
                let mut arrayinfo = ctx.ensure_ptr_info_arg0(op);
                if let Some(mut bound) = arrayinfo.getlenbound(None) {
                    let _ = bound.make_gt_const(const_index);
                    if let Some(crate::optimizeopt::info::PtrInfo::Array(a)) = arrayinfo.as_mut() {
                        a.lenbound = bound;
                    }
                }
            }
            ctx.arrayinfo_setitem(op, const_index as usize, op.pos);
            return OptimizationResult::Emit(op.clone());
        }

        // heap.py:319-324 lookup_cached + heap.py:308-314 cache_varindex_read
        // — variable-index cache via per-arraydescr submap.
        if let Some(descr) = op.descr.as_ref() {
            let descr_idx = descr.index();
            let arrayinfo = array_ref;
            let indexbox = ctx.get_box_replacement(op.arg(1));
            if let Some(submap) = self.get_cached_array_submap(descr_idx) {
                if let Some(cached) = submap.lookup_cached(arrayinfo, indexbox, ctx) {
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
            }
            self.arrayitem_submap(descr)
                .cache_varindex_read(arrayinfo, indexbox, op.pos);
        }

        // heap.py line 701: make_nonnull(op.getarg(0))
        vb_set(&mut self.known_nonnull, array_ref.0);
        ctx.make_nonnull(op.arg(0));
        OptimizationResult::Emit(op.clone())
    }

    fn optimize_setarrayitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // heapcache.py:224-230 _escape_from_write parity:
        let array_obj = ctx.get_box_replacement(op.arg(0));
        let stored_value = op.arg(2);
        self.escape_from_write(array_obj, stored_value);

        let key = match Self::arrayitem_key(op, ctx) {
            Some(k) => k,
            None => {
                // Non-constant index: force all lazy array stores and invalidate
                // both constant-index and variable-index caches via ArrayCachedItem.invalidate
                // (which RPython implements as: clear const_indexes entry + parent.clear_varindex).
                self.force_all_lazy_setarrayitems(ctx.current_pass_idx, ctx);
                let descr_idxs: Vec<u32> = self
                    .cached_arrayitems
                    .iter()
                    .map(|(idx, _, _)| *idx)
                    .collect();
                for descr_idx in descr_idxs {
                    let indexes: Vec<i64> = match self.get_cached_array_submap(descr_idx) {
                        Some(submap) => submap.const_indexes.keys().copied().collect(),
                        None => continue,
                    };
                    for index in indexes {
                        if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                            submap.invalidate_index(index, ctx);
                        }
                    }
                }
                // heap.py:316-317 cache_varindex_write -- cache this write so that
                // a subsequent read with the same variable index can hit.
                if let Some(descr) = op.descr.as_ref() {
                    let arrayinfo = ctx.get_box_replacement(op.arg(0));
                    let indexbox = ctx.get_box_replacement(op.arg(1));
                    let resbox = ctx.get_box_replacement(op.arg(2));
                    self.arrayitem_submap(descr)
                        .cache_varindex_write(arrayinfo, indexbox, resbox);
                }
                return OptimizationResult::Emit(op.clone());
            }
        };

        let (array, descr_idx, const_index) = key;
        let descr = op.descr.as_ref().unwrap();
        // array is already canonicalized by arrayitem_key
        let new_value = op.arg(2);

        // heap.py:77-101: do_setfield (shared by CachedField AND ArrayCachedItem)
        // Write-after-write check.
        {
            let cai = self.arrayitem_cache(descr, const_index);
            if let Some((lazy_obj, lazy_op)) = &cai.lazy_set {
                if *lazy_obj == array && lazy_op.arg(2) == new_value {
                    return OptimizationResult::Remove;
                }
            } else if let Some(entry) = cai._getfield(array, op.descr.as_ref().unwrap(), ctx) {
                match entry {
                    crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                        let cached_seen = ctx.get_box_replacement(pop.op);
                        if cached_seen == new_value {
                            let cached = pop.resolved;
                            ctx.force_op_from_preamble_op(&pop);
                            self.arrayitem_cache(descr, const_index)
                                .register_info(array);
                            ctx.arrayinfo_setitem(op, const_index as usize, cached);
                            return OptimizationResult::Remove;
                        }
                    }
                    crate::optimizeopt::info::FieldEntry::Value(cached) => {
                        let cached = ctx.get_box_replacement(cached);
                        if cached == new_value {
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
        }

        // heap.py:81-83: possible_aliasing → force_lazy_set
        let needs_force = self
            .get_cached_array_submap(descr_idx)
            .and_then(|s| s.const_indexes.get(&const_index))
            .map_or(false, |cai| cai.possible_aliasing(array));
        if needs_force {
            let lazy_data = self
                .get_cached_array_submap_mut(descr_idx)
                .and_then(|s| s.const_indexes.get_mut(&const_index))
                .and_then(|cai| cai.lazy_set.take());
            if let Some((lazy_obj, mut lazy_op)) = lazy_data {
                if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    submap.invalidate_index(const_index, ctx);
                }
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos;
                    if lazy_op.args.iter().any(|a| *a == ppos) {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                }
                let put_back_op = lazy_op.clone();
                Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                // put_field_back_to_info
                let final_value = lazy_op.arg(2);
                let descr = lazy_op.descr.clone();
                self.cache_arrayitem(lazy_obj, descr_idx, const_index, descr.as_ref());
                ctx.arrayinfo_setitem(&put_back_op, const_index as usize, final_value);
            }
        }

        // heap.py:84-101: recheck after force
        {
            let arr_descr = op.descr.as_ref().unwrap();
            if let Some(entry) = self
                .arrayitem_cache(descr, const_index)
                ._getfield(array, arr_descr, ctx)
            {
                match entry {
                    crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                        let cached_seen = ctx.get_box_replacement(pop.op);
                        if cached_seen == new_value {
                            let cached = pop.resolved;
                            ctx.force_op_from_preamble_op(&pop);
                            self.arrayitem_cache(descr, const_index)
                                .register_info(array);
                            ctx.arrayinfo_setitem(op, const_index as usize, cached);
                            self.arrayitem_cache(descr, const_index).lazy_set = None;
                            return OptimizationResult::Remove;
                        }
                    }
                    crate::optimizeopt::info::FieldEntry::Value(cached) => {
                        let cached_resolved = ctx.get_box_replacement(cached);
                        if cached_resolved == new_value {
                            self.arrayitem_cache(descr, const_index).lazy_set = None;
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
        }

        // heap.py:88-90 do_setfield common case:
        //     self._lazy_set = op
        // PyPy do_setfield does NOT touch opinfo._items here. The
        // ArrayPtrInfo write is deferred until force_lazy_set ->
        // put_field_back_to_info. getfield_from_cache checks `_lazy_set`
        // first, so reads still see the pending value before commit.
        let cai = self.arrayitem_cache(descr, const_index);
        cai.lazy_set = Some((array, op.clone()));
        // heap.py:759 submap.clear_varindex() — written index
        // potentially clobbers any varindex triple in the same submap.
        if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
            submap.clear_varindex();
        }

        OptimizationResult::Remove
    }

    /// Handle operations that may have side effects.
    /// Forces lazy sets and invalidates caches as needed.
    /// Tracks allocations for aliasing analysis.
    fn handle_side_effects(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let opcode = op.opcode;

        // Track allocations for aliasing analysis.
        // Allocated objects are always non-null.
        if opcode.is_malloc() {
            vb_set(&mut self.seen_allocation, op.pos.0);
            vb_set(&mut self.unescaped, op.pos.0);
            vb_set(&mut self.known_nonnull, op.pos.0);
            return OptimizationResult::Emit(op.clone());
        }

        // Note: postponed_op (from CallMayForce) must only be emitted at
        // GuardNotForced, not at arbitrary guards. RPython's emit() callback
        // calls emit_postponed_op() before every op, but the postpone→emit
        // cycle is specifically CallMayForce→GuardNotForced. Don't emit here.

        // Guards: force lazy sets but keep caches (guards don't mutate the heap).
        // RPython heap.py does NOT handle GuardNonnull — that is handled
        // exclusively by rewrite.py. Only track nullity for cache purposes.
        if opcode.is_guard() {
            if opcode == OpCode::GuardNonnull {
                vb_set(&mut self.known_nonnull, op.arg(0).0);
            }

            // GuardClass / GuardNonnullClass imply non-null.
            // GuardValue does NOT imply non-null: the guarded constant
            // may be NULL (rewrite.py:320 handles GUARD_VALUE(..., NULL)).
            match opcode {
                OpCode::GuardClass | OpCode::GuardNonnullClass => {
                    vb_set(&mut self.known_nonnull, op.arg(0).0);
                }
                _ => {}
            }

            // force_lazy_sets_for_guard is now called via emitting_operation
            // callback (which runs for ALL guards regardless of which pass emits
            // them). No need to force here — it was already done.
            return OptimizationResult::Emit(op.clone());
        }

        // Final operations (Jump, Finish): force everything.
        if opcode.is_final() {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            return OptimizationResult::Emit(op.clone());
        }

        // Calls: mark arguments as escaped, force lazy sets, and invalidate.
        if opcode.is_call() {
            let oopspec = Self::get_oopspecindex(op);
            match oopspec {
                // heap.py: DICT_LOOKUP caching — consecutive dict lookups
                // on the same dict with the same key can be deduplicated.
                OopSpecIndex::DictLookup => {
                    self.mark_escaped_varargs(op, ctx);
                    self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                    // Invalidate dict-related caches but keep field/array caches.
                    // Dict operations don't affect struct fields.
                    return OptimizationResult::Emit(op.clone());
                }
                OopSpecIndex::Arraycopy | OopSpecIndex::Arraymove => {
                    // ARRAYCOPY/ARRAYMOVE: only invalidate affected array entries.
                    // Call args: [func_addr, source, dest, source_start, dest_start, length, ...]
                    // args[2] = dest array, args[4] = dest_start, args[5] = length
                    self.mark_escaped_varargs(op, ctx);
                    self.force_all_lazy_sets(ctx.current_pass_idx, ctx);

                    let dest_ref = if op.args.len() > 2 {
                        op.arg(2)
                    } else {
                        OpRef::NONE
                    };
                    let dest_start = if op.args.len() > 4 {
                        ctx.get_constant_int(op.arg(4))
                    } else {
                        None
                    };
                    let length = if op.args.len() > 5 {
                        ctx.get_constant_int(op.arg(5))
                    } else {
                        None
                    };

                    if !dest_ref.is_none() {
                        self.invalidate_array_caches_for_copy(ctx, dest_ref, dest_start, length);
                    } else {
                        self.clean_caches(ctx);
                    }

                    return OptimizationResult::Emit(op.clone());
                }
                _ => {
                    self.mark_escaped_varargs(op, ctx);
                    // heapcache.py:337-369 clear_caches_varargs
                    // Plain residual calls preserve cache entries for
                    // unescaped allocations. Calls with explicit EffectInfo
                    // keep the more precise heap.py force_from_effectinfo path.
                    if op.descr.is_none() {
                        self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                        self.invalidate_caches_for_escaped(ctx);
                    } else if Self::call_has_random_effects(op) {
                        self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                        self.clean_caches(ctx);
                    } else {
                        // heap.py: force_from_effectinfo — selective cache
                        // invalidation using EffectInfo bitstrings.
                        self.force_from_effectinfo(op, ctx);
                    }
                    if Self::call_can_invalidate(op) {
                        self.seen_guard_not_invalidated = false;
                    }
                    return OptimizationResult::Emit(op.clone());
                }
            }
        }

        // Other side-effecting ops: force and invalidate.
        if !opcode.has_no_side_effect() && !opcode.is_ovf() {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            self.clean_caches(ctx);
            return OptimizationResult::Emit(op.clone());
        }

        // Pure / no-side-effect / overflow ops: pass through.
        OptimizationResult::Emit(op.clone())
    }

    fn dispatch_propagate(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            // ── Field reads ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldGcPureI
            | OpCode::GetfieldGcPureR
            | OpCode::GetfieldGcPureF => self.optimize_getfield(op, ctx),

            // ── Raw field reads/writes ──
            // Keep these conservative. The standard heap.py cache/postprocess
            // logic applies to GC field descriptors, while raw field traffic is
            // used by compatibility seams that intentionally reload state from
            // memory instead of carrying it through loop args.
            OpCode::GetfieldRawI | OpCode::GetfieldRawR | OpCode::GetfieldRawF => {
                OptimizationResult::Emit(op.clone())
            }
            OpCode::SetfieldRaw => OptimizationResult::Emit(op.clone()),

            // ── Field writes ──
            OpCode::SetfieldGc => self.optimize_setfield(op, ctx),

            // ── Array item reads ──
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcF => {
                self.optimize_getarrayitem(op, ctx)
            }

            // ── Raw array item reads/writes ──
            // Same rationale as raw fields above: keep exact ordering and
            // dynamic indices visible until we have RPython-style virtualizable
            // handling for these buffers.
            OpCode::GetarrayitemRawI | OpCode::GetarrayitemRawR | OpCode::GetarrayitemRawF => {
                OptimizationResult::Emit(op.clone())
            }

            // ── Array item writes ──
            OpCode::SetarrayitemGc => self.optimize_setarrayitem(op, ctx),
            OpCode::SetarrayitemRaw => OptimizationResult::Emit(op.clone()),

            // ── Interior field reads ──
            // info.py:682: "heapcache does not work for interiorfields"
            // RPython has no optimize_GETINTERIORFIELD_GC handler — falls
            // through to optimize_default (just emit). GETINTERIORFIELD has
            // no side effect so emitting_operation returns early (heap.py:428).
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => OptimizationResult::Emit(op.clone()),
            // SETINTERIORFIELD_GC: NOT matched here — falls through to
            // handle_side_effects (the `_` arm). RPython heap.py:463-464:
            // SETINTERIORFIELD_GC is NOT in the emitting_operation exclusion
            // list, so it triggers force_all_lazy_sets + clean_caches.

            // ARRAYLEN_GC / STRLEN / UNICODELEN: pure ops handled by OptPure
            // (resoperation.py:947-1056 _ALWAYS_PURE_FIRST..LAST). Heap CSE
            // would shadow OptPure's `_pure_operations[opnum]` table.

            // ── heap.py: Allocation tracking ──
            OpCode::New | OpCode::NewWithVtable | OpCode::NewArray | OpCode::NewArrayClear => {
                vb_set(&mut self.seen_allocation, op.pos.0);
                vb_set(&mut self.unescaped, op.pos.0);
                vb_set(&mut self.known_nonnull, op.pos.0);
                OptimizationResult::PassOn
            }

            // RPython heap.py: CALL_ASSEMBLER — force all lazy sets before
            // the call. The callee reads from the allocated objects passed
            // in the args array; any pending SetfieldGc must be flushed to
            // memory before execution transfers to the callee.
            //
            // Unlike force_all_lazy_setfields (which mirrors RPython's
            // emit_extra(emit=False) and drops non-virtualizable ops),
            // CALL_ASSEMBLER REQUIRES the SetfieldGc ops to reach the
            // compiled code so that forced-virtual objects have their
            // fields initialized before the callee reads them.
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                // heap.py:454-455: call_assembler always resets
                // _seen_guard_not_invalidated (can call arbitrary code).
                self.seen_guard_not_invalidated = false;
                self.mark_escaped_varargs(op, ctx);
                // heap.py:463-464: force_all_lazy_sets + clean_caches.
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                self.clean_caches(ctx);
                return OptimizationResult::Emit(op.clone());
            }

            // ── heap.py: CALL_MAY_FORCE — postpone until GUARD_NOT_FORCED ──
            // These calls may force virtualizable objects, so we defer emission
            // until the guard arrives, ensuring correct exception semantics.
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[opt-heap] postpone {:?} pos={:?} descr={:?}",
                        op.opcode, op.pos, op.descr
                    );
                }
                // RPython emitting_operation: calls go through
                // force_from_effectinfo (selective) or clean_caches,
                // NOT force_all_lazy. force_all_lazy is only in flush().
                self.mark_escaped_varargs(op, ctx);
                // Postpone the call — it will be emitted when GUARD_NOT_FORCED arrives.
                self.postponed_op = Some(op.clone());
                if Self::call_has_random_effects(op) {
                    self.clean_caches(ctx);
                } else {
                    self.force_from_effectinfo(op, ctx);
                }
                if Self::call_can_invalidate(op) {
                    self.seen_guard_not_invalidated = false;
                }
                return OptimizationResult::Remove;
            }

            // heap.py: GUARD_NOT_FORCED — emit the postponed call_may_force,
            // then handle as a guard. RPython uses force_lazy_sets_for_guard
            // (not force_all_lazy) — immutable caches survive.
            OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                if let Some(postponed) = self.postponed_op.take() {
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[opt-heap] emit postponed {:?} pos={:?} before {:?} pos={:?}",
                            postponed.opcode, postponed.pos, op.opcode, op.pos
                        );
                    }
                    // RPython emit_postponed_op: route through next_optimization
                    ctx.emit_extra(ctx.current_pass_idx, postponed);
                } else if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[opt-heap] no postponed op before {:?} pos={:?}",
                        op.opcode, op.pos
                    );
                }
                // RPython emitting_operation for guards:
                //   self.optimizer.pendingfields = self.force_lazy_sets_for_guard()
                let pending_virtual = self.force_lazy_sets_for_guard(ctx.current_pass_idx, ctx);
                for pending_op in pending_virtual {
                    if pending_op.opcode == OpCode::SetarrayitemGc {
                        let descr = pending_op.descr.as_ref().unwrap().clone();
                        if let Some(index) = ctx.get_constant_int(pending_op.arg(1)) {
                            let array = ctx.get_box_replacement(pending_op.arg(0));
                            let cai = self.arrayitem_cache(&descr, index);
                            cai.lazy_set = Some((array, pending_op));
                        } else {
                            ctx.emit(pending_op);
                        }
                    } else {
                        let descr = pending_op.descr.as_ref().unwrap().clone();
                        let obj = ctx.get_box_replacement(pending_op.arg(0));
                        let cf = self.field_cache(&descr);
                        cf.lazy_set = Some((obj, pending_op));
                    }
                }
                return OptimizationResult::Emit(op.clone());
            }

            // ── heap.py: COND_CALL handling ──
            OpCode::CondCallN => {
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                self.clean_caches(ctx);
                OptimizationResult::PassOn
            }

            // heap.py:530-535 optimize_GUARD_NO_EXCEPTION
            // (alias optimize_GUARD_EXCEPTION = optimize_GUARD_NO_EXCEPTION):
            //
            //     def optimize_GUARD_NO_EXCEPTION(self, op):
            //         if self.last_emitted_operation is REMOVED:
            //             return
            //         return self.emit(op)
            //     optimize_GUARD_EXCEPTION = optimize_GUARD_NO_EXCEPTION
            //
            // The REMOVED check is the only path the upstream guard arm
            // shortcuts on. last_emitted_operation is set to REMOVED only
            // by _optimize_CALL_DICT_LOOKUP (heap.py:527), which majit
            // does not yet port (see the field comment near the top of
            // OptHeap). Until that helper lands the REMOVED branch has no
            // producer, so the unconditional emit is the structurally
            // correct port.
            //
            // Returning Emit hands the guard back to the propagate_forward
            // wrapper, which (1) flushes self.postponed_op via the standard
            // emit() override and (2) routes the guard through
            // emit_operation. emit_operation then invokes
            // emitting_operation for OptHeap, whose guard branch
            // (heap.py:432-435) calls force_lazy_sets_for_guard — preserving
            // immutable cache entries and writing the lazy sets only into
            // the guard's pendingfields, never as new ops in the trace.
            OpCode::GuardNoException | OpCode::GuardException => {
                OptimizationResult::Emit(op.clone())
            }

            // ── GUARD_NOT_INVALIDATED deduplication ──
            OpCode::GuardNotInvalidated => {
                if self.seen_guard_not_invalidated {
                    OptimizationResult::Remove
                } else {
                    self.seen_guard_not_invalidated = true;
                    self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                    OptimizationResult::Emit(op.clone())
                }
            }

            // Quasi-immutable field: treat as read + guard_not_invalidated.
            // The QUASIIMMUT_FIELD op marks a field that rarely changes.
            // The optimizer replaces the field read with the cached value and
            // emits GUARD_NOT_INVALIDATED to ensure validity.
            OpCode::QuasiimmutField => {
                // RPython optimize_QUASIIMMUT_FIELD (heap.py:781):
                // Does NOT create a new GUARD_NOT_INVALIDATED — the tracer
                // already emitted one via generate_guard (pyjitpl.py:1087).
                // Records quasi_immutable_deps for invalidation tracking.
                let obj = op.arg(0);
                // RPython optimize_QUASIIMMUT_FIELD: collect quasi-immutable
                // dependencies. Add (obj_ptr, field_idx) to quasi_immutable_deps
                // for per-slot watcher registration after compilation.
                // field_idx comes from descr (GC object fields) or arg(1)
                // (namespace slot index).
                let (dep_field_idx, cache_field_key) = if let Some(descr) = &op.descr {
                    (
                        Some(Self::field_effect_index(descr)),
                        Some(Self::field_cache_identity(descr)),
                    )
                } else if op.args.len() > 1 {
                    let idx = ctx.get_constant_int(op.arg(1)).map(|v| v as u32);
                    (idx, idx.map(|v| v as usize))
                } else {
                    (None, None)
                };
                if let Some(idx) = dep_field_idx {
                    if let Some(dep_ptr) = ctx.get_constant_int(obj) {
                        ctx.add_quasi_immutable_dep((dep_ptr as u64, idx));
                    }
                }
                if let Some(key) = cache_field_key {
                    self.quasi_immut_cache.insert((obj, key), OpRef::NONE);
                }
                OptimizationResult::Remove
            }

            // ── heap.py: RAW_LOAD / RAW_STORE — virtualize.py handles ──
            //
            // PyPy heap.py does NOT cache RAW_LOAD/RAW_STORE. Raw
            // pointer arithmetic over `VirtualRawBuffer` /
            // `VirtualRawSlice` is handled by virtualize.py:358-385.
            // RAW_STORE is also listed in `emitting_operation`'s "no
            // effect on GC struct" list (heap.py:442). Falling through
            // to handle_side_effects matches the PyPy default
            // (`dispatch_opt(default=OptHeap.emit)` at heap.py:898).

            // ── GC_LOAD / GC_LOAD_INDEXED: generic memory loads ──
            // These could read from any field/array slot, so force all
            // pending lazy writes to ensure correct values.
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::GcLoadIndexedI
            | OpCode::GcLoadIndexedR
            | OpCode::GcLoadIndexedF => {
                self.force_all_lazy_setfields(ctx.current_pass_idx, ctx);
                self.force_all_lazy_setarrayitems(ctx.current_pass_idx, ctx);
                vb_set(&mut self.known_nonnull, op.arg(0).0);
                OptimizationResult::Emit(op.clone())
            }

            // ── Everything else: check for side effects ──
            _ => self.handle_side_effects(op, ctx),
        }
    }
}

impl Default for OptHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptHeap {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let result = self.dispatch_propagate(op, ctx);
        // RPython heap.py:417-425 emit() override parity:
        // Before emitting any new op, flush the postponed op. Then
        // postpone comparison/ovf ops (call_may_force already handled
        // in its own match arm).
        if let OptimizationResult::Emit(ref emit_op) = result {
            // Step 1: emit_postponed_op — flush previous postponed
            if let Some(postponed) = self.postponed_op.take() {
                ctx.emit_extra(ctx.current_pass_idx, postponed);
            }
            // Step 2: postpone comparison/ovf
            if emit_op.opcode.is_comparison() || emit_op.opcode.is_ovf() {
                self.postponed_op = Some(emit_op.clone());
                return OptimizationResult::Remove;
            }
        }
        result
    }

    fn setup(&mut self) {
        self.cached_fields.clear();
        self.immutable_cached_fields.clear();
        self.cached_arrayitems.clear();
        self.seen_guard_not_invalidated = false;
        self.postponed_op = None;
        self.immutable_array_descrs.clear();
        self.seen_allocation.clear();
        self.unescaped.clear();
        self.known_nonnull.clear();
        self.quasi_immut_cache.clear();
    }

    fn flush(&mut self, ctx: &mut OptContext) {
        // RPython heap.py: flush() = force_all_lazy_sets(); emit_postponed_op()
        self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
        // RPython emit_postponed_op: route through next_optimization
        if let Some(postponed) = self.postponed_op.take() {
            ctx.emit_extra(ctx.current_pass_idx, postponed);
        }
    }

    /// RPython heap.py: emitting_operation(op)
    /// Called for EVERY op about to be emitted, regardless of which pass emits it.
    /// This is how the heap optimizer forces lazy sets before guards even when
    /// the guard was emitted by an earlier pass (e.g., IntBounds).
    fn emitting_operation(&mut self, op: &Op, ctx: &mut OptContext, self_pass_idx: usize) {
        // heap.py:427-464: emitting_operation(op)
        //
        // RPython calls emitting_operation in heap pass context.
        // Save/restore current_pass_idx so internal methods
        // (force_from_effectinfo, etc.) use the correct heap index.
        let saved_pass_idx = ctx.current_pass_idx;
        ctx.current_pass_idx = self_pass_idx;

        // RPython early returns for side-effect-free operations:
        if op.opcode.has_no_side_effect() {
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        if op.opcode.is_ovf() {
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        // heap.py:432-434: guards → force lazy sets for guard
        if op.opcode.is_guard() {
            let pending_virtual = self.force_lazy_sets_for_guard(self_pass_idx, ctx);
            // heap.py:433: self.optimizer.pendingfields = pendingfields
            ctx.pending_for_guard = pending_virtual;
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        // heap.py:436-452: specific opcodes that don't affect GC caches
        match op.opcode {
            OpCode::SetfieldGc
            | OpCode::SetfieldRaw
            | OpCode::SetarrayitemGc
            | OpCode::SetarrayitemRaw
            | OpCode::SetinteriorfieldRaw
            | OpCode::RawStore
            | OpCode::Strsetitem
            | OpCode::Unicodesetitem
            | OpCode::DebugMergePoint
            | OpCode::JitDebug
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::Copystrcontent
            | OpCode::Copyunicodecontent
            | OpCode::CheckMemoryError => {
                ctx.current_pass_idx = saved_pass_idx;
                return;
            }
            _ => {}
        }
        // heap.py:453-463: calls → handle effects
        if op.opcode.is_call() {
            if op.opcode.is_call_assembler() {
                self.seen_guard_not_invalidated = false;
            } else {
                if Self::call_can_invalidate(op) {
                    self.seen_guard_not_invalidated = false;
                }
                if op.descr.is_none() {
                    self.force_all_lazy_sets(self_pass_idx, ctx);
                    self.invalidate_caches_for_escaped(ctx);
                    ctx.current_pass_idx = saved_pass_idx;
                    return;
                }
                if !Self::call_has_random_effects(op) {
                    self.force_from_effectinfo(op, ctx);
                    ctx.current_pass_idx = saved_pass_idx;
                    return;
                }
            }
        }
        // heap.py:464: everything else → force all lazy sets + clean caches
        self.force_all_lazy_sets(self_pass_idx, ctx);
        self.clean_caches(ctx);
        ctx.current_pass_idx = saved_pass_idx;
    }

    /// heap.py:360-377 OptHeap.produce_potential_short_preamble_ops(sb)
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &mut OptContext,
    ) {
        // heap.py:370-372:
        //     for descr in descrkeys:
        //         d = self.cached_fields[descr]
        //         d.produce_potential_short_preamble_ops(self.optimizer, sb, descr)
        let mut field_entries: Vec<_> = self
            .cached_fields
            .iter()
            .map(|(field_idx, descr, cf)| (descr, (*field_idx, cf)))
            .collect();
        sort_descr_item_refs_untranslated(&mut field_entries);
        for (descr, (field_idx, cf)) in field_entries {
            cf.produce_potential_short_preamble_ops(sb, descr, field_idx, ctx);
        }
        // heap.py:374-377:
        //     for descr, submap in self.cached_arrayitems.items():
        //         for index, d in submap.const_indexes.items():
        //             d.produce_potential_short_preamble_ops(self.optimizer, sb, descr, index)
        for (_, descr, submap) in &self.cached_arrayitems {
            for cai in submap.const_indexes.values() {
                cai.produce_potential_short_preamble_ops(sb, &descr, ctx);
            }
        }
    }

    fn name(&self) -> &'static str {
        "heap"
    }

    /// heap.py:825-846 OptHeap.serialize_optheap(available_boxes)
    fn export_cached_fields(
        &self,
        ctx: &mut OptContext,
        available_boxes: Option<&std::collections::HashMap<OpRef, ()>>,
    ) -> Vec<(OpRef, DescrRef, OpRef)> {
        let mut result = Vec::new();
        // heap.py:827-846: for descr, cf in cached_fields.iteritems():
        for (field_idx, descr, cf) in &self.cached_fields {
            // heap.py:830-831: if cf._lazy_set: continue
            if cf.lazy_set.is_some() {
                continue;
            }
            // heap.py:828: if descr.get_descr_index() == -1: continue
            if descr.get_descr_index() == -1 {
                continue;
            }
            // heap.py:833-834:
            //     parent_descr = descr.get_parent_descr()
            //     if not parent_descr.is_object(): continue
            let parent = descr.as_field_descr().and_then(|fd| fd.get_parent_descr());
            let is_object = parent
                .as_ref()
                .and_then(|pd| pd.as_size_descr())
                .map_or(false, |sd| sd.is_object());
            if !is_object {
                continue;
            }
            // heap.py:835-846: for i, box1 in enumerate(cf.cached_structs)
            for &obj in &cf.cached_structs {
                if obj.is_none() {
                    continue;
                }
                // heap.py:836: if not box1.is_constant() and box1 not in available_boxes: continue
                if let Some(ab) = available_boxes {
                    if !obj.is_constant() && !ab.contains_key(&obj) {
                        continue;
                    }
                }
                // heap.py:838-839: structinfo = cf.cached_infos[i]
                //                  box2 = structinfo.getfield(descr)
                let resolved = ctx.get_box_replacement(obj);
                let Some(val) = ctx
                    .get_ptr_info(resolved)
                    .and_then(|info| info.getfield(*field_idx))
                    .map(|entry| entry.as_seen_opref())
                    .or_else(|| {
                        ctx.get_const_info_mut(resolved, parent.clone())
                            .and_then(|info| info.getfield(*field_idx))
                            .map(|entry| entry.as_seen_opref())
                    })
                else {
                    continue;
                };
                // heap.py:844: box2 = box2.get_box_replacement()
                let val = ctx.get_box_replacement(val);
                // heap.py:845: if box2.is_constant() or box2 in available_boxes:
                if val.is_none() {
                    continue;
                }
                let val_ok = available_boxes.map_or(true, |ab| {
                    val.is_constant() || ctx.is_constant(val) || ab.contains_key(&val)
                });
                if val_ok {
                    result.push((obj, descr.clone(), val));
                }
            }
        }
        result
    }

    /// heap.py:870-883 OptHeap.deserialize_optheap (struct half)
    fn import_cached_fields(&mut self, entries: &[(OpRef, DescrRef, OpRef)], ctx: &mut OptContext) {
        use crate::optimizeopt::info::PtrInfo;
        for (box1, descr, box2) in entries {
            if box1.is_none() || box2.is_none() {
                continue;
            }
            let field_idx = Self::field_slot_index(descr);
            let resolved = ctx.get_box_replacement(*box1);
            // heap.py:872-873: parent_descr = descr.get_parent_descr()
            //                  assert parent_descr.is_object()
            let parent_descr = descr.as_field_descr().and_then(|fd| fd.get_parent_descr());
            debug_assert!(
                parent_descr
                    .as_ref()
                    .and_then(|pd| pd.as_size_descr())
                    .map_or(false, |sd| sd.is_object()),
                "deserialize_optheap: parent_descr must be is_object()"
            );
            // heap.py:874-881:
            //     if box1.is_constant():
            //         structinfo = info.ConstPtrInfo(box1)
            //     else:
            //         structinfo = box1.get_forwarded()
            //         if not isinstance(structinfo, info.AbstractVirtualPtrInfo):
            //             structinfo = info.InstancePtrInfo(parent_descr)
            //             structinfo.init_fields(parent_descr, descr.get_index())
            //             box1.set_forwarded(structinfo)
            let needs_install = !ctx.is_constant(resolved)
                && match ctx.get_ptr_info(resolved) {
                    Some(info) => !info.is_virtual(),
                    None => true,
                };
            if needs_install {
                // info.py:175-188 InstancePtrInfo + init_fields
                ctx.set_ptr_info(resolved, PtrInfo::instance(parent_descr.clone(), None));
            }
            // heap.py:882-883: cf = self.field_cache(descr)
            //                  structinfo.setfield(descr, box1, box2, optheap, cf=cf)
            self.cache_field(*box1, descr);
            if ctx.is_constant(resolved) {
                if let Some(info) = ctx.get_const_info_mut(resolved, parent_descr.clone()) {
                    info.setfield(field_idx, *box2);
                }
            } else if let Some(info) = ctx.get_ptr_info_mut(resolved) {
                info.setfield(field_idx, *box2);
            }
        }
    }

    /// heap.py:847-868 serialize_optheap(available_boxes) (array half)
    fn export_cached_arrayitems(
        &self,
        ctx: &mut OptContext,
        available_boxes: Option<&std::collections::HashMap<OpRef, ()>>,
    ) -> Vec<(OpRef, i64, DescrRef, OpRef)> {
        let mut result = Vec::new();
        for (_, descr, submap) in &self.cached_arrayitems {
            // heap.py:849: if descr.get_descr_index() == -1: continue
            if descr.get_descr_index() == -1 {
                continue;
            }
            for (&index, cai) in &submap.const_indexes {
                // heap.py:852: if cf._lazy_set: continue
                if cai.lazy_set.is_some() {
                    continue;
                }
                for &obj in &cai.cached_structs {
                    if obj.is_none() {
                        continue;
                    }
                    // heap.py:855: if not box1.is_constant() and box1 not in available_boxes: continue
                    if let Some(ab) = available_boxes {
                        if !obj.is_constant() && !ab.contains_key(&obj) {
                            continue;
                        }
                    }
                    // heap.py:858: if index >= 2**15: continue
                    if index >= (1 << 15) {
                        continue;
                    }
                    let resolved = ctx.get_box_replacement(obj);
                    // heap.py:860: box2 = arrayinfo.getitem(descr, index)
                    let Some(val) = ctx
                        .get_ptr_info(resolved)
                        .and_then(|info| info.getitem(index as usize))
                        .map(|entry| entry.as_seen_opref())
                        .or_else(|| {
                            ctx.get_const_info_array_mut(resolved, descr.clone())
                                .and_then(|info| info.getitem(index as usize))
                                .map(|entry| entry.as_seen_opref())
                        })
                    else {
                        continue;
                    };
                    // heap.py:865: box2 = box2.get_box_replacement()
                    let val = ctx.get_box_replacement(val);
                    // heap.py:866: if box2.is_constant() or box2 in available_boxes:
                    if val.is_none() {
                        continue;
                    }
                    let val_ok = available_boxes.map_or(true, |ab| {
                        val.is_constant() || ctx.is_constant(val) || ab.contains_key(&val)
                    });
                    if val_ok {
                        result.push((obj, index, descr.clone(), val));
                    }
                }
            }
        }
        result
    }

    /// heap.py:885-894 deserialize_optheap (array half)
    fn import_cached_arrayitems(
        &mut self,
        entries: &[(OpRef, i64, DescrRef, OpRef)],
        ctx: &mut OptContext,
    ) {
        use crate::optimizeopt::info::PtrInfo;
        for (box1, index, descr, box2) in entries {
            if box1.is_none() || box2.is_none() {
                continue;
            }
            let resolved = ctx.get_box_replacement(*box1);
            // heap.py:886-892:
            //     if box1.is_constant(): arrayinfo = info.ConstPtrInfo(box1)
            //     else:
            //         arrayinfo = box1.get_forwarded()
            //         if not isinstance(arrayinfo, info.AbstractVirtualPtrInfo):
            //             arrayinfo = info.ArrayPtrInfo(descr)
            //             box1.set_forwarded(arrayinfo)
            let needs_install = !ctx.is_constant(resolved)
                && match ctx.get_ptr_info(resolved) {
                    Some(info) => !info.is_virtual(),
                    None => true,
                };
            if needs_install {
                ctx.set_ptr_info(
                    resolved,
                    PtrInfo::array(
                        descr.clone(),
                        crate::optimizeopt::intutils::IntBound::nonnegative(),
                    ),
                );
            }
            // heap.py:893-894: cf = self.arrayitem_cache(descr, index)
            //                  arrayinfo.setitem(descr, index, box1, box2, optheap, cf=cf)
            let cai = self.arrayitem_cache(descr, *index);
            cai.register_info(*box1);
            if ctx.is_constant(resolved) {
                // info.py:746-748 ConstPtrInfo.setitem → _get_array_info
                if let Some(info) = ctx.get_const_info_array_mut(resolved, descr.clone()) {
                    info.setitem(*index as usize, *box2);
                }
            } else if let Some(info) = ctx.get_ptr_info_mut(resolved) {
                info.setitem(*index as usize, *box2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Upstream parity anchor:
    //! `rpython/jit/metainterp/test/test_heapcache.py` and
    //! `rpython/jit/metainterp/test/test_tracingopts.py`
    //! (`test_heapcache_interiorfields`, `test_heapcache_from_constant`, ...).
    //!
    //! Imported-short-field, arraycopy-range, and byte-array cases below are
    //! original Rust regressions for optimizer-state seams that upstream
    //! mostly covers only indirectly through larger integration tests.

    use std::sync::Arc;

    use majit_ir::{
        CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, FieldDescr, OopSpecIndex, Op, OpCode,
        OpRef, SizeDescr, Type,
    };

    use crate::optimizeopt::info::PtrInfo;
    use crate::optimizeopt::optimizer::Optimizer;
    use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

    use super::OptHeap;

    /// Test SizeDescr that pretends to wrap a struct with `is_object()` matching
    /// the constructor arg. Mirrors the PyPy `optimizer.py:480` dispatch test
    /// for `parent_descr.is_object()`.
    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
        is_object: bool,
    }

    impl Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            self.index
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            self.is_object
        }
    }

    /// Single shared parent SizeDescr for all test FieldDescrs. The exact
    /// instance doesn't matter — `ensure_ptr_info_arg0` only reads
    /// `is_object()`. We use a Struct (is_object=false) so the field branch
    /// constructs `PtrInfo::Struct` (the matchless case at heap.rs:1313).
    fn test_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0000,
            is_object: false,
        })
    }

    fn test_object_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0001,
            is_object: true,
        })
    }

    /// Minimal descriptor for tests, identified by its index. Implements
    /// `FieldDescr` with a synthetic Struct parent so the optimizer's
    /// `ensure_ptr_info_arg0` field branch can dispatch correctly.
    #[derive(Debug)]
    struct TestDescr(u32);

    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_descr())
        }
        fn offset(&self) -> usize {
            self.0 as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
    }

    #[derive(Debug)]
    struct ObjectTestDescr(u32);

    impl Descr for ObjectTestDescr {
        fn index(&self) -> u32 {
            self.0
        }

        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for ObjectTestDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_object_parent_descr())
        }
        fn offset(&self) -> usize {
            self.0 as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
    }

    /// Descriptor for immutable (green) fields. `is_always_pure()` returns true.
    #[derive(Debug)]
    struct ImmutableDescr(u32);

    impl Descr for ImmutableDescr {
        fn index(&self) -> u32 {
            self.0
        }

        fn is_always_pure(&self) -> bool {
            true
        }

        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for ImmutableDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_descr())
        }
        fn offset(&self) -> usize {
            self.0 as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
        fn is_immutable(&self) -> bool {
            true
        }
    }

    fn descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr(idx))
    }

    fn immutable_descr(idx: u32) -> DescrRef {
        Arc::new(ImmutableDescr(idx))
    }

    fn object_descr(idx: u32) -> DescrRef {
        Arc::new(ObjectTestDescr(idx))
    }

    #[derive(Debug)]
    struct ParentIndexedDescr {
        parent_idx: u32,
    }

    impl Descr for ParentIndexedDescr {
        fn index(&self) -> u32 {
            0
        }

        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for ParentIndexedDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_descr())
        }

        fn offset(&self) -> usize {
            self.parent_idx as usize * 8
        }

        fn field_size(&self) -> usize {
            8
        }

        fn field_type(&self) -> Type {
            Type::Ref
        }

        fn index_in_parent(&self) -> usize {
            self.parent_idx as usize
        }
    }

    #[test]
    fn field_cache_is_keyed_by_descr_identity_not_field_slot() {
        let descr_a: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 1 });
        let descr_b: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 1 });

        assert_eq!(OptHeap::field_slot_index(&descr_a), 1);
        assert_eq!(OptHeap::field_slot_index(&descr_b), 1);
        assert_ne!(
            OptHeap::field_cache_identity(&descr_a),
            OptHeap::field_cache_identity(&descr_b)
        );

        let mut heap = OptHeap::new();
        heap.cache_field(OpRef(0), &descr_a);
        heap.cache_field(OpRef(0), &descr_b);

        assert_eq!(heap.cached_fields.len(), 2);
    }

    fn initialize_imported_short_heap_field(
        heap: &mut OptHeap,
        ctx: &mut OptContext,
        object: OpRef,
        descr: &DescrRef,
        source: OpRef,
        resolved: OpRef,
        opcode: OpCode,
    ) {
        use crate::optimizeopt::info::{PreambleOp, PtrInfo};

        let mut preamble_op = Op::with_descr(opcode, &[object], descr.clone());
        preamble_op.pos = source;
        ctx.initialize_imported_short_preamble_builder(
            &[object, resolved],
            &[object, resolved],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: preamble_op.clone(),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            }],
        );
        ctx.set_ptr_info(object, PtrInfo::instance(None, None));
        ctx.get_ptr_info_mut(object).unwrap().set_preamble_field(
            OptHeap::field_slot_index(descr),
            PreambleOp {
                op: source,
                resolved,
                invented_name: false,
                preamble_op,
            },
        );
        heap.import_cached_fields(&[(object, descr.clone(), resolved)], ctx);
    }

    /// Call descriptor with default EffectInfo (non-random, non-elidable).
    /// heapcache.py:362-370 parity: plain calls with known effectinfo
    /// use invalidate_unescaped instead of full reset.
    fn plain_call_descr(idx: u32) -> DescrRef {
        Arc::new(majit_ir::SimpleCallDescr::new(
            idx,
            vec![],
            majit_ir::Type::Void,
            false,
            0,
            EffectInfo {
                // Write all fields/arrays so invalidation is triggered
                write_descrs_fields: u64::MAX,
                write_descrs_arrays: u64::MAX,
                ..EffectInfo::default()
            },
        ))
    }

    /// Helper: assign sequential positions to ops.  Also attaches a
    /// fresh `ResumeGuardDescr` to every guard op that lacks one, mirroring
    /// RPython's `optimizer.py:691 assert isinstance(last_descr,
    /// compile.ResumeGuardDescr)` invariant — every guard reaching the
    /// optimizer carries a head-of-chain descr, so `_copy_resume_data_from`
    /// can share via `make_resume_guard_copied_descr(prev)` without panicking
    /// on a missing donor.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
            if op.opcode.is_guard() && op.descr.is_none() {
                op.descr = Some(crate::compile::make_resume_guard_descr_typed(Vec::new()));
            }
        }
    }

    /// Run a single OptHeap pass over the given ops.
    ///
    /// Uses num_inputs=1024 so that high-numbered OpRef values used as
    /// input arguments in tests (e.g. OpRef(100), OpRef(500)) are treated
    /// as valid defined positions by the optimizer's undefined-ref filter.
    ///
    /// In production every recorded Box carries its intrinsic type via
    /// `trace_inputarg_types`, so the preamble exporter can recover a
    /// renamed inputarg's type without guessing. Unit-test inputs are
    /// anonymous stand-ins, so we seed Ref for every slot — the only
    /// use of the type in these tests is to populate
    /// `renamed_inputarg_types`, and Ref keeps heap/aliasing tests on
    /// the same path RPython exercises for pointer Boxes.
    fn run_heap_opt(ops: &mut [Op]) -> Vec<Op> {
        run_heap_opt_typed(ops, &[])
    }

    /// Like `run_heap_opt`, but declares specific OpRef slots as Int-typed.
    /// Use for tests whose anonymous high-numbered Boxes are bound to
    /// int-typed fields (setfield_gc value, int_ args, etc.). Without this
    /// the `trace_inputarg_types = vec![Ref; 1024]` default types every
    /// anonymous slot as Ref and the heap cache's MUST_ALIAS replacement
    /// `replace_op(getfield_gc_i.pos:Int, cached_value:Ref)` trips the
    /// `replace_op` cross-type assertion introduced for the Box.type
    /// invariant.
    fn run_heap_opt_typed(ops: &mut [Op], int_slots: &[u32]) -> Vec<Op> {
        assign_positions(ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptHeap::new()));
        let mut types = vec![Type::Ref; 1024];
        for &idx in int_slots {
            types[idx as usize] = Type::Int;
        }
        opt.trace_inputarg_types = types;
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024)
    }

    // ── Test 1: SETFIELD then GETFIELD → read from cache ──

    #[test]
    fn test_setfield_then_getfield_cached() {
        // setfield_gc(p0, i1, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d0)   <- should be eliminated, replaced by i1
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            // Terminate trace so lazy set is forced.
            Op::new(OpCode::Jump, &[]),
        ];
        // OpRef(101) is the Int value stored into the Int-typed field; the
        // cache replays it as the GetfieldGcI result.
        let result = run_heap_opt_typed(&mut ops, &[101]);

        // force_all_lazy_setfields emits the lazy SetfieldGc before Jump.
        // GetfieldGcI is eliminated (replaced by cached i1). SetfieldGc + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    fn test_imported_short_cached_fields_replays_into_heap() {
        let d = object_descr(55);
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_num_inputs(4, 2);
        initialize_imported_short_heap_field(
            &mut heap,
            &mut ctx,
            OpRef(0),
            &d,
            OpRef(100),
            OpRef(1),
            OpCode::GetfieldGcI,
        );

        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d);
        op.pos = OpRef(2);

        let result = heap.optimize_getfield(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(OpRef(2)), OpRef(1));
    }

    /// After consuming an imported short field, a cache invalidation followed
    /// by another getfield must emit the actual load (not reuse the stale
    /// preamble value).  This prevents null-pointer crashes when the
    /// preamble's cached value (e.g. a linked-list head) is no longer valid
    /// after a call/setfield that empties the container.
    #[test]
    fn test_imported_short_field_not_reused_after_invalidation() {
        let d_head = object_descr(10); // head field
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_num_inputs(4, 4);
        initialize_imported_short_heap_field(
            &mut heap,
            &mut ctx,
            OpRef(0),
            &d_head,
            OpRef(100),
            OpRef(1),
            OpCode::GetfieldGcR,
        );

        // First getfield on head: consumes the import, caches the value.
        let mut op1 = Op::with_descr(OpCode::GetfieldGcR, &[OpRef(0)], d_head.clone());
        op1.pos = OpRef(2);
        let result1 = heap.optimize_getfield(&op1, &mut ctx);
        assert!(matches!(result1, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(OpRef(2)), OpRef(1));

        // A call invalidates all mutable field caches.
        heap.clean_caches(&mut ctx);

        // Second getfield on head after invalidation: must NOT return the
        // stale preamble value.  The import was consumed, so it should emit.
        let mut op2 = Op::with_descr(OpCode::GetfieldGcR, &[OpRef(0)], d_head.clone());
        op2.pos = OpRef(3);
        let result2 = heap.optimize_getfield(&op2, &mut ctx);
        assert!(
            matches!(result2, OptimizationResult::Emit(_)),
            "getfield after invalidation must emit, not reuse stale import"
        );
    }

    #[test]
    fn test_getfield_does_not_deref_arbitrary_int_constant_base() {
        let d = immutable_descr(77);
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_num_inputs(4, 1);
        ctx.make_constant(OpRef(0), majit_ir::Value::Int(1));

        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d);
        op.pos = OpRef(1);

        let result = heap.optimize_getfield(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Emit(_)));
        assert_eq!(ctx.get_box_replacement(OpRef(1)), OpRef(1));
    }

    // ── Test 2: Two GETFIELDs on same object/field → second eliminated ──

    #[test]
    fn test_getfield_read_after_read() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d0)   <- eliminated, reuse i1
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only the first GETFIELD + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 3: SETFIELD then SETFIELD → first eliminated (write-after-write) ──

    #[test]
    fn test_setfield_write_after_write() {
        // setfield_gc(p0, i1, descr=d0)
        // setfield_gc(p0, i2, descr=d0)   <- first is dead
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(102)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // First SetfieldGc is dead (overwritten). Second is emitted as lazy set before Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 4: SETFIELD then CALL then GETFIELD → cache invalidated ──

    #[test]
    fn test_setfield_call_invalidates_cache() {
        // setfield_gc(p0, i1, descr=d0)
        // call_n(...)
        // i2 = getfield_gc_i(p0, descr=d0)   <- cache invalidated by call, must emit
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // force_all_lazy at call emits SetfieldGc + invalidates caches.
        // SetfieldGc + CALL + GETFIELD (re-emitted, cache was invalidated) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 5: SETFIELD on different objects → both cached independently ──

    #[test]
    fn test_setfield_different_objects() {
        // setfield_gc(p0, i1, descr=d0)
        // setfield_gc(p1, i2, descr=d0)  <- possible_aliasing: forces first lazy_set
        // i3 = getfield_gc_i(p0, descr=d0)   <- cached from forced set (i1)
        // i4 = getfield_gc_i(p1, descr=d0)   <- cached from second lazy_set (i2)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(200), OpRef(201)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(200)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[101, 201]);

        // RPython CachedField per-descr with aliasing analysis:
        // - setfield(p1) forces lazy_set(p0) → emit SETFIELD(p0), put_back p0
        // - invalidate_for_write(p1) removes p0 (input args can alias)
        // - getfield(p0): lazy_set is p1, UNKNOWN_ALIAS → force lazy_set(p1)
        //   → emit SETFIELD(p1), put_back p1. p0 entry gone → cache miss → emit GETFIELD(p0)
        // - getfield(p1): entry p1=i2 from put_back → cache hit → remove
        // Result: SETFIELD(p0) + SETFIELD(p1) + GETFIELD(p0) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 6: Array items: SETARRAYITEM then GETARRAYITEM → cached ──

    #[test]
    fn test_setarrayitem_then_getarrayitem_cached() {
        // setarrayitem_gc(p0, i_idx, i_val, descr=d0)
        // i2 = getarrayitem_gc_i(p0, i_idx, descr=d0)   <- eliminated
        // We need i_idx to be a known constant.
        let d = descr(0);
        let idx = OpRef(50);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx, OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        // We need to make the index a known constant in the context.
        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(3));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::Jump]);
    }

    #[test]
    fn test_getarrayitem_postprocess_updates_ptr_info() {
        let d = descr(0);
        let idx = OpRef(50);
        let mut op = Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone());
        op.pos = OpRef(200);

        let mut ctx = OptContext::new(256);
        ctx.make_constant(idx, majit_ir::Value::Int(3));
        ctx.set_ptr_info(OpRef(100), PtrInfo::virtual_array(d, 8, false));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Emit(_)));
        assert_eq!(
            ctx.get_ptr_info(OpRef(100))
                .and_then(|info| info.getitem(3))
                .and_then(|e| e.as_opref()),
            Some(OpRef(200))
        );
    }

    #[test]
    fn test_setarrayitem_postprocess_updates_ptr_info() {
        // heap.py:88-90 do_setfield common case: only sets `_lazy_set = op`.
        // The ArrayPtrInfo._items[index] write is deferred to
        // force_lazy_set -> put_field_back_to_info; until then a
        // subsequent getarrayitem on the same (array, index) reads
        // the value back via getfield_from_cache's _lazy_set check.
        let d = descr(0);
        let idx = OpRef(50);
        let op = Op::with_descr(
            OpCode::SetarrayitemGc,
            &[OpRef(100), idx, OpRef(101)],
            d.clone(),
        );

        let mut ctx = OptContext::new(256);
        ctx.make_constant(idx, majit_ir::Value::Int(3));
        ctx.set_ptr_info(OpRef(100), PtrInfo::virtual_array(d.clone(), 8, false));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        // _lazy_set holds the pending op; PtrInfo is NOT yet written.
        let cai = pass
            .get_cached_array_submap(d.index())
            .and_then(|s| s.const_indexes.get(&3))
            .expect("ArrayCachedItem must exist");
        assert!(
            cai.lazy_set.is_some(),
            "do_setfield should have stored _lazy_set"
        );
        // After flush() the lazy set is forced and PtrInfo._items[3]
        // becomes the rhs value via put_field_back_to_info.
        pass.flush(&mut ctx);
        assert_eq!(
            ctx.get_ptr_info(OpRef(100))
                .and_then(|info| info.getitem(3))
                .and_then(|e| e.as_opref()),
            Some(OpRef(101))
        );
    }

    // ── Test 7: Guard forces lazy sets ──

    #[test]
    fn test_guard_forces_lazy_setfield() {
        // setfield_gc(p0, i1, descr=d0)     <- lazy, not emitted yet
        // guard_true(i_cond)                <- forces the lazy set
        // i2 = getfield_gc_i(p0, descr=d0) <- still cached (guards don't invalidate)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::new(OpCode::GuardTrue, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        // OpRef(101) is the Int setfield value + guard condition OpRef(200)
        // stays as Ref (guards accept Ref-typed "any non-null" predicates
        // in this harness).
        let result = run_heap_opt_typed(&mut ops, &[101]);

        // SETFIELD (forced by guard) + GUARD_TRUE + Jump.
        // GETFIELD is eliminated (cache survives guards).
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 8: GUARD_NOT_INVALIDATED deduplication ──

    #[test]
    fn test_guard_not_invalidated_dedup() {
        let mut ops = vec![
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only one GUARD_NOT_INVALIDATED + Jump.
        let gni_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNotInvalidated)
            .count();
        assert_eq!(gni_count, 1);
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }

    // ── Test 9: Different field descriptors are independent ──

    #[test]
    fn test_different_field_descriptors() {
        // setfield_gc(p0, i1, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d1)   <- different descriptor, NOT cached
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d0),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // SetfieldGc(d0) emitted as lazy set + GETFIELD(d1, different descriptor) + Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 10: SETFIELD_RAW does not affect GC caches ──

    #[test]
    fn test_setfield_raw_no_effect_on_gc_cache() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_raw(p1, i2, descr=d1)     <- RAW, no effect on GC caches
        // i3 = getfield_gc_i(p0, descr=d0)   <- still cached from first read
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0.clone()),
            Op::with_descr(OpCode::SetfieldRaw, &[OpRef(200), OpRef(201)], d1),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + SETFIELD_RAW + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::SetfieldRaw);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 11: Writing same value is redundant ──

    #[test]
    fn test_setfield_same_value_redundant() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_gc(p0, i1, descr=d0)   <- writing back the same value, redundant
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            // pos=0 will be the GETFIELD result; setfield writes it back
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + Jump only. SETFIELD removed (writing same value).
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 12: Pure/overflow ops don't invalidate ──

    #[test]
    fn test_pure_ops_dont_invalidate() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // i2 = int_add(i1, i1)              <- pure, no invalidation
        // i3 = getfield_gc_i(p0, descr=d0)  <- still cached
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + INT_ADD + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 13: Ref and float field variants ──

    #[test]
    fn test_getfield_ref_cached() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    fn test_getfield_float_cached() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcF);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 14: Array write-after-write ──

    #[test]
    fn test_setarrayitem_write_after_write() {
        let d = descr(0);
        let idx = OpRef(50);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx, OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx, OpRef(102)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(5));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let result_opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(result_opcodes, vec![OpCode::Jump]);
    }

    // ── Test 15: Overflow ops don't invalidate caches ──

    #[test]
    fn test_overflow_ops_dont_invalidate() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::IntAddOvf, &[OpRef(0), OpRef(0)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + INT_ADD_OVF + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::IntAddOvf);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 16: Multiple fields on same object ──

    #[test]
    fn test_multiple_fields_same_object() {
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d0.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(102)], d1.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[101, 102]);

        // Both GETFIELDs eliminated (cached). Both lazy SetfieldGc emitted before Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Green field optimization tests ──

    // ── Test 17: Immutable field cache survives call invalidation ──

    #[test]
    fn test_immutable_field_survives_call() {
        // i1 = getfield_gc_i(p0, descr=immutable_d0)
        // call_n(...)                              <- invalidates mutable caches
        // i2 = getfield_gc_i(p0, descr=immutable_d0) <- still cached (immutable)
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + Jump. Second GETFIELD eliminated (immutable survives call).
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 18: Mutable field cache is still invalidated by call ──

    #[test]
    fn test_mutable_field_invalidated_by_call() {
        // i1 = getfield_gc_i(p0, descr=mutable_d0)
        // call_n(...)
        // i2 = getfield_gc_i(p0, descr=mutable_d0) <- re-emitted (mutable, invalidated)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + GETFIELD (re-emitted) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 19: Mixed immutable and mutable fields: only mutable invalidated ──

    #[test]
    fn test_mixed_immutable_mutable_fields() {
        // i1 = getfield_gc_i(p0, descr=immut_d0)
        // i2 = getfield_gc_i(p0, descr=mut_d1)
        // call_n(...)
        // i3 = getfield_gc_i(p0, descr=immut_d0)  <- cached (immutable survives)
        // i4 = getfield_gc_i(p0, descr=mut_d1)    <- re-emitted (mutable invalidated)
        let d_immut = immutable_descr(0);
        let d_mut = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d_immut.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d_mut.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d_immut.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d_mut.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD(immut) + GETFIELD(mut) + CALL + GETFIELD(mut, re-emitted) + Jump.
        // GETFIELD(immut) after call is eliminated.
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI); // immutable, first read
        assert_eq!(result[1].opcode, OpCode::GetfieldGcI); // mutable, first read
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::GetfieldGcI); // mutable, re-emitted
        assert_eq!(result[4].opcode, OpCode::Jump);
    }

    // ── Test 20: Immutable field from non-constant object still gets read-cache ──

    #[test]
    fn test_immutable_field_read_cache_no_constant() {
        // Even without a constant source object, immutable fields benefit from
        // read-after-read caching that survives side effects.
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only first GETFIELD + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 21: Immutable Ref and Float field variants survive call ──

    #[test]
    fn test_immutable_field_ref_survives_call() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_immutable_field_float_survives_call() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcF);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_short_preamble_ref_field_preserves_getfield_opcode() {
        let descr =
            majit_ir::make_field_descr(55, 8, majit_ir::Type::Ref, majit_ir::ArrayFlag::Pointer);
        let mut pass = OptHeap::new();
        pass.cache_field(OpRef(100), &descr);

        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef(100),
            OpRef(101),
        ]);
        // Register input args so produce_arg can resolve them.
        sb.add_short_input_arg(OpRef(100), majit_ir::Type::Int);
        sb.add_short_input_arg(OpRef(101), majit_ir::Type::Int);
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        // Seed PtrInfo._fields[idx] with the cached value so the
        // produce_potential_short_preamble_ops read path can find it.
        use crate::optimizeopt::info::PtrInfo;
        ctx.set_ptr_info(OpRef(100), PtrInfo::instance(None, None));
        ctx.get_ptr_info_mut(OpRef(100))
            .unwrap()
            .setfield(descr.index(), OpRef(101));
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let produced = sb.produced_ops();

        // Filter to heap-produced ops (exclude SameAsI from add_short_input_arg).
        let heap_ops: Vec<_> = produced
            .iter()
            .filter(|(_, p)| p.preamble_op.opcode != OpCode::SameAsI)
            .collect();
        assert_eq!(heap_ops.len(), 1);
        assert_eq!(heap_ops[0].1.preamble_op.opcode, OpCode::GetfieldGcR);
    }

    // ── Test 22: Immutable field survives multiple calls ──

    #[test]
    fn test_immutable_field_survives_multiple_calls() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::new(OpCode::CallN, &[OpRef(201)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + CALL + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 23: Different objects with same immutable descr are independent ──

    #[test]
    fn test_immutable_field_different_objects() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(200)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(300)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()), // cached
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(200)], d.clone()), // cached
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Both initial GETFIELDs + CALL + Jump. Both post-call GETFIELDs eliminated.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Aliasing analysis tests ──

    // ── Test 24: Two NEW objects don't alias — write to one preserves cache of the other ──

    // ── Test 25: Unknown-origin object write invalidates other unknown caches ──

    #[test]
    fn test_unknown_object_write_invalidates() {
        // p0 = InputRef(100), p1 = InputRef(200)  — both unknown origin
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_gc(p1, i20, descr=d0)   <- p1 is unknown, might alias p0
        // i2 = getfield_gc_i(p0, descr=d0) <- must re-emit (might have been clobbered)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(200), OpRef(20)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + SETFIELD + GETFIELD (re-emitted) + Jump.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "second GETFIELD must be re-emitted for unknown-origin objects"
        );
    }

    // ── Test 26: Unescaped allocation's cache survives call ──

    #[test]
    fn test_unescaped_invalidated_by_call_write() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(some_func)               <- p0 NOT passed to call
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit
        //
        // heap.py:545-546: check_write_descr_field → force_lazy_set(can_cache=False)
        // PyPy does a full invalidate on write, regardless of escape status.
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::with_descr(OpCode::CallN, &[OpRef(200)], plain_call_descr(100)),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // heap.py: force_lazy_set(can_cache=False) invalidates ALL entries.
        // GETFIELD must be re-emitted after call that writes the field.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count,
            1,
            "GETFIELD must be re-emitted after call writes field (full invalidate), got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test 27: Escaped allocation's cache is invalidated by call ──

    #[test]
    fn test_escaped_invalidated_by_call() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(p0)                       <- p0 is passed to call, escapes
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit (call might have modified p0)
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::with_descr(OpCode::CallN, &[OpRef(0)], plain_call_descr(100)), // pass p0 to call
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // NEW + SETFIELD + CALL + GETFIELD (re-emitted) + Jump.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETFIELD must be re-emitted after escape via call"
        );
    }

    // ── Test 28: SetfieldGc marks stored value as escaped ──

    #[test]
    fn test_setfield_marks_escape() {
        // p0 = new()       <- unescaped
        // p1 = new()       <- unescaped
        // setfield_gc(p0, i10, descr=d0)
        // setfield_gc(p1, p0, descr=d1)   <- p0 is stored into p1's field, p0 escapes
        // call_n(p1)                       <- p1 escapes; p0 already escaped via setfield
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit (p0 escaped)
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::new(OpCode::New, &[]), // pos=1 -> p1
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d0.clone()), // p0.f0 = i10
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(1), OpRef(0)], d1.clone()), // p1.f1 = p0 (p0 escapes)
            Op::with_descr(OpCode::CallN, &[OpRef(1)], plain_call_descr(100)), // call(p1) (p1 escapes)
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d0.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // p0 escaped via setfield, so its cache is invalidated by the call.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETFIELD must be re-emitted after p0 escaped via setfield"
        );
    }

    // ── Test 29: Seen-allocation cache survives write from unknown-origin object ──

    // ── Test 30: Different field descriptors are not affected by aliasing ──

    #[test]
    fn test_aliasing_different_fields_independent() {
        // Even with unknown-origin objects, writes to field d0 don't
        // invalidate caches for field d1.
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(200), OpRef(20)], d0.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()), // different field
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD(d1) + SETFIELD(d0) + Jump. Second GETFIELD eliminated.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(get_count, 1, "write to d0 should not invalidate d1 cache");
    }

    // ── Test 31: Unescaped array cache survives call ──

    #[test]
    fn test_unescaped_array_invalidated_by_call_write() {
        // p0 = new_array(5)
        // setarrayitem_gc(p0, idx, i10, descr=d0)
        // call_n(some_func)                <- p0 not passed
        // i1 = getarrayitem_gc_i(p0, idx, descr=d0) <- must re-emit
        //
        // heap.py:557-558: check_write_descr_array → force_lazy_setarrayitem_submap(can_cache=False)
        // PyPy does a full invalidate on write, regardless of escape status.
        let d = descr(0);
        let idx = OpRef(50);
        let mut ops = vec![
            Op::new(OpCode::NewArray, &[OpRef(5)]), // pos=0 -> p0
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), idx, OpRef(10)],
                d.clone(),
            ),
            Op::with_descr(OpCode::CallN, &[OpRef(200)], plain_call_descr(100)),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(0), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(3));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        let get_count = opcodes
            .iter()
            .filter(|&&o| o == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETARRAYITEM must be re-emitted after call writes array (full invalidate), got: {opcodes:?}"
        );
    }

    // ── Test 32: Multiple calls — unescaped object stays cached ──

    #[test]
    fn test_unescaped_invalidated_by_multiple_calls() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(f1)
        // call_n(f2)
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit
        //
        // heap.py: full invalidate on write call, regardless of escape status.
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::with_descr(OpCode::CallN, &[OpRef(200)], plain_call_descr(100)),
            Op::with_descr(OpCode::CallN, &[OpRef(201)], plain_call_descr(101)),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count,
            1,
            "GETFIELD must be re-emitted after calls that write field (full invalidate), got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Nullity tracking tests ──

    // ── Test 37: GuardNonnull after allocation is removed ──

    #[test]
    fn test_guard_nonnull_after_allocation() {
        // p0 = new()
        // guard_nonnull(p0)   <- redundant, allocation is always non-null
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after allocation should be removed"
        );
    }

    // ── Test 38: GuardNonnull after GuardNonnull is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_nonnull() {
        // guard_nonnull(p0)
        // guard_nonnull(p0)   <- redundant
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(nonnull_count, 2, "second guard_nonnull should be removed");
    }

    // ── Test 39: GuardNonnull after GuardClass is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_class() {
        // guard_class(p0, cls)  <- implies non-null
        // guard_nonnull(p0)     <- redundant
        let mut ops = vec![
            Op::new(OpCode::GuardClass, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after guard_class should be removed"
        );
    }

    // ── Test 40: GuardNonnull on unknown input arg is kept ──

    #[test]
    fn test_guard_nonnull_unknown_not_removed() {
        // guard_nonnull(p0)  <- first time seeing p0, must keep
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(nonnull_count, 1, "guard_nonnull on unknown should be kept");
    }

    // ── Test 41: Nonnull from allocation survives call ──

    #[test]
    fn test_known_nonnull_survives_call_for_allocation() {
        // p0 = new()
        // call_n(some_func)    <- invalidates caches, but not allocation nonnull
        // guard_nonnull(p0)    <- still redundant (allocation is always non-null)
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after allocation should be removed even after call"
        );
    }

    // ── Test 42: Nonnull from guard does NOT survive call ──

    #[test]
    fn test_known_nonnull_from_guard_invalidated_by_call() {
        // guard_nonnull(p0)
        // call_n(some_func)   <- invalidates guard-derived nonnull
        // guard_nonnull(p0)   <- must re-emit
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 2,
            "guard_nonnull after call should be re-emitted for non-allocation values"
        );
    }

    // ── Test 43: GuardNonnull after GuardNonnullClass is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_nonnull_class() {
        // guard_nonnull_class(p0, cls) <- implies non-null
        // guard_nonnull(p0)            <- redundant
        let mut ops = vec![
            Op::new(OpCode::GuardNonnullClass, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after guard_nonnull_class should be removed"
        );
    }

    // ── Test 44: GuardNonnull after GuardValue is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_value() {
        // guard_value(p0, c) <- implies non-null
        // guard_nonnull(p0)  <- redundant
        let mut ops = vec![
            Op::new(OpCode::GuardValue, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after guard_value should be removed"
        );
    }

    // ── Test 45: GuardNonnull after NewWithVtable is removed ──

    #[test]
    fn test_guard_nonnull_after_new_with_vtable() {
        let mut ops = vec![
            Op::new(OpCode::NewWithVtable, &[]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after new_with_vtable should be removed"
        );
    }

    // ── Test 46: GuardNonnull after NewArray is removed ──

    #[test]
    fn test_guard_nonnull_after_new_array() {
        let mut ops = vec![
            Op::new(OpCode::NewArray, &[OpRef(5)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after new_array should be removed"
        );
    }

    // ── Call descriptor with OopSpecIndex for arraycopy tests ──

    /// Call descriptor with configurable EffectInfo for testing.
    #[derive(Debug)]
    struct TestCallDescr {
        idx: u32,
        effect: EffectInfo,
    }

    impl Descr for TestCallDescr {
        fn index(&self) -> u32 {
            self.idx
        }
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[majit_ir::Type] {
            &[]
        }
        fn result_type(&self) -> majit_ir::Type {
            majit_ir::Type::Void
        }
        fn result_size(&self) -> usize {
            0
        }
        fn get_extra_info(&self) -> &EffectInfo {
            &self.effect
        }
    }

    fn call_descr(idx: u32, effect: EffectInfo) -> DescrRef {
        Arc::new(TestCallDescr { idx, effect })
    }

    fn arraycopy_descr(idx: u32) -> DescrRef {
        Arc::new(TestCallDescr {
            idx,
            effect: EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                oopspecindex: OopSpecIndex::Arraycopy,
                ..Default::default()
            },
        })
    }

    #[test]
    fn test_call_may_force_uses_effectinfo_to_keep_unaffected_cached_fields() {
        let d0 = descr(0);
        let call_d = call_descr(
            70,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                write_descrs_fields: 1u64 << 1,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0.clone()),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "CallMayForce with unrelated write bit should preserve cached GETFIELD"
        );
    }

    #[test]
    fn test_call_may_force_uses_effectinfo_to_invalidate_written_cached_fields() {
        let d0 = descr(0);
        let call_d = call_descr(
            71,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                write_descrs_fields: 1u64 << 0,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0.clone()),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "CallMayForce with matching write bit must invalidate cached GETFIELD"
        );
    }

    #[test]
    fn test_call_may_force_resets_guard_not_invalidated_when_call_can_invalidate() {
        let call_d = call_descr(
            72,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                can_invalidate: true,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNotInvalidated)
            .count();
        assert_eq!(
            guard_count, 2,
            "CallMayForce that can invalidate must keep the later GuardNotInvalidated"
        );
    }

    #[test]
    fn test_call_may_force_keeps_unaffected_variable_index_array_cache() {
        let d0 = descr(0);
        let idx = OpRef(50);
        let call_d = call_descr(
            73,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                write_descrs_arrays: 1u64 << 1,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d0.clone()),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let mut ctx = OptContext::new(ops.len() + 64);
        assign_positions(&mut ops);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        ctx.snapshot_boxes = snapshots;
        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "CallMayForce with unrelated array write bit should preserve variable-index cache"
        );
    }

    #[test]
    fn test_call_may_force_invalidates_written_variable_index_array_cache() {
        let d0 = descr(0);
        let idx = OpRef(50);
        let call_d = call_descr(
            74,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                write_descrs_arrays: 1u64 << 0,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d0.clone()),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let mut ctx = OptContext::new(ops.len() + 64);
        assign_positions(&mut ops);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        ctx.snapshot_boxes = snapshots;
        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "CallMayForce with matching array write bit must invalidate variable-index cache"
        );
    }

    // ── ARRAYCOPY cache invalidation tests ──

    // ── Test 47: ARRAYCOPY only invalidates destination range ──

    #[test]
    fn test_arraycopy_only_invalidates_dest_range() {
        // p_src = OpRef(100), p_dst = OpRef(200)
        // setarrayitem_gc(p_dst, idx=0, val=i10)   <- cache dest[0]
        // setarrayitem_gc(p_dst, idx=5, val=i11)   <- cache dest[5]
        // call_n(arraycopy_func, p_src, p_dst, src_start=0, dst_start=2, length=3)
        //   -> copies to dest[2..5], so dest[0] survives, dest[5] survives
        // getarrayitem_gc_i(p_dst, idx=0)           <- still cached
        // getarrayitem_gc_i(p_dst, idx=5)           <- still cached
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx0 = OpRef(60);
        let idx5 = OpRef(61);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63);
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            // pos=0: setarrayitem_gc(dst, idx=0, val)
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(200), idx0, OpRef(10)],
                d.clone(),
            ),
            // pos=1: setarrayitem_gc(dst, idx=5, val)
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(200), idx5, OpRef(11)],
                d.clone(),
            ),
            // pos=2: call_n(func, src, dst, src_start, dst_start, length)
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            // pos=3: getarrayitem_gc_i(dst, idx=0)
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx0], d.clone()),
            // pos=4: getarrayitem_gc_i(dst, idx=5)
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx5], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx0, majit_ir::Value::Int(0));
        ctx.make_constant(idx5, majit_ir::Value::Int(5));
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(2));
        ctx.make_constant(length_ref, majit_ir::Value::Int(3));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // Both GETARRAYITEMs should be eliminated (dest[0] and dest[5] are outside [2..5)).
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        let get_count = opcodes
            .iter()
            .filter(|&&o| o == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 0,
            "dest[0] and dest[5] outside copy range [2..5) should be cached, got: {opcodes:?}"
        );
    }

    // ── Test 48: ARRAYCOPY with unknown length invalidates all dest entries ──

    #[test]
    fn test_arraycopy_unknown_range_invalidates_all() {
        // setarrayitem_gc(p_dst, idx=0, val=i10)
        // call_n(arraycopy_func, src, dst, src_start, dst_start, length)
        //   length is NOT a constant -> invalidates all dest array entries
        // getarrayitem_gc_i(p_dst, idx=0)   <- must re-emit
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx0 = OpRef(60);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63); // NOT constant
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(200), idx0, OpRef(10)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx0], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx0, majit_ir::Value::Int(0));
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(2));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));
        // length_ref is NOT a constant

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // GETARRAYITEM must be re-emitted (unknown length invalidates all dest entries).
        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "unknown arraycopy length must invalidate all dest array entries"
        );
    }

    #[test]
    fn test_arraycopy_invalidates_dest_variable_index_cache() {
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx = OpRef(60);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63);
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx], d.clone()),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(2));
        ctx.make_constant(length_ref, majit_ir::Value::Int(3));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "arraycopy must invalidate variable-index cache for destination array"
        );
    }

    #[test]
    fn test_arraycopy_keeps_other_array_variable_index_cache() {
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx = OpRef(60);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63);
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(400), idx], d.clone()),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(400), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(2));
        ctx.make_constant(length_ref, majit_ir::Value::Int(3));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "arraycopy should preserve variable-index cache for unrelated arrays"
        );
    }

    #[test]
    fn test_arraycopy_preserves_unrelated_cached_fields() {
        let field_d = descr(0);
        let array_d = descr(1);
        let ac_d = arraycopy_descr(50);
        let idx0 = OpRef(60);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63);
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(500)], field_d.clone()),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(200), idx0, OpRef(10)],
                array_d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(500)], field_d),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx0, majit_ir::Value::Int(0));
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(0));
        ctx.make_constant(length_ref, majit_ir::Value::Int(1));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "arraycopy should not invalidate unrelated field caches"
        );
    }

    // ── Quasi-immutable field tests ──

    // ── Test 50: GC_LOAD forces lazy setfields ──

    #[test]
    fn test_gc_load_forces_lazy_setfields() {
        // setfield_gc(p0, i1, descr=d0)   <- lazy, not emitted yet
        // i2 = gc_load_i(p1, offset, size) <- generic load, forces all lazy writes
        // The SETFIELD must be emitted before the GC_LOAD.
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::new(OpCode::GcLoadI, &[OpRef(200), OpRef(8), OpRef(4)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // force_all_lazy_setfields at GcLoadI emits lazy SetfieldGc. SetfieldGc + GcLoadI + Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::GcLoadI);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 51: GC_LOAD marks base as nonnull ──

    #[test]
    fn test_gc_load_marks_nonnull() {
        // i1 = gc_load_i(p0, offset, size)  <- dereferences p0, so p0 is nonnull
        // guard_nonnull(p0)                  <- redundant
        let mut ops = vec![
            Op::new(OpCode::GcLoadI, &[OpRef(100), OpRef(8), OpRef(4)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after gc_load should be removed"
        );
    }

    // ── Test 52: QUASIIMMUT_FIELD on field 0 doesn't affect field 1 ──

    #[test]
    fn test_quasiimmut_field_different_field_not_cached() {
        // quasiimmut_field(p0, descr=d0)      <- marks field 0 as quasi-immut
        // i1 = getfield_gc_i(p0, descr=d1)   <- different field, NOT quasi-immut
        // call_n(some_func)
        // i2 = getfield_gc_i(p0, descr=d1)   <- must re-emit (d1 is mutable)
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::QuasiimmutField, &[OpRef(100)], d0),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GUARD_NOT_INVALIDATED + GETFIELD(d1) + CALL + GETFIELD(d1, re-emitted) + Jump.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "quasi-immut on field 0 should not affect field 1"
        );
    }

    // ── Test 53: Bytearray-as-array heap cache verification ──
    //
    // RPython treats bytearray as regular arrays with item_size=1.
    // Verify the heap cache works correctly with byte-sized array items.

    /// Array descriptor with item_size=1 (byte array).
    #[derive(Debug)]
    struct ByteArrayDescr(u32);

    impl Descr for ByteArrayDescr {
        fn index(&self) -> u32 {
            self.0
        }
        fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
            Some(self)
        }
    }

    impl majit_ir::ArrayDescr for ByteArrayDescr {
        fn base_size(&self) -> usize {
            8 // typical GC header
        }
        fn item_size(&self) -> usize {
            1 // byte-sized items
        }
        fn type_id(&self) -> u32 {
            0
        }
        fn item_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
    }

    fn byte_array_descr(idx: u32) -> DescrRef {
        Arc::new(ByteArrayDescr(idx))
    }

    #[test]
    fn test_bytearray_setitem_then_getitem_cached() {
        // setarrayitem_gc(p0, idx, val, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx, descr=byte_array)  <- eliminated
        let d = byte_array_descr(50);
        let idx = OpRef(60);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx, OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(5)); // byte index 5

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::Jump],
            "byte-array getitem should be cached after setitem; lazy set dropped at Jump"
        );
    }

    #[test]
    fn test_bytearray_different_indices_not_cached() {
        // setarrayitem_gc(p0, idx=5, val, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx=6, descr=byte_array)  <- NOT cached (different index)
        let d = byte_array_descr(50);
        let idx5 = OpRef(60);
        let idx6 = OpRef(61);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx5, OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx6], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx5, majit_ir::Value::Int(5));
        ctx.make_constant(idx6, majit_ir::Value::Int(6));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // GETARRAYITEM must be emitted (not cached — different index).
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert!(
            opcodes.contains(&OpCode::GetarrayitemGcI),
            "different byte-array index should not use cache: {:?}",
            opcodes
        );
    }

    #[test]
    fn test_bytearray_read_after_read_cached() {
        // i1 = getarrayitem_gc_i(p0, idx=3, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx=3, descr=byte_array)  <- eliminated (same read)
        let d = byte_array_descr(50);
        let idx = OpRef(60);
        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone()),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(3));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // Only one GETARRAYITEM + Jump: the second read is eliminated.
        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(get_count, 1, "byte-array read-after-read should be cached");
    }

    #[test]
    fn test_arraylen_caching() {
        // Two ARRAYLEN_GC on the same array → second eliminated by OptPure.
        // resoperation.py:1044 marks ARRAYLEN_GC as ALWAYS_PURE; OptPure's
        // _pure_operations CSE table handles dedup.
        let d = descr(42);
        let mut ops = vec![
            {
                let mut op = Op::new(OpCode::ArraylenGc, &[OpRef(100)]);
                op.descr = Some(d.clone());
                op
            },
            {
                let mut op = Op::new(OpCode::ArraylenGc, &[OpRef(100)]);
                op.descr = Some(d);
                op
            },
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::pure::OptPure::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        let len_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ArraylenGc)
            .count();
        assert_eq!(len_count, 1, "duplicate ARRAYLEN_GC should be cached");
    }

    /// heap.py:278-298 ArrayCachedItem._cannot_alias_via_content — two
    /// arrays with the same descr + length but differing constant items
    /// must be reported as unable to alias.
    #[test]
    fn test_cannot_alias_via_content_different_constants() {
        use crate::optimizeopt::OptContext;
        use crate::optimizeopt::info::{ArrayPtrInfo, FieldEntry};
        use crate::optimizeopt::intutils::IntBound;

        let mut ctx = OptContext::with_num_inputs_and_start_pos(64, 4, 0, 4);

        let arr_descr = descr(50);
        let op1 = OpRef(0);
        let op2 = OpRef(1);

        let const_10 = ctx.emit_constant_int(10);
        let const_20 = ctx.emit_constant_int(20);
        let const_30 = ctx.emit_constant_int(30);

        ctx.set_ptr_info(
            op1,
            PtrInfo::Array(ArrayPtrInfo {
                descr: arr_descr.clone(),
                lenbound: IntBound::from_constant(2),
                items: vec![FieldEntry::Value(const_10), FieldEntry::Value(const_20)],
                last_guard_pos: -1,
            }),
        );
        ctx.set_ptr_info(
            op2,
            PtrInfo::Array(ArrayPtrInfo {
                descr: arr_descr,
                lenbound: IntBound::from_constant(2),
                items: vec![FieldEntry::Value(const_10), FieldEntry::Value(const_30)],
                last_guard_pos: -1,
            }),
        );

        assert!(
            super::ArrayCachedItem::_cannot_alias_via_content(op1, op2, &mut ctx),
            "arrays with different constant at index 1 cannot alias"
        );
    }
}
