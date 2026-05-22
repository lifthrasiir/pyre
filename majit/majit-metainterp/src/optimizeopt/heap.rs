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

use majit_ir::{DescrRef, OopSpecIndex, Op, OpCode, OpRef, Value, descr::descr_identity};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// util.py:100-128 args_dict() / args_eq(): same_box semantics — identity
/// for non-Const boxes, value-equality for Const subclasses (history.py:204).
/// Two distinct Const slots holding the same value must hash and compare
/// equal so consecutive dict lookups with key `5` (encoded via different
/// const slots) hit the cache.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum DictArgKey {
    Const(Value),
    Op(OpRef),
}

impl DictArgKey {
    fn from_arg(arg: OpRef, ctx: &OptContext) -> Self {
        let resolved = ctx.get_box_replacement(arg);
        match ctx.get_constant(resolved) {
            Some(value) => DictArgKey::Const(value),
            None => DictArgKey::Op(resolved),
        }
    }
}

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
///   on-demand from `box._forwarded` / `ctx.const_infos[gcref]`.
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
        self.cached_structs.push(struct_opref);
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
    /// `cached_structs`, resolves each opref through `box._forwarded`
    /// OR `ctx.const_infos` (the latter mirrors `info.py:715-726
    /// ConstPtrInfo._get_info` which routes constant bases through
    /// `optheap.const_infos[gcref]`), and calls
    /// `info.clear_field(descr_idx)`.
    /// heap.py:189-194 `CachedField.invalidate(descr)` line-by-line:
    ///
    /// ```python
    /// def invalidate(self, descr):
    ///     if descr.is_always_pure():
    ///         return
    ///     for opinfo in self.cached_infos:
    ///         assert isinstance(opinfo, info.AbstractStructPtrInfo)
    ///         opinfo._fields[descr.get_index()] = None
    ///     self.cached_infos = []
    /// ```
    ///
    /// The `descr.is_always_pure()` short-circuit is performed inside
    /// the method, not the caller — symmetric with upstream. The slot
    /// index is derived from `descr` via `field_slot_index`
    /// (FieldDescr.index_in_parent when a parent SizeDescr is bound,
    /// else Descr::index). Callers no longer need to gate manually.
    fn invalidate(&mut self, descr: &DescrRef, ctx: &mut OptContext) {
        if descr.is_always_pure() {
            return;
        }
        let descr_idx = OptHeap::field_slot_index(descr);
        for &obj in &self.cached_structs {
            let resolved = ctx.get_box_replacement(obj);
            if let Some(b) = ctx.ensure_box(resolved) {
                ctx.with_ptr_info_mut(&b, |info| info.clear_field(descr_idx));
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
        let struct_box = ctx.get_box_replacement_box(struct_opref);
        if let Some(info) = struct_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
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
        let b1 = ctx.get_box_replacement_box(opref1);
        let b2 = ctx.get_box_replacement_box(opref2);
        let class1 = b1
            .as_ref()
            .and_then(|b| ctx.getptrinfo(b))
            .and_then(|i| i.get_known_class(ctx.cpu.as_ref()));
        let class2 = b2
            .as_ref()
            .and_then(|b| ctx.getptrinfo(b))
            .and_then(|i| i.get_known_class(ctx.cpu.as_ref()));
        matches!((class1, class2), (Some(c1), Some(c2)) if c1 != c2)
    }

    /// heap.py:206-226 CachedField._cannot_alias_via_content
    fn _cannot_alias_via_content(opref1: OpRef, opref2: OpRef, ctx: &mut OptContext) -> bool {
        // heap.py:207-210: both must be AbstractStructPtrInfo
        let b1 = ctx.get_box_replacement_box(opref1);
        let b2 = ctx.get_box_replacement_box(opref2);
        let (Some(info1), Some(info2)) = (
            b1.as_ref().and_then(|b| ctx.peek_ptr_info(b)),
            b2.as_ref().and_then(|b| ctx.peek_ptr_info(b)),
        ) else {
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

    /// heap.py:172-175 CachedField.put_field_back_to_info
    fn put_field_back_to_info(&mut self, op: &Op, ctx: &mut OptContext) {
        // info.py:203-211 opinfo.setfield(descr, struct, op, optheap, cf=self)
        // PyPy: `setfield(..., cf=cf)` calls `cf.register_info(struct, self)`
        // (info.py:209-210). The Rust port performs both halves here.
        let descr_idx = op
            .getdescr()
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
            let structbox_box = ctx.get_box_replacement_box(structbox);
            let cached_val = match structbox_box
                .as_ref()
                .and_then(|b| ctx.peek_ptr_info(b))
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
            op.pos.set(cached_val);
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
        self.cached_structs.push(array_opref);
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
        let b1 = ctx.get_box_replacement_box(opref1);
        let b2 = ctx.get_box_replacement_box(opref2);
        let len1 = match b1.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
            Some(PtrInfo::Array(v)) => v.lenbound.clone(),
            _ => return false,
        };
        let len2 = match b2.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
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
        let b1 = ctx.get_box_replacement_box(opref1);
        let b2 = ctx.get_box_replacement_box(opref2);
        let items1: Vec<FieldEntry> = match b1.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
            Some(PtrInfo::Array(a)) => a.items.clone(),
            _ => return false,
        };
        let items2: Vec<FieldEntry> = match b2.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
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
    /// `cached_structs` and routes through `box._forwarded` /
    /// `ctx.const_infos`. The `self.parent.clear_varindex()` half is
    /// performed by the caller (`ArrayCacheSubMap::invalidate_index`)
    /// because Rust forbids the back-pointer.
    fn invalidate(&mut self, ctx: &mut OptContext) {
        let index = self.index as usize;
        for &obj in &self.cached_structs {
            let resolved = ctx.get_box_replacement(obj);
            if let Some(b) = ctx.ensure_box(resolved) {
                ctx.with_ptr_info_mut(&b, |info| info.clear_item(index));
            }
            // info.py:728 ConstPtrInfo._get_array_info — only clear
            // an existing ArrayPtrInfo slot; do NOT create on miss.
            if let Some(info) = ctx.get_const_info_mut_if_exists(resolved) {
                info.clear_item(index);
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
        let array_box = ctx.get_box_replacement_box(array_opref);
        if let Some(info) = array_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
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
            let arraybox_box = ctx.get_box_replacement_box(arraybox);
            let cached_val = match arraybox_box
                .as_ref()
                .and_then(|b| ctx.peek_ptr_info(b))
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
            op.pos.set(cached_val);
            sb.add_heap_op(op);
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
    const_indexes: crate::optimizeopt::vec_assoc::VecAssoc<i64, ArrayCachedItem>,
    /// heap.py:305-306: cached_varindex_triples = None
    /// List of (arrayinfo, indexbox, resbox). RPython uses Python object
    /// identity for arrayinfo; majit uses the canonical array OpRef.
    cached_varindex_triples: Option<Vec<(OpRef, OpRef, OpRef)>>,
}

impl ArrayCacheSubMap {
    fn new() -> Self {
        ArrayCacheSubMap {
            const_indexes: crate::optimizeopt::vec_assoc::VecAssoc::new(),
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
    /// heap.py:332: cached_arrayitems OrderedDict keyed by array descr.
    cached_arrayitems: Vec<(u32, DescrRef, ArrayCacheSubMap)>,
    /// Whether we've already emitted a GUARD_NOT_INVALIDATED.
    seen_guard_not_invalidated: bool,
    /// Postponed operation: held back until the next GUARD_NO_EXCEPTION.
    /// RPython heap.py: `postponed_op` — delays emission of operations
    /// that may raise (CALL_MAY_FORCE, comparison ops) until we see
    /// a GUARD_NO_EXCEPTION, ensuring correct exception semantics.
    postponed_op: Option<Op>,
    // ── Aliasing analysis state — RPython: PtrInfo flags ──
    seen_allocation: Vec<bool>,
    unescaped: Vec<bool>,
    known_nonnull: Vec<bool>,
    /// heapcache.py:209/298-307/453-455 `box._heapc_deps` — per-Box
    /// dependency list. RPython attaches `_heapc_deps: list | None`
    /// as an attribute on the `RefFrontendOp` Box object itself;
    /// pyre keeps a side-table keyed by OpRef for the same effect.
    /// When an unescaped value is stored into an unescaped container,
    /// the value is recorded as a dependency of the container instead
    /// of being immediately escaped. When the container escapes later,
    /// all its dependencies are transitively escaped.
    heapc_deps: crate::optimizeopt::vec_assoc::VecAssoc<OpRef, Vec<OpRef>>,

    /// heap.py:27 Optimization.last_emitted_operation is REMOVED.
    /// Set to true when `_optimize_CALL_DICT_LOOKUP` folds a lookup;
    /// read by `optimize_GUARD_NO_EXCEPTION` to suppress the trailing guard.
    last_emitted_removed: bool,
    /// heap.py:337: cached_dict_reads — descr_identity(extradescrs[0]) → { [dict,key] → result_opref }.
    /// Consecutive dict lookups on the same dict+key are deduplicated.
    /// Inner key uses `DictArgKey` so Const args compare by value
    /// (util.py:100 args_dict / args_eq via history.py:204 same_box).
    cached_dict_reads: crate::optimizeopt::vec_assoc::VecAssoc<
        usize,
        crate::optimizeopt::vec_assoc::VecAssoc<[DictArgKey; 2], OpRef>,
    >,
    /// heap.py:560: corresponding_array_descrs — maps extradescrs[1] (entries
    /// array descr) → extradescrs[0] dict identity.
    ///
    /// PyPy stores the *array descr object* directly and resolves
    /// `arraydescr.ei_index` inside `effectinfo.check_write_descr_array(arraydescr)`
    /// at invalidation time (`effectinfo.py:220-222`).  Pyre keeps the
    /// `DescrRef` alive on the value side so the same lazy resolution can
    /// happen via `descr.get_ei_index()` at
    /// `force_from_effectinfo`; the `u32` key dedups by raw `descr.index()`
    /// to mirror PyPy's `dict[arraydescr]` "later registration wins" idiom
    /// (the registration is gated by a `cached_dict_reads` first-encounter
    /// check, so duplicates are rare anyway — see `_optimize_call_dict_lookup`).
    corresponding_array_descrs: crate::optimizeopt::vec_assoc::VecAssoc<u32, (DescrRef, usize)>,
    /// Fields known to be quasi-immutable: (obj, field_idx) -> cached value OpRef.
    /// Populated by QUASIIMMUT_FIELD, consumed by subsequent GETFIELD_GC_*.
    /// Survives calls (guarded by GUARD_NOT_INVALIDATED).
    quasi_immut_cache: crate::optimizeopt::vec_assoc::VecAssoc<FieldKey, OpRef>,
}

impl OptHeap {
    pub fn new() -> Self {
        OptHeap {
            cached_fields: Vec::new(),
            cached_arrayitems: Vec::new(),
            seen_guard_not_invalidated: false,
            postponed_op: None,
            seen_allocation: Vec::new(),
            unescaped: Vec::new(),
            known_nonnull: Vec::new(),
            heapc_deps: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            last_emitted_removed: false,
            cached_dict_reads: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            corresponding_array_descrs: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            quasi_immut_cache: crate::optimizeopt::vec_assoc::VecAssoc::new(),
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

    /// `effectinfo.py:529-532` `bitstrr = [descr.ei_index for descr in
    /// getattr(ei, '_readonly_descrs_' + key)]` — the bit position in
    /// each EI's `bitstring_*` is the descr's `ei_index`, set by
    /// `compute_bitstrings` (`effectinfo.py:526 descr.ei_index = …`).
    /// `effectinfo.py:496 descr.ei_index = sys.maxint` is the sentinel
    /// for descrs absent from any EI's raw set;
    /// `bitstring.py:18 if byte_number >= len(bitstring)` then makes
    /// `bitcheck` return false out of range. Pyre matches by returning
    /// `u32::MAX`, whose `byte_number = u32::MAX >> 3` is far past any
    /// realistic bitstring length so `bitcheck` shorts to false the
    /// same way.
    fn field_effect_index(descr: &DescrRef) -> u32 {
        descr.get_ei_index()
    }

    /// Same as [`field_effect_index`] for the array namespace
    /// (`effectinfo.py:307-311 add_array` writes `ei_index` onto the
    /// array descr; `compute_bitstrings` re-stamps in-place per
    /// `effectinfo.py:526 descr.ei_index = …`).
    fn array_effect_index(descr: &DescrRef) -> u32 {
        descr.get_ei_index()
    }

    /// heapcache.py:295-309 `_escape_box`: escape a box and transitively
    /// escape all its dependencies stored in `box._heapc_deps`.
    fn escape_box(&mut self, opref: OpRef) {
        vb_unset(&mut self.unescaped, opref.raw());
        if let Some(deps) = self.heapc_deps.remove(&opref) {
            for dep in deps {
                self.escape_box(dep);
            }
        }
    }

    /// heapcache.py:224-230 `_escape_from_write`: when storing a value
    /// into a container, append to `_get_deps(box)` if both are
    /// unescaped; otherwise escape the value immediately.
    fn escape_from_write(&mut self, container: OpRef, value: OpRef) {
        if vb_get(&self.unescaped, container.raw()) && vb_get(&self.unescaped, value.raw()) {
            self.heapc_deps
                .entry_or_insert_with(container, Vec::new)
                .push(value);
        } else if !value.is_none() {
            self.escape_box(value);
        }
    }

    /// Build the field cache key from a GETFIELD or SETFIELD op.
    ///
    /// For GETFIELD_GC_I/R/F: args = [obj], descr = field descriptor.
    /// For SETFIELD_GC: args = [obj, value], descr = field descriptor.
    fn field_key(op: &Op) -> Option<FieldKey> {
        let descr = op.getdescr()?;
        let obj = op.arg(0);
        Some((obj, Self::field_cache_identity(&descr)))
    }

    /// heap.py:409-415 arrayitem_cache: constant-index array cache key.
    /// Canonicalizes array and index through get_box_replacement.
    fn arrayitem_key(op: &Op, ctx: &mut OptContext) -> Option<ArrayItemKey> {
        let descr = op.getdescr()?;
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
    ///
    /// `descr_idx` is the registry-slot identity (`descr.index()`),
    /// matching every other cache-identity site in this file:
    /// `arrayitem_key` (`:878-884`), the varindex lookup (`:2485`),
    /// the lazy-set force probe (`:2586`), and the immutable-array
    /// flag (`:2447` insert / `:1282` query) all key on
    /// `descr.index()`.  Using `array_effect_index` here would publish
    /// the ei_index slot once the codewriter set it
    /// (`effectinfo.py:465 compute_bitstrings`), which puts insert and
    /// lookup in different identity domains — PyPy's
    /// `cached_arrayitems[descr]` (`heap.py:399`) avoids this by
    /// keying on the descriptor object itself, and reserves
    /// `descr.get_ei_index()` for the EffectInfo bitstring check
    /// (`effectinfo.py:217 check_write_descr_array`).
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
        self.arrayitem_submap(&descr)
            .const_indexes
            .entry_or_insert_with(index, || ArrayCachedItem::new(index))
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

    fn invalidate_arrayitem_cache(&mut self, descr_idx: u32, index: i64, ctx: &mut OptContext) {
        if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
            if let Some(cai) = submap.const_indexes.get_mut(&index) {
                cai.invalidate(ctx);
            }
        }
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
                    .entry_or_insert_with(index, || ArrayCachedItem::new(index))
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
        let orig_is_virtual = ctx
            .get_box_replacement_box(orig_val)
            .as_ref()
            .map_or(false, |b| ctx.is_virtual(b));
        if orig_is_virtual {
            return false;
        }

        // Non-virtual path: resolve forwarding and route after heap
        // optimizer.py:651-652 setarg loop parity.
        for i in 0..op.num_args() {
            op.setarg(i, ctx.get_box_replacement(op.arg(i)));
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
    fn force_all_lazy_setfields(&mut self, _heap_pass_idx: usize, ctx: &mut OptContext) {
        // heap.py:574-587 force_all_lazy_sets (field half) line-by-line:
        //
        //     items = self.cached_fields.items()
        //     if not we_are_translated():
        //         items.sort(...)
        //     for descr, cf in items:
        //         cf.force_lazy_set(self, descr)
        //
        // Pyre collects the sorted descrs first to avoid mutably
        // borrowing `cached_fields` and `self` simultaneously inside
        // the loop body.
        let mut descrs: Vec<DescrRef> = self
            .cached_fields
            .iter()
            .filter_map(|(_, descr, cf)| cf.lazy_set.as_ref().map(|_| descr.clone()))
            .collect();
        // we_are_translated() == False sort path for test stability.
        descrs.sort_by_key(|d| majit_ir::descr::descr_identity(d));
        for descr in descrs {
            self.force_lazy_set_field(&descr, true, ctx);
        }
    }

    fn force_all_lazy_setarrayitems(&mut self, _heap_pass_idx: usize, ctx: &mut OptContext) {
        // heap.py:600-606 force_all_lazy_sets (array half) line-by-line:
        //
        //     for descr, submap in self.cached_arrayitems.iteritems():
        //         self.force_lazy_setarrayitem_submap(submap, can_cache=True)
        //
        // force_lazy_setarrayitem_submap (heap.py:589-593) iterates
        // submap.const_indexes and calls cf.force_lazy_set per cai.
        let entries: Vec<(u32, DescrRef, Vec<i64>)> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, descr, submap)| {
                let mut indexes: Vec<i64> = submap.const_indexes.keys().copied().collect();
                indexes.sort();
                (*idx, descr.clone(), indexes)
            })
            .collect();
        for (descr_idx, descr, indexes) in entries {
            for index in indexes {
                // heap.py:591 cf.force_lazy_set(self, None, can_cache=True)
                self.force_lazy_set_array(descr_idx, index, true, ctx);
            }
        }
    }

    /// Force all pending lazy stores (both fields and array items).
    /// `heap_pass_idx`: this pass's own pipeline index for emit routing.
    fn force_all_lazy_sets(&mut self, heap_pass_idx: usize, ctx: &mut OptContext) {
        self.force_all_lazy_setfields(heap_pass_idx, ctx);
        self.force_all_lazy_setarrayitems(heap_pass_idx, ctx);
    }

    /// heap.py:580-586 force_lazy_setarrayitem(arraydescr, indexb=None, can_cache=True)
    ///
    /// Selectively force lazy array stores for a specific array descriptor.
    /// Only entries whose const index is within `indexb` (if provided) are forced.
    /// RPython uses this for variable-index GETARRAYITEM/SETARRAYITEM to avoid
    /// forcing ALL lazy stores — only those that could alias the variable index.
    fn force_lazy_setarrayitem(
        &mut self,
        descr: &DescrRef,
        indexb: Option<&crate::optimizeopt::intutils::IntBound>,
        can_cache: bool,
        ctx: &mut OptContext,
    ) {
        let pos = match self.cached_array_pos_for_descr(descr) {
            Some(pos) => pos,
            None => return,
        };
        let descr_idx = self.cached_arrayitems[pos].0;
        let indexes: Vec<i64> = self.cached_arrayitems[pos]
            .2
            .const_indexes
            .keys()
            .copied()
            .filter(|&idx| indexb.map_or(true, |b| b.contains(idx)))
            .collect();
        for index in indexes {
            self.force_lazy_set_array(descr_idx, index, can_cache, ctx);
        }
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
            let is_virtual = ctx
                .get_box_replacement_box(value_ref)
                .as_ref()
                .map_or(false, |b| ctx.is_virtual(b));
            if is_virtual {
                // heap.py:618-619: virtual value → pendingfields
                pendingfields.push(op);
                continue;
            }
            // heap.py:621: cf.force_lazy_set(self, descr) →
            // invalidate first, then emit_extra(op, emit=False),
            // then put_field_back_to_info restores the cache.
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..op.num_args() {
                op.setarg(i, ctx.get_box_replacement(op.arg(i)));
            }
            let final_value = op.arg(1);
            // heap.py:129,189-191: invalidate(descr) — purity self-gate
            // inside CachedField::invalidate (heap.py:189-194 parity).
            if let Some(cf) = self.get_cached_field_mut(&descr) {
                cf.invalidate(&descr, ctx);
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
            let is_virtual = ctx
                .get_box_replacement_box(value_ref)
                .as_ref()
                .map_or(false, |b| ctx.is_virtual(b));
            if is_virtual {
                // heap.py:634: pendingfields.append(op)
                pendingfields.push(op);
                continue;
            }

            // optimizer.py:651-652 setarg loop parity.
            for i in 0..op.num_args() {
                op.setarg(i, ctx.get_box_replacement(op.arg(i)));
            }
            let final_value = op.arg(2);
            let array_ref = op.arg(0);
            let descr = op.getdescr();
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
        for (_field_idx, descr, cf) in field_entries {
            // heap.py:384: `cf.invalidate(descr)` — purity self-gate
            // inside the method (heap.py:189-194). `_field_idx` unused
            // post-purity-lift; index now recomputed from `descr`.
            cf.invalidate(&descr, ctx);
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
        let descr_entries: Vec<(u32, bool)> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, descr, _)| (*idx, descr.is_always_pure()))
            .collect();
        for (descr_idx, is_pure) in descr_entries {
            if is_pure {
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
        self.cached_dict_reads.clear();
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
                    let idx = index as usize;
                    if let Some(b) = ctx.ensure_box(dest_resolved) {
                        ctx.with_ptr_info_mut(&b, |info| info.clear_item(idx));
                    }
                    if let Some(info) = ctx.get_const_info_mut_if_exists(dest_resolved) {
                        info.clear_item(idx);
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
        op.with_call_descr(|cd| cd.get_extra_info().oopspecindex)
            .unwrap_or(OopSpecIndex::None)
    }

    /// heap.py:480-528 _optimize_CALL_DICT_LOOKUP.
    ///
    /// Cache consecutive dict lookup calls on the same dict+key.
    /// FLAG_LOOKUP (0): always cache and reuse.
    /// FLAG_STORE  (1): don't cache new; reuse only if cached value ≥ 0.
    /// FLAG_DELETE (2+): never cache, never reuse.
    fn _optimize_call_dict_lookup(&mut self, op: &Op, ctx: &mut OptContext) -> bool {
        const FLAG_LOOKUP: i64 = 0;
        const FLAG_STORE: i64 = 1;

        // heap.py:497-500: flag_value = self.getintbound(op.getarg(4))
        //   if not flag_value.is_constant(): return False
        //   flag = flag_value.get_constant_int()
        if op.num_args() < 5 {
            return false;
        }
        let flag = match ctx.get_constant_int_or_bound(op.arg(4)) {
            Some(v) => v,
            None => return false,
        };
        if flag != FLAG_LOOKUP && flag != FLAG_STORE {
            return false;
        }

        // heap.py:504: descrs = op.getdescr().get_extra_info().extradescrs
        let extradescrs = op
            .with_call_descr(|cd| cd.get_extra_info().extradescrs.clone())
            .flatten();
        let descrs = match extradescrs {
            Some(ref d) if d.len() >= 2 => d,
            _ => return false,
        };
        let descr1_id = descr_identity(&descrs[0]);
        let descr2 = descrs[1].clone();

        // heap.py:506-511 try/except KeyError:
        //   try:
        //       d = self.cached_dict_reads[descr1]
        //   except KeyError:
        //       d = self.cached_dict_reads[descr1] = args_dict()
        //       self.corresponding_array_descrs[descrs[1]] = descr1
        // The corresponding_array_descrs registration fires ONLY on the
        // first encounter of `descr1` (the KeyError arm); repeated lookups
        // skip it. Mirror this — `or_default()` would unconditionally
        // re-register and diverge from RPython.
        if !self.cached_dict_reads.contains_key(&descr1_id) {
            self.cached_dict_reads
                .insert(descr1_id, crate::optimizeopt::vec_assoc::VecAssoc::new());
            self.corresponding_array_descrs
                .insert(descr2.index(), (descr2, descr1_id));
        }
        let d = self
            .cached_dict_reads
            .get_mut(&descr1_id)
            .expect("just inserted above when absent");

        // heap.py:513-514: key = [get_box_replacement(arg1), get_box_replacement(arg2)]
        // util.py:100/127 args_dict() compares args via same_box: identity for
        // non-Const, value-equality for Const (history.py:204). Encode each
        // arg through DictArgKey so two ConstInt slots with the same value
        // hash and compare equal.
        let key = [
            DictArgKey::from_arg(op.arg(1), ctx),
            DictArgKey::from_arg(op.arg(2), ctx),
        ];

        if let Some(&res_v) = d.get(&key) {
            // heap.py:523-525: flag != FLAG_LOOKUP → self.getintbound(res_v).known_ge_const(0)
            if flag != FLAG_LOOKUP {
                let resolved = ctx.get_box_replacement(res_v);
                let bound = ctx.get_int_bound(resolved);
                let known_ge_zero = bound.map_or(false, |b| b.known_ge_const(0));
                if !known_ge_zero {
                    return false;
                }
            }
            // heap.py:525-527: make_equal_to + last_emitted_operation = REMOVED
            let b_old = ctx
                .ensure_box(op.pos.get())
                .expect("body-namespace OpRef must have a BoxRef slot");
            let b_res = ctx
                .ensure_box(res_v)
                .expect("body-namespace OpRef must have a BoxRef slot");
            ctx.make_equal_to(&b_old, &b_res);
            self.last_emitted_removed = true;
            return true;
        }

        // heap.py:517-518: no hit — cache if FLAG_LOOKUP
        if flag == FLAG_LOOKUP {
            d.insert(key, op.pos.get());
        }
        false
    }

    /// heap.py:417-464 self.emit(op) → emitting_operation for is_call(op).
    /// Generic residual-call emit path: mark args escaped, then route
    /// through force_from_effectinfo / clean_caches / invalidate_for_escaped
    /// per the same descr-aware policy as the catch-all `_` arm.
    fn emit_residual_call(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let escaped_owners = self.call_argument_owner_closure(op, ctx);
        self.mark_escaped_varargs(op, ctx);
        // STRUCTURAL ADAPTATION: RPython heap.py relies purely on EffectInfo
        // here. Removing this pyre-specific direct-argument flush currently
        // breaks synthetic correctness (`comprehensions.py`, and then wider
        // dynasm/cranelift wrong-output failures). Keep it until the caller
        // materialization path is ported enough that direct call arguments
        // cannot observe stale lazy stores.
        self.force_call_argument_lazy_sets(&escaped_owners, ctx.current_pass_idx, ctx);
        // heapcache.py:337-369 clear_caches_varargs.
        // Plain residual calls preserve cache entries for unescaped
        // allocations. Calls with explicit EffectInfo keep the more
        // precise heap.py force_from_effectinfo path.
        if !op.has_descr() {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            self.clean_caches(ctx);
        } else if Self::call_has_random_effects(op) {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            self.clean_caches(ctx);
        } else {
            // heap.py:537-571 force_from_effectinfo — selective cache
            // invalidation using EffectInfo bitstrings.
            self.force_from_effectinfo(op, ctx);
        }
        if Self::call_can_invalidate(op) {
            self.seen_guard_not_invalidated = false;
        }
        OptimizationResult::Emit(op.clone())
    }

    /// Mark call arguments as escaped.
    /// heapcache.py:259-293 mark_escaped_varargs parity.
    /// ARRAYCOPY/ARRAYMOVE with constant indices and known array descriptor
    /// do NOT escape arguments.
    fn mark_escaped_varargs(&mut self, op: &Op, ctx: &mut OptContext) {
        let oopspec = Self::get_oopspecindex(op);
        // heapcache.py:277 `descr.get_extra_info().single_write_descr_array
        // is not None`. The check predicates on the dedicated
        // `single_write_descr_array` field, NOT on the `bitstring_write*`
        // bit-count — `effectinfo.py:201-206 set_single_write_descr_array`
        // populates the field in addition to flipping the array bit, and
        // heapcache reads back the field directly.
        let has_single_write_descr = op
            .with_call_descr(|cd| cd.get_extra_info().single_write_descr_array.is_some())
            .unwrap_or(false);
        if oopspec == OopSpecIndex::Arraycopy
            && has_single_write_descr
            && op.num_args() >= 6
            && ctx.is_constant(op.arg(3))
            && ctx.is_constant(op.arg(4))
            && ctx.is_constant(op.arg(5))
        {
            return;
        }
        if oopspec == OopSpecIndex::Arraymove
            && has_single_write_descr
            && op.num_args() >= 5
            && ctx.is_constant(op.arg(2))
            && ctx.is_constant(op.arg(3))
            && ctx.is_constant(op.arg(4))
        {
            return;
        }
        for &arg in op.getarglist().iter() {
            self.escape_box(arg);
        }
    }

    fn call_argument_owner_closure(&self, op: &Op, ctx: &OptContext) -> Vec<OpRef> {
        let mut owners = Vec::new();
        let mut stack: Vec<OpRef> = op
            .getarglist()
            .iter()
            .map(|arg| ctx.get_box_replacement(*arg))
            .collect();
        while let Some(owner) = stack.pop() {
            if owners.contains(&owner) {
                continue;
            }
            owners.push(owner);
            if let Some(deps) = self.heapc_deps.get(&owner) {
                stack.extend(deps.iter().map(|dep| ctx.get_box_replacement(*dep)));
            }
        }
        owners
    }

    fn emit_postponed_if_referenced(
        &mut self,
        op: &Op,
        heap_pass_idx: usize,
        ctx: &mut OptContext,
    ) {
        let needs_postponed = self.postponed_op.as_ref().map_or(false, |postponed| {
            op.getarglist()
                .iter()
                .any(|arg| *arg == postponed.pos.get())
        });
        if needs_postponed {
            if let Some(p) = self.postponed_op.take() {
                ctx.emit_extra(heap_pass_idx, p);
            }
        }
    }

    /// Flush lazy stores for objects that escape as direct residual-call
    /// arguments. EffectInfo bitstrings describe global heap effects, but a
    /// callee can still read fields from objects it receives explicitly.
    ///
    /// Keep this selective: unrelated lazy stores may represent loop-carried
    /// state that should stay pending until the regular guard/JUMP flush.
    ///
    /// Do not remove as a cosmetic PyPy-parity cleanup. The direct parity
    /// change was tested with `python3 pyre/check.py --synthetic-only` and
    /// caused correctness failures.
    fn force_call_argument_lazy_sets(
        &mut self,
        escaped_owners: &[OpRef],
        heap_pass_idx: usize,
        ctx: &mut OptContext,
    ) {
        let mut field_entries: Vec<_> = self
            .cached_fields
            .iter_mut()
            .map(|(field_idx, descr, cf)| (*field_idx, descr.clone(), cf))
            .collect();
        sort_descr_entries_untranslated(&mut field_entries);
        let pending_fields: Vec<(u32, DescrRef, OpRef, Op)> = field_entries
            .into_iter()
            .filter_map(|(field_idx, descr, cf)| match cf.lazy_set.as_ref() {
                Some((obj, _)) => {
                    let owner = ctx.get_box_replacement(*obj);
                    if escaped_owners.contains(&owner) {
                        cf.lazy_set
                            .take()
                            .map(|(_, op)| (field_idx, descr, owner, op))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();

        for (field_idx, descr, obj, mut pending_op) in pending_fields {
            // heap.py:189-194 invalidate(descr) — purity self-gate
            // inside the method.
            if let Some(cf) = self.get_cached_field_mut(&descr) {
                cf.invalidate(&descr, ctx);
            }
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..pending_op.num_args() {
                pending_op.setarg(i, ctx.get_box_replacement(pending_op.arg(i)));
            }
            self.emit_postponed_if_referenced(&pending_op, heap_pass_idx, ctx);
            let final_value = pending_op.arg(1);
            let put_back_op = pending_op.clone();
            ctx.emit_extra(heap_pass_idx, pending_op);
            self.cache_field(obj, &descr);
            ctx.structinfo_setfield(&put_back_op, field_idx, final_value);
        }

        let mut pending_arrays = Vec::new();
        for (descr_idx, _, submap) in &mut self.cached_arrayitems {
            let mut index_entries: Vec<_> = submap
                .const_indexes
                .iter_mut()
                .map(|(&index, cai)| (index, cai))
                .collect();
            sort_array_index_entries_untranslated(&mut index_entries);
            for (index, cai) in index_entries {
                if cai.lazy_set.as_ref().map_or(false, |(obj, _)| {
                    let owner = ctx.get_box_replacement(*obj);
                    escaped_owners.contains(&owner)
                }) {
                    if let Some((obj, op)) = cai.lazy_set.take() {
                        pending_arrays.push((*descr_idx, index, ctx.get_box_replacement(obj), op));
                    }
                }
            }
        }
        for (descr_idx, index, _obj, mut pending_op) in pending_arrays {
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..pending_op.num_args() {
                pending_op.setarg(i, ctx.get_box_replacement(pending_op.arg(i)));
            }
            self.invalidate_arrayitem_cache(descr_idx, index, ctx);
            self.emit_postponed_if_referenced(&pending_op, heap_pass_idx, ctx);
            let final_value = pending_op.arg(2);
            let array_ref = pending_op.arg(0);
            let descr = pending_op.getdescr();
            let put_back_op = pending_op.clone();
            ctx.emit_extra(heap_pass_idx, pending_op);
            self.cache_arrayitem(array_ref, descr_idx, index, descr.as_ref());
            ctx.arrayinfo_setitem(&put_back_op, index as usize, final_value);
        }
    }

    /// heap.py: check if a call has random effects (EffectInfo).
    /// Calls with HAS_RANDOM_EFFECTS invalidate all caches.
    /// Calls without it only invalidate non-immutable/non-unescaped entries.
    fn call_has_random_effects(op: &Op) -> bool {
        op.with_call_descr(|cd| cd.get_extra_info().has_random_effects())
            .unwrap_or(true) // conservative: assume random effects if unknown
    }

    /// heap.py: check if a call can invalidate quasi-immutable fields.
    fn call_can_invalidate(op: &Op) -> bool {
        op.with_call_descr(|cd| cd.get_extra_info().check_can_invalidate())
            .unwrap_or(true)
    }

    /// heap.py: check if a call forces virtual/virtualizable objects.
    fn call_forces_virtual(op: &Op) -> bool {
        op.with_call_descr(|cd| cd.get_extra_info().check_forces_virtual_or_virtualizable())
            .unwrap_or(false)
    }

    /// heap.py: force_from_effectinfo(effectinfo)
    ///
    /// Selective cache invalidation based on EffectInfo bitstrings.
    /// Instead of invalidating all caches, only force/invalidate
    /// fields and arrays that the call may read or write.
    fn force_from_effectinfo(&mut self, op: &Op, ctx: &mut OptContext) {
        // heapcache.py:259-293: escape call arguments first
        self.mark_escaped_varargs(op, ctx);

        let __descr_arc_ei = op.getdescr();
        let ei = match __descr_arc_ei.as_ref().and_then(|d| d.as_call_descr()) {
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
        for (_field_idx, descr) in field_entries {
            let effect_idx = Self::field_effect_index(&descr);
            if ei.check_readonly_descr_field(effect_idx) {
                // heap.py:543-544 cf.force_lazy_set(self, fielddescr)
                // [can_cache=True].
                self.force_lazy_set_field(&descr, true, ctx);
            }
            if ei.check_write_descr_field(effect_idx) {
                // heap.py:545-546 cf.force_lazy_set(self, fielddescr,
                //                                   can_cache=False).
                self.force_lazy_set_field(&descr, false, ctx);
                // heap.py:547-552 del self.cached_dict_reads[fielddescr]
                if !descr.is_always_pure() {
                    self.cached_dict_reads.remove(&descr_identity(&descr));
                }
            }
        }

        // heap.py:554-558: for arraydescr, submap in self.cached_arrayitems.items()
        // Bitstring bit position resolves through `descr.get_ei_index()`
        // directly; `effectinfo.py:526 descr.ei_index = …` stamps the
        // slot in-place at `compute_bitstrings` time, and
        // `effectinfo.py:496 descr.ei_index = sys.maxint` is the
        // sentinel for descrs absent from any EI's raw set.
        let array_descrs: Vec<(u32, DescrRef, u32)> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, descr, _)| (*idx, descr.clone(), descr.get_ei_index()))
            .collect();
        for (descr_idx, descr, effect_idx) in array_descrs {
            let read = ei.check_readonly_descr_array(effect_idx);
            let write = ei.check_write_descr_array(effect_idx);
            if !read && !write {
                continue;
            }
            let indexes: Vec<i64> = match self.get_cached_array_submap(descr_idx) {
                Some(submap) => submap.const_indexes.keys().copied().collect(),
                None => continue,
            };
            for index in indexes {
                if read {
                    // heap.py:555-556 force_lazy_setarrayitem_submap(submap)
                    // [can_cache=True] → cf.force_lazy_set per index.
                    self.force_lazy_set_array(descr_idx, index, true, ctx);
                }
                if write {
                    // heap.py:557-558 force_lazy_setarrayitem_submap(submap,
                    // can_cache=False) → cf.force_lazy_set per index.
                    self.force_lazy_set_array(descr_idx, index, false, ctx);
                }
            }
            if write {
                // heap.py:592 force_lazy_setarrayitem_submap explicit
                // `if not can_cache: submap.clear_varindex()` — pyre
                // already calls invalidate_index per cai inside
                // force_lazy_set_array, which clears varindex via
                // submap.invalidate_index → submap.clear_varindex; this
                // outer clear_varindex is a defensive sweep for the
                // entire submap matching the upstream's batched call.
                if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    submap.clear_varindex();
                }
            }
        }

        // heap.py:560-563: invalidate cached_dict_reads via corresponding_array_descrs.
        // PyPy `effectinfo.check_write_descr_array(arraydescr)` reads
        // `arraydescr.ei_index` (`effectinfo.py:220-222`); pyre's lift
        // resolves `descr.get_ei_index()` directly per
        // `effectinfo.py:526 descr.ei_index = …` in-place stamp.
        let array_ids_to_clear: Vec<usize> = self
            .corresponding_array_descrs
            .iter()
            .filter_map(|(_, (arr_descr, dict_id))| {
                let effect_idx = arr_descr.get_ei_index();
                if ei.check_write_descr_array(effect_idx) {
                    Some(*dict_id)
                } else {
                    None
                }
            })
            .collect();
        for dict_id in array_ids_to_clear {
            self.cached_dict_reads.remove(&dict_id);
        }
    }

    // ── Handlers for specific opcodes ──

    fn optimize_getfield(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return OptimizationResult::Emit(op.clone()),
        };
        let descr = op.getdescr().unwrap();
        let field_idx = Self::field_slot_index(&descr);

        // heap.py:640-643: constant_fold — pure getfield on constant object.
        //   if descr.is_always_pure() and self.get_constant_box(arg0):
        //       resbox = self.optimizer.constant_fold(op)
        //       self.optimizer.make_constant(op, resbox)
        if descr.is_always_pure() {
            // BoxRef shim for `get_constant_box` — read-only.
            let arg0_box = ctx.get_box_replacement_box(op.arg(0));
            if arg0_box
                .as_ref()
                .and_then(|b| ctx.get_constant_box(b))
                .is_some()
            {
                if let Some(value) = ctx.constant_fold(&op) {
                    ctx.make_constant(op.pos.get(), value);
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
        if let Some(cf) = self.get_cached_field(&descr) {
            if let Some((lazy_obj, lazy_op)) = &cf.lazy_set {
                if *lazy_obj == obj {
                    // MUST_ALIAS: lazy_set targets the same struct → return rhs
                    let cached = lazy_op.arg(1);
                    let b_old = ctx
                        .ensure_box(op.pos.get())
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    let b_cached = ctx
                        .ensure_box(cached)
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    ctx.make_equal_to(&b_old, &b_cached);
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
                if let Some(entry) = cf._getfield(obj, &descr, field_idx, ctx) {
                    // heap.py:182-186: isinstance(res, PreambleOp)
                    match entry {
                        crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                            // heap.py:185-186:
                            //     res = optheap.optimizer.force_op_from_preamble(res)
                            //     opinfo.setfield(descr, None, res, optheap=optheap)
                            // Force first (use_box / potential_extra_ops side
                            // effects) and store the returned `preamble_op.op`,
                            // then walk forwarding for the body replace.
                            let cached = ctx.force_op_from_preamble_op(&pop);
                            ctx.structinfo_setfield(op, field_idx, cached);
                            self.field_cache(&descr).register_info(obj);
                            let cached = ctx.get_box_replacement(cached);
                            let b_old = ctx
                                .ensure_box(op.pos.get())
                                .expect("body-namespace OpRef must have a BoxRef slot");
                            let b_cached = ctx
                                .ensure_box(cached)
                                .expect("body-namespace OpRef must have a BoxRef slot");
                            ctx.make_equal_to(&b_old, &b_cached);
                            return OptimizationResult::Remove;
                        }
                        crate::optimizeopt::info::FieldEntry::Value(cached) => {
                            if !cached.is_none() {
                                let cached = ctx.get_box_replacement(cached);
                                let b_old = ctx
                                    .ensure_box(op.pos.get())
                                    .expect("body-namespace OpRef must have a BoxRef slot");
                                let b_cached = ctx
                                    .ensure_box(cached)
                                    .expect("body-namespace OpRef must have a BoxRef slot");
                                ctx.make_equal_to(&b_old, &b_cached);
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
                .get_cached_field_mut(&descr)
                .and_then(|cf| cf.lazy_set.take());
            if let Some((lazy_obj, mut lazy_op)) = lazy_data {
                // heap.py:189-194 invalidate(descr) — purity self-gate
                // inside the method.
                if let Some(cf) = self.get_cached_field_mut(&descr) {
                    cf.invalidate(&descr, ctx);
                }
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos.get();
                    if lazy_op.getarglist().iter().any(|a| *a == ppos) {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                }
                Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                // can_cache=True: put_field_back_to_info
                let final_value = lazy_op.arg(1);
                let lazy_descr = lazy_op.getdescr().unwrap().clone();
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

        // Check read cache (after import).
        if let Some(cf) = self.get_cached_field(&descr) {
            if let Some(entry) = cf._getfield(obj, &descr, field_idx, ctx) {
                match entry {
                    crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                        // heap.py:185-186 force-then-setfield (see above).
                        let cached = ctx.force_op_from_preamble_op(&pop);
                        ctx.structinfo_setfield(op, field_idx, cached);
                        self.field_cache(&descr).register_info(obj);
                        let cached = ctx.get_box_replacement(cached);
                        let b_old = ctx
                            .ensure_box(op.pos.get())
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        let b_cached = ctx
                            .ensure_box(cached)
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        ctx.make_equal_to(&b_old, &b_cached);
                        return OptimizationResult::Remove;
                    }
                    crate::optimizeopt::info::FieldEntry::Value(cached) => {
                        if !cached.is_none() {
                            let cached = ctx.get_box_replacement(cached);
                            let b_old = ctx
                                .ensure_box(op.pos.get())
                                .expect("body-namespace OpRef must have a BoxRef slot");
                            let b_cached = ctx
                                .ensure_box(cached)
                                .expect("body-namespace OpRef must have a BoxRef slot");
                            ctx.make_equal_to(&b_old, &b_cached);
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
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_qi = ctx
                    .ensure_box(qi_cached)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_qi);
                return OptimizationResult::Remove;
            }
            // First read after QUASIIMMUT_FIELD: emit the load, then cache
            // the result so it survives calls (unlike normal mutable fields).
            self.quasi_immut_cache.insert(key, op.pos.get());
            self.cache_field(obj, &descr);
            ctx.structinfo_setfield(op, field_idx, op.pos.get());
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
        vb_set(&mut self.known_nonnull, struct_ref.raw());
        self.cache_field(obj, &descr);
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
            ctx.structinfo_setfield(op, field_idx, op.pos.get());
        }
        // Virtualizable Ref fields (linked list head) need a null guard.
        let is_vable_ref =
            is_vable_field && matches!(op.opcode, OpCode::GetfieldGcR | OpCode::GetfieldGcPureR);
        if is_vable_ref {
            ctx.emit(op.clone());
            let zero_ref = ctx.make_constant_int(0);
            let cmp_pos = ctx.alloc_op_position_typed(OpCode::IntNe.result_type());
            let mut cmp_op = Op::new(OpCode::IntNe, &[op.pos.get(), zero_ref]);
            cmp_op.pos.set(cmp_pos);
            ctx.emit(cmp_op);
            // unroll.py:409 parity: synthetic guards inherit
            // rd_resume_position from patchguardop (the optimizer's
            // running GUARD_FUTURE_CONDITION). Without this, the guard
            // arrives at store_final_boxes_in_guard with -1 and would
            // be silently dropped under the patchguardop-only fallback.
            let guard_op = Op::new(OpCode::GuardTrue, &[cmp_pos]);
            if let Some(ref patch) = ctx.patchguardop {
                guard_op
                    .rd_resume_position
                    .set(patch.rd_resume_position.get());
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
        let descr = op.getdescr().unwrap();
        let (raw_obj, _) = key;
        // heap.py:78 ensure_ptr_info_arg0 — install structinfo as a
        // side effect; canonical OpRef for the cache key.
        let obj = ctx.get_box_replacement(raw_obj);
        let _ = ctx.ensure_ptr_info_arg0(op);
        // heapcache.py:224-230 _escape_from_write — pyre-specific
        // escape tracking outside the do_setfield contract.
        self.escape_from_write(obj, op.arg(1));
        // heap.py:77-101 do_setfield line-by-line.
        self.do_setfield_field(op, &descr, obj, ctx)
    }

    /// heap.py:77-101 `AbstractCachedEntry.do_setfield(optheap, op)`
    /// line-by-line. Lives on `OptHeap` rather than on `CachedField`
    /// because of the Rust borrow restriction: the upstream method
    /// calls `self.force_lazy_set(optheap, descr)` which would require
    /// holding `&mut cf` (cf lives inside `optheap.cached_fields`) and
    /// `&mut optheap` simultaneously. The body is identical to
    /// upstream's; the receiver is the parent container.
    ///
    /// ```python
    /// def do_setfield(self, optheap, op):
    ///     structinfo = optheap.ensure_ptr_info_arg0(op)
    ///     arg1 = get_box_replacement(self._get_rhs_from_set_op(op))
    ///     if self.possible_aliasing(structinfo):
    ///         self.force_lazy_set(optheap, op.getdescr())
    ///         assert not self.possible_aliasing(structinfo)
    ///     cached_field = self._getfield(structinfo, op.getdescr(), optheap, False)
    ///     if cached_field is not None:
    ///         cached_field = cached_field.get_box_replacement()
    ///     if not cached_field or not cached_field.same_box(arg1):
    ///         # common case: store the 'op' as lazy_set
    ///         self._lazy_set = op
    ///     else:
    ///         # cancel out — value already there
    ///         self._getfield(structinfo, op.getdescr(), optheap)
    ///         self._lazy_set = None
    /// ```
    ///
    /// The pyre `_getfield` returns `Option<FieldEntry>`; the
    /// `FieldEntry::Preamble` branch is handled by emitting the
    /// preamble op + registering the info before the cancel — that's
    /// the pyre-specific Preamble plumbing parity (info.py:259-272
    /// `produce_short_preamble_ops`).
    fn do_setfield_field(
        &mut self,
        op: &Op,
        descr: &DescrRef,
        obj: OpRef,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // heap.py:80 arg1 = get_box_replacement(self._get_rhs_from_set_op(op))
        let arg1 = ctx.get_box_replacement(Self::field_get_rhs(op));
        let field_idx = Self::field_slot_index(descr);
        // heap.py:81-83 if self.possible_aliasing(structinfo):
        //                  self.force_lazy_set(optheap, op.getdescr())
        let needs_force = self
            .get_cached_field(&descr)
            .map_or(false, |cf| cf.possible_aliasing(obj));
        if needs_force {
            // heap.py:122-145 force_lazy_set(self, optheap, descr)
            // [can_cache=True]. Lifted into `force_lazy_set_field` per
            // upstream's CachedField method.
            self.force_lazy_set_field(descr, true, ctx);
        }
        // heap.py:85 cached_field = self._getfield(structinfo, op.getdescr(),
        //                                          optheap, False)
        // heap.py:86-87 if cached_field is not None:
        //                   cached_field = cached_field.get_box_replacement()
        // heap.py:88-101 lazy_set vs cancel dispatch (with pyre's
        // Preamble/Value FieldEntry split).
        if let Some(entry) = self
            .field_cache(descr)
            ._getfield(obj, descr, field_idx, ctx)
        {
            match entry {
                crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                    let cached_seen = ctx.get_box_replacement(pop.op);
                    if cached_seen == arg1 {
                        let cached = ctx.force_op_from_preamble_op(&pop);
                        ctx.structinfo_setfield(op, field_idx, cached);
                        self.field_cache(descr).register_info(obj);
                        // heap.py:100 self._lazy_set = None
                        self.field_cache(descr).lazy_set = None;
                        return OptimizationResult::Remove;
                    }
                }
                crate::optimizeopt::info::FieldEntry::Value(cached) => {
                    let cached_resolved = ctx.get_box_replacement(cached);
                    if cached_resolved == arg1 {
                        // heap.py:100 self._lazy_set = None
                        self.field_cache(descr).lazy_set = None;
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        // heap.py:89-91 common case: self._lazy_set = op
        let cf = self.field_cache(descr);
        cf.lazy_set = Some((obj, op.clone()));
        OptimizationResult::Remove
    }

    /// heap.py:169-170 CachedField._get_rhs_from_set_op — the new
    /// value of a SETFIELD_GC is its second arg.
    fn field_get_rhs(op: &Op) -> OpRef {
        op.arg(1)
    }

    /// heap.py:300 ArrayCachedItem._get_rhs_from_set_op — the new
    /// value of a SETARRAYITEM_GC is its third arg.
    fn array_get_rhs(op: &Op) -> OpRef {
        op.arg(2)
    }

    /// heap.py:122-145 `AbstractCachedEntry.force_lazy_set(optheap,
    /// descr, can_cache=True)` array-side line-by-line. Body identical
    /// to `force_lazy_set_field` modulo:
    /// - cai entry locator (submap by descr_idx, const_index by index).
    /// - cai.invalidate() also invalidates parent.clear_varindex via
    ///   submap.invalidate_index (heap.py:266 parent.clear_varindex()
    ///   inside ArrayCachedItem.invalidate; pyre lifts this to the
    ///   caller-side per the `cai.invalidate` doc at heap.rs:516).
    /// - `_get_rhs_from_set_op` uses op.arg(2) at the put_back step.
    fn force_lazy_set_array(
        &mut self,
        descr_idx: u32,
        const_index: i64,
        can_cache: bool,
        ctx: &mut OptContext,
    ) {
        // heap.py:123 op = self._lazy_set
        let lazy_data = self
            .get_cached_array_submap_mut(descr_idx)
            .and_then(|s| s.const_indexes.get_mut(&const_index))
            .and_then(|cai| cai.lazy_set.take());
        match lazy_data {
            Some((lazy_obj, mut lazy_op)) => {
                // heap.py:127 self.invalidate(descr) — cai.invalidate +
                // parent.clear_varindex (lifted into submap.invalidate_index).
                if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    submap.invalidate_index(const_index, ctx);
                }
                // heap.py:128 self._lazy_set = None — already done via take().
                // heap.py:130-134 emit postponed_op if referenced.
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos.get();
                    if lazy_op.getarglist().iter().any(|a| *a == ppos) {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                }
                // heap.py:135 optheap.emit_extra(op, emit=False)
                let put_back_op = lazy_op.clone();
                Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                // heap.py:136-137 if not can_cache: return
                if !can_cache {
                    return;
                }
                // heap.py:141-143 put_field_back_to_info — array-side write.
                let final_value = lazy_op.arg(2);
                let lazy_descr = put_back_op.getdescr();
                self.cache_arrayitem(lazy_obj, descr_idx, const_index, lazy_descr.as_ref());
                ctx.arrayinfo_setitem(&put_back_op, const_index as usize, final_value);
            }
            None => {
                // heap.py:144-145 elif not can_cache: self.invalidate(descr)
                if !can_cache {
                    if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                        submap.invalidate_index(const_index, ctx);
                    }
                }
            }
        }
    }

    /// heap.py:77-101 `AbstractCachedEntry.do_setfield` array-side
    /// line-by-line port. Body identical to `do_setfield_field` modulo
    /// `_get_rhs_from_set_op` (arg2 vs arg1) and the entry locator
    /// (ArrayCachedItem in submap keyed by const_index vs CachedField
    /// keyed by descr). Lives on OptHeap for the same Rust borrow
    /// reason — cai is owned by `optheap.cached_arrayitems[descr_idx]
    /// .const_indexes[const_index]`.
    fn do_setfield_array(
        &mut self,
        op: &Op,
        descr: &DescrRef,
        array: OpRef,
        descr_idx: u32,
        const_index: i64,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // heap.py:80 arg1 = get_box_replacement(self._get_rhs_from_set_op(op))
        let arg1 = ctx.get_box_replacement(Self::array_get_rhs(op));
        // heap.py:81-83 if self.possible_aliasing(structinfo):
        //                  self.force_lazy_set(optheap, op.getdescr())
        let needs_force = self
            .get_cached_array_submap(descr_idx)
            .and_then(|s| s.const_indexes.get(&const_index))
            .map_or(false, |cai| cai.possible_aliasing(array));
        if needs_force {
            // heap.py:122-145 force_lazy_set(self, optheap, descr) [can_cache=True]
            // extracted as force_lazy_set_array per AbstractCachedEntry parity.
            self.force_lazy_set_array(descr_idx, const_index, true, ctx);
        }
        // heap.py:85 cached_field = self._getfield(structinfo, descr, optheap, False)
        // heap.py:86-87 get_box_replacement
        // heap.py:88-101 lazy_set vs cancel dispatch (pyre's
        // Preamble/Value FieldEntry split).
        if let Some(entry) = self
            .arrayitem_cache(descr, const_index)
            ._getfield(array, descr, ctx)
        {
            match entry {
                crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                    let cached_seen = ctx.get_box_replacement(pop.op);
                    if cached_seen == arg1 {
                        let cached = ctx.force_op_from_preamble_op(&pop);
                        self.arrayitem_cache(descr, const_index)
                            .register_info(array);
                        ctx.arrayinfo_setitem(op, const_index as usize, cached);
                        // heap.py:100 self._lazy_set = None
                        self.arrayitem_cache(descr, const_index).lazy_set = None;
                        return OptimizationResult::Remove;
                    }
                }
                crate::optimizeopt::info::FieldEntry::Value(cached) => {
                    let cached_resolved = ctx.get_box_replacement(cached);
                    if cached_resolved == arg1 {
                        // heap.py:100 self._lazy_set = None
                        self.arrayitem_cache(descr, const_index).lazy_set = None;
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        // heap.py:89-91 common case: self._lazy_set = op
        let cai = self.arrayitem_cache(descr, const_index);
        cai.lazy_set = Some((array, op.clone()));
        OptimizationResult::Remove
    }

    /// heap.py:122-145 `AbstractCachedEntry.force_lazy_set(optheap,
    /// descr, can_cache=True)` line-by-line for the CachedField path.
    /// Lives on `OptHeap` rather than `CachedField` due to the Rust
    /// borrow restriction (cf is owned by `optheap.cached_fields`, so
    /// `&mut cf` and `&mut optheap` can't coexist — see do_setfield_field).
    ///
    /// ```python
    /// def force_lazy_set(self, optheap, descr, can_cache=True):
    ///     op = self._lazy_set
    ///     if op is not None:
    ///         self.invalidate(descr)
    ///         self._lazy_set = None
    ///         if optheap.postponed_op:
    ///             for a in op.getarglist():
    ///                 if a is optheap.postponed_op:
    ///                     optheap.emit_postponed_op()
    ///                     break
    ///         optheap.emit_extra(op, emit=False)
    ///         if not can_cache:
    ///             return
    ///         opinfo = optheap.ensure_ptr_info_arg0(op)
    ///         self.put_field_back_to_info(op, opinfo, optheap)
    ///     elif not can_cache:
    ///         self.invalidate(descr)
    /// ```
    ///
    /// `put_field_back_to_info` is heap.py:146-158; pyre's
    /// `cache_field` + `ctx.structinfo_setfield` is the line-by-line
    /// inline port.
    fn force_lazy_set_field(&mut self, descr: &DescrRef, can_cache: bool, ctx: &mut OptContext) {
        // heap.py:123 op = self._lazy_set
        let lazy_data = self
            .get_cached_field_mut(descr)
            .and_then(|cf| cf.lazy_set.take());
        match lazy_data {
            Some((lazy_obj, mut lazy_op)) => {
                // heap.py:127 self.invalidate(descr) — purity self-gate
                // inside CachedField::invalidate.
                if let Some(cf) = self.get_cached_field_mut(descr) {
                    cf.invalidate(descr, ctx);
                }
                // heap.py:128 self._lazy_set = None — already done via
                // `take()` above.

                // heap.py:130-134 emit postponed_op if referenced.
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos.get();
                    if lazy_op.getarglist().iter().any(|&a| a == ppos) {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                }
                // heap.py:135 optheap.emit_extra(op, emit=False)
                let put_back_op = lazy_op.clone();
                Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                // heap.py:136-137 if not can_cache: return
                if !can_cache {
                    return;
                }
                // heap.py:141-143 put_field_back_to_info(op, opinfo, optheap)
                let final_value = lazy_op.arg(1);
                let lazy_descr = put_back_op.getdescr().unwrap();
                let lazy_field_idx = Self::field_slot_index(&lazy_descr);
                self.cache_field(lazy_obj, &lazy_descr);
                ctx.structinfo_setfield(&put_back_op, lazy_field_idx, final_value);
            }
            None => {
                // heap.py:144-145 elif not can_cache: self.invalidate(descr)
                if !can_cache {
                    if let Some(cf) = self.get_cached_field_mut(descr) {
                        cf.invalidate(descr, ctx);
                    }
                }
            }
        }
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
            let descr = op.getdescr().unwrap();
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
                        let b_old = ctx
                            .ensure_box(op.pos.get())
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        let b_cached = ctx
                            .ensure_box(cached)
                            .expect("body-namespace OpRef must have a BoxRef slot");
                        ctx.make_equal_to(&b_old, &b_cached);
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
                    if let Some(entry) = cai._getfield(array, &op.getdescr().unwrap(), ctx) {
                        match entry {
                            crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                                // heap.py:243-249 ArrayCachedItem._getfield:
                                //   res = optheap.optimizer.force_op_from_preamble(res)
                                //   opinfo.setitem(descr, index, None, res, optheap=optheap)
                                let cached = ctx.force_op_from_preamble_op(&pop);
                                self.arrayitem_cache(&descr, const_index)
                                    .register_info(array);
                                ctx.arrayinfo_setitem(op, const_index as usize, cached);
                                let cached = ctx.get_box_replacement(cached);
                                let b_old = ctx
                                    .ensure_box(op.pos.get())
                                    .expect("body-namespace OpRef must have a BoxRef slot");
                                let b_cached = ctx
                                    .ensure_box(cached)
                                    .expect("body-namespace OpRef must have a BoxRef slot");
                                ctx.make_equal_to(&b_old, &b_cached);
                                return OptimizationResult::Remove;
                            }
                            crate::optimizeopt::info::FieldEntry::Value(cached) => {
                                if !cached.is_none() {
                                    let cached = ctx.get_box_replacement(cached);
                                    let b_old = ctx
                                        .ensure_box(op.pos.get())
                                        .expect("body-namespace OpRef must have a BoxRef slot");
                                    let b_cached = ctx
                                        .ensure_box(cached)
                                        .expect("body-namespace OpRef must have a BoxRef slot");
                                    ctx.make_equal_to(&b_old, &b_cached);
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
                        let ppos = postponed.pos.get();
                        if lazy_op.getarglist().iter().any(|a| *a == ppos) {
                            if let Some(p) = self.postponed_op.take() {
                                ctx.emit_extra(ctx.current_pass_idx, p);
                            }
                        }
                    }
                    Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                    // can_cache=True: put_field_back_to_info
                    let final_value = lazy_op.arg(2);
                    let descr = lazy_op.getdescr();
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
                .ensure_box(array)
                .and_then(|b| {
                    ctx.with_ptr_info_mut(&b, |info| info.take_preamble_item(const_index as usize))
                })
                .flatten()
                .or_else(|| {
                    ctx.get_const_info_mut_if_exists(array)
                        .and_then(|info| info.take_preamble_item(const_index as usize))
                });
            if let Some(pop) = pop {
                // heap.py:243-249 force-then-setitem (see above).
                let cached = ctx.force_op_from_preamble_op(&pop);
                self.arrayitem_cache(&descr, const_index)
                    .register_info(array);
                ctx.arrayinfo_setitem(op, const_index as usize, cached);
                let cached = ctx.get_box_replacement(cached);
                let b_old = ctx
                    .ensure_box(op.pos.get())
                    .expect("body-namespace OpRef must have a BoxRef slot");
                let b_cached = ctx
                    .ensure_box(cached)
                    .expect("body-namespace OpRef must have a BoxRef slot");
                ctx.make_equal_to(&b_old, &b_cached);
                return OptimizationResult::Remove;
            }
            if let Some(cai) = self
                .get_cached_array_submap(descr_idx)
                .and_then(|s| s.const_indexes.get(&const_index))
            {
                if let Some(entry) = cai._getfield(array, &op.getdescr().unwrap(), ctx) {
                    match entry {
                        crate::optimizeopt::info::FieldEntry::Preamble(pop) => {
                            // heap.py:243-249 force-then-setitem (see above).
                            let cached = ctx.force_op_from_preamble_op(&pop);
                            self.arrayitem_cache(&descr, const_index)
                                .register_info(array);
                            ctx.arrayinfo_setitem(op, const_index as usize, cached);
                            let cached = ctx.get_box_replacement(cached);
                            let b_old = ctx
                                .ensure_box(op.pos.get())
                                .expect("body-namespace OpRef must have a BoxRef slot");
                            let b_cached = ctx
                                .ensure_box(cached)
                                .expect("body-namespace OpRef must have a BoxRef slot");
                            ctx.make_equal_to(&b_old, &b_cached);
                            return OptimizationResult::Remove;
                        }
                        crate::optimizeopt::info::FieldEntry::Value(cached) => {
                            if !cached.is_none() {
                                let cached = ctx.get_box_replacement(cached);
                                let b_old = ctx
                                    .ensure_box(op.pos.get())
                                    .expect("body-namespace OpRef must have a BoxRef slot");
                                let b_cached = ctx
                                    .ensure_box(cached)
                                    .expect("body-namespace OpRef must have a BoxRef slot");
                                ctx.make_equal_to(&b_old, &b_cached);
                                return OptimizationResult::Remove;
                            }
                        }
                    }
                }
            }
            self.cache_arrayitem(array, descr_idx, const_index, op.getdescr().as_ref());
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
            vb_set(&mut self.known_nonnull, array_ref.raw());
            if const_index >= 0 {
                ctx.with_ensured_ptr_info_arg0(op, |mut arrayinfo| {
                    if let Some(mut bound) = arrayinfo.getlenbound(None) {
                        let _ = bound.make_gt_const(const_index);
                        if let Some(mut handle) = arrayinfo.as_mut() {
                            if let crate::optimizeopt::info::PtrInfo::Array(a) = &mut *handle {
                                a.lenbound = bound;
                            }
                        }
                    }
                });
            }
            ctx.arrayinfo_setitem(op, const_index as usize, op.pos.get());
            return OptimizationResult::Emit(op.clone());
        }

        // heap.py:690-701: variable-index GETARRAYITEM_GC path.
        //   self.force_lazy_setarrayitem(op.getdescr(), self.getintbound(op.getarg(1)))
        //   submap = self.arrayitem_submap(op.getdescr(), create_if_nonexistant=False)
        //   cached_result = submap.lookup_cached(arrayinfo, indexop)
        if let Some(descr) = op.getdescr() {
            // heap.py:692-693: force lazy stores for this descr within the index bound
            let indexb = ctx.getintbound(op.arg(1));
            self.force_lazy_setarrayitem(&descr, Some(&indexb), true, ctx);

            let descr_idx = descr.index();
            let arrayinfo = array_ref;
            let indexbox = ctx.get_box_replacement(op.arg(1));
            if let Some(submap) = self.get_cached_array_submap(descr_idx) {
                if let Some(cached) = submap.lookup_cached(arrayinfo, indexbox, ctx) {
                    let b_old = ctx
                        .ensure_box(op.pos.get())
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    let b_cached = ctx
                        .ensure_box(cached)
                        .expect("body-namespace OpRef must have a BoxRef slot");
                    ctx.make_equal_to(&b_old, &b_cached);
                    return OptimizationResult::Remove;
                }
            }
            self.arrayitem_submap(&descr)
                .cache_varindex_read(arrayinfo, indexbox, op.pos.get());
        }

        // heap.py line 701: make_nonnull(op.getarg(0))
        vb_set(&mut self.known_nonnull, array_ref.raw());
        // OpRef → BoxRef shim until this caller migrates (Phase D-2).
        if let Some(arg0_box) = ctx.ensure_box(op.arg(0)) {
            ctx.make_nonnull(&arg0_box);
        }
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
                // heap.py:762-767: variable index SETARRAYITEM_GC
                //   self.force_lazy_setarrayitem(op.getdescr(), indexb, can_cache=False)
                //   submap.cache_varindex_write(arrayinfo, ...)
                //   return self.emit(op)
                if let Some(descr) = op.getdescr() {
                    let indexb = ctx.getintbound(op.arg(1));
                    self.force_lazy_setarrayitem(&descr, Some(&indexb), false, ctx);
                    let arrayinfo = ctx.get_box_replacement(op.arg(0));
                    let indexbox = ctx.get_box_replacement(op.arg(1));
                    let resbox = ctx.get_box_replacement(op.arg(2));
                    self.arrayitem_submap(&descr)
                        .cache_varindex_write(arrayinfo, indexbox, resbox);
                }
                return OptimizationResult::Emit(op.clone());
            }
        };

        let (array, descr_idx, const_index) = key;
        let descr = op.getdescr().unwrap();
        // heap.py:77-101 ArrayCachedItem.do_setfield (shared body via
        // AbstractCachedEntry).
        let result = self.do_setfield_array(op, &descr, array, descr_idx, const_index, ctx);
        // heap.py:761 `submap.clear_varindex()` AFTER do_setfield (called
        // at the optimize_SETARRAYITEM_GC site — outside do_setfield).
        if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
            submap.clear_varindex();
        }
        result
    }

    /// Handle operations that may have side effects.
    /// Forces lazy sets and invalidates caches as needed.
    /// Tracks allocations for aliasing analysis.
    fn handle_side_effects(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let opcode = op.opcode;

        // Track allocations for aliasing analysis.
        // Allocated objects are always non-null.
        if opcode.is_malloc() {
            vb_set(&mut self.seen_allocation, op.pos.get().raw());
            vb_set(&mut self.unescaped, op.pos.get().raw());
            vb_set(&mut self.known_nonnull, op.pos.get().raw());
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
                vb_set(&mut self.known_nonnull, op.arg(0).raw());
            }

            // GuardClass / GuardNonnullClass imply non-null.
            // GuardValue does NOT imply non-null: the guarded constant
            // may be NULL (rewrite.py:320 handles GUARD_VALUE(..., NULL)).
            match opcode {
                OpCode::GuardClass | OpCode::GuardNonnullClass => {
                    vb_set(&mut self.known_nonnull, op.arg(0).raw());
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
                // heap.py:472-475: DICT_LOOKUP caching — consecutive dict
                // lookups on the same dict with the same key can be
                // deduplicated. On a miss the call falls through to
                // self.emit(op) → emitting_operation → force_from_effectinfo,
                // identical to a non-DICT_LOOKUP residual call.
                OopSpecIndex::DictLookup => {
                    if self._optimize_call_dict_lookup(op, ctx) {
                        return OptimizationResult::Remove;
                    }
                    return self.emit_residual_call(op, ctx);
                }
                OopSpecIndex::Arraycopy | OopSpecIndex::Arraymove => {
                    // ARRAYCOPY/ARRAYMOVE: only invalidate affected array entries.
                    // Call args: [func_addr, source, dest, source_start, dest_start, length, ...]
                    // args[2] = dest array, args[4] = dest_start, args[5] = length
                    self.mark_escaped_varargs(op, ctx);
                    self.force_all_lazy_sets(ctx.current_pass_idx, ctx);

                    let dest_ref = if op.num_args() > 2 {
                        op.arg(2)
                    } else {
                        OpRef::NONE
                    };
                    let dest_start = if op.num_args() > 4 {
                        ctx.get_constant_int(op.arg(4))
                    } else {
                        None
                    };
                    let length = if op.num_args() > 5 {
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
                _ => return self.emit_residual_call(op, ctx),
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
                vb_set(&mut self.seen_allocation, op.pos.get().raw());
                vb_set(&mut self.unescaped, op.pos.get().raw());
                vb_set(&mut self.known_nonnull, op.pos.get().raw());
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
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[opt-heap] postpone {:?} pos={:?} descr={:?}",
                        op.opcode,
                        op.pos.get(),
                        op.descr
                    );
                }
                // RPython emitting_operation: calls go through
                // force_from_effectinfo (selective) or clean_caches,
                // NOT force_all_lazy. force_all_lazy is only in flush().
                let escaped_owners = self.call_argument_owner_closure(op, ctx);
                self.mark_escaped_varargs(op, ctx);
                // See `emit_residual_call`: this pyre-specific flush is
                // required for current synthetic correctness.
                self.force_call_argument_lazy_sets(&escaped_owners, ctx.current_pass_idx, ctx);
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
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[opt-heap] emit postponed {:?} pos={:?} before {:?} pos={:?}",
                            postponed.opcode,
                            postponed.pos.get(),
                            op.opcode,
                            op.pos.get()
                        );
                    }
                    // RPython emit_postponed_op: route through next_optimization
                    ctx.emit_extra(ctx.current_pass_idx, postponed);
                } else if crate::majit_log_enabled() {
                    eprintln!(
                        "[opt-heap] no postponed op before {:?} pos={:?}",
                        op.opcode,
                        op.pos.get()
                    );
                }
                // RPython emitting_operation for guards:
                //   self.optimizer.pendingfields = self.force_lazy_sets_for_guard()
                let pending_virtual = self.force_lazy_sets_for_guard(ctx.current_pass_idx, ctx);
                for pending_op in pending_virtual {
                    if pending_op.opcode == OpCode::SetarrayitemGc {
                        let descr = pending_op.getdescr().unwrap().clone();
                        if let Some(index) = ctx.get_constant_int(pending_op.arg(1)) {
                            let array = ctx.get_box_replacement(pending_op.arg(0));
                            let cai = self.arrayitem_cache(&descr, index);
                            cai.lazy_set = Some((array, pending_op));
                        } else {
                            ctx.emit(pending_op);
                        }
                    } else {
                        let descr = pending_op.getdescr().unwrap().clone();
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

            // heap.py:530-535 optimize_GUARD_NO_EXCEPTION / optimize_GUARD_EXCEPTION.
            // When _optimize_CALL_DICT_LOOKUP folds a lookup, it sets
            // last_emitted_removed; the trailing GUARD_NO_EXCEPTION is dead.
            OpCode::GuardNoException | OpCode::GuardException => {
                if self.last_emitted_removed {
                    self.last_emitted_removed = false;
                    OptimizationResult::Remove
                } else {
                    OptimizationResult::Emit(op.clone())
                }
            }

            // heap.py:810-825 optimize_GUARD_NOT_INVALIDATED
            OpCode::GuardNotInvalidated => {
                if self.seen_guard_not_invalidated {
                    OptimizationResult::Remove
                } else {
                    self.seen_guard_not_invalidated = true;
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
                let (dep_field_idx, cache_field_key) = if let Some(descr) = op.getdescr() {
                    (
                        Some(Self::field_effect_index(&descr)),
                        Some(Self::field_cache_identity(&descr)),
                    )
                } else if op.num_args() > 1 {
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
                vb_set(&mut self.known_nonnull, op.arg(0).raw());
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
        // heap.py:417-425 emit() override parity:
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
                // optimizer.py:84-87 — postponed ops do NOT call
                // Optimization.emit, so line 86's `last_emitted_operation = op`
                // does NOT fire. Leave `last_emitted_removed` intact so a
                // GUARD_NO_EXCEPTION trailing a folded DICT_LOOKUP that came
                // before this comparison still observes REMOVED.
                return OptimizationResult::Remove;
            }
        }
        // optimizer.py:84-92 — `Optimization.emit(op)` and
        // `Optimization.emit_result(opt_result)` BOTH set
        // `self.last_emitted_operation = op` before returning. The
        // PASS_OP_ON path (line 87) reaches that assignment too, so
        // every dispatch result that is downstreamed (Emit non-postponed,
        // PassOn, Replace, Restart) overwrites the REMOVED sentinel.
        // Remove / InvalidLoop / postponed-Emit do not.
        match &result {
            OptimizationResult::Emit(_)
            | OptimizationResult::PassOn
            | OptimizationResult::Replace(_)
            | OptimizationResult::Restart(_) => {
                self.last_emitted_removed = false;
            }
            OptimizationResult::Remove | OptimizationResult::InvalidLoop => {}
        }
        result
    }

    fn setup(&mut self) {
        self.cached_fields.clear();
        self.cached_arrayitems.clear();
        self.seen_guard_not_invalidated = false;
        self.postponed_op = None;
        self.seen_allocation.clear();
        self.unescaped.clear();
        self.known_nonnull.clear();
        self.last_emitted_removed = false;
        self.cached_dict_reads.clear();
        self.corresponding_array_descrs.clear();
        self.quasi_immut_cache.clear();
    }

    fn flush(&mut self, ctx: &mut OptContext) {
        // heap.py:348-352 flush():
        //   self.cached_dict_reads.clear()
        //   self.corresponding_array_descrs.clear()
        //   self.force_all_lazy_sets()
        //   self.emit_postponed_op()
        self.cached_dict_reads.clear();
        self.corresponding_array_descrs.clear();
        self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
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
                if !op.has_descr() {
                    // RPython: all calls have descriptors. Conservative
                    // fallback for pyre calls that lack an EffectInfo.
                    self.force_all_lazy_sets(self_pass_idx, ctx);
                    self.clean_caches(ctx);
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
        available_boxes: Option<&[OpRef]>,
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
                    if !obj.is_constant() && !ab.contains(&obj) {
                        continue;
                    }
                }
                // heap.py:838-839: structinfo = cf.cached_infos[i]
                //                  box2 = structinfo.getfield(descr)
                let resolved = ctx.get_box_replacement(obj);
                let resolved_box = ctx.get_box_replacement_box(resolved);
                let Some(val) = resolved_box
                    .as_ref()
                    .and_then(|b| ctx.peek_ptr_info(b))
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
                    val.is_constant() || ctx.is_constant(val) || ab.contains(&val)
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
            let resolved_is_virtual = ctx
                .get_box_replacement_box(resolved)
                .as_ref()
                .map_or(false, |b| ctx.is_virtual(b));
            let needs_install = !ctx.is_constant(resolved) && !resolved_is_virtual;
            if needs_install {
                // info.py:175-188 InstancePtrInfo + init_fields
                if let Some(b) = ctx.ensure_box(resolved) {
                    ctx.set_ptr_info(&b, PtrInfo::instance(parent_descr.clone(), None));
                }
            }
            // heap.py:882-883: cf = self.field_cache(&descr)
            //                  structinfo.setfield(descr, box1, box2, optheap, cf=cf)
            self.cache_field(*box1, descr);
            if ctx.is_constant(resolved) {
                if let Some(info) = ctx.get_const_info_mut(resolved, parent_descr.clone()) {
                    info.setfield(field_idx, *box2);
                }
            } else {
                let box2 = *box2;
                if let Some(b) = ctx.ensure_box(resolved) {
                    ctx.with_ptr_info_mut(&b, |info| info.setfield(field_idx, box2));
                }
            }
        }
    }

    /// heap.py:847-868 serialize_optheap(available_boxes) (array half)
    fn export_cached_arrayitems(
        &self,
        ctx: &mut OptContext,
        available_boxes: Option<&[OpRef]>,
    ) -> Vec<(OpRef, i64, DescrRef, OpRef)> {
        let mut result = Vec::new();
        for (_, descr, submap) in &self.cached_arrayitems {
            // heap.py:849: if descr.get_descr_index() == -1: continue
            if descr.get_descr_index() == -1 {
                continue;
            }
            for (&index, cai) in submap.const_indexes.iter() {
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
                        if !obj.is_constant() && !ab.contains(&obj) {
                            continue;
                        }
                    }
                    // heap.py:858: if index >= 2**15: continue
                    if index >= (1 << 15) {
                        continue;
                    }
                    let resolved = ctx.get_box_replacement(obj);
                    let resolved_box = ctx.get_box_replacement_box(resolved);
                    // heap.py:860: box2 = arrayinfo.getitem(descr, index)
                    let Some(val) = resolved_box
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
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
                        val.is_constant() || ctx.is_constant(val) || ab.contains(&val)
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
            let resolved_is_virtual = ctx
                .get_box_replacement_box(resolved)
                .as_ref()
                .map_or(false, |b| ctx.is_virtual(b));
            let needs_install = !ctx.is_constant(resolved) && !resolved_is_virtual;
            if needs_install {
                if let Some(b) = ctx.ensure_box(resolved) {
                    ctx.set_ptr_info(
                        &b,
                        PtrInfo::array(
                            descr.clone(),
                            crate::optimizeopt::intutils::IntBound::nonnegative(),
                        ),
                    );
                }
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
            } else {
                let idx = *index as usize;
                let box2 = *box2;
                if let Some(b) = ctx.ensure_box(resolved) {
                    ctx.with_ptr_info_mut(&b, |info| info.setitem(idx, box2));
                }
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
    use std::sync::atomic::{AtomicU32, Ordering};

    use majit_ir::{
        CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, FieldDescr, OopSpecIndex, Op, OpCode,
        OpRef, SizeDescr, Type, bitstring,
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
    ///
    /// `ei_index` mirrors PyPy production field descrs (`history.py:498
    /// FieldDescr.ei_index`); default `u32::MAX` matches PyPy
    /// `effectinfo.py:496 descr.ei_index = sys.maxint` for descrs absent
    /// from any EI's raw set. Tests that need a specific `ei_index`
    /// (i.e., the descr appears in some EI's `_*_descrs_*` raw set) call
    /// `descr.set_ei_index(N)` before constructing the EI, mirroring
    /// `effectinfo.py:526 descr.ei_index = mapping.setdefault(...)`.
    #[derive(Debug)]
    struct TestDescr {
        index: u32,
        ei_index: AtomicU32,
    }

    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn get_ei_index(&self) -> u32 {
            self.ei_index.load(Ordering::Relaxed)
        }
        fn set_ei_index(&self, idx: u32) {
            self.ei_index.store(idx, Ordering::Relaxed);
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
            self.index as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
    }

    #[derive(Debug)]
    struct ObjectTestDescr {
        index: u32,
        ei_index: AtomicU32,
    }

    impl Descr for ObjectTestDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn get_ei_index(&self) -> u32 {
            self.ei_index.load(Ordering::Relaxed)
        }
        fn set_ei_index(&self, idx: u32) {
            self.ei_index.store(idx, Ordering::Relaxed);
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
            self.index as usize * 8
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
    struct ImmutableDescr {
        index: u32,
        ei_index: AtomicU32,
    }

    impl Descr for ImmutableDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn get_ei_index(&self) -> u32 {
            self.ei_index.load(Ordering::Relaxed)
        }
        fn set_ei_index(&self, idx: u32) {
            self.ei_index.store(idx, Ordering::Relaxed);
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
            self.index as usize * 8
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
        Arc::new(TestDescr {
            index: idx,
            ei_index: AtomicU32::new(u32::MAX),
        })
    }

    fn immutable_descr(idx: u32) -> DescrRef {
        Arc::new(ImmutableDescr {
            index: idx,
            ei_index: AtomicU32::new(u32::MAX),
        })
    }

    fn object_descr(idx: u32) -> DescrRef {
        Arc::new(ObjectTestDescr {
            index: idx,
            ei_index: AtomicU32::new(u32::MAX),
        })
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
        heap.cache_field(OpRef::int_op(0), &descr_a);
        heap.cache_field(OpRef::int_op(0), &descr_b);

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
        preamble_op.pos.set(source);
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
        let object_box = ctx.ensure_box(object).unwrap();
        ctx.set_ptr_info(&object_box, PtrInfo::instance(None, None));
        ctx.with_ptr_info_mut(&object_box, |info| {
            info.set_preamble_field(
                OptHeap::field_slot_index(descr),
                PreambleOp {
                    op: source,
                    invented_name: false,
                    preamble_op,
                },
            );
        })
        .unwrap();
        let _ = resolved;
        heap.import_cached_fields(&[(object, descr.clone(), resolved)], ctx);
    }

    /// Call descriptor with default EffectInfo (non-random, non-elidable).
    /// Test helper for "residual call with unknown heap effects". Mirrors
    /// PyPy `effectinfo.MOST_GENERAL` (`effectinfo.py:271-273`):
    /// `extraeffect=EF_RANDOM_EFFECTS`, all six raw sets `None`, all six
    /// bitstrings `None`, `can_invalidate=True`. Production sites and
    /// tests use this whenever no analyzer info is available — invalidation
    /// flows through `dispatch_emit:2631/2766 call_has_random_effects ==
    /// true → clean_caches`, mirroring `heap.py:551`'s top-level
    /// random-effects branch. The previous saturated-bitstring fallback
    /// (`CanRaise + raw=Some(empty) + bitstring=Some(0xff;8)`) was a
    /// pyre-only shape PyPy never produces — `effectinfo_from_writeanalyze`
    /// (`effectinfo.py:285`) force-promotes top_set inputs to RandomEffects
    /// before constructing the EI.
    fn plain_call_descr(idx: u32) -> DescrRef {
        Arc::new(majit_ir::SimpleCallDescr::new(
            idx,
            vec![],
            majit_ir::Type::Void,
            false,
            0,
            EffectInfo::MOST_GENERAL,
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
            // Slice P6a: type-tag op.pos via the result-type-aware factory
            // so the OpRef variant carries `Box.type` (history.py:220 +
            // resoperation.py:1693) at priority 0 in `opref_type`. The
            // intrinsic `op.type_` field set at `Op::new` is the
            // authoritative source, surfaced via `Op::result_type`.
            op.pos.set(OpRef::op_typed(i as u32, op.result_type()));
            if op.opcode.is_guard() && !op.has_descr() {
                op.setdescr(crate::compile::make_resume_guard_descr_typed(Vec::new()));
            }
        }
    }

    /// Run a single OptHeap pass over the given ops.
    ///
    /// Uses num_inputs=1024 so that high-numbered OpRef values used as
    /// input arguments in tests (e.g. OpRef::int_op(100), OpRef::int_op(500)) are treated
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
    /// `make_equal_to(getfield_gc_i.pos:Int, cached_value:Ref)` trips the
    /// `make_equal_to` cross-type assertion introduced for the Box.type
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
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024)
    }

    // ── Test 1: SETFIELD then GETFIELD → read from cache ──

    #[test]
    fn test_setfield_then_getfield_cached() {
        // setfield_gc(p0, i1, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d0)   <- should be eliminated, replaced by i1
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_int(101)],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            // Terminate trace so lazy set is forced.
            Op::new(OpCode::Jump, &[]),
        ];
        // OpRef::input_arg_int(101) is the Int value stored into the Int-typed field; the
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
        // `inputarg_from_tp(arg.type)` per opencoder.py:259 — base is a Ref
        // pointer, cached_value is the Int field result of GetfieldGcI.
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref, Type::Int]);
        let p0 = OpRef::input_arg_typed(0, Type::Ref);
        let p1 = OpRef::input_arg_typed(1, Type::Int);
        let cached_op_key = OpRef::int_op(100);
        initialize_imported_short_heap_field(
            &mut heap,
            &mut ctx,
            p0,
            &d,
            cached_op_key,
            p1,
            OpCode::GetfieldGcI,
        );

        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[p0], d);
        op.pos.set(pos2);

        let result = heap.optimize_getfield(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(pos2), p1);
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
        // `inputarg_from_tp(arg.type)` per opencoder.py:259 — base and
        // cached_value are both Ref (GetfieldGcR result type).
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref, Type::Ref]);
        let p0 = OpRef::input_arg_typed(0, Type::Ref);
        let p1 = OpRef::input_arg_typed(1, Type::Ref);
        let cached_op_key = OpRef::int_op(100);
        initialize_imported_short_heap_field(
            &mut heap,
            &mut ctx,
            p0,
            &d_head,
            cached_op_key,
            p1,
            OpCode::GetfieldGcR,
        );

        // First getfield on head: consumes the import, caches the value.
        let pos2 = ctx.reserve_pos_typed(Type::Ref);
        let mut op1 = Op::with_descr(OpCode::GetfieldGcR, &[p0], d_head.clone());
        op1.pos.set(pos2);
        let result1 = heap.optimize_getfield(&op1, &mut ctx);
        assert!(matches!(result1, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(pos2), p1);

        // A call invalidates all mutable field caches.
        heap.clean_caches(&mut ctx);

        // Second getfield on head after invalidation: must NOT return the
        // stale preamble value.  The import was consumed, so it should emit.
        let pos3 = ctx.reserve_pos_typed(Type::Ref);
        let mut op2 = Op::with_descr(OpCode::GetfieldGcR, &[p0], d_head.clone());
        op2.pos.set(pos3);
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
        // base is promoted to an Int constant via make_constant — Int input
        // tag matches the constant value type.
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Int]);
        let p0 = OpRef::input_arg_typed(0, Type::Int);
        ctx.make_constant(p0, majit_ir::Value::Int(1));

        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[p0], d);
        op.pos.set(pos1);

        let result = heap.optimize_getfield(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Emit(_)));
        assert_eq!(ctx.get_box_replacement(pos1), pos1);
    }

    // ── Test 2: Two GETFIELDs on same object/field → second eliminated ──

    #[test]
    fn test_getfield_read_after_read() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d0)   <- eliminated, reuse i1
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_ref(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_ref(102)],
                d.clone(),
            ),
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_ref(101)],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_int(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(200), OpRef::input_arg_int(201)],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(200)], d.clone()),
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
        let idx = OpRef::int_op(50);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(100), idx, OpRef::int_op(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(100), idx],
                d.clone(),
            ),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        let idx = OpRef::int_op(50);
        let mut op = Op::with_descr(
            OpCode::GetarrayitemGcI,
            &[OpRef::ref_op(100), idx],
            d.clone(),
        );
        op.pos.set(OpRef::int_op(200));

        let mut ctx = OptContext::new(256);
        ctx.make_constant(idx, majit_ir::Value::Int(3));
        let pos100 = ctx
            .ensure_box(OpRef::ref_op(100))
            .expect("body-namespace OpRef must have a BoxRef slot");
        ctx.set_ptr_info(&pos100, PtrInfo::virtual_array(d, 8, false));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Emit(_)));
        let arr_box = ctx
            .get_box_replacement_box(OpRef::int_op(100))
            .expect("array box");
        assert_eq!(
            ctx.peek_ptr_info(&arr_box)
                .and_then(|info| info.getitem(3))
                .and_then(|e| e.as_opref()),
            Some(OpRef::int_op(200))
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
        let idx = OpRef::int_op(50);
        let op = Op::with_descr(
            OpCode::SetarrayitemGc,
            &[OpRef::int_op(100), idx, OpRef::int_op(101)],
            d.clone(),
        );

        let mut ctx = OptContext::new(256);
        ctx.make_constant(idx, majit_ir::Value::Int(3));
        let pos100 = ctx
            .ensure_box(OpRef::int_op(100))
            .expect("body-namespace OpRef must have a BoxRef slot");
        ctx.set_ptr_info(&pos100, PtrInfo::virtual_array(d.clone(), 8, false));

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
        let arr_box = ctx
            .get_box_replacement_box(OpRef::int_op(100))
            .expect("array box");
        assert_eq!(
            ctx.peek_ptr_info(&arr_box)
                .and_then(|info| info.getitem(3))
                .and_then(|e| e.as_opref()),
            Some(OpRef::int_op(101))
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_int(101)],
                d.clone(),
            ),
            Op::new(OpCode::GuardTrue, &[OpRef::input_arg_int(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        // resoperation.py:719 InputArgInt — guard_true expects an int
        // condition box (truthiness via INT_IS_TRUE / GUARD_TRUE family).
        let result = run_heap_opt_typed(&mut ops, &[101, 200]);

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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_ref(101)],
                d0,
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d1),
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
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldRaw,
                &[OpRef::input_arg_ref(200), OpRef::input_arg_ref(201)],
                d1,
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d0),
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
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            // assign_positions stamps GETFIELD's pos as IntOp(0); the
            // setfield's value arg must match that variant for the
            // store-load elision to recognize identity.
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::int_op(0)],
                d.clone(),
            ),
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
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(0)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d),
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
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef::input_arg_ref(100)], d.clone()),
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
        let idx = OpRef::int_op(50);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(100), idx, OpRef::int_op(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(100), idx, OpRef::int_op(102)],
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::new(OpCode::IntAddOvf, &[OpRef::int_op(0), OpRef::int_op(0)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d),
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_int(101)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_int(102)],
                d1.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d1.clone(),
            ),
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
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d_immut.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d_mut.clone(),
            ),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d_immut.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d_mut.clone(),
            ),
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
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef::input_arg_ref(100)], d.clone()),
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
        // history.py:182 PtrInfo applies to ref-typed boxes; the field
        // descr is Type::Ref so the field source is ref-typed too.
        pass.cache_field(OpRef::ref_op(100), &descr);

        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::ref_op(100),
            OpRef::ref_op(101),
        ]);
        // Register input args so produce_arg can resolve them.
        sb.add_short_input_arg(OpRef::ref_op(100), majit_ir::Type::Ref);
        sb.add_short_input_arg(OpRef::ref_op(101), majit_ir::Type::Ref);
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        // Seed PtrInfo._fields[idx] with the cached value so the
        // produce_potential_short_preamble_ops read path can find it.
        use crate::optimizeopt::info::PtrInfo;
        let pos100 = ctx
            .ensure_box(OpRef::ref_op(100))
            .expect("body-namespace OpRef must have a BoxRef slot");
        ctx.set_ptr_info(&pos100, PtrInfo::instance(None, None));
        ctx.with_ptr_info_mut(&pos100, |info| {
            info.setfield(descr.index(), OpRef::ref_op(101));
        })
        .unwrap();
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let produced = sb.produced_ops(&mut ctx);

        // Filter to heap-produced ops (exclude SameAs* from add_short_input_arg).
        let heap_ops: Vec<_> = produced
            .iter()
            .filter(|(_, p)| {
                !matches!(
                    p.preamble_op.opcode,
                    OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
                )
            })
            .collect();
        assert_eq!(heap_ops.len(), 1);
        assert_eq!(heap_ops[0].1.preamble_op.opcode, OpCode::GetfieldGcR);
    }

    // ── Test 22: Immutable field survives multiple calls ──

    #[test]
    fn test_immutable_field_survives_multiple_calls() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(201)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(200)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(300)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()), // cached
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(200)], d.clone()), // cached
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
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(200), OpRef::input_arg_ref(20)],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d.clone()),
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::ref_op(0), OpRef::input_arg_ref(10)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[OpRef::input_arg_ref(200)],
                plain_call_descr(100),
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::ref_op(0)], d.clone()),
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::ref_op(0), OpRef::input_arg_ref(10)],
                d.clone(),
            ),
            Op::with_descr(OpCode::CallN, &[OpRef::ref_op(0)], plain_call_descr(100)), // pass p0 to call
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::ref_op(0)], d.clone()),
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::ref_op(0), OpRef::input_arg_ref(10)],
                d0.clone(),
            ), // p0.f0 = i10
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::ref_op(1), OpRef::ref_op(0)],
                d1.clone(),
            ), // p1.f1 = p0 (p0 escapes)
            Op::with_descr(OpCode::CallN, &[OpRef::ref_op(1)], plain_call_descr(100)), // call(p1) (p1 escapes)
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::ref_op(0)], d0.clone()),
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
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d1.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(200), OpRef::input_arg_ref(20)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d1.clone(),
            ), // different field
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
        let idx = OpRef::int_op(50);
        let mut ops = vec![
            Op::new(OpCode::NewArray, &[OpRef::int_op(5)]), // pos=0 -> p0
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), idx, OpRef::int_op(10)],
                d.clone(),
            ),
            Op::with_descr(OpCode::CallN, &[OpRef::int_op(200)], plain_call_descr(100)),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef::ref_op(0), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(3));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::ref_op(0), OpRef::input_arg_ref(10)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[OpRef::input_arg_ref(200)],
                plain_call_descr(100),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[OpRef::input_arg_ref(201)],
                plain_call_descr(101),
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::ref_op(0)], d.clone()),
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
            Op::new(OpCode::GuardNonnull, &[OpRef::ref_op(0)]),
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
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
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
            Op::new(
                OpCode::GuardClass,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_ref(101)],
            ),
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
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
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
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
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::ref_op(0)]),
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
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
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
            Op::new(
                OpCode::GuardNonnullClass,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_ref(101)],
            ),
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
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
            Op::new(
                OpCode::GuardValue,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_ref(101)],
            ),
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
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
            Op::new(OpCode::GuardNonnull, &[OpRef::ref_op(0)]),
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
            Op::new(OpCode::NewArray, &[OpRef::input_arg_ref(5)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::ref_op(0)]),
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
        let d1 = descr(1);
        let call_d = call_descr(
            70,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                // `effectinfo.py:131 _write_descrs_fields = frozenset({descr1})`
                // — Arc-identity raw set carrying d1, paired with the
                // legacy `descr.index()`-keyed bitstring so the fixture
                // works both pre- and post-`compute_bitstrings`.
                _write_descrs_fields: Some(vec![d1]),
                write_descrs_fields: Some(bitstring::make_bitstring(&[1])),
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d0.clone(),
            ),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef::input_arg_ref(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d0),
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
        // PyPy `effectinfo.py:526 mapping.setdefault(...)` would assign
        // `d0.ei_index = 0` once compute_bitstrings has run, since d0 is
        // the only descr in `_write_descrs_fields`. Tests skip
        // compute_bitstrings, so we set it manually so the bitstring at
        // bit 0 actually corresponds to d0's encoding.
        d0.set_ei_index(0);
        let call_d = call_descr(
            71,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                _write_descrs_fields: Some(vec![d0.clone()]),
                write_descrs_fields: Some(bitstring::make_bitstring(&[0])),
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d0.clone(),
            ),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef::input_arg_ref(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::input_arg_ref(100)], d0),
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
            Op::with_descr(OpCode::CallMayForceN, &[OpRef::input_arg_ref(200)], call_d),
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
        let d1 = descr(1);
        let idx = OpRef::int_op(50);
        let call_d = call_descr(
            73,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                _write_descrs_arrays: Some(vec![d1]),
                write_descrs_arrays: Some(bitstring::make_bitstring(&[1])),
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(100), idx],
                d0.clone(),
            ),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef::int_op(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef::int_op(100), idx], d0),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        // d0 is the only descr in `_write_descrs_arrays`, so PyPy
        // `effectinfo.py:526` would assign `d0.ei_index = 0`. Tests skip
        // compute_bitstrings, so we set it manually.
        d0.set_ei_index(0);
        let idx = OpRef::int_op(50);
        let call_d = call_descr(
            74,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                _write_descrs_arrays: Some(vec![d0.clone()]),
                write_descrs_arrays: Some(bitstring::make_bitstring(&[0])),
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(100), idx],
                d0.clone(),
            ),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef::int_op(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef::int_op(100), idx], d0),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        // p_src = OpRef::int_op(100), p_dst = OpRef::int_op(200)
        // setarrayitem_gc(p_dst, idx=0, val=i10)   <- cache dest[0]
        // setarrayitem_gc(p_dst, idx=5, val=i11)   <- cache dest[5]
        // call_n(arraycopy_func, p_src, p_dst, src_start=0, dst_start=2, length=3)
        //   -> copies to dest[2..5], so dest[0] survives, dest[5] survives
        // getarrayitem_gc_i(p_dst, idx=0)           <- still cached
        // getarrayitem_gc_i(p_dst, idx=5)           <- still cached
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx0 = OpRef::int_op(60);
        let idx5 = OpRef::int_op(61);
        let dst_start_ref = OpRef::int_op(62);
        let length_ref = OpRef::int_op(63);
        let src_start_ref = OpRef::int_op(64);

        let mut ops = vec![
            // pos=0: setarrayitem_gc(dst, idx=0, val)
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(200), idx0, OpRef::int_op(10)],
                d.clone(),
            ),
            // pos=1: setarrayitem_gc(dst, idx=5, val)
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(200), idx5, OpRef::int_op(11)],
                d.clone(),
            ),
            // pos=2: call_n(func, src, dst, src_start, dst_start, length)
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef::int_op(300),
                    OpRef::int_op(100),
                    OpRef::int_op(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            // pos=3: getarrayitem_gc_i(dst, idx=0)
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(200), idx0],
                d.clone(),
            ),
            // pos=4: getarrayitem_gc_i(dst, idx=5)
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(200), idx5],
                d.clone(),
            ),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        let idx0 = OpRef::int_op(60);
        let dst_start_ref = OpRef::int_op(62);
        let length_ref = OpRef::int_op(63); // NOT constant
        let src_start_ref = OpRef::int_op(64);

        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(200), idx0, OpRef::int_op(10)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef::int_op(300),
                    OpRef::int_op(100),
                    OpRef::int_op(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(200), idx0],
                d.clone(),
            ),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        let idx = OpRef::int_op(60);
        let dst_start_ref = OpRef::int_op(62);
        let length_ref = OpRef::int_op(63);
        let src_start_ref = OpRef::int_op(64);

        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(200), idx],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef::int_op(300),
                    OpRef::int_op(100),
                    OpRef::int_op(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(200), idx],
                d.clone(),
            ),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        let idx = OpRef::int_op(60);
        let dst_start_ref = OpRef::int_op(62);
        let length_ref = OpRef::int_op(63);
        let src_start_ref = OpRef::int_op(64);

        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(400), idx],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef::int_op(300),
                    OpRef::int_op(100),
                    OpRef::int_op(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(400), idx],
                d.clone(),
            ),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        let idx0 = OpRef::int_op(60);
        let dst_start_ref = OpRef::int_op(62);
        let length_ref = OpRef::int_op(63);
        let src_start_ref = OpRef::int_op(64);

        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::int_op(500)], field_d.clone()),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(200), idx0, OpRef::int_op(10)],
                array_d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef::int_op(300),
                    OpRef::int_op(100),
                    OpRef::int_op(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef::int_op(500)], field_d),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::input_arg_ref(100), OpRef::input_arg_ref(101)],
                d.clone(),
            ),
            Op::new(
                OpCode::GcLoadI,
                &[
                    OpRef::input_arg_ref(200),
                    OpRef::input_arg_ref(8),
                    OpRef::input_arg_ref(4),
                ],
            ),
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
            Op::new(
                OpCode::GcLoadI,
                &[
                    OpRef::input_arg_ref(100),
                    OpRef::input_arg_ref(8),
                    OpRef::input_arg_ref(4),
                ],
            ),
            Op::new(OpCode::GuardNonnull, &[OpRef::input_arg_ref(100)]),
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
            Op::with_descr(OpCode::QuasiimmutField, &[OpRef::input_arg_ref(100)], d0),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d1.clone(),
            ),
            Op::new(OpCode::CallN, &[OpRef::input_arg_ref(200)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[OpRef::input_arg_ref(100)],
                d1.clone(),
            ),
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
        let idx = OpRef::int_op(60);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(100), idx, OpRef::int_op(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(100), idx],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(5)); // byte index 5

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        let idx5 = OpRef::int_op(60);
        let idx6 = OpRef::int_op(61);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::int_op(100), idx5, OpRef::int_op(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(100), idx6],
                d.clone(),
            ),
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
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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
        let idx = OpRef::int_op(60);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(100), idx],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[OpRef::int_op(100), idx],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(3));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(i, ctx.get_box_replacement(resolved.arg(i)));
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

    /// `resoperation.py:1044` lists `ARRAYLEN_GC` inside the
    /// `_ALWAYS_PURE_FIRST.._ALWAYS_PURE_LAST` band; CSE of always-pure
    /// ops is `optimizeopt/pure.py:316`'s `_pure_operations[opnum]`
    /// table, not heap.py.  This test wires `OptPure` to confirm the
    /// dedup path; running with `OptHeap` alone would (correctly) leave
    /// both reads in place — heap.py has no parallel cache.
    #[test]
    fn test_arraylen_caching_via_optpure() {
        let d = descr(42);
        let mut ops = vec![
            {
                let mut op = Op::new(OpCode::ArraylenGc, &[OpRef::int_op(100)]);
                op.setdescr(d.clone());
                op
            },
            {
                let mut op = Op::new(OpCode::ArraylenGc, &[OpRef::int_op(100)]);
                op.setdescr(d);
                op
            },
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::pure::OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);
        let len_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ArraylenGc)
            .count();
        assert_eq!(
            len_count, 1,
            "duplicate ARRAYLEN_GC should be cached by OptPure"
        );
    }

    /// heap.py:278-298 ArrayCachedItem._cannot_alias_via_content — two
    /// arrays with the same descr + length but differing constant items
    /// must be reported as unable to alias.
    #[test]
    fn test_cannot_alias_via_content_different_constants() {
        use crate::optimizeopt::OptContext;
        use crate::optimizeopt::info::{ArrayPtrInfo, FieldEntry};
        use crate::optimizeopt::intutils::IntBound;

        // `inputarg_from_tp(arg.type)` per opencoder.py:259 — both arrays are
        // Ref-typed inputargs.
        let mut ctx = OptContext::with_inputarg_types(64, &[Type::Ref, Type::Ref]);

        let arr_descr = descr(50);
        let op1 = OpRef::input_arg_typed(0, Type::Ref);
        let op2 = OpRef::input_arg_typed(1, Type::Ref);

        let const_10 = ctx.emit_constant_int(10);
        let const_20 = ctx.emit_constant_int(20);
        let const_30 = ctx.emit_constant_int(30);

        let op1_box = ctx
            .ensure_box(op1)
            .expect("body-namespace OpRef must have a BoxRef slot");
        let op2_box = ctx
            .ensure_box(op2)
            .expect("body-namespace OpRef must have a BoxRef slot");
        ctx.set_ptr_info(
            &op1_box,
            PtrInfo::Array(ArrayPtrInfo {
                descr: arr_descr.clone(),
                lenbound: IntBound::from_constant(2),
                items: vec![FieldEntry::Value(const_10), FieldEntry::Value(const_20)],
                last_guard_pos: -1,
            }),
        );
        ctx.set_ptr_info(
            &op2_box,
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

    fn dict_lookup_descr(idx: u32, extra0: DescrRef, extra1: DescrRef) -> DescrRef {
        Arc::new(TestCallDescr {
            idx,
            effect: EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                oopspecindex: OopSpecIndex::DictLookup,
                extradescrs: Some(vec![extra0, extra1]),
                ..Default::default()
            },
        })
    }

    /// heap.py:480-528 parity: FLAG_LOOKUP deduplicates consecutive dict lookups.
    #[test]
    fn test_dict_lookup_cache_flag_lookup() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        let mut ctx = OptContext::new(256);

        // Build args: [func_addr, dict, key, hash, flag=FLAG_LOOKUP(0)]
        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let key = OpRef::input_arg_typed(1, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag = ctx.make_constant_int(0); // FLAG_LOOKUP

        // First lookup — not cached yet, should return false (emit).
        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(OpCode::CallI, &[func_addr, dict, key, hash, flag]);
        op1.setdescr(descr.clone());
        op1.pos.set(pos1);
        assert!(!heap._optimize_call_dict_lookup(&op1, &mut ctx));

        // Second lookup with same dict+key — should be cached.
        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op2 = Op::new(OpCode::CallI, &[func_addr, dict, key, hash, flag]);
        op2.setdescr(descr.clone());
        op2.pos.set(pos2);
        assert!(heap._optimize_call_dict_lookup(&op2, &mut ctx));
        assert_eq!(ctx.get_box_replacement(pos2), pos1);
        assert!(heap.last_emitted_removed);
    }

    /// heap.py:495-499 parity: FLAG_STORE reuses only if cached value >= 0.
    #[test]
    fn test_dict_lookup_cache_flag_store_nonneg() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        let mut ctx = OptContext::new(256);

        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let key = OpRef::input_arg_typed(1, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag_lookup = ctx.make_constant_int(0); // FLAG_LOOKUP
        let flag_store = ctx.make_constant_int(1); // FLAG_STORE

        // Seed cache with FLAG_LOOKUP, then try FLAG_STORE.
        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(OpCode::CallI, &[func_addr, dict, key, hash, flag_lookup]);
        op1.setdescr(descr.clone());
        op1.pos.set(pos1);
        // Pretend the result is known >= 0.
        // OpRef → BoxRef shim until this caller migrates (Phase D-2).
        // `reserve_pos_typed` skips extending box_pool when the pool is
        // empty (test baseline), so lazy-allocate here.
        let pos1_box = ctx
            .ensure_box(pos1)
            .expect("body-namespace OpRef must have a BoxRef slot");
        ctx.setintbound(
            &pos1_box,
            &crate::optimizeopt::intutils::IntBound::from_constant(5),
        );
        assert!(!heap._optimize_call_dict_lookup(&op1, &mut ctx));

        // FLAG_STORE with known non-negative cached value → reuse.
        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op2 = Op::new(OpCode::CallI, &[func_addr, dict, key, hash, flag_store]);
        op2.setdescr(descr.clone());
        op2.pos.set(pos2);
        assert!(heap._optimize_call_dict_lookup(&op2, &mut ctx));
        assert_eq!(ctx.get_box_replacement(pos2), pos1);
    }

    /// heap.py:390 parity: clean_caches clears cached_dict_reads.
    #[test]
    fn test_dict_lookup_cache_cleared_by_clean_caches() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        let mut ctx = OptContext::new(256);

        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let key = OpRef::input_arg_typed(1, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag = ctx.make_constant_int(0);

        // Seed cache.
        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(OpCode::CallI, &[func_addr, dict, key, hash, flag]);
        op1.setdescr(descr.clone());
        op1.pos.set(pos1);
        heap._optimize_call_dict_lookup(&op1, &mut ctx);
        assert!(!heap.cached_dict_reads.is_empty());

        // clean_caches should clear it.
        heap.clean_caches(&mut ctx);
        assert!(heap.cached_dict_reads.is_empty());

        // Second lookup after clean — should NOT be cached.
        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op2 = Op::new(OpCode::CallI, &[func_addr, dict, key, hash, flag]);
        op2.setdescr(descr.clone());
        op2.pos.set(pos2);
        assert!(!heap._optimize_call_dict_lookup(&op2, &mut ctx));
    }

    /// util.py:100/127 args_dict() / args_eq parity: same_box treats two
    /// distinct ConstInt slots holding the same value as equal
    /// (history.py:204 Const.same_box → same_constant). Two consecutive
    /// dict lookups whose constant key arg is encoded via different const
    /// slots must hit the cache.
    #[test]
    fn test_dict_lookup_cache_key_same_box_for_constants() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        let mut ctx = OptContext::new(256);

        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag = ctx.make_constant_int(0);

        // Two distinct ConstInt slots, same value. RPython's args_dict()
        // hashes them via _get_hash_() (value-based for Const), so they
        // collide as the same key.
        let key_a = ctx.make_constant_int(7);
        let key_b = ctx.make_constant_int(7);
        assert_ne!(key_a, key_b, "make_constant_int must mint fresh slots");

        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(OpCode::CallI, &[func_addr, dict, key_a, hash, flag]);
        op1.setdescr(descr.clone());
        op1.pos.set(pos1);
        assert!(!heap._optimize_call_dict_lookup(&op1, &mut ctx));

        // Same value via a different const slot — must hit the cache.
        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op2 = Op::new(OpCode::CallI, &[func_addr, dict, key_b, hash, flag]);
        op2.setdescr(descr.clone());
        op2.pos.set(pos2);
        assert!(heap._optimize_call_dict_lookup(&op2, &mut ctx));
        assert_eq!(ctx.get_box_replacement(pos2), pos1);
    }

    /// optimizer.py:84-87 parity: every Optimization.emit overwrite path —
    /// including the PASS_OP_ON return at line 87 — sets
    /// `last_emitted_operation = op` first (line 86). Heap arms returning
    /// PassOn (NEW family, COND_CALL_N) hit that path in RPython, so a
    /// REMOVED sentinel left over from `_optimize_CALL_DICT_LOOKUP` must
    /// be cleared by the time a downstream GUARD_NO_EXCEPTION is dispatched
    /// — otherwise the guard is wrongly removed across an intervening op.
    #[test]
    fn test_pass_on_clears_last_emitted_removed_flag() {
        let mut heap = OptHeap::new();
        heap.setup();
        let mut ctx = OptContext::new(256);

        // Seed the REMOVED sentinel as if a DICT_LOOKUP cache hit just fired.
        heap.last_emitted_removed = true;

        // OpCode::New is a PassOn arm in heap.rs (allocation tracking only).
        let new_pos = ctx.reserve_pos_typed(Type::Ref);
        let mut new_op = Op::new(OpCode::New, &[]);
        new_op.pos.set(new_pos);
        let result = heap.propagate_forward(&new_op, &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
        assert!(
            !heap.last_emitted_removed,
            "PassOn must clear last_emitted_removed (optimizer.py:86 fires before line 87 PASS_OP_ON)"
        );

        // Now a GUARD_NO_EXCEPTION must be emitted, not removed.
        let mut guard = Op::new(OpCode::GuardNoException, &[]);
        guard.pos.set(ctx.reserve_pos_typed(Type::Void));
        let guard_result = heap.propagate_forward(&guard, &mut ctx);
        assert!(
            matches!(guard_result, OptimizationResult::Emit(_)),
            "GUARD_NO_EXCEPTION after a PassOn-emitted op must NOT be removed"
        );
    }

    /// FLAG_DELETE (2+) should never cache or reuse.
    #[test]
    fn test_dict_lookup_cache_flag_delete_no_cache() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        let mut ctx = OptContext::new(256);

        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let key = OpRef::input_arg_typed(1, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag_delete = ctx.make_constant_int(2); // FLAG_DELETE

        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(OpCode::CallI, &[func_addr, dict, key, hash, flag_delete]);
        op1.setdescr(descr.clone());
        op1.pos.set(pos1);
        assert!(!heap._optimize_call_dict_lookup(&op1, &mut ctx));
        assert!(heap.cached_dict_reads.is_empty());
    }
}
