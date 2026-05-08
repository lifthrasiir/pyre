//! Resume data: encodes the mapping from guard fail_args to full interpreter state.
//!
//! When a guard fails, the JIT needs to reconstruct the interpreter's full state
//! (program counter, local variables, stack contents) from the values stored in
//! the DeadFrame. Resume data provides this mapping.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/resume.py`.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use majit_backend::{
    ExitFrameLayout, ExitPendingFieldLayout, ExitRecoveryLayout, ExitValueSourceLayout,
    ExitVirtualLayout,
};
use majit_ir::{Const, GcRef, OpRef, Type};

/// resume.py:656-670: element kind from arraydescr.
/// 0=ref (is_array_of_pointers), 1=int, 2=float (is_array_of_floats).
fn array_kind_from_descr(arraydescr: Option<&majit_ir::DescrRef>) -> u8 {
    arraydescr
        .and_then(|d| d.as_array_descr())
        .map(|ad| {
            if ad.is_array_of_pointers() {
                0u8
            } else if ad.is_array_of_floats() {
                2u8
            } else {
                1u8
            }
        })
        .unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════
// RPython resume.py:96-139 — structural port (i16 tags).
// ═══════════════════════════════════════════════════════════════

// resume.py:96-97
#[derive(Debug)]
pub struct TagOverflow;

// resume.py:99-104
pub fn tag(value: i32, tagbits: u8) -> Result<i16, TagOverflow> {
    debug_assert!(tagbits <= 3);
    let sx = value >> 13;
    if sx != 0 && sx != -1 {
        return Err(TagOverflow);
    }
    Ok(((value << 2) | tagbits as i32) as i16)
}

// resume.py:106-109
pub fn untag(value: i16) -> (i32, u8) {
    let widened = value as i32;
    let tagbits = (widened & TAGMASK as i32) as u8;
    (widened >> 2, tagbits)
}

// resume.py:111-113
#[inline]
pub fn tagged_eq(x: i16, y: i16) -> bool {
    (x as i32) == (y as i32)
}

// resume.py:115-121
pub fn tagged_list_eq(tl1: &[i16], tl2: &[i16]) -> bool {
    if tl1.len() != tl2.len() {
        return false;
    }
    tl1.iter().zip(tl2.iter()).all(|(&a, &b)| tagged_eq(a, b))
}

// resume.py:123-132
pub const TAGCONST: u8 = 0;
pub const TAGINT: u8 = 1;
pub const TAGBOX: u8 = 2;
pub const TAGVIRTUAL: u8 = 3;
const TAGMASK: u8 = 3;

pub const UNASSIGNED: i16 = ((-1i32 << 13) << 2 | TAGBOX as i32) as i16;
pub const UNASSIGNEDVIRTUAL: i16 = ((-1i32 << 13) << 2 | TAGVIRTUAL as i32) as i16;
pub const NULLREF: i16 = ((-1i32 << 2) | TAGCONST as i32) as i16;
pub const UNINITIALIZED_TAG: i16 = ((-2i32 << 2) | TAGCONST as i32) as i16;
pub const TAG_CONST_OFFSET: i32 = 0;

/// Ordered livebox map: OpRef → i16 tag.
///
/// resume.py:137/370: RPython uses `dict` keyed by the actual Box object.
/// In Python 3 that dict is insertion-ordered, and `_number_virtuals`
/// iterates it directly. `IndexMap<OpRef, i16>` mirrors both properties:
/// the full typed OpRef variant participates in identity
/// (resoperation.py:38 `same_box`) and iteration follows first insertion
/// order. A raw-position Vec would collapse `InputArgInt(0)` and
/// `IntOp(0)`, which are distinct RPython Box objects.
pub struct LiveboxMap {
    entries: IndexMap<majit_ir::OpRef, i16>,
}

impl LiveboxMap {
    pub fn new() -> Self {
        Self {
            entries: IndexMap::new(),
        }
    }

    #[inline(always)]
    pub fn get(&self, opref: majit_ir::OpRef) -> Option<i16> {
        self.entries.get(&opref).copied()
    }

    #[inline(always)]
    pub fn insert(&mut self, opref: majit_ir::OpRef, value: i16) {
        self.entries.insert(opref, value);
    }

    #[inline(always)]
    pub fn contains_key(&self, opref: majit_ir::OpRef) -> bool {
        self.entries.contains_key(&opref)
    }

    /// Iterate over all (typed OpRef, tag) pairs in RPython dict insertion
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (majit_ir::OpRef, i16)> + '_ {
        self.entries.iter().map(|(opref, v)| (*opref, *v))
    }
}

impl Default for LiveboxMap {
    fn default() -> Self {
        Self::new()
    }
}

// resume.py:134-139
pub struct NumberingState {
    pub writer: crate::resumecode::Writer,
    pub liveboxes: LiveboxMap,
    pub num_boxes: i32,
    pub num_virtuals: i32,
    /// RPython Box.type parity: type of each TAGBOX livebox, captured at
    /// numbering time when env.get_type() is called. Eliminates the need
    /// for post-hoc type inference cascades in store_final_boxes_in_guard.
    ///
    /// Keyed by the typed OpRef (resoperation.py:719-739 InputArg{Int,
    /// Ref,Float}, resoperation.py:564-638 *Op mixins) so that
    /// `InputArgRef(0)` and `RefOp(0)` do not collapse onto the same
    /// raw u32 — pyre's flat-OpRef stand-in for PyPy's `box is box`
    /// identity. See `LiveboxMap` (resume.rs:98) for the matching
    /// typed-key convention.
    pub livebox_types: std::collections::HashMap<majit_ir::OpRef, majit_ir::Type>,
}

impl NumberingState {
    pub fn new(size: usize) -> Self {
        NumberingState {
            writer: crate::resumecode::Writer::new(size),
            liveboxes: LiveboxMap::new(),
            num_boxes: 0,
            num_virtuals: 0,
            livebox_types: std::collections::HashMap::new(),
        }
    }
    pub fn append_short(&mut self, item: i16) {
        self.writer.append_short(item as i32);
    }
    pub fn append_int(&mut self, item: i64) {
        self.writer.append_int(item);
    }
    pub fn patch_current_size(&mut self, index: usize) {
        self.writer.patch_current_size(index);
    }
    pub fn create_numbering(&self) -> Vec<u8> {
        self.writer.create_numbering()
    }
}

/// RPython snapshot: the state captured at a guard point.
/// Corresponds to RPython's SnapshotIterator output:
/// snapshot_iter.vable_array, snapshot_iter.vref_array, snapshot_iter.framestack.
///
/// NOTE: RPython does not have this struct. It uses `trace.get_snapshot_iter(position)`
/// which returns a lazy iterator over the trace's snapshot data (opencoder.py).
/// We use an eager struct because pyre's tracing records fail_args directly
/// on guard ops rather than using RPython's snapshot log format.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Virtualizable field boxes (resume.py:234-241).
    pub vable_array: Vec<SnapshotBox>,
    /// Virtualref pairs (resume.py:243-247). Length must be even.
    pub vref_array: Vec<SnapshotBox>,
    /// Frame chain (resume.py:249-253). Multiple frames for inlined calls.
    pub framestack: Vec<SnapshotFrame>,
}

/// A snapshot entry corresponding to one RPython Box.
///
/// RPython carries `box.type` on the Box object itself. Pyre's typed
/// `OpRef` enum (resoperation.py:719/727/739 InputArg{Int,Float,Ref},
/// resoperation.py:564-638 *Op mixin variants) carries the same type
/// tag intrinsically, so SnapshotBox copies it from the OpRef variant
/// at construction time. The explicit `tp` field stays around so the
/// SnapshotBox API can answer `box.type` without re-decoding the
/// variant on every read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnapshotBox {
    pub opref: majit_ir::OpRef,
    pub tp: Option<majit_ir::Type>,
}

impl SnapshotBox {
    pub fn untyped(opref: majit_ir::OpRef) -> Self {
        SnapshotBox { opref, tp: None }
    }

    pub fn typed(opref: majit_ir::OpRef, tp: majit_ir::Type) -> Self {
        SnapshotBox {
            opref,
            tp: Some(tp),
        }
    }

    pub fn map_opref(self, f: impl FnOnce(majit_ir::OpRef) -> majit_ir::OpRef) -> Self {
        SnapshotBox {
            opref: f(self.opref),
            tp: self.tp,
        }
    }
}

impl From<majit_ir::OpRef> for SnapshotBox {
    fn from(opref: majit_ir::OpRef) -> Self {
        SnapshotBox::untyped(opref)
    }
}

/// One frame in a snapshot's frame chain.
#[derive(Debug, Clone)]
pub struct SnapshotFrame {
    /// Index into the jitcode table (resume.py:250 jitcode_index).
    pub jitcode_index: i32,
    /// Bytecode program counter (resume.py:250 pc).
    pub pc: i32,
    /// Live boxes for this frame's registers (resume.py:253).
    pub boxes: Vec<SnapshotBox>,
}

impl Snapshot {
    /// Create a simple single-frame snapshot (pyre common case).
    ///
    /// `jitcode_index` identifies the code this frame is running — the
    /// index into `METAINTERP_SD.jitcodes`. Required so the decoder's
    /// `frame_value_count_at(jitcode_index, pc)` query resolves the
    /// frame's liveness in the correct jitcode instead of silently
    /// falling through the pc-out-of-range LiveVars path on `jitcodes[0]`.
    pub fn single_frame(jitcode_index: i32, pc: i32, boxes: Vec<majit_ir::OpRef>) -> Self {
        Self::single_frame_boxes(
            jitcode_index,
            pc,
            boxes.into_iter().map(SnapshotBox::from).collect(),
        )
    }

    pub fn single_frame_boxes(jitcode_index: i32, pc: i32, boxes: Vec<SnapshotBox>) -> Self {
        Snapshot {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            framestack: vec![SnapshotFrame {
                jitcode_index,
                pc,
                boxes,
            }],
        }
    }

    /// Create a multi-frame snapshot from per-frame (jitcode_index, pc,
    /// boxes) tuples. Read-side ordering matches upstream: after
    /// `SnapshotIterator.__init__` calls `self.framestack.reverse()`
    /// (`opencoder.py:217`), `framestack[0]` is the outermost/caller
    /// frame and the last element is the innermost/callee — as asserted
    /// by `test_opencoder.py:123-130` (jc_index=2 at `framestack[0]`,
    /// jc_index=4 at `framestack[1]`). Input tuples for this factory
    /// follow the same caller-first order.
    pub fn multi_frame(frames: Vec<(i32, i32, Vec<majit_ir::OpRef>)>) -> Self {
        Self::multi_frame_boxes(
            frames
                .into_iter()
                .map(|(jitcode_index, pc, boxes)| {
                    (
                        jitcode_index,
                        pc,
                        boxes.into_iter().map(SnapshotBox::from).collect(),
                    )
                })
                .collect(),
        )
    }

    pub fn multi_frame_boxes(frames: Vec<(i32, i32, Vec<SnapshotBox>)>) -> Self {
        Snapshot {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            framestack: frames
                .into_iter()
                .map(|(jitcode_index, pc, boxes)| SnapshotFrame {
                    jitcode_index,
                    pc,
                    boxes,
                })
                .collect(),
        }
    }

    /// Estimated encoded size for NumberingState capacity hint.
    pub fn estimated_size(&self) -> usize {
        let frame_size: usize = self.framestack.iter().map(|f| f.boxes.len() + 2).sum();
        self.vable_array.len() + self.vref_array.len() + frame_size + 4
    }
}

/// Re-export BoxEnv from majit-ir.
pub use majit_ir::BoxEnv;

/// Simple BoxEnv implementation backed by constant/type HashMaps.
/// Used in tests and for simple snapshot numbering.
pub struct SimpleBoxEnv {
    pub constants: HashMap<u32, (i64, majit_ir::Type)>,
    pub replacements: HashMap<u32, majit_ir::OpRef>,
    pub types: HashMap<u32, majit_ir::Type>,
    pub virtuals: std::collections::HashSet<u32>,
    pub virtual_fields: HashMap<u32, majit_ir::VirtualFieldsInfo>,
}

impl SimpleBoxEnv {
    pub fn new() -> Self {
        SimpleBoxEnv {
            constants: HashMap::new(),
            replacements: HashMap::new(),
            types: HashMap::new(),
            virtuals: std::collections::HashSet::new(),
            virtual_fields: HashMap::new(),
        }
    }
}

impl BoxEnv for SimpleBoxEnv {
    fn get_box_replacement(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
        self.replacements
            .get(&opref.raw())
            .copied()
            .unwrap_or(opref)
    }
    fn is_const(&self, opref: majit_ir::OpRef) -> bool {
        self.constants.contains_key(&opref.raw())
    }
    fn get_const(&self, opref: majit_ir::OpRef) -> (i64, majit_ir::Type) {
        self.constants
            .get(&opref.raw())
            .copied()
            .unwrap_or((0, majit_ir::Type::Int))
    }
    fn get_type(&self, opref: majit_ir::OpRef) -> majit_ir::Type {
        self.types
            .get(&opref.raw())
            .copied()
            .unwrap_or(majit_ir::Type::Int)
    }
    fn is_virtual_ref(&self, opref: majit_ir::OpRef) -> bool {
        self.virtuals.contains(&opref.raw())
    }
    fn is_virtual_raw(&self, opref: majit_ir::OpRef) -> bool {
        self.virtuals.contains(&opref.raw())
    }
    fn get_virtual_fields(&self, opref: majit_ir::OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        self.virtual_fields.get(&opref.raw()).cloned()
    }
}

// resume.py:123-132 — tag constants (i64 widened for rd_numb encoding).
// Same values as the i16 TAGCONST/TAGINT/TAGBOX/TAGVIRTUAL above.
const TAGMASK_I64: i64 = TAGMASK as i64;
// resume.py:130 `NULLREF = tag(-1, TAGCONST)` — pre-shift num for
// `Const::Ref(NULL)`. Shared by the i16 `NULLREF` constant and the
// i64 `getconst_i64`/`decode_box` pair.
const ENCODED_NULLREF: i64 = -1;
const ENCODED_UNINITIALIZED: i64 = -2;
const ENCODED_UNAVAILABLE: i64 = -3;

// Two low bits are reserved for the tag.
const INLINE_TAGGED_MIN: i64 = -(1_i64 << 61);
const INLINE_TAGGED_MAX: i64 = (1_i64 << 61) - 1;

/// compile.py:853-876 `ResumeGuardDescr` storage.
///
/// Canonical, guard-owned resume payload (`storage.rd_numb/rd_consts/
/// rd_virtuals/rd_pendingfields`). Shared via `Arc<ResumeStorage>` so
/// every reader — `StoredResumeData`, `StoredExitLayout`, bridge
/// retrace, blackhole resume, GC root walker — observes the **same**
/// pool, matching RPython's guard-owned `ResumeGuardDescr` singleton.
///
/// `rd_consts` uses `UnsafeCell` because the GC root walker
/// (framework.py `root_walker.walk_roots` parity) rewrites `Const::Ref`
/// slots in place during minor collection, and the pyre runtime is
/// single-threaded so `Mutex` overhead is unnecessary.
pub struct ResumeStorage {
    /// resume.py:466 `storage.rd_numb` — packed byte stream (NUMBERING
    /// lltype equivalent). Immutable once installed.
    pub rd_numb: Vec<u8>,
    /// resume.py:467 `storage.rd_consts` — shared constant pool.
    ///
    /// Interior mutability: the minor-collection root walker visits
    /// `Const::Ref` slots to update forwarded GCREFs, so the slice
    /// must remain mutable after the `Arc` is shared. `UnsafeCell`
    /// is sound here because pyre is single-threaded and the walker
    /// holds exclusive access for the duration of each GC cycle.
    pub rd_consts: UnsafeCell<Vec<Const>>,
    /// compile.py:858 `storage.rd_virtuals` — live `RdVirtualInfo`
    /// entries describing virtual objects to materialize on resume.
    pub rd_virtuals: Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>,
    /// resume.py:468 `storage.rd_pendingfields` — pending field writes
    /// replayed during blackhole resume.
    pub rd_pendingfields: Vec<majit_ir::GuardPendingFieldEntry>,
}

// Pyre is single-threaded; UnsafeCell prevents auto-Send/Sync so
// provide them explicitly (matches RPython's non-thread-safe
// ResumeGuardDescr).
unsafe impl Send for ResumeStorage {}
unsafe impl Sync for ResumeStorage {}

impl std::fmt::Debug for ResumeStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let consts_len = unsafe { (*self.rd_consts.get()).len() };
        f.debug_struct("ResumeStorage")
            .field("rd_numb_len", &self.rd_numb.len())
            .field("rd_consts_len", &consts_len)
            .field("rd_virtuals_len", &self.rd_virtuals.len())
            .field("rd_pendingfields_len", &self.rd_pendingfields.len())
            .finish()
    }
}

impl ResumeStorage {
    pub fn new(
        rd_numb: Vec<u8>,
        rd_consts: Vec<Const>,
        rd_virtuals: Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>,
        rd_pendingfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) -> Arc<Self> {
        Arc::new(ResumeStorage {
            rd_numb,
            rd_consts: UnsafeCell::new(rd_consts),
            rd_virtuals,
            rd_pendingfields,
        })
    }

    /// Empty storage (pre-finalization placeholder).
    pub fn empty() -> Arc<Self> {
        Self::new(Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }

    /// Snapshot the constant pool (for readers that need an owned
    /// copy — e.g. legacy `Vec<Const>`-typed APIs before their
    /// migration to the shared storage handle).
    pub fn rd_consts_snapshot(&self) -> Vec<Const> {
        unsafe { (*self.rd_consts.get()).clone() }
    }

    /// Borrow `rd_consts` for reading. Safety: the root walker holds
    /// the only writer and runs during GC, outside of reader scope.
    pub fn rd_consts(&self) -> &[Const] {
        unsafe { &*self.rd_consts.get() }
    }

    /// Internal accessor for the GC root walker. SAFETY: caller must
    /// ensure exclusive access — only the minor-collection walker in
    /// `MetaInterp::walk_rd_consts_refs` uses this.
    pub(crate) unsafe fn rd_consts_mut_for_gc(&self) -> &mut Vec<Const> {
        unsafe { &mut *self.rd_consts.get() }
    }
}

/// resume.py: ResumeGuardDescr storage fields.
///
/// `rd_numb` is a flat encoded numbering section (resume.py:466):
/// 1. items_resume_section (total rd_numb length)
/// 2. count (number of liveboxes, resume.py:921)
/// 3. number of frames
/// 4. per-frame `(pc, slot_count, slot_sources...)`
///
/// Fields match RPython's `ResumeGuardDescr`:
/// - `rd_numb`: encoded numbering (resume.py:466)
/// - `rd_consts`: shared constant pool (resume.py:467)
/// - `rd_virtuals`: live VirtualInfo objects (compile.py:858)
/// - `rd_pendingfields`: pending field writes (resume.py:468)
#[derive(Debug, Clone)]
pub struct EncodedResumeData {
    /// resume.py:466 storage.rd_numb — flat encoded numbering section.
    pub rd_numb: Vec<i64>,
    /// resume.py:467 storage.rd_consts — shared constant pool.
    ///
    /// RPython stores `list[Const]` (history.py:220/261/307). We keep the
    /// same shape so Ref entries stay visible to the minor-collection root
    /// walker (framework.py `root_walker.walk_roots` parity).
    pub rd_consts: Vec<Const>,
    /// resume.py:468 storage.rd_pendingfields — pending field writes.
    pub rd_pendingfields: Vec<EncodedPendingFieldWrite>,
    /// compile.py:858 storage.rd_virtuals — live VirtualInfo objects.
    pub rd_virtuals: Vec<VirtualInfo>,
    /// resume.py:411 liveboxes — compact TAGBOX(n) → original FailArg index.
    /// In RPython, liveboxes[n] is the Box object that was assigned TAGBOX(n).
    /// Here, liveboxes[n] is the original deadframe slot index.
    pub liveboxes: Vec<usize>,
    /// Per-frame slot count — equivalent to jitcode liveness info.
    /// RPython uses jitcode.get_live_vars_info(pc) at decode time;
    /// we store the counts at encode time since this path lacks jitcodes.
    pub frame_sizes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct DecodedResumeLayout {
    vable_array: Vec<ResumeValueSource>,
    vref_array: Vec<ResumeValueSource>,
    frames: Vec<FrameInfo>,
    virtuals: Vec<VirtualInfo>,
    pending_fields: Vec<PendingFieldInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeValueKind {
    FailArg,
    Constant,
    Virtual,
    Uninitialized,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeValueLayoutSummary {
    pub kind: ResumeValueKind,
    pub fail_arg_index: usize,
    pub raw_fail_arg_position: Option<usize>,
    /// RPython parity: `Const.value` raw bits (getint/getref_base/getfloatstorage
    /// all project to `i64`).
    pub constant: Option<i64>,
    /// RPython parity: `Const.type` — paired with `constant` so the summary
    /// round-trips back into a typed `ResumeValueSource::Constant(Const)`.
    pub constant_type: Option<Type>,
    pub virtual_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeFrameLayoutSummary {
    pub trace_id: Option<u64>,
    pub header_pc: Option<u64>,
    pub source_guard: Option<(u64, u32)>,
    /// resume.py:250 jitcode_index — index into metainterp_sd.jitcodes[].
    pub jitcode_index: i32,
    pub pc: u64,
    pub slot_sources: Vec<ResumeValueKind>,
    pub slot_layouts: Vec<ResumeValueLayoutSummary>,
    pub slot_types: Option<Vec<Type>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeVirtualKind {
    Object,
    Struct,
    Array,
    ArrayStruct,
    RawBuffer,
    /// resume.py:763 VStrPlainInfo — virtual plain string.
    StrPlain,
    /// resume.py:781 VStrConcatInfo — virtual concatenated string.
    StrConcat,
    /// resume.py:801 VStrSliceInfo — virtual string slice.
    StrSlice,
    /// resume.py:817 VUniPlainInfo — virtual plain unicode string.
    UniPlain,
    /// resume.py:836 VUniConcatInfo — virtual concatenated unicode.
    UniConcat,
    /// resume.py:856 VUniSliceInfo — virtual unicode slice.
    UniSlice,
}

#[derive(Debug, Clone)]
pub enum ResumeVirtualLayoutSummary {
    Object {
        /// resume.py:615 self.descr — live SizeDescr, preserved across summary round-trip.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        /// info.py:318 _known_class — vtable pointer.
        known_class: Option<i64>,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    Struct {
        /// resume.py:631 self.typedescr — live SizeDescr, preserved across summary round-trip.
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py:643-684 AbstractVArrayInfo
    Array {
        /// resume.py:646: self.arraydescr
        arraydescr: Option<majit_ir::DescrRef>,
        descr_index: u32,
        /// resume.py:680-683: VArrayInfoClear.clear=True / VArrayInfoNotClear.clear=False
        clear: bool,
        items: Vec<ResumeValueLayoutSummary>,
    },
    /// resume.py:736 VArrayStructInfo(arraydescr, size, fielddescrs)
    ArrayStruct {
        /// resume.py:739: self.arraydescr
        arraydescr: Option<majit_ir::DescrRef>,
        descr_index: u32,
        /// resume.py:740: self.fielddescrs
        fielddescrs: Vec<majit_ir::DescrRef>,
        element_fields: Vec<Vec<(u32, ResumeValueLayoutSummary)>>,
    },
    RawBuffer {
        /// resume.py:694: self.func
        func: i64,
        size: usize,
        /// resume.py:695: self.offsets
        offsets: Vec<usize>,
        /// resume.py:697: self.descrs
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        /// resume.py:693: fieldnums (decoded)
        values: Vec<ResumeValueLayoutSummary>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingFieldLayoutSummary {
    pub descr_index: u32,
    pub item_index: Option<usize>,
    pub is_array_item: bool,
    pub target_kind: ResumeValueKind,
    pub value_kind: ResumeValueKind,
    pub target: ResumeValueLayoutSummary,
    pub value: ResumeValueLayoutSummary,
}

#[derive(Debug, Clone)]
pub struct ResumeLayoutSummary {
    pub num_frames: usize,
    pub frame_pcs: Vec<u64>,
    pub frame_slot_counts: Vec<usize>,
    pub frame_layouts: Vec<ResumeFrameLayoutSummary>,
    pub num_virtuals: usize,
    pub virtual_kinds: Vec<ResumeVirtualKind>,
    pub virtual_layouts: Vec<ResumeVirtualLayoutSummary>,

    pub pending_field_count: usize,
    pub pending_field_layouts: Vec<PendingFieldLayoutSummary>,
    pub const_pool_size: usize,
}

impl ResumeValueLayoutSummary {
    pub(crate) fn raw_fail_arg_position(&self) -> usize {
        if let Some(position) = self.raw_fail_arg_position {
            return position;
        }
        self.fail_arg_index
    }

    fn to_resume_source(&self) -> ResumeValueSource {
        match self.kind {
            ResumeValueKind::FailArg => ResumeValueSource::FailArg(self.raw_fail_arg_position()),
            ResumeValueKind::Constant => {
                let raw = self.constant.expect("missing constant value");
                // Constant kind requires both raw and type to be set; the
                // .expect() above already enforces raw, mirror it for type.
                let tp = self.constant_type.expect("missing constant type");
                ResumeValueSource::Constant(majit_ir::Const::from_raw_i64(raw, tp))
            }
            ResumeValueKind::Virtual => {
                ResumeValueSource::Virtual(self.virtual_index.expect("missing virtual index"))
            }
            ResumeValueKind::Uninitialized => ResumeValueSource::Uninitialized,
            ResumeValueKind::Unavailable => ResumeValueSource::Unavailable,
        }
    }

    fn to_exit_source(&self, virtual_offset: usize) -> ExitValueSourceLayout {
        match self.kind {
            ResumeValueKind::FailArg => {
                ExitValueSourceLayout::ExitValue(self.raw_fail_arg_position())
            }
            ResumeValueKind::Constant => {
                ExitValueSourceLayout::Constant(self.constant.expect("missing constant value"))
            }
            ResumeValueKind::Virtual => ExitValueSourceLayout::Virtual(
                self.virtual_index.expect("missing virtual index") + virtual_offset,
            ),
            ResumeValueKind::Uninitialized => ExitValueSourceLayout::Uninitialized,
            ResumeValueKind::Unavailable => ExitValueSourceLayout::Unavailable,
        }
    }
}

impl ResumeFrameLayoutSummary {
    fn to_frame_info(&self) -> FrameInfo {
        FrameInfo {
            jitcode_index: self.jitcode_index,
            pc: self.pc,
            slot_map: self
                .slot_layouts
                .iter()
                .map(|slot| slot.to_resume_source())
                .collect(),
        }
    }

    fn to_exit_frame_layout(&self, virtual_offset: usize) -> ExitFrameLayout {
        ExitFrameLayout {
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            source_guard: self.source_guard,
            pc: self.pc,
            jitcode_index: self.jitcode_index,
            slots: self
                .slot_layouts
                .iter()
                .map(|slot| slot.to_exit_source(virtual_offset))
                .collect(),
            slot_types: self.slot_types.clone(),
        }
    }

    /// Build a `ResumeFrameLayoutSummary` from a backend-origin `ExitFrameLayout`.
    ///
    /// Each `ExitValueSourceLayout` slot is converted to the corresponding
    /// `ResumeValueLayoutSummary`, preserving slot types when present.
    pub fn from_exit_frame_layout(exit_frame: &ExitFrameLayout) -> Self {
        let slot_layouts: Vec<ResumeValueLayoutSummary> = exit_frame
            .slots
            .iter()
            .map(ResumeValueLayoutSummary::from_exit_value_source)
            .collect();
        let slot_sources: Vec<ResumeValueKind> = slot_layouts.iter().map(|s| s.kind).collect();

        Self {
            trace_id: exit_frame.trace_id,
            header_pc: exit_frame.header_pc,
            source_guard: exit_frame.source_guard,
            jitcode_index: exit_frame.jitcode_index,
            pc: exit_frame.pc,
            slot_sources,
            slot_layouts,
            slot_types: exit_frame.slot_types.clone(),
        }
    }
}

impl ResumeValueLayoutSummary {
    /// Build a `ResumeValueLayoutSummary` from a backend-origin `ExitValueSourceLayout`.
    ///
    /// Backend-origin constants are always `Int` (numeric or tagged) — the
    /// backend layer is untyped, so we default the Const type to Int here.
    pub fn from_exit_value_source(source: &ExitValueSourceLayout) -> Self {
        match source {
            ExitValueSourceLayout::ExitValue(index) => Self {
                kind: ResumeValueKind::FailArg,
                fail_arg_index: *index,
                raw_fail_arg_position: Some(*index),
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
            ExitValueSourceLayout::Constant(value) => Self {
                kind: ResumeValueKind::Constant,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: Some(*value),
                constant_type: Some(Type::Int),
                virtual_index: None,
            },
            ExitValueSourceLayout::Virtual(index) => Self {
                kind: ResumeValueKind::Virtual,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: Some(*index),
            },
            ExitValueSourceLayout::Uninitialized => Self {
                kind: ResumeValueKind::Uninitialized,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
            ExitValueSourceLayout::Unavailable => Self {
                kind: ResumeValueKind::Unavailable,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
        }
    }
}

impl ResumeVirtualLayoutSummary {
    fn to_virtual_info(&self) -> VirtualInfo {
        match self {
            ResumeVirtualLayoutSummary::Object {
                descr,
                type_id,
                descr_index,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => VirtualInfo::VirtualObj {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_resume_source()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Struct {
                typedescr,
                type_id,
                descr_index,
                fields,
                fielddescrs,
                descr_size,
            } => VirtualInfo::VStruct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_resume_source()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Array {
                arraydescr,
                descr_index,
                clear,
                items,
            } => VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                descr_index: *descr_index,
                clear: *clear,
                items: items
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                arraydescr,
                descr_index,
                fielddescrs,
                element_fields,
                ..
            } => VirtualInfo::VArrayStruct {
                arraydescr: arraydescr.clone(),
                descr_index: *descr_index,
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| (*field_descr, source.to_resume_source()))
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => VirtualInfo::VRawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
        }
    }

    fn to_exit_virtual_layout(&self, virtual_offset: usize) -> ExitVirtualLayout {
        match self {
            ResumeVirtualLayoutSummary::Object {
                descr,
                type_id,
                descr_index,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => ExitVirtualLayout::Object {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_exit_source(virtual_offset)))
                    .collect(),
                target_slot: None,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Struct {
                typedescr,
                type_id,
                descr_index,
                fields,
                fielddescrs,
                descr_size,
            } => ExitVirtualLayout::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (*field_descr, source.to_exit_source(virtual_offset))
                    })
                    .collect(),
                target_slot: None,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Array {
                arraydescr,
                descr_index,
                clear,
                items,
            } => ExitVirtualLayout::Array {
                descr_index: *descr_index,
                clear: *clear,
                // resume.py:656-670: element type from arraydescr
                kind: array_kind_from_descr(arraydescr.as_ref()),
                items: items
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                arraydescr,
                descr_index,
                fielddescrs,
                element_fields,
            } => ExitVirtualLayout::ArrayStruct {
                descr_index: *descr_index,
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| {
                                (*field_descr, source.to_exit_source(virtual_offset))
                            })
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => ExitVirtualLayout::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
        }
    }
}

impl PendingFieldLayoutSummary {
    fn to_pending_field_info(&self) -> PendingFieldInfo {
        PendingFieldInfo {
            descr_index: self.descr_index,
            target: self.target.to_resume_source(),
            value: self.value.to_resume_source(),
            item_index: self.item_index,
        }
    }

    fn to_exit_pending_field_layout(&self, virtual_offset: usize) -> ExitPendingFieldLayout {
        ExitPendingFieldLayout {
            descr_index: self.descr_index,
            item_index: self.item_index,
            is_array_item: self.is_array_item,
            target: self.target.to_exit_source(virtual_offset),
            value: self.value.to_exit_source(virtual_offset),
            field_offset: 0,
            field_size: 0,
            field_type: majit_ir::Type::Int,
        }
    }
}

impl ResumeLayoutSummary {
    pub fn to_resume_data(&self) -> ResumeData {
        ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: self
                .frame_layouts
                .iter()
                .map(|frame| frame.to_frame_info())
                .collect(),
            virtuals: self
                .virtual_layouts
                .iter()
                .map(|virt| virt.to_virtual_info())
                .collect(),
            pending_fields: self
                .pending_field_layouts
                .iter()
                .map(|pending| pending.to_pending_field_info())
                .collect(),
        }
    }

    pub fn to_exit_recovery_layout(&self) -> ExitRecoveryLayout {
        self.to_exit_recovery_layout_with_caller_prefix(None)
    }

    pub fn to_exit_recovery_layout_with_caller_prefix(
        &self,
        caller_prefix: Option<&ExitRecoveryLayout>,
    ) -> ExitRecoveryLayout {
        if self.frame_layouts.is_empty() {
            return caller_prefix.cloned().unwrap_or(ExitRecoveryLayout {
                vable_array: Vec::new(),
                vref_array: Vec::new(),
                frames: Vec::new(),
                virtual_layouts: Vec::new(),
                pending_field_layouts: Vec::new(),
            });
        }

        let prefix_frame_count = caller_prefix
            .map(|layout| layout.frames.len().saturating_sub(self.frame_layouts.len()))
            .unwrap_or(0);
        let preserve_prefix = prefix_frame_count > 0;

        let mut frames = caller_prefix
            .map(|layout| layout.frames[..prefix_frame_count].to_vec())
            .unwrap_or_default();
        // RPython parity: rd_virtuals is stored once on the guard descriptor
        // and never replaced (compile.py:866, resume.py:492). Always preserve
        // caller_prefix's virtual_layouts — they originate from
        // build_guard_metadata and must not be overwritten.
        let mut virtual_layouts = caller_prefix
            .map(|layout| layout.virtual_layouts.clone())
            .unwrap_or_default();
        let mut pending_field_layouts = if preserve_prefix {
            caller_prefix
                .map(|layout| layout.pending_field_layouts.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let virtual_offset = virtual_layouts.len();

        frames.extend(
            self.frame_layouts
                .iter()
                .map(|frame| frame.to_exit_frame_layout(virtual_offset)),
        );
        virtual_layouts.extend(
            self.virtual_layouts
                .iter()
                .map(|virt| virt.to_exit_virtual_layout(virtual_offset)),
        );
        pending_field_layouts.extend(
            self.pending_field_layouts
                .iter()
                .map(|pending| pending.to_exit_pending_field_layout(virtual_offset)),
        );

        ExitRecoveryLayout {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames,
            virtual_layouts,
            pending_field_layouts,
        }
    }

    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let resume_data = self.to_resume_data();
        let virtuals = resume_data.materialize_virtuals(fail_values);
        let pending_fields =
            ResumeData::resolve_pending_field_writes(&resume_data.pending_fields, fail_values);
        let frames = self
            .frame_layouts
            .iter()
            .map(|frame| ReconstructedFrame {
                trace_id: frame.trace_id,
                header_pc: frame.header_pc,
                source_guard: frame.source_guard,
                pc: frame.pc,
                jitcode_index: frame.jitcode_index,
                slot_types: frame.slot_types.clone(),
                values: frame
                    .slot_layouts
                    .iter()
                    .map(|slot| {
                        ResumeData::resolve_frame_slot_source(&slot.to_resume_source(), fail_values)
                    })
                    .collect(),
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals,
            pending_fields,
        }
    }

    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    pub fn reconstruct_frame(
        &self,
        frame_index: usize,
        fail_values: &[i64],
    ) -> Option<ReconstructedFrame> {
        let frame = self.frame_layouts.get(frame_index)?;
        Some(ReconstructedFrame {
            trace_id: frame.trace_id,
            header_pc: frame.header_pc,
            source_guard: frame.source_guard,
            pc: frame.pc,
            jitcode_index: frame.jitcode_index,
            slot_types: frame.slot_types.clone(),
            values: frame
                .slot_layouts
                .iter()
                .map(|slot| {
                    ResumeData::resolve_frame_slot_source(&slot.to_resume_source(), fail_values)
                })
                .collect(),
        })
    }

    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        self.to_resume_data().materialize_virtuals(fail_values)
    }

    pub fn resolve_pending_field_writes(
        &self,
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        let resume_data = self.to_resume_data();
        ResumeData::resolve_pending_field_writes(&resume_data.pending_fields, fail_values)
    }
}

fn can_inline_tagged(value: i64) -> bool {
    (INLINE_TAGGED_MIN..=INLINE_TAGGED_MAX).contains(&value)
}

// `encode_tagged_source` has been promoted to a method on
// `ResumeDataLoopMemo` (see `ResumeDataLoopMemo::encode_tagged_source`)
// so it can share `self.consts` with `getconst`/`newconst` — matching
// RPython's single `self.consts: list[Const]` pool (resume.py:147).

/// resume.py:99-104 tag() — i64 widened variant for rd_numb encoding.
fn tag_i64(value: i64, tagbits: u8) -> i64 {
    debug_assert!(tagbits <= 3);
    debug_assert!(
        can_inline_tagged(value),
        "tagged resume value {value} exceeds inline range"
    );
    (value << 2) | tagbits as i64
}

/// resume.py:106-109 untag() — i64 widened variant for rd_numb decoding.
fn untag_i64(encoded: i64) -> (i64, u8) {
    ((encoded >> 2), (encoded & TAGMASK_I64) as u8)
}

fn encode_len(value: usize) -> i64 {
    i64::try_from(value).expect("resume length exceeds i64")
}

fn decode_len(value: i64) -> usize {
    usize::try_from(value).expect("negative or oversized resume length")
}

fn encode_u64(value: u64) -> i64 {
    value as i64
}

fn decode_u64(value: i64) -> u64 {
    value as u64
}

/// Describes how to reconstruct a single frame in the interpreter's call stack.
///
/// Each frame has a bytecode position (pc) and a set of named/indexed slots
/// that map to tagged resume sources.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameInfo {
    /// resume.py:250 jitcode_index — index into metainterp_sd.jitcodes[].
    pub jitcode_index: i32,
    /// Bytecode position (program counter) for this frame.
    pub pc: u64,
    /// Mapping from slot index to a tagged resume source.
    pub slot_map: Vec<FrameSlotSource>,
}

/// Complete resume data for a guard exit point.
///
/// Contains enough information to reconstruct the full interpreter state
/// from the values stored in a DeadFrame.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeData {
    /// resume.py: snapshot_iter.vable_array / virtualizable_boxes
    pub vable_array: Vec<ResumeValueSource>,
    /// resume.py: snapshot_iter.vref_array / virtualref_boxes
    pub vref_array: Vec<ResumeValueSource>,
    /// Stack of frames, outermost first.
    /// For a simple non-inlined trace, this has exactly one entry.
    pub frames: Vec<FrameInfo>,
    /// Virtual object descriptions for virtualized state.
    /// Each entry maps a fail_arg position to a virtual object that needs
    /// to be materialized when resuming.
    pub virtuals: Vec<VirtualInfo>,
    /// Deferred heap writes that must be replayed when resuming.
    ///
    /// Mirrors RPython's `rd_pendingfields`, which applies writes after
    /// virtuals and boxes have been reconstructed.
    pub pending_fields: Vec<PendingFieldInfo>,
}

/// Tagged source for a value that must be reconstructed on resume.
///
/// This is the majit equivalent of the tagged numbering used by
/// `rpython/jit/metainterp/resume.py`. Each `Constant` entry carries a
/// full `majit_ir::Const` (Int/Float/Ref) so the encoder's `getconst`
/// dispatch (resume.py:157-188) can route through the shared pool
/// (`ResumeDataLoopMemo.consts`) without losing type information.
#[derive(Debug, Clone, PartialEq)]
pub enum ResumeValueSource {
    /// Value comes from the deadframe fail-args array.
    FailArg(usize),
    /// Value is a compile-time constant — carries the Const so that the
    /// type survives encoding, matching RPython's `Const` object.
    Constant(majit_ir::Const),
    /// Value is a virtual object that must be materialized on resume.
    Virtual(usize),
    /// Value exists conceptually but is still uninitialized.
    ///
    /// Mirrors RPython's `UNINITIALIZED` tag used for string/unicode content.
    Uninitialized,
    /// Slot is not live at this guard.
    Unavailable,
}

impl ResumeValueSource {
    pub fn kind(&self) -> ResumeValueKind {
        match self {
            ResumeValueSource::FailArg(_) => ResumeValueKind::FailArg,
            ResumeValueSource::Constant(_) => ResumeValueKind::Constant,
            ResumeValueSource::Virtual(_) => ResumeValueKind::Virtual,
            ResumeValueSource::Uninitialized => ResumeValueKind::Uninitialized,
            ResumeValueSource::Unavailable => ResumeValueKind::Unavailable,
        }
    }

    pub fn layout_summary(&self) -> ResumeValueLayoutSummary {
        match self {
            ResumeValueSource::FailArg(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::FailArg,
                fail_arg_index: *index,
                raw_fail_arg_position: Some(*index),
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
            ResumeValueSource::Constant(c) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Constant,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: Some(c.as_raw_i64()),
                constant_type: Some(c.get_type()),
                virtual_index: None,
            },
            ResumeValueSource::Virtual(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Virtual,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: Some(*index),
            },
            ResumeValueSource::Uninitialized => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Uninitialized,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
            ResumeValueSource::Unavailable => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Unavailable,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
        }
    }
}

/// Source for a resumed frame slot.
pub type FrameSlotSource = ResumeValueSource;

/// Description of a virtual object that needs materialization on resume.
///
/// Mirrors RPython's AbstractVirtualInfo hierarchy:
/// - VirtualInfo (NEW_WITH_VTABLE)
/// - VStructInfo (NEW / plain struct)
/// - VArrayInfoClear / VArrayInfoNotClear (NEW_ARRAY)
/// - VArrayStructInfo (array of structs with interior fields)
/// - VRawBufferInfo (raw memory buffer)
#[derive(Debug, Clone)]
pub enum VirtualInfo {
    /// resume.py:612 VirtualInfo(descr, fielddescrs).
    VirtualObj {
        /// resume.py:615 self.descr — live SizeDescr.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        /// info.py:318 _known_class — vtable pointer.
        known_class: Option<i64>,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py:628 VStructInfo(typedescr, fielddescrs).
    VStruct {
        /// resume.py:631 self.typedescr — the full SizeDescr.
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py:643-684 AbstractVArrayInfo (from NEW_ARRAY).
    VArray {
        /// resume.py:646: self.arraydescr
        arraydescr: Option<majit_ir::DescrRef>,
        /// Array descriptor index (serialization compat).
        descr_index: u32,
        /// resume.py:680-683: VArrayInfoClear.clear / VArrayInfoNotClear.clear
        clear: bool,
        /// Element values.
        items: Vec<VirtualFieldSource>,
    },
    /// resume.py:736 VArrayStructInfo (from arrays with interior field access).
    VArrayStruct {
        /// resume.py:739: self.arraydescr
        arraydescr: Option<majit_ir::DescrRef>,
        /// Array descriptor index (serialization compat).
        descr_index: u32,
        /// resume.py:740: self.fielddescrs — live InteriorFieldDescr objects
        /// for setinteriorfield dispatch.
        fielddescrs: Vec<majit_ir::DescrRef>,
        /// Per-element fields: outer Vec = elements, inner Vec = (field_index, source).
        element_fields: Vec<Vec<(u32, VirtualFieldSource)>>,
    },
    /// resume.py:692 VRawBufferInfo(func, size, offsets, descrs).
    VRawBuffer {
        /// resume.py:694: self.func — raw malloc function pointer.
        func: i64,
        /// Size of the buffer in bytes.
        size: usize,
        /// resume.py:695: self.offsets — byte offsets for each stored value.
        offsets: Vec<usize>,
        /// resume.py:697: self.descrs — per-entry ArrayDescr snapshots.
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        /// resume.py:693: fieldnums — per-entry source (decoded from tagged fieldnums).
        values: Vec<VirtualFieldSource>,
    },
    /// resume.py: VRawSliceInfo — a slice into a virtual raw buffer.
    VRawSlice {
        /// Offset from the parent raw buffer.
        offset: i64,
        /// Source of the parent buffer.
        parent: VirtualFieldSource,
    },
    /// resume.py:763 VStrPlainInfo — virtual string (known characters).
    VStrPlain {
        /// Character values (as OpRef sources).
        chars: Vec<VirtualFieldSource>,
    },
    /// resume.py:781 VStrConcatInfo — virtual string concat (left + right).
    VStrConcat {
        left: Box<VirtualFieldSource>,
        right: Box<VirtualFieldSource>,
    },
    /// resume.py:801 VStrSliceInfo — virtual string slice.
    VStrSlice {
        source: Box<VirtualFieldSource>,
        start: Box<VirtualFieldSource>,
        length: Box<VirtualFieldSource>,
    },
    /// resume.py:817 VUniPlainInfo — virtual unicode string.
    VUniPlain { chars: Vec<VirtualFieldSource> },
    /// resume.py:836 VUniConcatInfo — virtual unicode concat.
    VUniConcat {
        left: Box<VirtualFieldSource>,
        right: Box<VirtualFieldSource>,
    },
    /// resume.py:856 VUniSliceInfo — virtual unicode slice.
    VUniSlice {
        source: Box<VirtualFieldSource>,
        start: Box<VirtualFieldSource>,
        length: Box<VirtualFieldSource>,
    },
}

// PartialEq/Eq: compare by data fields, skip descr/typedescr (Arc<dyn Descr>).
impl PartialEq for VirtualInfo {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                VirtualInfo::VirtualObj {
                    descr: None,
                    type_id: a1,
                    descr_index: a2,
                    fields: a3,
                    fielddescrs: a4,
                    descr_size: a5,
                    ..
                },
                VirtualInfo::VirtualObj {
                    descr: None,
                    type_id: b1,
                    descr_index: b2,
                    fields: b3,
                    fielddescrs: b4,
                    descr_size: b5,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5,
            (
                VirtualInfo::VStruct {
                    type_id: a1,
                    descr_index: a2,
                    fields: a3,
                    fielddescrs: a4,
                    descr_size: a5,
                    ..
                },
                VirtualInfo::VStruct {
                    type_id: b1,
                    descr_index: b2,
                    fields: b3,
                    fielddescrs: b4,
                    descr_size: b5,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5,
            (
                VirtualInfo::VArray {
                    arraydescr: _,
                    descr_index: a1,
                    clear: a_clear,
                    items: a2,
                },
                VirtualInfo::VArray {
                    arraydescr: _,
                    descr_index: b1,
                    clear: b_clear,
                    items: b2,
                },
            ) => a1 == b1 && a_clear == b_clear && a2 == b2,
            _ => false,
        }
    }
}
impl Eq for VirtualInfo {}

impl VirtualInfo {
    /// Iterate over all field sources in this virtual.
    /// resume.py: visitor_walk_recursive walks all box references in a virtual.
    pub fn field_sources(&self) -> Vec<&VirtualFieldSource> {
        match self {
            VirtualInfo::VirtualObj { fields, .. } | VirtualInfo::VStruct { fields, .. } => {
                fields.iter().map(|(_, src)| src).collect()
            }
            VirtualInfo::VArray { items, .. } => items.iter().collect(),
            VirtualInfo::VArrayStruct { element_fields, .. } => element_fields
                .iter()
                .flat_map(|el| el.iter().map(|(_, src)| src))
                .collect(),
            VirtualInfo::VRawBuffer { values, .. } => values.iter().collect(),
            VirtualInfo::VRawSlice { parent, .. } => vec![parent],
            VirtualInfo::VStrPlain { chars } | VirtualInfo::VUniPlain { chars } => {
                chars.iter().collect()
            }
            VirtualInfo::VStrConcat { left, right } | VirtualInfo::VUniConcat { left, right } => {
                vec![left.as_ref(), right.as_ref()]
            }
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
            }
            | VirtualInfo::VUniSlice {
                source,
                start,
                length,
            } => vec![source.as_ref(), start.as_ref(), length.as_ref()],
        }
    }

    pub fn kind(&self) -> ResumeVirtualKind {
        match self {
            VirtualInfo::VirtualObj { .. } => ResumeVirtualKind::Object,
            VirtualInfo::VStruct { .. } => ResumeVirtualKind::Struct,
            VirtualInfo::VArray { .. } => ResumeVirtualKind::Array,
            VirtualInfo::VArrayStruct { .. } => ResumeVirtualKind::ArrayStruct,
            VirtualInfo::VRawBuffer { .. } | VirtualInfo::VRawSlice { .. } => {
                ResumeVirtualKind::RawBuffer
            }
            VirtualInfo::VStrPlain { .. } => ResumeVirtualKind::StrPlain,
            VirtualInfo::VStrConcat { .. } => ResumeVirtualKind::StrConcat,
            VirtualInfo::VStrSlice { .. } => ResumeVirtualKind::StrSlice,
            VirtualInfo::VUniPlain { .. } => ResumeVirtualKind::UniPlain,
            VirtualInfo::VUniConcat { .. } => ResumeVirtualKind::UniConcat,
            VirtualInfo::VUniSlice { .. } => ResumeVirtualKind::UniSlice,
        }
    }

    pub fn layout_summary(&self) -> ResumeVirtualLayoutSummary {
        match self {
            VirtualInfo::VirtualObj {
                descr,
                type_id,
                descr_index,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => ResumeVirtualLayoutSummary::Object {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.layout_summary()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            VirtualInfo::VStruct {
                typedescr,
                type_id,
                descr_index,
                fields,
                fielddescrs,
                descr_size,
            } => ResumeVirtualLayoutSummary::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.layout_summary()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            VirtualInfo::VArray {
                arraydescr,
                descr_index,
                clear,
                items,
            } => ResumeVirtualLayoutSummary::Array {
                arraydescr: arraydescr.clone(),
                descr_index: *descr_index,
                clear: *clear,
                items: items.iter().map(|source| source.layout_summary()).collect(),
            },
            VirtualInfo::VArrayStruct {
                arraydescr,
                descr_index,
                fielddescrs,
                element_fields,
            } => ResumeVirtualLayoutSummary::ArrayStruct {
                arraydescr: arraydescr.clone(),
                descr_index: *descr_index,
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| (*field_descr, source.layout_summary()))
                            .collect()
                    })
                    .collect(),
            },
            VirtualInfo::VRawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => ResumeVirtualLayoutSummary::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values.iter().map(|src| src.layout_summary()).collect(),
            },
            // String/unicode virtual infos — represented as structs
            // with synthetic field indices for reconstruction.
            VirtualInfo::VRawSlice { .. }
            | VirtualInfo::VStrPlain { .. }
            | VirtualInfo::VStrConcat { .. }
            | VirtualInfo::VStrSlice { .. }
            | VirtualInfo::VUniPlain { .. }
            | VirtualInfo::VUniConcat { .. }
            | VirtualInfo::VUniSlice { .. } => ResumeVirtualLayoutSummary::Struct {
                typedescr: None,
                type_id: 0,
                descr_index: 0,
                fields: vec![],
                fielddescrs: vec![],
                descr_size: 0,
            },
        }
    }
}

/// Source of a virtual object's field value.
pub type VirtualFieldSource = ResumeValueSource;

/// Convert a tagged fieldnum (i16, resume.py encoding) to a VirtualFieldSource.
///
/// resume.py:1552-1596 decode_int/decode_ref: tagged values encode where
/// each field value comes from at resume time.
///
/// `consts` is the rd_consts array. `count` is the number of fail_args
/// (used for negative TAGBOX indices). Both come from the containing
/// ResumeGuardDescr / EncodedResumeData.
pub fn tagged_to_source(tagged: i16, consts: &[majit_ir::Const], count: i32) -> VirtualFieldSource {
    if tagged_eq(tagged, UNASSIGNED) {
        return ResumeValueSource::Unavailable;
    }
    if tagged_eq(tagged, UNINITIALIZED_TAG) {
        return ResumeValueSource::Uninitialized;
    }
    if tagged_eq(tagged, NULLREF) {
        // history.py:361 CONST_NULL = ConstPtr(null). resume.py:1589 parity.
        return ResumeValueSource::Constant(majit_ir::Const::Ref(majit_ir::GcRef::NULL));
    }
    let (num, tag_bits) = untag(tagged);
    match tag_bits {
        TAGCONST => {
            let idx = (num - TAG_CONST_OFFSET) as usize;
            if idx < consts.len() {
                // resume.py:1568 self.consts[num - TAG_CONST_OFFSET] — the
                // Const object carries its type (ConstInt/ConstFloat/ConstPtr).
                ResumeValueSource::Constant(consts[idx])
            } else {
                ResumeValueSource::Constant(majit_ir::Const::Int(0))
            }
        }
        // resume.py:1581 ConstInt(num) — always Int type for TAGINT.
        TAGINT => ResumeValueSource::Constant(majit_ir::Const::Int(num as i64)),
        TAGBOX => {
            let mut idx = num;
            if idx < 0 {
                idx += count;
            }
            ResumeValueSource::FailArg(idx as usize)
        }
        TAGVIRTUAL => ResumeValueSource::Virtual(num as usize),
        _ => ResumeValueSource::Unavailable,
    }
}

/// Convert an `RdVirtualInfo` (IR-level, from compile.rs/pyjitpl.rs)
/// to a `VirtualInfo` (resume-level, used by ResumeDataDirectReader).
///
/// `consts` and `count` are needed to decode tagged fieldnums.
pub fn rd_virtual_to_virtual_info(
    rd: &majit_ir::RdVirtualInfo,
    consts: &[majit_ir::Const],
    count: i32,
) -> VirtualInfo {
    match rd {
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            type_id,
            descr_index,
            known_class,
            fielddescrs,
            fieldnums,
            descr_size,
        } => {
            let fields = fielddescrs
                .iter()
                .zip(fieldnums.iter())
                .map(|(fd, &tagged)| (fd.index, tagged_to_source(tagged, consts, count)))
                .collect();
            VirtualInfo::VirtualObj {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            }
        }
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            type_id,
            descr_index,
            fielddescrs,
            fieldnums,
            descr_size,
        } => {
            let fields = fielddescrs
                .iter()
                .zip(fieldnums.iter())
                .map(|(fd, &tagged)| (fd.index, tagged_to_source(tagged, consts, count)))
                .collect();
            VirtualInfo::VStruct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            }
        }
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            arraydescr,
            descr_index,
            fieldnums,
            ..
        } => {
            let items = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count))
                .collect();
            VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                descr_index: *descr_index,
                clear: true,
                items,
            }
        }
        majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            arraydescr,
            descr_index,
            fieldnums,
            ..
        } => {
            let items = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count))
                .collect();
            VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                descr_index: *descr_index,
                clear: false,
                items,
            }
        }
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            descr_index,
            size,
            fielddescrs: rd_fielddescrs,
            fieldnums,
            ..
        } => {
            // resume.py:736-740: VArrayStructInfo(arraydescr, size, fielddescrs)
            // fieldnums is flat: size * len(fielddescrs) entries
            let num_fields = rd_fielddescrs.len().max(1);
            let mut element_fields = Vec::with_capacity(*size);
            for chunk in fieldnums.chunks(num_fields) {
                // resume.py:754: for j in range(len(self.fielddescrs)):
                let elem: Vec<(u32, VirtualFieldSource)> = chunk
                    .iter()
                    .enumerate()
                    .map(|(j, &tagged)| (j as u32, tagged_to_source(tagged, consts, count)))
                    .collect();
                element_fields.push(elem);
            }
            while element_fields.len() < *size {
                element_fields.push(vec![]);
            }
            VirtualInfo::VArrayStruct {
                arraydescr: arraydescr.clone(),
                descr_index: *descr_index,
                fielddescrs: rd_fielddescrs.clone(),
                element_fields,
            }
        }
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            assert_eq!(offsets.len(), descrs.len());
            assert_eq!(offsets.len(), fieldnums.len());
            let values = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count))
                .collect();
            VirtualInfo::VRawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values,
            }
        }
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            let parent = fieldnums
                .first()
                .map(|&tagged| tagged_to_source(tagged, consts, count))
                .unwrap_or(ResumeValueSource::Unavailable);
            VirtualInfo::VRawSlice {
                offset: *offset as i64,
                parent,
            }
        }
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums } => {
            let chars = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count))
                .collect();
            VirtualInfo::VStrPlain { chars }
        }
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums } => {
            let left = Box::new(tagged_to_source(fieldnums[0], consts, count));
            let right = Box::new(tagged_to_source(fieldnums[1], consts, count));
            VirtualInfo::VStrConcat { left, right }
        }
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums } => {
            let source = Box::new(tagged_to_source(fieldnums[0], consts, count));
            let start = Box::new(tagged_to_source(fieldnums[1], consts, count));
            let length = Box::new(tagged_to_source(fieldnums[2], consts, count));
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
            }
        }
        majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let chars = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count))
                .collect();
            VirtualInfo::VUniPlain { chars }
        }
        majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
            let left = Box::new(tagged_to_source(fieldnums[0], consts, count));
            let right = Box::new(tagged_to_source(fieldnums[1], consts, count));
            VirtualInfo::VUniConcat { left, right }
        }
        majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
            let source = Box::new(tagged_to_source(fieldnums[0], consts, count));
            let start = Box::new(tagged_to_source(fieldnums[1], consts, count));
            let length = Box::new(tagged_to_source(fieldnums[2], consts, count));
            VirtualInfo::VUniSlice {
                source,
                start,
                length,
            }
        }
        majit_ir::RdVirtualInfo::Empty => VirtualInfo::VirtualObj {
            descr: None,
            type_id: 0,
            descr_index: 0,
            known_class: None,
            fields: vec![],
            fielddescrs: vec![],
            descr_size: 0,
        },
    }
}

/// Deferred heap write to replay during resume.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingFieldInfo {
    /// Descriptor index identifying the target field or array descriptor.
    pub descr_index: u32,
    /// Source of the object/array pointer to update.
    pub target: ResumeValueSource,
    /// Source of the value to write.
    pub value: ResumeValueSource,
    /// Array item index. `None` means a plain field write.
    pub item_index: Option<usize>,
}

impl PendingFieldInfo {
    pub fn layout_summary(&self) -> PendingFieldLayoutSummary {
        PendingFieldLayoutSummary {
            descr_index: self.descr_index,
            item_index: self.item_index,
            is_array_item: self.item_index.is_some(),
            target_kind: self.target.kind(),
            value_kind: self.value.kind(),
            target: self.target.layout_summary(),
            value: self.value.layout_summary(),
        }
    }
}

/// Concrete pending heap write reconstructed from resume data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPendingFieldWrite {
    /// Descriptor index identifying the field or array descriptor.
    pub descr_index: u32,
    /// Concrete object/array pointer.
    pub target: MaterializedValue,
    /// Concrete value to write.
    pub value: MaterializedValue,
    /// Array item index. `None` means a plain field write.
    pub item_index: Option<usize>,
}

/// Encoded pending field write stored alongside an encoded resume snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPendingFieldWrite {
    pub descr_index: u32,
    pub target: i64,
    pub value: i64,
    pub item_index: Option<usize>,
}

impl EncodedResumeData {
    pub fn encode(rd: &ResumeData) -> Self {
        Self::from_semantic(
            &rd.vable_array,
            &rd.vref_array,
            &rd.frames,
            &rd.virtuals,
            &rd.pending_fields,
        )
    }

    /// resume.py:231-267 number + resume.py:380-468 finish
    ///
    /// Walks all frames via _number_boxes, assigning compact sequential
    /// TAGBOX numbers to unique liveboxes (resume.py:199-226).
    ///
    /// Unlike `ResumeDataLoopMemo::encode_shared`, this is a single-shot
    /// encoder (no cross-guard dedup). It builds a local memo that shares
    /// the same encoding logic so the tagged output is bit-compatible with
    /// the shared path. Used by tests and standalone embedders that don't
    /// carry a memo instance.
    fn from_semantic(
        vable_array: &[ResumeValueSource],
        vref_array: &[ResumeValueSource],
        frames: &[FrameInfo],
        virtuals: &[VirtualInfo],
        pending_fields: &[PendingFieldInfo],
    ) -> Self {
        let mut memo = ResumeDataLoopMemo::new();
        let mut rd_numb = Vec::new();
        // resume.py:138 numb_state.liveboxes — compact TAGBOX numbering state.
        let mut liveboxes: Vec<usize> = Vec::new();
        let mut box_map: HashMap<usize, usize> = HashMap::new();

        // resume.py:234-235: reserve slots for items_resume_section and count.
        rd_numb.push(0); // [0] = items_resume_section (patched later)
        rd_numb.push(0); // [1] = count (patched later)
        rd_numb.push(encode_len(vable_array.len()));
        for source in vable_array {
            let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }
        // resume.py:243-247: vref_array (pairs).
        assert!(
            vref_array.len() % 2 == 0,
            "vref_array must have even length (pairs)"
        );
        rd_numb.push(encode_len(vref_array.len() / 2));
        for source in vref_array {
            let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }

        // resume.py:249-253: per-frame encoding via _number_boxes.
        let mut frame_sizes = Vec::with_capacity(frames.len());
        for frame in frames {
            rd_numb.push(frame.jitcode_index as i64);
            rd_numb.push(encode_u64(frame.pc));
            // resume.py:253 _number_boxes(snapshot_iter, iter_array(snapshot), numb_state)
            for source in &frame.slot_map {
                let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
                rd_numb.push(tagged);
            }
            frame_sizes.push(frame.slot_map.len());
        }

        // compile.py:858 rd_virtuals — stored as live objects, not serialized.
        let rd_virtuals = virtuals.to_vec();

        // resume.py:412-418: visitor_walk_recursive — register virtual field boxes.
        for vinfo in &rd_virtuals {
            for source in vinfo.field_sources() {
                if let ResumeValueSource::FailArg(index) = source {
                    box_map.entry(*index).or_insert_with(|| {
                        let n = liveboxes.len();
                        liveboxes.push(*index);
                        n
                    });
                }
            }
        }

        // resume.py:420-430: walk pending fields — register + encode.
        let rd_pendingfields: Vec<_> = pending_fields
            .iter()
            .map(|pending| EncodedPendingFieldWrite {
                descr_index: pending.descr_index,
                target: memo.encode_tagged_source(&pending.target, &mut liveboxes, &mut box_map),
                value: memo.encode_tagged_source(&pending.value, &mut liveboxes, &mut box_map),
                item_index: pending.item_index,
            })
            .collect();
        let rd_consts = memo.take_consts();

        // resume.py:260: numb_state.patch_current_size(0) → items_resume_section
        rd_numb[0] = encode_len(rd_numb.len());
        // resume.py:464: numb_state.patch(1, len(liveboxes)) → count
        rd_numb[1] = encode_len(liveboxes.len());

        EncodedResumeData {
            rd_numb,
            rd_consts,
            rd_pendingfields,
            rd_virtuals,
            liveboxes,
            frame_sizes,
        }
    }

    /// resume.py:916-923 AbstractResumeDataReader._init — decode rd_numb.
    fn decode_layout(&self) -> DecodedResumeLayout {
        let mut cursor = 0usize;
        // resume.py:919 items_resume_section
        let items_resume_section = self.next_word(&mut cursor);
        assert_eq!(
            decode_len(items_resume_section),
            self.rd_numb.len(),
            "resume item count mismatch"
        );
        // resume.py:921 self.count — number of liveboxes in the deadframe.
        let _count = decode_len(self.next_word(&mut cursor));

        let vable_count = decode_len(self.next_word(&mut cursor));
        let mut vable_array = Vec::with_capacity(vable_count);
        for _ in 0..vable_count {
            vable_array.push(self.decode_box(self.next_word(&mut cursor)));
        }
        let vref_count = decode_len(self.next_word(&mut cursor));
        let mut vref_array = Vec::with_capacity(vref_count * 2);
        for _ in 0..(vref_count * 2) {
            vref_array.push(self.decode_box(self.next_word(&mut cursor)));
        }
        // resume.py:1049-1055: frame section.
        // Per-frame: jitcode_index, pc, [tagged_values...].
        // RPython uses jitcode.get_live_vars_info(pc) for frame boundary;
        // we use self.frame_sizes[] stored at encode time.
        let items_resume_len = decode_len(items_resume_section);
        let mut frames = Vec::new();
        let mut frame_idx = 0usize;
        while cursor < items_resume_len {
            let jitcode_index = self.next_word(&mut cursor) as i32;
            let pc = decode_u64(self.next_word(&mut cursor));
            let slot_count = if frame_idx < self.frame_sizes.len() {
                self.frame_sizes[frame_idx]
            } else {
                // Single-frame fallback: consume all remaining items.
                items_resume_len - cursor
            };
            let mut slot_map = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                slot_map.push(self.decode_box(self.next_word(&mut cursor)));
            }
            frames.push(FrameInfo {
                jitcode_index,
                pc,
                slot_map,
            });
            frame_idx += 1;
        }

        // compile.py:858 rd_virtuals — live objects, not deserialized from rd_numb.
        let virtuals = self.rd_virtuals.clone();

        assert_eq!(
            cursor,
            self.rd_numb.len(),
            "resume decoder left trailing data"
        );
        // resume.py:926 _prepare_pendingfields
        let pending_fields = self
            .rd_pendingfields
            .iter()
            .map(|pending| PendingFieldInfo {
                descr_index: pending.descr_index,
                target: self.decode_box(pending.target),
                value: self.decode_box(pending.value),
                item_index: pending.item_index,
            })
            .collect();
        DecodedResumeLayout {
            vable_array,
            vref_array,
            frames,
            virtuals,
            pending_fields,
        }
    }

    /// resume.py:919 resumecodereader.next_item()
    fn next_word(&self, cursor: &mut usize) -> i64 {
        let word = self
            .rd_numb
            .get(*cursor)
            .copied()
            .expect("truncated encoded resume data");
        *cursor += 1;
        word
    }

    /// resume.py:1240-1270 decode_box — decode a tagged value from rd_numb.
    fn decode_box(&self, encoded: i64) -> ResumeValueSource {
        let (value, tag) = untag_i64(encoded);
        match tag {
            // resume.py:1257 ConstInt(num).
            TAGINT => ResumeValueSource::Constant(majit_ir::Const::Int(value)),
            // resume.py:1261 self.liveboxes[num] — compact TAGBOX → original FailArg.
            TAGBOX => {
                let compact_idx = decode_len(value);
                let original_idx = self.liveboxes[compact_idx];
                ResumeValueSource::FailArg(original_idx)
            }
            TAGVIRTUAL => ResumeValueSource::Virtual(decode_len(value)),
            TAGCONST => match value {
                // resume.py:1552-1596 decode_ref: `if tagged_eq(tagged,
                // NULLREF): return CONST_NULL`. The i64 decoder mirrors
                // the i16 `decode_box`'s NULLREF fast-path (resume.rs
                // line 4363) so encoder/decoder stay symmetric.
                ENCODED_NULLREF => {
                    ResumeValueSource::Constant(majit_ir::Const::Ref(majit_ir::GcRef::NULL))
                }
                ENCODED_UNINITIALIZED => ResumeValueSource::Uninitialized,
                ENCODED_UNAVAILABLE => ResumeValueSource::Unavailable,
                index if index >= 0 => {
                    // resume.py:1555/1571/1583 self.consts[num - TAG_CONST_OFFSET]
                    // — the Const carries its own type.
                    let c = *self
                        .rd_consts
                        .get(decode_len(index))
                        .expect("resume const pool index out of bounds");
                    ResumeValueSource::Constant(c)
                }
                other => panic!("unknown CONST-tagged resume sentinel {other}"),
            },
            other => panic!("unknown resume tag {other}"),
        }
    }

    /// Decode this encoded snapshot back into a `ResumeData`.
    pub fn decode(&self) -> ResumeData {
        let layout = self.decode_layout();
        ResumeData {
            vable_array: layout.vable_array,
            vref_array: layout.vref_array,
            frames: layout.frames,
            virtuals: layout.virtuals,
            pending_fields: layout.pending_fields,
        }
    }

    /// Return a compact summary of this snapshot's frame/jitframe layout.
    pub fn layout_summary(&self) -> ResumeLayoutSummary {
        let layout = self.decode_layout();
        ResumeLayoutSummary {
            num_frames: layout.frames.len(),
            frame_pcs: layout.frames.iter().map(|frame| frame.pc).collect(),
            frame_slot_counts: layout
                .frames
                .iter()
                .map(|frame| frame.slot_map.len())
                .collect(),
            frame_layouts: layout
                .frames
                .iter()
                .map(|frame| ResumeFrameLayoutSummary {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    jitcode_index: frame.jitcode_index,
                    pc: frame.pc,
                    slot_sources: frame.slot_map.iter().map(ResumeValueSource::kind).collect(),
                    slot_layouts: frame
                        .slot_map
                        .iter()
                        .map(|source| source.layout_summary())
                        .collect(),
                    slot_types: None,
                })
                .collect(),
            num_virtuals: layout.virtuals.len(),
            virtual_kinds: layout.virtuals.iter().map(VirtualInfo::kind).collect(),
            virtual_layouts: layout
                .virtuals
                .iter()
                .map(|virt| virt.layout_summary())
                .collect(),

            pending_field_count: layout.pending_fields.len(),
            pending_field_layouts: layout
                .pending_fields
                .iter()
                .map(|pending| pending.layout_summary())
                .collect(),
            const_pool_size: self.rd_consts.len(),
        }
    }

    /// Reconstruct the full interpreter state directly from the encoded snapshot.
    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let layout = self.decode_layout();
        let virtuals = ResumeData::materialize_virtuals_from_infos(&layout.virtuals, fail_values);
        let pending_fields =
            ResumeData::resolve_pending_field_writes(&layout.pending_fields, fail_values);
        let frames = layout
            .frames
            .iter()
            .map(|frame| ReconstructedFrame {
                trace_id: None,
                header_pc: None,
                source_guard: None,
                pc: frame.pc,
                jitcode_index: frame.jitcode_index,
                slot_types: None,
                values: frame
                    .slot_map
                    .iter()
                    .map(|slot| ResumeData::resolve_frame_slot_source(slot, fail_values))
                    .collect(),
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals,
            pending_fields,
        }
    }

    /// Reconstruct only the interpreter frames from the encoded snapshot.
    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    /// Materialize virtual objects referenced by this encoded snapshot.
    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        let layout = self.decode_layout();
        ResumeData::materialize_virtuals_from_infos(&layout.virtuals, fail_values)
    }

    /// Resolve pending heap writes referenced by this encoded snapshot.
    pub fn resolve_pending_field_writes(
        &self,
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        let layout = self.decode_layout();
        ResumeData::resolve_pending_field_writes(&layout.pending_fields, fail_values)
    }
}

impl ResumeData {
    /// Create a simple ResumeData for a single-frame trace.
    pub fn simple(pc: u64, num_slots: usize) -> Self {
        let slot_map: Vec<FrameSlotSource> = (0..num_slots).map(FrameSlotSource::FailArg).collect();
        ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: vec![FrameInfo {
                jitcode_index: 0,
                pc,
                slot_map,
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    /// Encode this resume snapshot into a compact RPython-style numbering.
    pub fn encode(&self) -> EncodedResumeData {
        EncodedResumeData::from_semantic(
            &self.vable_array,
            &self.vref_array,
            &self.frames,
            &self.virtuals,
            &self.pending_fields,
        )
    }

    fn decode_layout(&self) -> DecodedResumeLayout {
        self.encode().decode_layout()
    }

    /// Reconstruct the full resume state from fail_args data.
    ///
    /// `fail_values` are the concrete values extracted from the DeadFrame.
    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let decoded = self.decode_layout();
        let materialized_virtuals =
            Self::materialize_virtuals_from_infos(&decoded.virtuals, fail_values);
        let frames = decoded
            .frames
            .iter()
            .map(|frame| {
                let values = frame
                    .slot_map
                    .iter()
                    .map(|slot| Self::resolve_frame_slot_source(slot, fail_values))
                    .collect();
                ReconstructedFrame {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: frame.pc,
                    jitcode_index: frame.jitcode_index,
                    slot_types: None,
                    values,
                }
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals: materialized_virtuals,
            pending_fields: Self::resolve_pending_field_writes(
                &decoded.pending_fields,
                fail_values,
            ),
        }
    }

    /// Reconstruct frame slots from fail_args data.
    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    /// Materialize virtual objects from resume data.
    ///
    /// When a guard fails and some fail_args slots hold virtual objects
    /// (objects that were never allocated during optimized execution),
    /// this method resolves each VirtualInfo into a `MaterializedVirtual`.
    ///
    /// `fail_values` are the concrete i64 values from the DeadFrame.
    ///
    /// Mirrors RPython's `ResumeDataVirtualAdder.finish()` → `virtual_materialize()`.
    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        let decoded = self.decode_layout();
        Self::materialize_virtuals_from_infos(&decoded.virtuals, fail_values)
    }

    fn materialize_virtuals_from_infos(
        virtuals: &[VirtualInfo],
        fail_values: &[i64],
    ) -> Vec<MaterializedVirtual> {
        let mut result = Vec::with_capacity(virtuals.len());
        // First pass: create empty materialized shells (handles forward references)
        for vinfo in virtuals {
            result.push(MaterializedVirtual::from_info(vinfo));
        }

        // Second pass: resolve field sources
        for (i, vinfo) in virtuals.iter().enumerate() {
            result[i].resolve_fields(vinfo, fail_values);
        }

        result
    }

    /// Resolve pending heap writes into concrete values.
    pub fn resolve_pending_field_writes(
        pending_fields: &[PendingFieldInfo],
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        pending_fields
            .iter()
            .map(|pending| ResolvedPendingFieldWrite {
                descr_index: pending.descr_index,
                target: Self::resolve_materialized_source(&pending.target, fail_values),
                value: Self::resolve_materialized_source(&pending.value, fail_values),
                item_index: pending.item_index,
            })
            .collect()
    }

    /// Resolve a single VirtualFieldSource to a concrete i64 value.
    ///
    /// For `Virtual(idx)` references, returns 0 as a placeholder —
    /// the actual object address is only known after allocation by the
    /// interpreter's allocator.
    pub fn resolve_field_source(source: &VirtualFieldSource, fail_values: &[i64]) -> i64 {
        match Self::resolve_materialized_source(source, fail_values) {
            MaterializedValue::Value(value) => value,
            MaterializedValue::VirtualRef(_) => 0,
        }
    }

    pub fn resolve_materialized_source(
        source: &VirtualFieldSource,
        fail_values: &[i64],
    ) -> MaterializedValue {
        match source {
            ResumeValueSource::FailArg(idx) => {
                MaterializedValue::Value(fail_values.get(*idx).copied().unwrap_or(0))
            }
            ResumeValueSource::Constant(c) => MaterializedValue::Value(c.as_raw_i64()),
            ResumeValueSource::Virtual(idx) => MaterializedValue::VirtualRef(*idx),
            ResumeValueSource::Uninitialized | ResumeValueSource::Unavailable => {
                MaterializedValue::Value(0)
            }
        }
    }

    /// Resolve a single frame-slot source into a reconstructed value.
    pub fn resolve_frame_slot_source(
        source: &FrameSlotSource,
        fail_values: &[i64],
    ) -> ReconstructedValue {
        match source {
            ResumeValueSource::FailArg(idx) => {
                ReconstructedValue::Value(fail_values.get(*idx).copied().unwrap_or(0))
            }
            ResumeValueSource::Constant(c) => ReconstructedValue::Value(c.as_raw_i64()),
            ResumeValueSource::Virtual(idx) => ReconstructedValue::Virtual(*idx),
            ResumeValueSource::Uninitialized => ReconstructedValue::Uninitialized,
            ResumeValueSource::Unavailable => ReconstructedValue::Unavailable,
        }
    }
}

/// A reconstructed interpreter frame from resume data.
#[derive(Debug, Clone)]
pub struct ReconstructedFrame {
    /// Compiled trace identifier for this frame, when known.
    pub trace_id: Option<u64>,
    /// Trace header pc associated with this frame, when known.
    pub header_pc: Option<u64>,
    /// Source guard this frame's trace is attached to, when known.
    pub source_guard: Option<(u64, u32)>,
    /// Program counter for this frame.
    pub pc: u64,
    /// resume.py:1051: jitcode index for CodeObject lookup.
    pub jitcode_index: i32,
    /// Typed layout of the reconstructed slots, when known.
    pub slot_types: Option<Vec<Type>>,
    /// Reconstructed values for each slot.
    pub values: Vec<ReconstructedValue>,
}

impl ReconstructedFrame {
    /// Lossy conversion: extract integer values, dropping virtual/unavailable info.
    pub fn lossy_values(&self) -> Vec<i64> {
        self.values
            .iter()
            .map(ReconstructedValue::lossy_i64)
            .collect()
    }
}

/// Full reconstructed state for a guard recovery.
#[derive(Debug, Clone)]
pub struct ReconstructedState {
    /// Reconstructed interpreter frames, outermost first.
    pub frames: Vec<ReconstructedFrame>,
    /// Materialized virtual objects referenced by frame slots.
    pub virtuals: Vec<MaterializedVirtual>,
    /// Deferred heap writes that the interpreter must replay after reconstruction.
    pub pending_fields: Vec<ResolvedPendingFieldWrite>,
}

/// Reconstructed slot value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructedValue {
    /// Concrete raw value, including ints, refs, and float bits.
    Value(i64),
    /// Reference to a materialized virtual in `ReconstructedState.virtuals`.
    Virtual(usize),
    /// Slot exists but remains uninitialized.
    Uninitialized,
    /// Slot is dead/unavailable at this guard.
    Unavailable,
}

impl ReconstructedValue {
    /// Lossy conversion used by the current integer-only compatibility layer.
    pub fn lossy_i64(&self) -> i64 {
        match self {
            ReconstructedValue::Value(value) => *value,
            ReconstructedValue::Virtual(_)
            | ReconstructedValue::Uninitialized
            | ReconstructedValue::Unavailable => 0,
        }
    }
}

/// A materialized virtual object, ready for the interpreter to allocate.
///
/// After a guard failure, virtual objects must be allocated on the heap
/// and their fields populated from the DeadFrame values. This struct
/// holds the resolved field values for a single virtual object.
///
/// Mirrors RPython's `_materialize_virtual()` in resume.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterializedValue {
    Value(i64),
    VirtualRef(usize),
}

impl MaterializedValue {
    pub fn resolve_with_refs(&self, materialized_refs: &[Option<GcRef>]) -> Option<i64> {
        match self {
            MaterializedValue::Value(value) => Some(*value),
            MaterializedValue::VirtualRef(index) => materialized_refs
                .get(*index)
                .copied()
                .flatten()
                .map(|gc_ref| gc_ref.as_usize() as i64),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MaterializedVirtual {
    /// Object with vtable — resume.py:612 VirtualInfo.
    /// Carries `descr` (resume.py:615 self.descr) so the deopt path can
    /// `allocate_with_vtable(descr=self.descr)` and replay fields generically,
    /// without special-casing the vtable at the JIT-state layer.
    Obj {
        /// SizeDescr for allocation (exposes vtable + obj_size).
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        /// (field_descr_index, concrete_value).
        fields: Vec<(u32, MaterializedValue)>,
    },
    /// Plain struct — resume.py:628 VStructInfo.
    Struct {
        /// resume.py:631 self.typedescr.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, MaterializedValue)>,
    },
    /// Array.
    Array {
        descr_index: u32,
        items: Vec<MaterializedValue>,
    },
    /// Array of structs.
    ArrayStruct {
        descr_index: u32,
        /// Per-element: Vec<(field_index, value)>.
        elements: Vec<Vec<(u32, MaterializedValue)>>,
    },
    /// Raw buffer.
    RawBuffer {
        func: i64,
        size: usize,
        offsets: Vec<usize>,
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        values: Vec<MaterializedValue>,
    },
}

impl MaterializedVirtual {
    /// Create an empty shell from a VirtualInfo (forward-reference safe).
    fn from_info(info: &VirtualInfo) -> Self {
        match info {
            VirtualInfo::VirtualObj {
                descr,
                type_id,
                descr_index,
                ..
            } => MaterializedVirtual::Obj {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: Vec::new(),
            },
            VirtualInfo::VStruct {
                typedescr,
                type_id,
                descr_index,
                ..
            } => MaterializedVirtual::Struct {
                descr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: Vec::new(),
            },
            VirtualInfo::VArray {
                arraydescr: _,
                descr_index,
                clear: _,
                items,
            } => MaterializedVirtual::Array {
                descr_index: *descr_index,
                items: vec![MaterializedValue::Value(0); items.len()],
            },
            VirtualInfo::VArrayStruct {
                arraydescr: _,
                descr_index,
                fielddescrs: _,
                element_fields,
            } => MaterializedVirtual::ArrayStruct {
                descr_index: *descr_index,
                elements: vec![Vec::new(); element_fields.len()],
            },
            VirtualInfo::VRawBuffer {
                func,
                size,
                offsets,
                descrs,
                ..
            } => MaterializedVirtual::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: vec![MaterializedValue::Value(0); offsets.len()],
            },
            VirtualInfo::VRawSlice { .. } => MaterializedVirtual::Struct {
                descr: None,
                type_id: 0,
                descr_index: 0,
                fields: Vec::new(),
            },
            // resume.py:763-870 VStr/VUni*Info — virtual string shells
            // reserved for future vstring.py port. Represented as struct
            // shells for now (zero fields) so the materializer doesn't
            // walk into them; actual allocate_string / string_setitem /
            // concat_strings / slice_string still live on the roadmap.
            VirtualInfo::VStrPlain { .. }
            | VirtualInfo::VStrConcat { .. }
            | VirtualInfo::VStrSlice { .. }
            | VirtualInfo::VUniPlain { .. }
            | VirtualInfo::VUniConcat { .. }
            | VirtualInfo::VUniSlice { .. } => MaterializedVirtual::Struct {
                descr: None,
                type_id: 0,
                descr_index: 0,
                fields: Vec::new(),
            },
        }
    }

    /// Resolve fields from fail_values.
    fn resolve_fields(&mut self, info: &VirtualInfo, fail_values: &[i64]) {
        match (self, info) {
            (
                MaterializedVirtual::Obj { fields, .. },
                VirtualInfo::VirtualObj {
                    fields: src_fields, ..
                },
            )
            | (
                MaterializedVirtual::Struct { fields, .. },
                VirtualInfo::VStruct {
                    fields: src_fields, ..
                },
            ) => {
                *fields = src_fields
                    .iter()
                    .map(|(idx, src)| {
                        (
                            *idx,
                            ResumeData::resolve_materialized_source(src, fail_values),
                        )
                    })
                    .collect();
            }
            (
                MaterializedVirtual::Array { items, .. },
                VirtualInfo::VArray {
                    items: src_items, ..
                },
            ) => {
                *items = src_items
                    .iter()
                    .map(|src| ResumeData::resolve_materialized_source(src, fail_values))
                    .collect();
            }
            (
                MaterializedVirtual::ArrayStruct { elements, .. },
                VirtualInfo::VArrayStruct {
                    element_fields: src_elems,
                    ..
                },
            ) => {
                *elements = src_elems
                    .iter()
                    .map(|elem_fields| {
                        elem_fields
                            .iter()
                            .map(|(idx, src)| {
                                (
                                    *idx,
                                    ResumeData::resolve_materialized_source(src, fail_values),
                                )
                            })
                            .collect()
                    })
                    .collect();
            }
            (
                MaterializedVirtual::RawBuffer { values, .. },
                VirtualInfo::VRawBuffer {
                    values: src_values, ..
                },
            ) => {
                *values = src_values
                    .iter()
                    .map(|src| ResumeData::resolve_materialized_source(src, fail_values))
                    .collect();
            }
            _ => {} // type mismatch — should not happen
        }
    }

    pub fn resolve_with_refs(
        &self,
        materialized_refs: &[Option<GcRef>],
    ) -> Option<MaterializedVirtual> {
        match self {
            MaterializedVirtual::Obj {
                descr,
                type_id,
                descr_index,
                fields,
            } => Some(MaterializedVirtual::Obj {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(idx, value)| {
                        Some((
                            *idx,
                            MaterializedValue::Value(value.resolve_with_refs(materialized_refs)?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::Struct {
                descr,
                type_id,
                descr_index,
                fields,
            } => Some(MaterializedVirtual::Struct {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(idx, value)| {
                        Some((
                            *idx,
                            MaterializedValue::Value(value.resolve_with_refs(materialized_refs)?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::Array { descr_index, items } => Some(MaterializedVirtual::Array {
                descr_index: *descr_index,
                items: items
                    .iter()
                    .map(|value| {
                        Some(MaterializedValue::Value(
                            value.resolve_with_refs(materialized_refs)?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::ArrayStruct {
                descr_index,
                elements,
            } => Some(MaterializedVirtual::ArrayStruct {
                descr_index: *descr_index,
                elements: elements
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(idx, value)| {
                                Some((
                                    *idx,
                                    MaterializedValue::Value(
                                        value.resolve_with_refs(materialized_refs)?,
                                    ),
                                ))
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => Some(MaterializedVirtual::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|value| {
                        Some(MaterializedValue::Value(
                            value.resolve_with_refs(materialized_refs)?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
        }
    }
}

/// Builder for constructing ResumeData during trace compilation.
///
/// resume.py:298-493 ResumeDataVirtualAdder.finish() is implemented
/// across two functions in majit:
/// - `store_final_boxes_in_guard` (mod.rs) — numbering + rd_numb/rd_consts
/// - `store_final_boxes_in_guard` (optimizer.rs) — virtual expansion + rd_virtuals
pub struct ResumeDataVirtualAdder {
    vable_array: Vec<ResumeValueSource>,
    vref_array: Vec<ResumeValueSource>,
    frames: Vec<FrameInfoBuilder>,
    virtuals: Vec<VirtualInfo>,
    pending_fields: Vec<PendingFieldInfo>,
}

struct FrameInfoBuilder {
    jitcode_index: i32,
    pc: u64,
    slot_map: Vec<FrameSlotSource>,
}

impl ResumeDataVirtualAdder {
    /// Create a new builder.
    pub fn new() -> Self {
        ResumeDataVirtualAdder {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: Vec::new(),
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    pub fn set_vable_array(&mut self, values: Vec<ResumeValueSource>) {
        self.vable_array = values;
    }

    /// Push a new frame onto the stack.
    /// resume.py:249-252: jitcode_index, pc per frame.
    pub fn push_frame(&mut self, jitcode_index: i32, pc: u64) {
        self.frames.push(FrameInfoBuilder {
            jitcode_index,
            pc,
            slot_map: Vec::new(),
        });
    }

    /// Map a slot in the current frame to a fail_arg index.
    pub fn map_slot(&mut self, slot_idx: usize, fail_arg_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::FailArg(fail_arg_idx));
    }

    /// Set a slot in the current frame to a tagged source.
    pub fn set_slot_source(&mut self, slot_idx: usize, source: FrameSlotSource) {
        let frame = self.frames.last_mut().expect("no frame pushed");
        while frame.slot_map.len() <= slot_idx {
            frame.slot_map.push(FrameSlotSource::Unavailable);
        }
        frame.slot_map[slot_idx] = source;
    }

    /// Set a frame slot to a compile-time constant.
    pub fn set_slot_constant(&mut self, slot_idx: usize, constant: majit_ir::Const) {
        self.set_slot_source(slot_idx, FrameSlotSource::Constant(constant));
    }

    /// Set a frame slot to reference a virtual object.
    pub fn set_slot_virtual(&mut self, slot_idx: usize, virtual_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Virtual(virtual_idx));
    }

    /// Mark a frame slot as present but uninitialized.
    pub fn set_slot_uninitialized(&mut self, slot_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Uninitialized);
    }

    /// Mark a frame slot as dead/unavailable.
    pub fn set_slot_unavailable(&mut self, slot_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Unavailable);
    }

    /// Add a virtual object description. Returns the index in the virtuals array.
    pub fn add_virtual(&mut self, info: VirtualInfo) -> usize {
        let idx = self.virtuals.len();
        self.virtuals.push(info);
        idx
    }

    /// Convenience: add a virtual object (NEW_WITH_VTABLE).
    pub fn add_virtual_obj(
        &mut self,
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        known_class: Option<i64>,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    ) -> usize {
        self.add_virtual(VirtualInfo::VirtualObj {
            descr,
            type_id,
            descr_index,
            known_class,
            fields,
            fielddescrs,
            descr_size,
        })
    }

    /// Convenience: add a virtual struct (NEW).
    pub fn add_virtual_struct(
        &mut self,
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    ) -> usize {
        self.add_virtual(VirtualInfo::VStruct {
            typedescr,
            type_id,
            descr_index,
            fields,
            fielddescrs,
            descr_size,
        })
    }

    /// Convenience: add a virtual array (NEW_ARRAY).
    pub fn add_virtual_array(
        &mut self,
        arraydescr: Option<majit_ir::DescrRef>,
        descr_index: u32,
        clear: bool,
        items: Vec<VirtualFieldSource>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VArray {
            arraydescr,
            descr_index,
            clear,
            items,
        })
    }

    /// resume.py:332: visit_varraystruct(arraydescr, size, fielddescrs)
    ///                 → VArrayStructInfo(arraydescr, size, fielddescrs)
    pub fn add_virtual_array_struct(
        &mut self,
        arraydescr: Option<majit_ir::DescrRef>,
        descr_index: u32,
        fielddescrs: Vec<majit_ir::DescrRef>,
        element_fields: Vec<Vec<(u32, VirtualFieldSource)>>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VArrayStruct {
            arraydescr,
            descr_index,
            fielddescrs,
            element_fields,
        })
    }

    /// Convenience: add a virtual raw buffer.
    pub fn add_virtual_raw_buffer(
        &mut self,
        func: i64,
        size: usize,
        offsets: Vec<usize>,
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        values: Vec<VirtualFieldSource>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VRawBuffer {
            func,
            size,
            offsets,
            descrs,
            values,
        })
    }

    /// Add a deferred field write to replay on resume.
    pub fn add_pending_field_write(
        &mut self,
        descr_index: u32,
        target: ResumeValueSource,
        value: ResumeValueSource,
    ) {
        self.pending_fields.push(PendingFieldInfo {
            descr_index,
            target,
            value,
            item_index: None,
        });
    }

    /// Add a deferred array item write to replay on resume.
    pub fn add_pending_arrayitem_write(
        &mut self,
        descr_index: u32,
        target: ResumeValueSource,
        item_index: usize,
        value: ResumeValueSource,
    ) {
        self.pending_fields.push(PendingFieldInfo {
            descr_index,
            target,
            value,
            item_index: Some(item_index),
        });
    }

    /// resume.py: visit_vrawslice(offset) — add virtual raw slice.
    pub fn add_virtual_raw_slice(&mut self, offset: i64, parent: VirtualFieldSource) -> usize {
        self.add_virtual(VirtualInfo::VRawSlice { offset, parent })
    }

    /// Build the final ResumeData.
    pub fn build(self) -> ResumeData {
        ResumeData {
            vable_array: self.vable_array,
            vref_array: self.vref_array,
            frames: self
                .frames
                .into_iter()
                .map(|f| FrameInfo {
                    jitcode_index: f.jitcode_index,
                    pc: f.pc,
                    slot_map: f.slot_map,
                })
                .collect(),
            virtuals: self.virtuals,
            pending_fields: self.pending_fields,
        }
    }
}

impl Default for ResumeDataVirtualAdder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fail-arg compression ─────────────────────────────────────────────

/// Shared resume data storage that deduplicates common snapshot sections
/// across multiple guards in the same trace.
///
/// RPython's `ResumeDataLoopMemo` shares constant pools and frame sections
/// across guards. We use a shared `ResumeEncoder` state so that the same
/// large constant only appears once in the pool.
/// RPython resume.py:142 ResumeDataLoopMemo.
/// Shared constant pool + box numbering cache across all guards in a loop.
///
/// NOTE: RPython's ResumeDataLoopMemo also stores `metainterp_sd` and `cpu`
/// (for box allocation during rebuild). pyre doesn't need these: the BoxEnv
/// trait provides box access. The canonical decoder lives in
/// `majit_ir::resumedata::rebuild_from_numbering`.
/// RPython's nvirtuals/nvholes/nvreused stats are kept for monitoring.
pub struct ResumeDataLoopMemo {
    /// resume.py:147 — shared constant pool.
    /// RPython stores Const objects (with type INT/REF/FLOAT).
    /// We store (value, type) pairs to preserve type information.
    consts: Vec<majit_ir::Const>,
    /// resume.py:148 — large integers (outside TAGINT range) → tagged const.
    large_ints: HashMap<i64, i16>,
    /// resume.py:149 — ref pointers → tagged const.
    refs: HashMap<i64, i16>,
    /// resume.py:147 self.consts — constant pool for encode_shared.
    /// Becomes storage.rd_consts (resume.py:467).
    ///
    /// resume.py:150-151 — cached box/virtual numbering.
    pub cached_boxes: HashMap<OpRef, i32>,
    pub cached_virtuals: HashMap<OpRef, i32>,
    /// resume.py:153-155 — statistics.
    pub nvirtuals: usize,
    pub nvholes: usize,
    pub nvreused: usize,
}

impl ResumeDataLoopMemo {
    pub fn new() -> Self {
        ResumeDataLoopMemo {
            consts: Vec::new(),
            large_ints: HashMap::new(),
            refs: HashMap::new(),
            cached_boxes: HashMap::new(),
            cached_virtuals: HashMap::new(),
            nvirtuals: 0,
            nvholes: 0,
            nvreused: 0,
        }
    }

    /// resume.py:199-226 `_number_boxes` + resume.py:209 `getconst` parity.
    ///
    /// Encode one `ResumeValueSource` into the i64 tagged form written to
    /// `rd_numb`. `liveboxes` / `box_map` track compact TAGBOX numbering
    /// identical to RPython's `numb_state.liveboxes` dict.
    ///
    /// Pushes Ref/Float/large-Int constants through `getconst_i64`, which
    /// shares `self.consts` with the i16 `getconst` path — so there is
    /// exactly one pool per memo (RPython parity: `self.consts: list[Const]`).
    pub fn encode_tagged_source(
        &mut self,
        source: &ResumeValueSource,
        liveboxes: &mut Vec<usize>,
        box_map: &mut HashMap<usize, usize>,
    ) -> i64 {
        match source {
            // resume.py:214-224: new box → liveboxes[box] = tag(num_boxes, TAGBOX)
            ResumeValueSource::FailArg(index) => {
                let compact = *box_map.entry(*index).or_insert_with(|| {
                    let n = liveboxes.len();
                    liveboxes.push(*index);
                    n
                });
                tag_i64(encode_len(compact), TAGBOX)
            }
            // resume.py:209: isinstance(box, Const) → self.getconst(box).
            ResumeValueSource::Constant(c) => self.getconst_i64(c),
            // resume.py:219-221: virtual → tag(num_virtuals, TAGVIRTUAL)
            ResumeValueSource::Virtual(index) => tag_i64(encode_len(*index), TAGVIRTUAL),
            ResumeValueSource::Uninitialized => tag_i64(ENCODED_UNINITIALIZED, TAGCONST),
            ResumeValueSource::Unavailable => tag_i64(ENCODED_UNAVAILABLE, TAGCONST),
        }
    }

    /// resume.py:157-183 getconst(const) — tag a constant value.
    /// Unified entry point matching RPython's getconst(const) which
    /// dispatches on const.type (INT, REF, FLOAT).
    pub fn getconst(&mut self, val: i64, tp: majit_ir::Type) -> i16 {
        match tp {
            majit_ir::Type::Int => self.getconst_int(val),
            majit_ir::Type::Ref => self.getconst_ref(val),
            majit_ir::Type::Float => self.getconst_float(val),
            majit_ir::Type::Void => self.newconst(val, tp),
        }
    }

    /// resume.py:158-172 getconst for INT type.
    pub fn getconst_int(&mut self, val: i64) -> i16 {
        // Try inline TAGINT (-8191..8190 in RPython's i16 range).
        let shifted = val >> 13;
        if shifted == 0 || shifted == -1 {
            return ((val << 2) | TAGINT as i64) as i16;
        }
        // Large int: check cache.
        if let Some(&tagged) = self.large_ints.get(&val) {
            return tagged;
        }
        let tagged = self.newconst(val, majit_ir::Type::Int);
        self.large_ints.insert(val, tagged);
        tagged
    }

    /// resume.py:173-182 getconst for REF type.
    pub fn getconst_ref(&mut self, val: i64) -> i16 {
        if val == 0 {
            return NULLREF;
        }
        if let Some(&tagged) = self.refs.get(&val) {
            return tagged;
        }
        let tagged = self.newconst(val, majit_ir::Type::Ref);
        self.refs.insert(val, tagged);
        tagged
    }

    /// resume.py:183 getconst fallback for FLOAT type.
    pub fn getconst_float(&mut self, val: i64) -> i16 {
        // FLOAT constants always go to the pool (no inline encoding).
        // RPython: return self._newconst(const)
        self.newconst(val, majit_ir::Type::Float)
    }

    /// resume.py:185 _newconst — add to consts pool, return TAGCONST-tagged.
    fn newconst(&mut self, val: i64, tp: majit_ir::Type) -> i16 {
        let index = self.consts.len() as i32 + TAG_CONST_OFFSET;
        self.consts.push(majit_ir::Const::from_raw_i64(val, tp));
        ((index << 2) | TAGCONST as i32) as i16
    }

    /// resume.py:161-188 getconst — i64-sized variant used by the rd_numb
    /// encoder (`encode_shared`). Shares the pool (`self.consts`) with the
    /// i16 variant (`getconst`) so there is exactly one `rd_consts` per
    /// memo, matching RPython's single `self.consts: list[Const]`.
    fn getconst_i64(&mut self, c: &majit_ir::Const) -> i64 {
        match c {
            // resume.py:163-167: try tag(val, TAGINT).
            majit_ir::Const::Int(value) if can_inline_tagged(*value) => tag_i64(*value, TAGINT),
            majit_ir::Const::Int(value) => {
                // resume.py:168-172 large int.
                if let Some(&tagged_i16) = self.large_ints.get(value) {
                    let (num, _) = untag(tagged_i16);
                    return tag_i64(encode_len((num - TAG_CONST_OFFSET) as usize), TAGCONST);
                }
                let index = self.consts.len();
                self.consts.push(majit_ir::Const::Int(*value));
                // Also publish through the i16 cache so that a later
                // `getconst_int(value)` returns the same pool slot
                // (resume.py:171 self.large_ints[val] = tagged).
                let tagged_i16 =
                    ((((index as i32) + TAG_CONST_OFFSET) << 2) | TAGCONST as i32) as i16;
                self.large_ints.insert(*value, tagged_i16);
                tag_i64(encode_len(index), TAGCONST)
            }
            majit_ir::Const::Ref(gcref) => {
                // resume.py:174-176 val = 0 → NULLREF sentinel (no pool
                // entry allocated). `NULLREF = tag(-1, TAGCONST)` —
                // encoder emits `tag_i64(-1, TAGCONST)` and the
                // matching decoder in `decode_box` recognizes
                // `ENCODED_NULLREF` before the positive-index branch.
                let raw = gcref.as_usize() as i64;
                if raw == 0 {
                    return tag_i64(ENCODED_NULLREF, TAGCONST);
                }
                if let Some(&tagged_i16) = self.refs.get(&raw) {
                    let (num, _) = untag(tagged_i16);
                    return tag_i64(encode_len((num - TAG_CONST_OFFSET) as usize), TAGCONST);
                }
                let index = self.consts.len();
                self.consts.push(majit_ir::Const::Ref(*gcref));
                let tagged_i16 =
                    ((((index as i32) + TAG_CONST_OFFSET) << 2) | TAGCONST as i32) as i16;
                self.refs.insert(raw, tagged_i16);
                tag_i64(encode_len(index), TAGCONST)
            }
            majit_ir::Const::Float(v) => {
                // resume.py:183 _newconst (no dedup for floats in RPython).
                let index = self.consts.len();
                self.consts.push(majit_ir::Const::Float(*v));
                tag_i64(encode_len(index), TAGCONST)
            }
        }
    }

    /// resume.py:261-262 num_cached_boxes — length of the box dedup cache.
    pub fn num_cached_boxes(&self) -> usize {
        self.cached_boxes.len()
    }

    /// resume.py:275-276 num_cached_virtuals — length of the virtual dedup cache.
    pub fn num_cached_virtuals(&self) -> usize {
        self.cached_virtuals.len()
    }

    /// resume.py:264 assign_number_to_box — returns a negative number.
    /// resume.py:264-273 assign_number_to_box(box, boxes).
    ///
    /// RPython version mutates `boxes` list:
    /// - cached: `boxes[-num - 1] = box`
    /// - new: `boxes.append(box); num = -len(boxes)`
    pub fn assign_number_to_box(&mut self, box_id: OpRef, boxes: &mut Vec<OpRef>) -> i32 {
        if let Some(&num) = self.cached_boxes.get(&box_id) {
            // resume.py:268: boxes[-num - 1] = box
            let idx = (-num - 1) as usize;
            if idx < boxes.len() {
                boxes[idx] = box_id;
            }
            return num;
        }
        // resume.py:270-271: boxes.append(box); num = -len(boxes)
        boxes.push(box_id);
        let num = -(boxes.len() as i32);
        self.cached_boxes.insert(box_id, num);
        num
    }

    /// resume.py:264-273 variant for `_number_virtuals`: boxes is `Vec<Option<OpRef>>`.
    /// RPython's `new_liveboxes = [None] * memo.num_cached_boxes()`.
    pub fn assign_number_to_box_opt(
        &mut self,
        box_id: OpRef,
        boxes: &mut Vec<Option<OpRef>>,
    ) -> i32 {
        if let Some(&num) = self.cached_boxes.get(&box_id) {
            let idx = (-num - 1) as usize;
            if idx < boxes.len() {
                boxes[idx] = Some(box_id);
            }
            return num;
        }
        boxes.push(Some(box_id));
        let num = -(boxes.len() as i32);
        self.cached_boxes.insert(box_id, num);
        num
    }

    /// resume.py:278 assign_number_to_virtual — returns a negative number.
    pub fn assign_number_to_virtual(&mut self, box_id: OpRef) -> i32 {
        if let Some(&num) = self.cached_virtuals.get(&box_id) {
            return num;
        }
        // resume.py:283: num = self.cached_virtuals[box] = -len(self.cached_virtuals) - 1
        let num = -(self.num_cached_virtuals() as i32) - 1;
        self.cached_virtuals.insert(box_id, num);
        num
    }

    /// resume.py:290-293 update_counters(profiler).
    ///
    /// Roll the memo's cumulative NVIRTUALS / NVHOLES / NVREUSED into the
    /// caller-supplied profiler. Called from optimizeopt/optimizer.py:557
    /// once per trace compilation. The caller owns the profiler state;
    /// the memo only exposes its counters.
    pub fn update_counters(&self, profiler: &crate::jitprof::JitProfiler) {
        profiler.count(crate::pyjitpl::counters::NVIRTUALS, self.nvirtuals);
        profiler.count(crate::pyjitpl::counters::NVHOLES, self.nvholes);
        profiler.count(crate::pyjitpl::counters::NVREUSED, self.nvreused);
    }

    /// resume.py:286 clear_box_virtual_numbers.
    pub fn clear_box_virtual_numbers(&mut self) {
        self.cached_boxes.clear();
        self.cached_virtuals.clear();
    }

    /// Access the shared constant pool (value, type) pairs. Parity with
    /// RPython `memo.consts` list access (resume.py:147).
    pub fn consts(&self) -> &[majit_ir::Const] {
        &self.consts
    }

    /// Take ownership of the shared constant pool — used by single-shot
    /// encoders that discard the memo after encoding.
    pub fn take_consts(&mut self) -> Vec<majit_ir::Const> {
        std::mem::take(&mut self.consts)
    }

    /// resume.py:370-374 register_box — add a non-const, non-seen box to
    /// new_liveboxes with `UNASSIGNED`. The virtual classification is
    /// applied separately by `register_virtual_fields` (resume.py:359),
    /// which overwrites the entry with `UNASSIGNEDVIRTUAL` (or a
    /// pre-numbered tag from `liveboxes_from_env`). RPython's
    /// `register_box` does not consult `env.is_virtual` — see
    /// resume.py:370-374:
    ///     if (box is not None and not isinstance(box, Const)
    ///         and box not in self.liveboxes_from_env
    ///         and box not in self.liveboxes):
    ///         self.liveboxes[box] = UNASSIGNED
    fn register_box(
        &self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &mut LiveboxMap,
    ) {
        if opref.is_none() {
            return;
        }
        // resume.py:371 — constants are handled by _gettagged
        // (TAGCONST/TAGINT) and don't need livebox slots.
        let is_c = env.is_const(opref);
        let in_env = liveboxes_from_env.contains_key(opref);
        let in_new = new_liveboxes.contains_key(opref);
        if is_c || in_env || in_new {
            return;
        }
        new_liveboxes.insert(opref, UNASSIGNED);
    }

    /// resume.py:359-368 register_virtual_fields — stamp a virtual
    /// box's livebox tag and queue it for visitor_walk_recursive.
    ///
    /// `tagged = liveboxes_from_env.get(virtualbox, UNASSIGNEDVIRTUAL)`
    /// then `self.liveboxes[virtualbox] = tagged` unconditionally
    /// (overwriting any UNASSIGNED a prior `register_box` may have
    /// installed). The pre-numbered branch lets a virtual that was
    /// already TAGVIRTUAL'd in numbering keep its negative index.
    fn register_virtual_box(
        &self,
        virtualbox: majit_ir::OpRef,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &mut LiveboxMap,
    ) {
        let tagged = liveboxes_from_env
            .get(virtualbox)
            .unwrap_or(UNASSIGNEDVIRTUAL);
        new_liveboxes.insert(virtualbox, tagged);
    }

    /// resume.py:454-509 `_number_virtuals(liveboxes, num_env_virtuals)`.
    ///
    /// Walks `new_liveboxes` in insertion order, converts UNASSIGNED /
    /// UNASSIGNEDVIRTUAL tags into real negative numbers via
    /// `assign_number_to_box` / `assign_number_to_virtual`, then
    /// materializes each virtual's fieldnums through
    /// `env.make_virtual_info` and stores them into the returned
    /// `rd_virtuals` Vec.
    ///
    /// `liveboxes` is extended in place with the freshly numbered boxes
    /// (resume.py:484 `liveboxes.extend(new_liveboxes)`). Returns
    /// `(rd_virtuals, nholes)` where nholes is used for the
    /// `_invalidation_needed` heuristic check by the caller.
    #[allow(clippy::too_many_arguments)]
    fn _number_virtuals(
        &mut self,
        liveboxes: &mut Vec<Option<majit_ir::OpRef>>,
        new_liveboxes: &mut LiveboxMap,
        virtual_fields: &std::collections::HashMap<majit_ir::OpRef, majit_ir::VirtualFieldsInfo>,
        num_env_virtuals: usize,
        numb_state: &NumberingState,
        env: &dyn majit_ir::BoxEnv,
    ) -> (Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>, usize) {
        // resume.py:460: new_liveboxes = [None] * memo.num_cached_boxes()
        let mut new_boxes_list: Vec<Option<majit_ir::OpRef>> = vec![None; self.num_cached_boxes()];
        let mut count = 0;
        // Iterate in insertion order (RPython dict iteration = insertion order).
        // resoperation.py:38 same_box parity: keys carry the typed OpRef
        // each entry was inserted with so virtual numbering preserves
        // `box.type` (history.py:220) without round-tripping through
        // `OpRef::from_raw` and collapsing variants to `Untyped`.
        let keys: Vec<(OpRef, i16)> = new_liveboxes.iter().collect();
        for (opref_id, tagged) in keys {
            let (_, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                // resume.py:472-473: index = assign_number_to_box; liveboxes[box] = tag(index, TAGBOX)
                let index = self.assign_number_to_box_opt(opref_id, &mut new_boxes_list);
                if let Ok(t) = tag(index, TAGBOX) {
                    new_liveboxes.insert(opref_id, t);
                }
                count += 1;
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                if tagged_eq(tagged, UNASSIGNEDVIRTUAL) {
                    // resume.py:479-480: index = assign_number_to_virtual; liveboxes[box] = tag(index, TAGVIRTUAL)
                    let index = self.assign_number_to_virtual(opref_id);
                    if let Ok(t) = tag(index, TAGVIRTUAL) {
                        new_liveboxes.insert(opref_id, t);
                    }
                }
            }
        }
        // resume.py:483-484: new_liveboxes.reverse(); liveboxes.extend(new_liveboxes)
        new_boxes_list.reverse();
        for box_id in &new_boxes_list {
            liveboxes.push(*box_id);
        }
        let nholes = new_boxes_list.len() - count;

        // resume.py:488-506: create rd_virtuals
        // resume.py:500-501: make_virtual_info(info, fieldnums) via BoxEnv dispatch
        let mut rd_virtuals: Vec<std::rc::Rc<majit_ir::RdVirtualInfo>> = Vec::new();
        if !virtual_fields.is_empty() {
            // resume.py:491: length = num_env_virtuals + memo.num_cached_virtuals()
            let length = num_env_virtuals + self.num_cached_virtuals();
            // PRE-EXISTING-ADAPTATION: resume.py:492 uses `[None] * length` —
            // holes are represented as Python `None` in the list. Pyre's
            // descr-side `rd_virtuals: Arc<[Rc<RdVirtualInfo>]>` (compile.py:855
            // `_attrs_`) wraps the whole array in Option but the INNER
            // element type is not itself optional.  We use the
            // `RdVirtualInfo::Empty` sentinel variant to mark hole slots;
            // downstream consumers (compile.rs:644, compiler.rs:10952,
            // state.rs:3180, eval.rs:2680, resume.rs:1766) match `Empty`
            // and treat it as `None` equivalent.  Functional parity is
            // preserved; the structural divergence stays isolated to
            // this one type.
            rd_virtuals.resize(length, std::rc::Rc::new(majit_ir::RdVirtualInfo::Empty));
            // resume.py:493-494: memo.nvirtuals += length; memo.nvholes += length - len(vfieldboxes)
            self.nvirtuals += length;
            self.nvholes += length - virtual_fields.len();

            for (&opref_id, vf) in virtual_fields {
                // resume.py:496: num, _ = untag(self.liveboxes[virtualbox])
                // Check both numb_state.liveboxes (env virtuals) and
                // new_liveboxes (nested virtuals discovered via worklist).
                let opref = opref_id;
                let tagged = numb_state
                    .liveboxes
                    .get(opref)
                    .or_else(|| new_liveboxes.get(opref))
                    .unwrap_or(UNASSIGNEDVIRTUAL);
                let (num, _) = untag(tagged);
                // RPython uses Python negative indexing: virtuals[-1] = virtuals[len-1].
                // Negative nums come from assign_number_to_virtual for nested virtuals.
                let num_idx = if num >= 0 {
                    num as usize
                } else {
                    (rd_virtuals.len() as i32 + num) as usize
                };
                if num_idx < rd_virtuals.len() {
                    // resume.py:500: fieldnums = [self._gettagged(box) for box in fieldboxes]
                    let fieldnums: Vec<i16> = vf
                        .field_oprefs
                        .iter()
                        .map(|&opref| {
                            // resume.py:560-568 _gettagged with pyre-specific fallback
                            // to cached_boxes/cached_virtuals when the local
                            // liveboxes entries are still UNASSIGNED/UNASSIGNEDVIRTUAL.
                            if opref.is_none() {
                                return UNINITIALIZED_TAG;
                            }
                            if env.is_const(opref) {
                                let (val, tp) = env.get_const(opref);
                                return self.getconst(val, tp);
                            }
                            if let Some(t) = numb_state.liveboxes.get(opref) {
                                return t;
                            }
                            if let Some(t) = new_liveboxes.get(opref) {
                                if tagged_eq(t, UNASSIGNED) {
                                    if let Some(&num) = self.cached_boxes.get(&opref) {
                                        return tag(num, TAGBOX).unwrap_or(UNASSIGNED);
                                    }
                                }
                                if tagged_eq(t, UNASSIGNEDVIRTUAL) {
                                    if let Some(&num) = self.cached_virtuals.get(&opref) {
                                        return tag(num, TAGVIRTUAL).unwrap_or(UNASSIGNEDVIRTUAL);
                                    }
                                }
                                return t;
                            }
                            UNASSIGNED
                        })
                        .collect();
                    let reused = env.virtual_info_would_be_reused(opref_id, &fieldnums);
                    // resume.py:501: vinfo = self.make_virtual_info(info, fieldnums)
                    if let Some(rd_virt) = env.make_virtual_info(opref_id, fieldnums) {
                        if reused {
                            // resume.py:504-505: cached `_cached_vinfo` reused.
                            self.nvreused += 1;
                        }
                        rd_virtuals[num_idx] = rd_virt;
                    }
                }
            }
        }
        (rd_virtuals, nholes)
    }

    /// resume.py:520-558 `_add_pending_fields(pending_setfields)`.
    ///
    /// Tags the target/value boxes of each pending SETFIELD_GC/SETARRAYITEM_GC
    /// operation so the resume path can replay them against rehydrated
    /// struct instances. RPython decodes descr/opnum/itemindex from a
    /// `ResOperation` inline; pyre has already split the op into a
    /// `GuardPendingFieldEntry` by the time finish() is called, so this
    /// method only tags the target and value OpRefs in place.
    fn _add_pending_fields(
        &mut self,
        pending_setfields: &mut [majit_ir::GuardPendingFieldEntry],
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &LiveboxMap,
    ) {
        for pf in pending_setfields.iter_mut() {
            let target = env.get_box_replacement(pf.target);
            let value = env.get_box_replacement(pf.value);
            // resume.py:548-549 num = self._gettagged(box); fieldnum = self._gettagged(fieldbox)
            pf.target_tagged = self._gettagged(target, env, liveboxes_from_env, new_liveboxes);
            pf.value_tagged = self._gettagged(value, env, liveboxes_from_env, new_liveboxes);
        }
    }

    /// resume.py:570-574 `_add_optimizer_sections(numb_state, liveboxes, liveboxes_from_env)`.
    ///
    /// Delegates to bridgeopt.py:63-122 `serialize_optimizer_knowledge(optimizer,
    /// numb_state, liveboxes, liveboxes_from_env, memo)`. Emits three
    /// serialized sections on every guard (RPython emits zeros when the
    /// optheap/optrewrite caches are empty; the deserializer relies on the
    /// sections always being present):
    ///
    /// 1. known-class bitfield per Ref livebox (bridgeopt.py:74-90)
    /// 2. heap field + array item triples (bridgeopt.py:92-108)
    /// 3. loopinvariant call results (bridgeopt.py:113-122)
    ///
    /// RPython's `memo` is `self`; `numb_state.liveboxes` plays the role of
    /// the caller's `liveboxes_from_env` (the dict-like live-set). Pyre
    /// additionally carries an explicit `new_liveboxes` map so the per-
    /// guard tagged numbers assigned during `_number_virtuals` line up with
    /// the optimizer_knowledge lookup.
    fn _add_optimizer_sections(
        &mut self,
        numb_state: &mut NumberingState,
        liveboxes: &[Option<majit_ir::OpRef>],
        new_liveboxes: &LiveboxMap,
        env: &dyn majit_ir::BoxEnv,
        optimizer_knowledge: Option<&OptimizerKnowledgeForResume>,
    ) {
        // resume.py:572-574: serialize_optimizer_knowledge(
        //     self.optimizer, numb_state, liveboxes, liveboxes_from_env, self.memo)
        crate::optimizeopt::bridgeopt::serialize_optimizer_knowledge(
            self,
            numb_state,
            liveboxes,
            new_liveboxes,
            env,
            optimizer_knowledge,
        );
    }

    /// resume.py:511-518 `_invalidation_needed(nliveboxes, nholes)`.
    ///
    /// Heuristic for when the shared memo's cached-box dedup should be
    /// flushed after a successful resume encoding:
    ///
    /// ```python
    /// def _invalidation_needed(self, nliveboxes, nholes):
    ///     failargs_limit = memo.metainterp_sd.options.failargs_limit
    ///     if nliveboxes > (failargs_limit // 2):
    ///         if nholes > nliveboxes // 3:
    ///             return True
    ///     return False
    /// ```
    ///
    /// pyre uses the IR's compile-time FAILARGS_LIMIT (majit_ir:value.rs:201)
    /// as the metainterp option isn't wired yet. Matches RPython semantics
    /// exactly: "lots of live boxes, many of them holes" → invalidate.
    fn _invalidation_needed(&self, nliveboxes: usize, nholes: usize) -> bool {
        // resume.py:514-517
        let failargs_limit = majit_ir::FAILARGS_LIMIT;
        if nliveboxes > failargs_limit / 2 && nholes > nliveboxes / 3 {
            return true;
        }
        false
    }

    /// resume.py:560-568 _gettagged — resolve an OpRef to its tagged number.
    /// Looks up in liveboxes_from_env first, then new_liveboxes, then constant.
    pub(crate) fn _gettagged(
        &mut self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &LiveboxMap,
    ) -> i16 {
        if opref.is_none() {
            return UNINITIALIZED_TAG;
        }
        // resume.py:563-564: isinstance(box, Const) → getconst
        if env.is_const(opref) {
            let (val, tp) = env.get_const(opref);
            return self.getconst(val, tp);
        }
        // resume.py:566-567: liveboxes_from_env → existing tag
        if let Some(tagged) = liveboxes_from_env.get(opref) {
            return tagged;
        }
        if let Some(tagged) = new_liveboxes.get(opref) {
            // Resolve UNASSIGNED to real cached number
            if tagged_eq(tagged, UNASSIGNED) {
                if let Some(&num) = self.cached_boxes.get(&opref) {
                    return tag(num, TAGBOX).unwrap_or(UNASSIGNED);
                }
            }
            if tagged_eq(tagged, UNASSIGNEDVIRTUAL) {
                if let Some(&num) = self.cached_virtuals.get(&opref) {
                    return tag(num, TAGVIRTUAL).unwrap_or(UNASSIGNEDVIRTUAL);
                }
            }
            return tagged;
        }
        UNASSIGNED
    }

    /// resume.py:192-226 _number_boxes — tag each box in a snapshot section.
    ///
    /// Exact port of RPython's `_number_boxes(self, iter, iterator, numb_state)`.
    ///
    /// `env` provides box access matching RPython's box operations:
    /// - `get_box_replacement(opref)` → forwarded OpRef (resume.py:202)
    /// - `is_const(opref)` → isinstance(box, Const) (resume.py:204)
    /// - `get_const(opref)` → (value, type) for constants
    /// - `get_type(opref)` → box.type ('i', 'r', 'f') (resume.py:211,214)
    /// - `is_virtual_ref(opref)` → getptrinfo(box).is_virtual() (resume.py:212-213)
    /// - `is_virtual_raw(opref)` → getrawptrinfo(box).is_virtual() (resume.py:215-216)
    /// resume.py:192-226 `_number_boxes` — tag each box in a snapshot section.
    pub fn _number_boxes(
        &mut self,
        boxes: &[SnapshotBox],
        numb_state: &mut NumberingState,
        env: &dyn BoxEnv,
    ) -> Result<(), TagOverflow> {
        for &snapshot_box in boxes {
            let raw_opref = snapshot_box.opref;
            if raw_opref.is_none() {
                numb_state.append_short(NULLREF);
                continue;
            }
            let opref = env.get_box_replacement(raw_opref);
            if opref.is_none() {
                numb_state.append_short(NULLREF);
                continue;
            }
            // resume.py:204: isinstance(box, Const) → getconst
            if env.is_const(opref) {
                let (val, tp) = env.get_const(opref);
                let tagged = self.getconst(val, tp);
                numb_state.append_short(tagged);
                continue;
            }
            // resume.py:206-208: liveboxes
            if let Some(tagged) = numb_state.liveboxes.get(opref) {
                numb_state.append_short(tagged);
                continue;
            }
            let box_type = snapshot_box.tp.unwrap_or_else(|| env.get_type(opref));
            let is_virtual = match box_type {
                majit_ir::Type::Ref => env.is_virtual_ref(opref),
                majit_ir::Type::Int => env.is_virtual_raw(opref),
                _ => false,
            };
            let tagged = if is_virtual {
                let t = tag(numb_state.num_virtuals, TAGVIRTUAL)?;
                numb_state.num_virtuals += 1;
                t
            } else {
                // RPython Box.type parity: capture type alongside TAGBOX
                // assignment. This is the equivalent of Box.type being
                // intrinsic — the type is determined once at numbering time.
                //
                // Typed OpRef variants (resoperation.py:719-739
                // InputArg{Int,Ref,Float}, resoperation.py:564-638 *Op
                // mixins) already carry the type intrinsically. The
                // `livebox_types` HashMap is the legacy side-table that
                // must agree with `opref.ty()` whenever the latter is
                // populated — a divergence here would mean an Untyped
                // OpRef leaked into the live set with a stale type
                // annotation, which is the kind of encoder/decoder
                // mismatch we want to fail-loud on (epic #171 will
                // retire the side-table once Untyped is gone).
                if let Some(intrinsic_tp) = opref.ty() {
                    debug_assert_eq!(
                        intrinsic_tp, box_type,
                        "livebox numbering: typed OpRef {:?} intrinsic type {:?} \
                         disagrees with snapshot/env type {:?}",
                        opref, intrinsic_tp, box_type
                    );
                }
                numb_state.livebox_types.insert(opref, box_type);
                let t = tag(numb_state.num_boxes, TAGBOX)?;
                numb_state.num_boxes += 1;
                t
            };
            numb_state.liveboxes.insert(opref, tagged);
            numb_state.append_short(tagged);
        }
        Ok(())
    }

    /// resume.py:228-256 number() — serialize a guard's full snapshot.
    ///
    /// Output format (in NumberingState):
    /// ```text
    /// [0]  size (patched later)
    /// [1]  number of failargs (patched later)
    /// [2]  vable_array_length  (0 if no virtualizable)
    ///      [tagged boxes for vable_array]
    /// [n]  vref_array_length   (0 if no virtualrefs)
    ///      [tagged boxes for vref_array]
    /// [m]  frame0_pc frame0_slots...
    /// [m+] frame1_pc frame1_slots...
    /// ...
    /// ```
    ///
    /// `frames` is a list of (pc, fail_args_slice) for each frame.
    /// In pyre (single frame), this is typically one frame.
    /// resume.py:228-256 number() — serialize a guard's full snapshot.
    ///
    /// Exact port of RPython's `number(self, position, trace, ...)`.
    ///
    /// `snapshot` describes the guard's state:
    /// - `vable_array`: virtualizable field boxes
    /// - `vref_array`: virtualref pairs
    /// - `framestack`: list of (jitcode_index, pc, boxes) per frame
    ///
    /// `env` implements BoxEnv for box operations.
    ///
    /// Returns `Err(TagOverflow)` if any box index exceeds the tag range.
    /// RPython: raises TagOverflow → caller does compile.giveup().
    ///
    /// NOTE: Slot 1 (number of failargs) is left as 0 here.
    /// RPython patches it later in ResumeDataVirtualAdder.finish()
    /// (resume.py:433). Callers must call
    /// `numb_state.writer.patch(1, num_liveboxes)` after finish().
    pub fn number(
        &mut self,
        snapshot: &Snapshot,
        env: &dyn BoxEnv,
        minimum_virtualizable_size: i64,
    ) -> Result<NumberingState, TagOverflow> {
        let size_hint = snapshot.estimated_size();
        let mut numb_state = NumberingState::new(size_hint);

        // resume.py:231-232: patch later
        numb_state.append_int(0); // slot 0: size of resume section
        numb_state.append_int(0); // slot 1: number of failargs (patched by finish())

        // resume.py:236-239: if minimum_virtualizable_size != -1: the
        // virtualizable itself is one entry in the array too, so use '>'.
        if minimum_virtualizable_size != -1 {
            debug_assert!(
                snapshot.vable_array.len() as i64 > minimum_virtualizable_size,
                "vable_array length {} not > minimum_virtualizable_size {}",
                snapshot.vable_array.len(),
                minimum_virtualizable_size
            );
        }

        // resume.py:240-241 virtualizable array.
        //
        // Upstream shape is `virtualizable_boxes = read_boxes(...);`
        // `virtualizable_boxes.append(virtualizable_box)` (pyjitpl.py:3302-3306),
        // i.e. payload first, identity last. The snapshot already carries the
        // tracing-time Box identities in that order, so line-by-line parity is
        // to run the whole array through `_number_boxes()` unchanged.
        numb_state.append_int(snapshot.vable_array.len() as i64);
        self._number_boxes(&snapshot.vable_array, &mut numb_state, env)?;

        // resume.py:243-247: virtualref array
        let vref_len = snapshot.vref_array.len();
        debug_assert!(vref_len & 1 == 0, "vref_array length must be even");
        numb_state.append_int((vref_len >> 1) as i64);
        self._number_boxes(&snapshot.vref_array, &mut numb_state, env)?;

        // resume.py:249-253: frame chain.
        // Per-frame: jitcode_index, pc, [tagged_values...].
        // RPython uses jitcode.get_live_vars_info(pc) at decode time
        // to know how many tagged values each frame has.
        for frame in &snapshot.framestack {
            numb_state.append_int(frame.jitcode_index as i64);
            numb_state.append_int(frame.pc as i64);
            self._number_boxes(&frame.boxes, &mut numb_state, env)?;
        }

        // resume.py:254: patch total size
        numb_state.patch_current_size(0);

        Ok(numb_state)
    }

    /// resume.py:389-452 ResumeDataVirtualAdder.finish() — exact port.
    ///
    /// `numb_state`: output of `number()`
    /// `env`: BoxEnv for resolving box properties (constants, types).
    ///   Virtual fields are discovered via `env.get_virtual_fields()`,
    ///   matching RPython's `visitor_walk_recursive` callback pattern.
    /// `pending_setfields`: resume.py:428-442 register_box + visitor_walk_recursive,
    ///   resume.py:520-558 _add_pending_fields tagging.
    ///   target_tagged/value_tagged are filled in-place.
    /// `optimizer_knowledge`: bridgeopt.py:63 serialize_optimizer_knowledge.
    ///   Heap field triples and known-class info for bridge compilation.
    ///
    /// Returns `(rd_numb, rd_consts, rd_virtuals, liveboxes, livebox_types)`.
    /// `livebox_types` maps typed OpRef → Type, captured at numbering time
    /// (RPython Box.type parity).
    pub fn finish(
        &mut self,
        mut numb_state: NumberingState,
        env: &dyn majit_ir::BoxEnv,
        pending_setfields: &mut [majit_ir::GuardPendingFieldEntry],
        optimizer_knowledge: Option<&OptimizerKnowledgeForResume>,
    ) -> (
        Vec<u8>,
        Vec<majit_ir::Const>,
        Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>,
        Vec<majit_ir::OpRef>,
        std::collections::HashMap<majit_ir::OpRef, majit_ir::Type>,
    ) {
        let num_env_virtuals = numb_state.num_virtuals;

        // resume.py:410-426: split liveboxes_from_env into TAGBOX/TAGVIRTUAL
        let mut liveboxes: Vec<Option<majit_ir::OpRef>> = vec![None; numb_state.num_boxes as usize];

        // resume.py:413: self.vfieldboxes collected by virtual walk
        // resume.py:408: self.liveboxes — newly discovered boxes from field walk
        let mut new_liveboxes = LiveboxMap::new();

        // resume.py:414-426: iterate liveboxes_from_env, discover virtual
        // fields. RPython walks the dict in insertion order; pyre's
        // `LiveboxMap` is built on `IndexMap` (resume.rs:98) so the
        // `.iter()` sequence already matches that order, which the
        // virtual worklist drain below relies on for byte-identical
        // visitor_walk_recursive sequencing. Sorting by tag would
        // observably re-order virtuals across builds.
        //
        // TAGBOX placement at `liveboxes[i] = opref` uses the
        // tag-derived index (resume.py:417), so it is iteration-order-
        // invariant; only the TAGVIRTUAL worklist push order matters.
        //
        // resoperation.py:38 same_box parity: iter() yields the typed
        // OpRef each entry was inserted with, so consumers no longer
        // need to round-trip through `OpRef::from_raw` to recover
        // `box.type` (history.py:220).

        // Collect virtual fields discovered via env.get_virtual_fields()
        // (resume.py:419-426 visitor_walk_recursive pattern). Keyed by
        // typed OpRef so the same_box (resoperation.py:38) identity is
        // preserved end-to-end through the worklist drain.
        let mut virtual_fields: HashMap<majit_ir::OpRef, majit_ir::VirtualFieldsInfo> =
            HashMap::new();

        // resume.py:419-426: visitor_walk_recursive — worklist for nested virtuals.
        let mut virtual_worklist: Vec<majit_ir::OpRef> = Vec::new();

        for (opref, tagged) in numb_state.liveboxes.iter() {
            let (i, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                if (i as usize) < liveboxes.len() {
                    liveboxes[i as usize] = Some(opref);
                }
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                virtual_worklist.push(opref);
            }
        }

        // Worklist-based recursive virtual discovery (RPython visitor_walk_recursive).
        // Process each virtual: register its field boxes, and if any field is
        // itself a virtual, add it to the worklist for later processing.
        let mut worklist_idx = 0;
        while worklist_idx < virtual_worklist.len() {
            let opref_id = virtual_worklist[worklist_idx];
            worklist_idx += 1;

            if virtual_fields.contains_key(&opref_id) {
                continue; // already_seen_virtual
            }
            let vf_result = env.get_virtual_fields(opref_id);
            if let Some(vf) = vf_result {
                // resume.py:362-368: register_virtual_fields
                for &field_opref in &vf.field_oprefs {
                    // resume.py:370-374: register_box (UNASSIGNED for
                    // non-virtual fields).
                    self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                    // resume.py:419-426 visitor_walk_recursive: if the
                    // field is a virtual, register_virtual_fields
                    // overwrites the UNASSIGNED stamp with the env-
                    // pre-numbered tag (or UNASSIGNEDVIRTUAL).
                    let resolved = env.get_box_replacement(field_opref);
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved)
                        && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                    {
                        self.register_virtual_box(
                            resolved,
                            &numb_state.liveboxes,
                            &mut new_liveboxes,
                        );
                        virtual_worklist.push(resolved);
                    }
                }
                virtual_fields.insert(opref_id, vf);
            }
        }

        // resume.py:428-442: process pending_setfields — register_box on
        // target and value, then visitor_walk_recursive on virtual fieldbox.
        for pf in pending_setfields.iter() {
            let box_opref = env.get_box_replacement(pf.target);
            let fieldbox = env.get_box_replacement(pf.value);
            // resume.py:438-439: self.register_box(box); self.register_box(fieldbox)
            self.register_box(box_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
            self.register_box(fieldbox, env, &numb_state.liveboxes, &mut new_liveboxes);
            // resume.py:440-442 — info.visitor_walk_recursive requires
            // the fieldbox to be a virtual:
            //     info = getptrinfo(fieldbox)
            //     assert info is not None and info.is_virtual()
            // A non-virtual fieldbox in pending_setfields is an
            // invariant violation in the optheap producer side; the
            // previous silent skip masked it.
            let vf = env.get_virtual_fields(fieldbox).unwrap_or_else(|| {
                panic!(
                    "pending_setfields fieldbox {:?} (target={:?}) is not virtual \
                     (resume.py:441 assert info is not None and info.is_virtual())",
                    fieldbox, box_opref
                )
            });
            for &field_opref in &vf.field_oprefs {
                self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                let resolved = env.get_box_replacement(field_opref);
                if !resolved.is_none()
                    && !virtual_fields.contains_key(&resolved)
                    && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                {
                    self.register_virtual_box(resolved, &numb_state.liveboxes, &mut new_liveboxes);
                    virtual_worklist.push(resolved);
                }
            }
            virtual_fields.insert(fieldbox, vf);
        }

        // resume.py:440-442 parity: drain worklist for nested virtuals
        // discovered from pending_setfields. RPython's visitor_walk_recursive
        // recursively processes all levels; our worklist pattern resumes here.
        while worklist_idx < virtual_worklist.len() {
            let opref_id = virtual_worklist[worklist_idx];
            worklist_idx += 1;
            if virtual_fields.contains_key(&opref_id) {
                continue;
            }
            if let Some(vf) = env.get_virtual_fields(opref_id) {
                for &field_opref in &vf.field_oprefs {
                    self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                    let resolved = env.get_box_replacement(field_opref);
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved)
                        && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                    {
                        self.register_virtual_box(
                            resolved,
                            &numb_state.liveboxes,
                            &mut new_liveboxes,
                        );
                        virtual_worklist.push(resolved);
                    }
                }
                virtual_fields.insert(opref_id, vf);
            }
        }

        // resume.py:454-509 self._number_virtuals(liveboxes, num_env_virtuals)
        let (rd_virtuals, nholes) = self._number_virtuals(
            &mut liveboxes,
            &mut new_liveboxes,
            &virtual_fields,
            num_env_virtuals as usize,
            &numb_state,
            env,
        );

        // resume.py:508-509: if self._invalidation_needed(...): memo.clear_box_virtual_numbers()
        if self._invalidation_needed(liveboxes.len(), nholes) {
            self.clear_box_virtual_numbers();
        }

        // resume.py:445 self._add_pending_fields(pending_setfields)
        self._add_pending_fields(
            pending_setfields,
            env,
            &numb_state.liveboxes,
            &new_liveboxes,
        );

        // resume.py:447: numb_state.patch(1, len(liveboxes))
        numb_state.writer.patch(1, liveboxes.len() as i32);

        // resume.py:449: self._add_optimizer_sections(numb_state, liveboxes, liveboxes_from_env)
        self._add_optimizer_sections(
            &mut numb_state,
            &liveboxes,
            &new_liveboxes,
            env,
            optimizer_knowledge,
        );

        // resume.py:450-451: storage.rd_numb, storage.rd_consts
        let rd_numb = numb_state.create_numbering();
        let rd_consts = self.consts.clone();

        // Resolve each livebox through the forwarding chain so the backend
        // sees the final concrete OpRef (not an optimizer-internal alias).
        let ordered_liveboxes: Vec<majit_ir::OpRef> = liveboxes
            .into_iter()
            .map(|opt| {
                opt.map(|opref| env.get_box_replacement(opref))
                    .unwrap_or(majit_ir::OpRef::NONE)
            })
            .collect();

        // Merge livebox_types: numbering-time types + types for boxes
        // discovered during virtual field walking.
        let mut all_livebox_types = numb_state.livebox_types;
        for &opref in &ordered_liveboxes {
            if !opref.is_none() && !all_livebox_types.contains_key(&opref) {
                all_livebox_types.insert(opref, env.get_type(opref));
            }
        }
        (
            rd_numb,
            rd_consts,
            rd_virtuals,
            ordered_liveboxes,
            all_livebox_types,
        )
    }

    /// resume.py:452-468 finish (on ResumeDataVirtualAdder) — encode with shared pool.
    pub fn encode_shared(&mut self, rd: &ResumeData) -> EncodedResumeData {
        let mut rd_numb = Vec::new();
        // resume.py:138 compact TAGBOX numbering state.
        let mut liveboxes: Vec<usize> = Vec::new();
        let mut box_map: HashMap<usize, usize> = HashMap::new();

        // resume.py:234-235: reserve slots
        rd_numb.push(0); // [0] = items_resume_section
        rd_numb.push(0); // [1] = count
        rd_numb.push(encode_len(rd.vable_array.len()));
        for source in &rd.vable_array {
            let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }
        // resume.py:243-247: vref_array (pairs).
        assert!(
            rd.vref_array.len() % 2 == 0,
            "vref_array must have even length (pairs)"
        );
        rd_numb.push(encode_len(rd.vref_array.len() / 2));
        for source in &rd.vref_array {
            let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }

        // resume.py:249-253: per-frame: jitcode_index, pc, [tagged_values...].
        let mut frame_sizes = Vec::with_capacity(rd.frames.len());
        for frame in &rd.frames {
            rd_numb.push(frame.jitcode_index as i64);
            rd_numb.push(encode_u64(frame.pc));
            for source in &frame.slot_map {
                let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
                rd_numb.push(tagged);
            }
            frame_sizes.push(frame.slot_map.len());
        }

        let rd_virtuals = rd.virtuals.clone();

        // resume.py:412-418: register virtual field boxes.
        for vinfo in &rd_virtuals {
            for source in vinfo.field_sources() {
                if let ResumeValueSource::FailArg(index) = source {
                    box_map.entry(*index).or_insert_with(|| {
                        let n = liveboxes.len();
                        liveboxes.push(*index);
                        n
                    });
                }
            }
        }

        // resume.py:420-430: walk pending fields — register + encode.
        // Collect first, then encode — can't hold iter borrow and call
        // `&mut self` method in the same expression.
        let pending_fields_snapshot: Vec<_> = rd.pending_fields.iter().cloned().collect();
        let rd_pendingfields: Vec<_> = pending_fields_snapshot
            .into_iter()
            .map(|pending| EncodedPendingFieldWrite {
                descr_index: pending.descr_index,
                target: self.encode_tagged_source(&pending.target, &mut liveboxes, &mut box_map),
                value: self.encode_tagged_source(&pending.value, &mut liveboxes, &mut box_map),
                item_index: pending.item_index,
            })
            .collect();

        // resume.py:260 patch_current_size, resume.py:464 patch count
        rd_numb[0] = encode_len(rd_numb.len());
        rd_numb[1] = encode_len(liveboxes.len());

        EncodedResumeData {
            rd_numb,
            // resume.py:451 storage.rd_consts = self.memo.consts — single pool.
            rd_consts: self.consts.clone(),
            rd_pendingfields,
            rd_virtuals,
            liveboxes,
            frame_sizes,
        }
    }

    /// Number of entries in the shared constant pool.
    pub fn num_shared_consts(&self) -> usize {
        self.consts.len()
    }
}

impl Default for ResumeDataLoopMemo {
    fn default() -> Self {
        Self::new()
    }
}

/// resume.py: AbstractResumeDataReader — reads resume data to
/// reconstruct interpreter state after a guard failure.
///
/// Two concrete implementations in RPython:
/// - ResumeDataBoxReader: creates boxes (for blackhole interpreter)
/// - ResumeDataDirectReader: reads values directly (for fast path)
pub struct ResumeDataReader<'a> {
    /// The resume data to read from.
    resume_data: &'a ResumeData,
    /// Fail argument values from the guard failure.
    fail_values: &'a [i64],
    /// Materialized virtuals (lazily populated).
    virtuals: Vec<Option<i64>>,
}

impl<'a> ResumeDataReader<'a> {
    /// resume.py: AbstractResumeDataReader.__init__
    pub fn new(resume_data: &'a ResumeData, fail_values: &'a [i64]) -> Self {
        let num_virtuals = resume_data.virtuals.len();
        ResumeDataReader {
            resume_data,
            fail_values,
            virtuals: vec![None; num_virtuals],
        }
    }

    /// resume.py: _decode_box — decode a tagged value reference.
    pub fn decode_frame_slot(&self, source: &FrameSlotSource) -> i64 {
        self.decode_value(source)
    }

    /// Decode a ResumeValueSource to a concrete value.
    pub fn decode_value(&self, source: &ResumeValueSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(idx) => self.fail_values.get(*idx).copied().unwrap_or(0),
            ResumeValueSource::Constant(c) => c.as_raw_i64(),
            ResumeValueSource::Virtual(vidx) => {
                self.virtuals.get(*vidx).copied().flatten().unwrap_or(0)
            }
            ResumeValueSource::Uninitialized | ResumeValueSource::Unavailable => 0,
        }
    }

    /// resume.py: consume_boxes — read all frame slots for one frame.
    pub fn read_frame_slots(&self, frame_idx: usize) -> Vec<i64> {
        if frame_idx >= self.resume_data.frames.len() {
            return vec![];
        }
        let frame = &self.resume_data.frames[frame_idx];
        frame
            .slot_map
            .iter()
            .map(|source| self.decode_frame_slot(source))
            .collect()
    }

    /// Number of frames in the resume data.
    pub fn num_frames(&self) -> usize {
        self.resume_data.frames.len()
    }

    /// PC for a given frame.
    pub fn frame_pc(&self, frame_idx: usize) -> u64 {
        self.resume_data
            .frames
            .get(frame_idx)
            .map(|f| f.pc)
            .unwrap_or(0)
    }
}

/// resume.py:576-728 VirtualInfo parity.
/// Describes a virtual object's fields for materialization.
/// RPython uses a class hierarchy (VirtualInfo, VStructInfo, VArrayInfoClear, etc.).
/// We use a single struct with tagged field values.
#[derive(Debug, Clone, Default)]
pub struct VirtualFieldValues {
    /// Descriptor (type/class) for the virtual object.
    pub descr: Option<majit_ir::DescrRef>,
    /// Known class pointer (ob_type for NewWithVtable).
    pub known_class: Option<i64>,
    /// Tagged field values (i16 tags referencing consts/boxes/other virtuals).
    pub fieldnums: Vec<i16>,
}

/// resume.py:554-557 — tagged pending field entry.
/// RPython stores (lldescr, num, fieldnum, itemindex) where num and fieldnum
/// are tagged references into the numbering system.
#[derive(Debug, Clone)]
pub struct TaggedPendingField {
    pub descr_index: u32,
    pub item_index: i32,
    /// Tagged reference to target box (from _gettagged).
    pub num: i16,
    /// Tagged reference to value box (from _gettagged).
    pub fieldnum: i16,
}

/// bridgeopt.py:63 — optimizer knowledge for resume data encoding.
/// Passed into finish() for _add_optimizer_sections.
pub struct OptimizerKnowledgeForResume {
    /// (obj_opref, descr_index, val_opref) heap field triples.
    /// bridgeopt.py:96-101
    pub heap_fields: Vec<(majit_ir::OpRef, i32, majit_ir::OpRef)>,
    /// (array_opref, index, descr_index, val_opref) heap array item quads.
    /// bridgeopt.py:102-108
    pub heap_arrayitems: Vec<(majit_ir::OpRef, i64, i32, majit_ir::OpRef)>,
    /// (const_func_ptr, result_opref) loop-invariant call results.
    pub loopinvariant_results: Vec<(i64, majit_ir::OpRef)>,
}

impl OptimizerKnowledgeForResume {
    pub fn is_empty(&self) -> bool {
        self.heap_fields.is_empty()
            && self.heap_arrayitems.is_empty()
            && self.loopinvariant_results.is_empty()
    }
}

/// bridgeopt.py:44-61 decode_box return type.
///
/// RPython's decode_box returns actual Const/Box objects. Two Rust
/// variants mirror that: `LiveBox` for `Box` references (TAGBOX) and
/// `Const` for all Const subtypes (TAGINT/TAGCONST/NULLREF), collapsing
/// into a single `majit_ir::Const` that carries its own type.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodedBox {
    /// TAGBOX → liveboxes[num] (bridge inputarg / optimizer box).
    LiveBox(majit_ir::OpRef),
    /// TAGINT / TAGCONST / NULLREF — all Const subtypes.
    Const(majit_ir::Const),
}

/// bridgeopt.py:44-61 decode_box: untag a tagged value from rd_numb.
///
/// Line-by-line port of PyPy's decode_box(). Returns DecodedBox to
/// preserve the Const vs Box distinction that RPython encodes via
/// Python class hierarchy.
pub fn decode_box(
    tagged: i16,
    rd_consts: &[majit_ir::Const],
    liveboxes: &[majit_ir::OpRef],
) -> DecodedBox {
    let (num, tag_type) = untag(tagged);
    // NB: the TAGVIRTUAL case can't happen here, because this code runs after
    // virtuals are already forced again.
    match tag_type {
        TAGCONST => {
            if tagged_eq(tagged, NULLREF) {
                // bridgeopt.py:51: box = CONST_NULL (history.py:361).
                DecodedBox::Const(majit_ir::Const::Ref(majit_ir::GcRef::NULL))
            } else {
                // bridgeopt.py:54: box = resumestorage.rd_consts[num - TAG_CONST_OFFSET]
                // — direct list index, IndexError on out-of-range. A
                // bridgeopt-knowledge stream that names a const slot
                // outside `rd_consts` is corrupt, and silently swapping
                // in CONST_NULL hides the corruption from the optimizer.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                let c = rd_consts.get(idx).copied().unwrap_or_else(|| {
                    panic!(
                        "bridgeopt decode_box TAGCONST out-of-range: idx={} \
                         rd_consts.len()={} (corrupt knowledge stream — see \
                         bridgeopt.py:54)",
                        idx,
                        rd_consts.len()
                    )
                });
                DecodedBox::Const(c)
            }
        }
        // bridgeopt.py:56: box = ConstInt(num)
        TAGINT => DecodedBox::Const(majit_ir::Const::Int(num as i64)),
        TAGBOX => {
            // bridgeopt.py:58: box = liveboxes[num] — direct list index,
            // IndexError on out-of-range. See TAGCONST comment above.
            let idx = num as usize;
            let lb = *liveboxes.get(idx).unwrap_or_else(|| {
                panic!(
                    "bridgeopt decode_box TAGBOX out-of-range: idx={} \
                     liveboxes.len()={} (corrupt knowledge stream — see \
                     bridgeopt.py:58)",
                    idx,
                    liveboxes.len()
                )
            });
            DecodedBox::LiveBox(lb)
        }
        _ => {
            // bridgeopt.py:60: raise AssertionError("unreachable")
            unreachable!("bridgeopt decode_box: unexpected tag type {}", tag_type);
        }
    }
}

// VirtualFieldInfo removed: replaced by majit_ir::VirtualFieldsInfo.
// finish() now discovers virtual fields via env.get_virtual_fields().

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};

    #[test]
    fn livebox_map_preserves_box_identity_and_insertion_order() {
        let mut liveboxes = LiveboxMap::new();
        let input = majit_ir::OpRef::input_arg_int(0);
        let op = majit_ir::OpRef::int_op(0);

        liveboxes.insert(input, UNASSIGNED);
        liveboxes.insert(op, UNASSIGNEDVIRTUAL);

        assert_eq!(liveboxes.get(input), Some(UNASSIGNED));
        assert_eq!(liveboxes.get(op), Some(UNASSIGNEDVIRTUAL));
        assert_eq!(
            liveboxes.iter().collect::<Vec<_>>(),
            vec![(input, UNASSIGNED), (op, UNASSIGNEDVIRTUAL)]
        );
    }

    #[test]
    fn test_simple_resume_data() {
        let rd = ResumeData::simple(42, 3);
        let fail_values = vec![10, 20, 30];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].pc, 42);
        assert_eq!(
            frames[0].values,
            vec![
                ReconstructedValue::Value(10),
                ReconstructedValue::Value(20),
                ReconstructedValue::Value(30),
            ]
        );
    }

    #[test]
    fn test_resume_data_with_gaps() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                jitcode_index: 0,
                pc: 100,
                slot_map: vec![
                    FrameSlotSource::FailArg(2),
                    FrameSlotSource::Unavailable,
                    FrameSlotSource::FailArg(0),
                ],
            }],
            virtuals: Vec::new(),
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            pending_fields: Vec::new(),
        };
        let fail_values = vec![10, 20, 30];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(
            frames[0].values,
            vec![
                ReconstructedValue::Value(30),
                ReconstructedValue::Unavailable,
                ReconstructedValue::Value(10),
            ]
        );
        assert_eq!(frames[0].lossy_values(), vec![30, 0, 10]);
    }

    #[test]
    fn test_multi_frame_resume() {
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    jitcode_index: 0,
                    pc: 10,
                    slot_map: vec![FrameSlotSource::FailArg(0), FrameSlotSource::FailArg(1)],
                },
                FrameInfo {
                    jitcode_index: 1,
                    pc: 20,
                    slot_map: vec![FrameSlotSource::FailArg(2), FrameSlotSource::FailArg(3)],
                },
            ],
            virtuals: Vec::new(),
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            pending_fields: Vec::new(),
        };
        let fail_values = vec![1, 2, 3, 4];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].pc, 10);
        assert_eq!(
            frames[0].values,
            vec![ReconstructedValue::Value(1), ReconstructedValue::Value(2)]
        );
        assert_eq!(frames[1].pc, 20);
        assert_eq!(
            frames[1].values,
            vec![ReconstructedValue::Value(3), ReconstructedValue::Value(4)]
        );
    }

    #[test]
    fn test_builder() {
        let mut builder = ResumeDataVirtualAdder::new();
        builder.push_frame(0, 42);
        builder.map_slot(0, 0);
        builder.map_slot(2, 1); // gap at slot 1
        let rd = builder.build();

        assert_eq!(rd.frames.len(), 1);
        assert_eq!(rd.frames[0].pc, 42);
        assert_eq!(
            rd.frames[0].slot_map,
            vec![
                FrameSlotSource::FailArg(0),
                FrameSlotSource::Unavailable,
                FrameSlotSource::FailArg(1),
            ]
        );
    }

    #[test]
    fn test_memo_number_simple() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.constants
            .insert(OpRef::const_int(1).raw(), (42i64, majit_ir::Type::Int));
        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![OpRef::const_int(1), OpRef::int_op(1), OpRef::int_op(2)],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // Should have: [size, num_failargs, 0(vable), 0(vref), 0(jitcode), 8(pc), tagged...]
        let items = crate::resumecode::unpack_all(&numb_state.create_numbering());
        // items[0] = total size
        assert!(items[0] > 0);
        // items[1] = num_failargs: 0 (not patched yet — RPython patches in finish())
        // After finish: patch(1, numb_state.liveboxes.len()) would set to 2.
        assert_eq!(items[1], 0);
        // items[2] = vable_array_length = 0
        assert_eq!(items[2], 0);
        // items[3] = vref_array_length = 0
        assert_eq!(items[3], 0);
        // items[4] = jitcode_index = 0
        assert_eq!(items[4], 0);
        // items[5] = pc = 8
        assert_eq!(items[5], 8);
        // items[6] = OpRef::const_int(1) tagged as TAGINT(42) since 42 fits in 13 bits
        let (val, tagbits) = untag(items[6] as i16);
        assert_eq!(tagbits, TAGINT);
        assert_eq!(val, 42);
        // items[7] = OpRef::int_op(1) tagged as TAGBOX(0) — first live box
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[8] = OpRef::int_op(2) tagged as TAGBOX(1) — second live box
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_number_rebuild_roundtrip() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.constants
            .insert(OpRef::const_int(1).raw(), (42i64, majit_ir::Type::Int));
        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![OpRef::const_int(1), OpRef::int_op(1), OpRef::int_op(2)],
        );
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // RPython: ResumeDataVirtualAdder.finish() patches slot 1 with num_boxes.
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, memo.consts(), &fail_arg_types, None);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(rebuilt_frames[0].pc, 8);
        assert_eq!(rebuilt_frames[0].values.len(), 3);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Const(majit_ir::Const::Int(42))
        );
        assert_eq!(
            rebuilt_frames[0].values[1],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(
            rebuilt_frames[0].values[2],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn test_number_rebuild_with_virtual() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2); // OpRef::ref_op(2) is virtual (Ref type)
        env.types.insert(2, majit_ir::Type::Ref);
        let snapshot = Snapshot::single_frame(
            0,
            10,
            vec![OpRef::int_op(1), OpRef::ref_op(2), OpRef::int_op(3)],
        );
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // RPython: finish() patches with len(newboxes) which is num_boxes
        // (not liveboxes which includes virtuals).
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, memo.consts(), &fail_arg_types, None);
        assert_eq!(num_failargs, 2); // OpRef::int_op(1) and OpRef::int_op(3) are boxes
        assert_eq!(rebuilt_frames[0].values.len(), 3);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(rebuilt_frames[0].values[1], RebuiltValue::Virtual(0));
        assert_eq!(
            rebuilt_frames[0].values[2],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn test_memo_number_with_virtual() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2);
        env.types.insert(2, majit_ir::Type::Ref);
        let snapshot = Snapshot::single_frame(
            0,
            10,
            vec![OpRef::int_op(1), OpRef::ref_op(2), OpRef::int_op(3)],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let items = crate::resumecode::unpack_all(&numb_state.create_numbering());
        // items[1] = num_failargs: 0 (not patched — RPython patches in finish())
        assert_eq!(items[1], 0);
        // items[6] = OpRef::int_op(1) → TAGBOX(0)
        let (val, tagbits) = untag(items[6] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[7] = OpRef::ref_op(2) → TAGVIRTUAL(0)
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGVIRTUAL);
        assert_eq!(val, 0);
        // items[8] = OpRef::int_op(3) → TAGBOX(1)
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_multi_frame_snapshot() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.constants
            .insert(OpRef::const_int(0).raw(), (99i64, majit_ir::Type::Int));

        let snapshot = Snapshot {
            vable_array: vec![],
            vref_array: vec![],
            framestack: vec![
                SnapshotFrame {
                    jitcode_index: 0,
                    pc: 10,
                    boxes: vec![OpRef::int_op(1).into(), OpRef::const_int(0).into()],
                },
                SnapshotFrame {
                    jitcode_index: 1,
                    pc: 20,
                    boxes: vec![OpRef::int_op(2).into(), OpRef::int_op(3).into()],
                },
            ],
        };

        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        // Multi-frame encoding: no box_count, RPython parity.
        let items = crate::resumecode::unpack_all(&rd_numb);
        assert_eq!(items[1], 3); // num_failargs: 3 boxes patched
        // Frame 0: items[4]=jitcode(0), items[5]=pc(10), items[6..7]=tagged
        assert_eq!(items[4], 0);
        assert_eq!(items[5], 10);
        // Frame 1: items[8]=jitcode(1), items[9]=pc(20), items[10..11]=tagged
        assert_eq!(items[8], 1);
        assert_eq!(items[9], 20);

        // Roundtrip with liveness-based closure.
        let rd_consts: Vec<majit_ir::Const> = memo.consts().to_vec();
        let frame_count = |jitcode_index: i32, _pc: i32| -> usize {
            match jitcode_index {
                0 => 2, // Frame 0 has 2 boxes
                1 => 2, // Frame 1 has 2 boxes
                _ => 0,
            }
        };
        let fail_arg_types = vec![
            majit_ir::Type::Int,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
        ];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, &fail_arg_types, Some(&frame_count));
        assert_eq!(num_failargs, 3);
        assert_eq!(rebuilt_frames.len(), 2);
        assert_eq!(rebuilt_frames[0].jitcode_index, 0);
        assert_eq!(rebuilt_frames[0].pc, 10);
        assert_eq!(rebuilt_frames[0].values.len(), 2);
        assert_eq!(rebuilt_frames[1].jitcode_index, 1);
        assert_eq!(rebuilt_frames[1].pc, 20);
        assert_eq!(rebuilt_frames[1].values.len(), 2);
    }

    #[test]
    fn test_finish_produces_rd_numb_and_liveboxes() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.constants
            .insert(OpRef::const_int(1).raw(), (42i64, majit_ir::Type::Int));
        env.virtuals.insert(2);
        env.types.insert(2, majit_ir::Type::Ref);

        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![
                OpRef::const_int(1),
                OpRef::int_op(1),
                OpRef::ref_op(2),
                OpRef::int_op(3),
            ],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let (rd_numb, rd_consts, _rd_virtuals, liveboxes, _livebox_types) =
            memo.finish(numb_state, &env, &mut [], None);

        // liveboxes should contain only TAGBOX entries: OpRef::int_op(1) and OpRef::int_op(3)
        assert_eq!(liveboxes.len(), 2);
        assert_eq!(liveboxes[0], OpRef::int_op(1)); // box #0
        assert_eq!(liveboxes[1], OpRef::int_op(3)); // box #1

        // rd_numb should be valid
        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, &fail_arg_types, None);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Const(majit_ir::Const::Int(42))
        );
        assert_eq!(
            rebuilt_frames[0].values[1],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(rebuilt_frames[0].values[2], RebuiltValue::Virtual(0));
        assert_eq!(
            rebuilt_frames[0].values[3],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn test_number_virtualizable_array_preserves_payload_then_identity_order() {
        use majit_ir::OpRef;

        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.types.insert(7, majit_ir::Type::Ref);
        env.types.insert(1, majit_ir::Type::Int);

        let snapshot = Snapshot {
            // pyjitpl.py:3302-3306 parity: payload slots first,
            // virtualizable identity (`virtualizable_boxes[-1]`) last.
            vable_array: vec![OpRef::int_op(1).into(), OpRef::ref_op(7).into()],
            vref_array: vec![],
            framestack: vec![SnapshotFrame {
                jitcode_index: 0,
                pc: 8,
                boxes: vec![OpRef::int_op(1).into()],
            }],
        };

        let numb_state = memo.number(&snapshot, &env, 0).unwrap();
        let items = crate::resumecode::unpack_all(&numb_state.create_numbering());

        assert_eq!(items[2], 2);
        let (val, tagbits) = untag(items[3] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);

        let (val, tagbits) = untag(items[4] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);

        assert_eq!(items[5], 0);
        assert_eq!(items[6], 0);
        assert_eq!(items[7], 8);

        // The frame slot reuses the payload tag because numbering follows
        // Box identity exactly: upstream dedups only when the same Box object
        // appears twice, and in this test we passed the same OpRef twice.
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
    }

    #[test]
    fn blackhole_from_resumedata_accepts_runtime_jitcode_without_canonical_pair() {
        use crate::blackhole::BlackholeInterpBuilder;
        use crate::jitcode::{
            BC_ABORT, BC_CATCH_EXCEPTION, BC_LIVE, BC_RVMPROF_CODE, JitCodeBuilder,
        };

        let mut writer = crate::resumecode::Writer::new(6);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count: no failargs
        writer.append_int(0); // vable_array length
        writer.append_int(0); // vref_array length
        writer.append_int(0); // jitcode_pos
        writer.append_int(0); // pc
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut runtime = JitCodeBuilder::default().finish();
        runtime.body_mut().code = vec![BC_LIVE, 0, 0, BC_ABORT];
        runtime.body_mut().c_num_regs_i = 1;
        runtime.body_mut().constants_i = vec![321];
        // Hand-crafted body bypasses the builder's `start_instr` path,
        // so populate `startpoints` explicitly: BC_LIVE at 0,
        // BC_ABORT at 3.  RPython `jitcode.py:85-90` asserts
        // `pc in self._startpoints`.
        runtime.body_mut().startpoints = Some([0_usize, 3].into_iter().collect());
        let runtime = std::sync::Arc::new(runtime);

        let mut builder = BlackholeInterpBuilder::new();
        builder.setup_cached_control_opcodes(
            BC_LIVE as i32,
            BC_CATCH_EXCEPTION as i32,
            BC_RVMPROF_CODE as i32,
        );
        let resolve_jitcode = |_jitcode_pos: i32, _pc: i32| -> Option<ResolvedJitCode> {
            Some(ResolvedJitCode::new(runtime.clone(), 0))
        };

        let all_liveness: Vec<u8> = vec![0, 0, 0];
        let (bh, virtualizable_ptr) = blackhole_from_resumedata(
            &mut builder,
            &resolve_jitcode,
            &rd_numb,
            &[],
            &all_liveness,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            &NullAllocator,
        )
        .expect("runtime-only jitcode should still resume");

        assert_eq!(virtualizable_ptr, 0);
        assert!(std::sync::Arc::ptr_eq(&bh.jitcode, &runtime));
        assert_eq!(bh.position, 0);
        assert_eq!(bh.registers_i, vec![0, 321]);
    }

    struct TestVirtualizableInfo;

    impl VirtualizableInfo for TestVirtualizableInfo {
        fn get_total_size(&self, _virtualizable: i64) -> usize {
            0
        }

        fn reset_token_gcref(&self, _virtualizable: i64) {}

        fn write_from_resume_data_partial(
            &self,
            _virtualizable: i64,
            _reader: &mut ResumeDataDirectReader,
        ) {
        }
    }

    #[test]
    #[should_panic(expected = "vable_size > 0")]
    fn test_consume_vref_and_vable_asserts_zero_vable_size_when_vinfo_present() {
        let mut writer = crate::resumecode::Writer::new(3);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count
        writer.append_int(0); // vable_size
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut reader =
            ResumeDataDirectReader::new(&rd_numb, &[], &[], &[], None, None, &NullAllocator);
        reader.consume_vref_and_vable(None, Some(&TestVirtualizableInfo), None);
    }

    #[test]
    #[should_panic(expected = "load_next_value_of_type: unexpected type Void")]
    fn test_next_value_of_type_rejects_void() {
        let mut reader =
            ResumeDataDirectReader::new(&[0, 0], &[], &[], &[], None, None, &NullAllocator);
        let _ = reader.next_value_of_type(majit_ir::Type::Void);
    }
}

// ═══════════════════════════════════════════════════════════════
// resume.py:901-1039 AbstractResumeDataReader
// resume.py:1354-1601 ResumeDataDirectReader
//
// Direct reader that decodes resume data and fills blackhole
// interpreter registers with concrete values from the deadframe.
// ═══════════════════════════════════════════════════════════════

use crate::blackhole::BlackholeInterpreter;
use crate::resumecode::Reader;

/// RPython virtualref_info interface for resume data consumption.
///
/// Corresponds to `metainterp_sd.virtualref_info` (VirtualRefInfo).
pub trait VRefInfo {
    /// resume.py:1397 vrefinfo.continue_tracing(vref, virtual)
    fn continue_tracing(&self, vref: i64, virtual_ref: i64);
}

/// RPython virtualizable_info interface for resume data consumption.
///
/// Corresponds to `jitdriver_sd.virtualizable_info` (VirtualizableInfo).
pub trait VirtualizableInfo {
    /// resume.py:1406 vinfo.get_total_size(virtualizable)
    fn get_total_size(&self, virtualizable: i64) -> usize;

    /// resume.py:1407 vinfo.reset_token_gcref(virtualizable)
    fn reset_token_gcref(&self, virtualizable: i64);

    /// resume.py:1408 vinfo.write_from_resume_data_partial(virtualizable, self)
    ///
    /// Read fields from the resume reader and write them into the virtualizable.
    fn write_from_resume_data_partial(
        &self,
        virtualizable: i64,
        reader: &mut ResumeDataDirectReader,
    );
}

/// RPython greenfield_info interface for resume data consumption.
///
/// Corresponds to `jitdriver_sd.greenfield_info`.
pub trait GreenfieldInfo {}

/// resume.py:1354 ResumeDataDirectReader
///
/// Reads encoded resume data (rd_numb) and fills blackhole interpreter
/// resume.py:874-899 AbstractVirtualCache / get_VirtualCache_class
///
/// ```text
/// class AbstractVirtualCache(object):
///     pass
///
/// def get_VirtualCache_class(suffix):
///     class VirtualCache(AbstractVirtualCache):
///         def __init__(self, virtuals_ptr_cache, virtuals_int_cache):
///             self.virtuals_ptr_cache = virtuals_ptr_cache
///             self.virtuals_int_cache = virtuals_int_cache
///
///         def get_ptr(self, i):  return self.virtuals_ptr_cache[i]
///         def get_int(self, i):  return self.virtuals_int_cache[i]
///         def set_ptr(self, i, v): self.virtuals_ptr_cache[i] = v
///         def set_int(self, i, v): self.virtuals_int_cache[i] = v
/// ```
///
/// RPython generates two flavours of this class — one for
/// `ResumeDataDirectReader` (raw `i64` slots) and one for the future
/// `ResumeDataBoxReader` (boxed). majit only emits the direct flavour at
/// runtime, so a single struct backs both.
#[derive(Default)]
pub struct VirtualCache {
    pub virtuals_ptr_cache: Vec<i64>,
    pub virtuals_int_cache: Vec<i64>,
}

impl VirtualCache {
    pub fn new() -> Self {
        VirtualCache::default()
    }

    /// resume.py:882-884 __init__
    pub fn from_caches(virtuals_ptr_cache: Vec<i64>, virtuals_int_cache: Vec<i64>) -> Self {
        VirtualCache {
            virtuals_ptr_cache,
            virtuals_int_cache,
        }
    }

    /// resume.py:886-887 get_ptr
    #[inline]
    pub fn get_ptr(&self, i: usize) -> i64 {
        self.virtuals_ptr_cache[i]
    }

    /// resume.py:889-890 get_int
    #[inline]
    pub fn get_int(&self, i: usize) -> i64 {
        self.virtuals_int_cache[i]
    }

    /// resume.py:892-893 set_ptr
    #[inline]
    pub fn set_ptr(&mut self, i: usize, v: i64) {
        self.virtuals_ptr_cache[i] = v;
    }

    /// resume.py:895-896 set_int
    #[inline]
    pub fn set_int(&mut self, i: usize, v: i64) {
        self.virtuals_int_cache[i] = v;
    }

    /// `len(self.virtuals_ptr_cache)` — both halves stay the same length.
    pub fn len(&self) -> usize {
        self.virtuals_ptr_cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.virtuals_ptr_cache.is_empty()
    }
}

/// registers directly from the deadframe's fail_args values.
///
/// Combines AbstractResumeDataReader (resume.py:901) mixin with
/// ResumeDataDirectReader (resume.py:1354) concrete class.
pub struct ResumeDataDirectReader<'a> {
    // AbstractResumeDataReader fields (resume.py:909-922)
    /// resume.py:918 resumecodereader
    pub resumecodereader: Reader<'a>,
    /// resume.py:919 items_resume_section — total items in resume section
    pub items_resume_section: i32,
    /// resume.py:921 count — number of failargs
    pub count: i32,
    /// resume.py:922 consts — constant pool from rd_consts.
    ///
    /// RPython stores `list[Const]` where each `Const` carries its own type
    /// (history.py:220 ConstInt / :261 ConstFloat / :307 ConstPtr). In the
    /// Rust port we represent each entry as `(raw_i64, Type)` so that
    /// `ConstPtr.getref_base()` parity (returning a GC-tracked pointer) can
    /// be surfaced to the minor-collection root walker — see
    /// `walk_rd_consts_refs` on `MetaInterp`. Raw `Vec<i64>` hid the
    /// Ref-typed entries from the GC and caused nursery use-after-free in
    /// TAGCONST decode paths (resume.py:1557 decode_int / :1566 decode_ref /
    /// :1578 decode_float).
    pub consts: &'a [majit_ir::Const],

    // ResumeDataDirectReader fields (resume.py:1364-1367)
    /// resume.py:1366 deadframe — raw fail_args values
    pub deadframe: &'a [i64],
    /// pyre flat-deadframe adaptation: original type of each deadframe slot.
    /// RPython's CPU exposes typed getters (get_ref_value/get_int_value/...);
    /// pyre passes a flat raw slice and needs slot kinds to emulate
    /// load_box_from_cpu(kind) for TAGBOX decode.
    pub deadframe_types: Option<&'a [majit_ir::Type]>,

    // resume.py:1358 resume_after_guard_not_forced
    //   0: not a GUARD_NOT_FORCED
    //   1: in handle_async_forcing
    //   2: resuming from the GUARD_NOT_FORCED
    pub resume_after_guard_not_forced: u8,

    // resume.py:909 rd_virtuals
    rd_virtuals: Option<&'a [VirtualInfo]>,

    /// resume.py:910 virtuals_cache — lazy-allocated virtual objects.
    /// Wraps both the ptr and int half so callers go through the RPython
    /// `VirtualCache` API (`get_ptr`/`set_ptr`/`get_int`/`set_int`).
    pub virtuals_cache: VirtualCache,

    /// resume.py:1367 — CPU allocation backend.
    /// RPython uses self.cpu (from metainterp_sd.cpu) for allocate_with_vtable etc.
    allocator: &'a dyn BlackholeAllocator,

    /// resume.py:1022 `self.metainterp_sd.liveness_info` — shared
    /// packed `all_liveness` buffer used by `_prepare_next_section` /
    /// `enumerate_vars`. RPython reaches it through `self.metainterp_sd`;
    /// pyre holds the slice directly because `ResumeDataDirectReader`
    /// lives outside the `MetaInterpStaticData` ownership graph.
    pub all_liveness: &'a [u8],

    /// resume.py:1404: virtualizable pointer read by consume_vable_info.
    /// Stored so the caller (blackhole_from_resumedata) can access it
    /// after consume_vref_and_vable completes.
    pub virtualizable_ptr: i64,
}

/// resume.py:1433-1456 CPU allocation interface for virtual materialization.
///
/// ResumeDataDirectReader calls these methods when TAGVIRTUAL values
/// need to be lazily allocated during decode_ref/decode_int.
pub trait BlackholeAllocator {
    /// resume.py:1437-1439 allocate_with_vtable(known_class, descr) →
    ///   exec_new_with_vtable(self.cpu, descr)
    fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        let _ = (descr, vtable);
        0
    }
    /// resume.py:1441-1442 allocate_struct(typedescr) → cpu.bh_new(typedescr)
    fn allocate_struct(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        let _ = typedescr;
        0
    }
    /// resume.py:1444 allocate_array(length, arraydescr, clear)
    fn allocate_array(&self, length: usize, arraydescr: &majit_ir::DescrRef, clear: bool) -> i64 {
        let _ = (length, arraydescr, clear);
        0
    }
    /// resume.py:1509 setfield
    fn setfield(&self, struct_ptr: i64, field_descr: u32, value: i64) {
        let _ = (struct_ptr, field_descr, value);
    }
    /// resume.py:1531 setarrayitem_int(array, i, value, arraydescr)
    fn setarrayitem_int(&self, array: i64, index: usize, value: i64, descr: &majit_ir::DescrRef) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1535 setarrayitem_ref(array, i, value, arraydescr)
    fn setarrayitem_ref(&self, array: i64, index: usize, value: i64, descr: &majit_ir::DescrRef) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1539 setarrayitem_float(array, i, value, arraydescr)
    fn setarrayitem_float(&self, array: i64, index: usize, value: i64, descr: &majit_ir::DescrRef) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1520-1529 setinteriorfield(index, array, fieldnum, descr)
    /// RPython passes the live descr object; backend reads offset/size/type from it.
    fn setinteriorfield_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    fn setinteriorfield_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    fn setinteriorfield_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1452 allocate_raw_buffer(func, size)
    fn allocate_raw_buffer(&self, func: i64, size: usize) -> i64 {
        let _ = (func, size);
        0
    }
    /// resume.py:1543 setrawbuffer_item(buffer, fieldnum, offset, descr)
    fn setrawbuffer_item(
        &self,
        buffer: i64,
        offset: usize,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let _ = (buffer, offset, value, descr);
    }
    /// resume.py:1509-1528 setfield — write field value at known offset.
    fn setfield_typed(
        &self,
        struct_ptr: i64,
        value: i64,
        descr: u32,
        field_offset: usize,
        field_size: usize,
    ) {
        let _ = (field_offset, field_size);
        self.setfield(struct_ptr, descr, value);
    }
    /// pendingfields: setarrayitem dispatch by descr_index (legacy u32 path).
    fn setarrayitem_typed(&self, array: i64, index: usize, value: i64, descr: u32) {
        let _ = (array, index, value, descr);
    }
    /// Pyre-specific: box a raw int to a PyObject ref.
    ///
    /// RPython equivalent: cpu.get_ref_value always returns GCREF because
    /// the jitframe stores typed values. Pyre's deadframe is untyped i64;
    /// when a slot typed as Int is read through decode_ref, this method
    /// wraps it into a valid GCREF (W_IntObject).
    fn box_int(&self, value: i64) -> i64 {
        value // default: return raw value (override in pyre allocator)
    }
    /// Pyre-specific: box raw float bits to a PyObject ref.
    fn box_float(&self, value: i64) -> i64 {
        value
    }
}

/// Default no-op allocator.
pub struct NullAllocator;
impl BlackholeAllocator for NullAllocator {}

impl VirtualInfo {
    /// resume.py:576 kind attribute — REF for object/struct/array/string,
    /// INT for raw buffers.
    pub fn is_about_raw(&self) -> bool {
        matches!(
            self,
            VirtualInfo::VRawBuffer { .. } | VirtualInfo::VRawSlice { .. }
        )
    }

    /// resume.py:618/634/650 allocate(decoder, index)
    ///
    /// Allocate a virtual object and fill in its fields from the decoder.
    /// Sets virtuals_cache_ptr[index] before filling fields (for recursive refs).
    pub fn allocate(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64 {
        match self {
            VirtualInfo::VirtualObj {
                fields,
                fielddescrs,
                descr,
                known_class,
                ..
            } => {
                // resume.py:619 allocate_with_vtable(descr=self.descr)
                let vtable = known_class.unwrap_or(0) as usize;
                let obj = descr
                    .as_ref()
                    .map(|d| allocator.allocate_with_vtable(d, vtable))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, obj);
                for (i, (field_descr, source)) in fields.iter().enumerate() {
                    let fd = fielddescrs.get(i);
                    let field_type = fd.map(|fd| fd.field_type).unwrap_or(majit_ir::Type::Ref);
                    let value = match field_type {
                        majit_ir::Type::Ref => decoder.decode_field_source(source),
                        majit_ir::Type::Float => decoder.decode_field_source_float(source),
                        _ => decoder.decode_field_source_int(source),
                    };
                    // resume.py:1509-1528 setfield uses the descr's byte
                    // offset/size rather than a symbolic `field_descr` id.
                    // PyreBlackholeAllocator only implements setfield_typed
                    // (pyre objects are raw Rust structs); the default
                    // `setfield` in the trait is a no-op. Always route
                    // through setfield_typed so the field actually lands.
                    let field_offset = fd.map(|fd| fd.offset).unwrap_or(0);
                    let field_size = fd.map(|fd| fd.field_size).unwrap_or(8);
                    allocator.setfield_typed(obj, value, *field_descr, field_offset, field_size);
                }
                obj
            }
            VirtualInfo::VStruct {
                typedescr,
                fields,
                fielddescrs,
                ..
            } => {
                // resume.py:635 allocate_struct(self.typedescr)
                let obj = typedescr
                    .as_ref()
                    .map(|d| allocator.allocate_struct(d))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, obj);
                for (i, (field_descr, source)) in fields.iter().enumerate() {
                    let fd = fielddescrs.get(i);
                    let field_type = fd.map(|fd| fd.field_type).unwrap_or(majit_ir::Type::Ref);
                    let value = match field_type {
                        majit_ir::Type::Ref => decoder.decode_field_source(source),
                        majit_ir::Type::Float => decoder.decode_field_source_float(source),
                        _ => decoder.decode_field_source_int(source),
                    };
                    let field_offset = fd.map(|fd| fd.offset).unwrap_or(0);
                    let field_size = fd.map(|fd| fd.field_size).unwrap_or(8);
                    allocator.setfield_typed(obj, value, *field_descr, field_offset, field_size);
                }
                obj
            }
            VirtualInfo::VArray {
                arraydescr,
                clear,
                items,
                ..
            } => {
                let length = items.len();
                // resume.py:653: array = decoder.allocate_array(length, arraydescr, self.clear)
                let array = arraydescr
                    .as_ref()
                    .map(|d| allocator.allocate_array(length, d, *clear))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, array);
                // resume.py:656-670: dispatch by arraydescr element type
                let is_pointers = arraydescr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .map_or(false, |ad| ad.is_array_of_pointers());
                let is_floats = arraydescr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .map_or(false, |ad| ad.is_array_of_floats());
                if let Some(ad) = arraydescr.as_ref() {
                    for (i, source) in items.iter().enumerate() {
                        if is_pointers {
                            // resume.py:659: decoder.setarrayitem_ref(array, i, num, arraydescr)
                            let value = decoder.decode_field_source(source);
                            allocator.setarrayitem_ref(array, i, value, ad);
                        } else if is_floats {
                            // resume.py:664: decoder.setarrayitem_float(array, i, num, arraydescr)
                            let value = decoder.decode_field_source_float(source);
                            allocator.setarrayitem_float(array, i, value, ad);
                        } else {
                            // resume.py:669: decoder.setarrayitem_int(array, i, num, arraydescr)
                            let value = decoder.decode_field_source_int(source);
                            allocator.setarrayitem_int(array, i, value, ad);
                        }
                    }
                }
                array
            }
            // resume.py:748-760: VArrayStructInfo.allocate
            VirtualInfo::VArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
                ..
            } => {
                let size = element_fields.len();
                // resume.py:749: array = decoder.allocate_array(self.size, self.arraydescr, clear=True)
                let array = arraydescr
                    .as_ref()
                    .map(|d| allocator.allocate_array(size, d, true))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, array);
                // resume.py:752-759:
                //   for i in range(self.size):
                //       for j in range(len(self.fielddescrs)):
                //           num = self.fieldnums[p]
                //           if not tagged_eq(num, UNINITIALIZED):
                //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                //           p += 1
                for (i, fields) in element_fields.iter().enumerate() {
                    debug_assert_eq!(
                        fields.len(),
                        fielddescrs.len(),
                        "VArrayStruct element_fields[{i}] has {} fields but {} fielddescrs",
                        fields.len(),
                        fielddescrs.len()
                    );
                    for (j, &(_, ref source)) in fields.iter().enumerate() {
                        if matches!(source, VirtualFieldSource::Uninitialized) {
                            continue;
                        }
                        // resume.py:757: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                        decoder.setinteriorfield(i, array, source, &fielddescrs[j], allocator);
                    }
                }
                array
            }
            _ => {
                decoder.virtuals_cache.set_ptr(index, 0);
                0
            }
        }
    }

    /// resume.py:701 VRawBufferInfo.allocate_int / VRawSliceInfo.allocate_int
    pub fn allocate_int(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64 {
        match self {
            VirtualInfo::VRawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => {
                assert_eq!(offsets.len(), descrs.len());
                assert_eq!(offsets.len(), values.len());
                // resume.py:703: buffer = decoder.allocate_raw_buffer(self.func, self.size)
                let buffer = allocator.allocate_raw_buffer(*func, *size);
                // resume.py:704
                decoder.virtuals_cache.set_int(index, buffer);
                // resume.py:705-708: for i in range(len(self.offsets)):
                //     offset = self.offsets[i]; descr = self.descrs[i]
                //     decoder.setrawbuffer_item(buffer, fieldnums[i], offset, descr)
                for i in 0..offsets.len() {
                    let descr = &descrs[i];
                    let source = &values[i];
                    // resume.py:1543-1550: dispatch by descr kind
                    assert!(
                        descr.item_type != 0,
                        "raw buffer entry must not be pointer type"
                    );
                    let value = if descr.item_type == 2 {
                        // resume.py:1545: descr.is_array_of_floats() → decode_float
                        decoder.decode_field_source_float(source)
                    } else {
                        // resume.py:1549: else → decode_int
                        decoder.decode_field_source_int(source)
                    };
                    allocator.setrawbuffer_item(buffer, offsets[i], value, descr);
                }
                buffer
            }
            VirtualInfo::VRawSlice { offset, parent } => {
                // resume.py:723-725 — parent is an INT virtual (raw buffer)
                let parent_val = decoder.decode_field_source_int(parent);
                let result = parent_val + *offset;
                decoder.virtuals_cache.set_int(index, result);
                result
            }
            _ => panic!("allocate_int called on non-raw virtual"),
        }
    }
}

impl<'a> ResumeDataDirectReader<'a> {
    /// resume.py:1364 __init__
    pub fn new(
        rd_numb: &'a [u8],
        rd_consts: &'a [majit_ir::Const],
        all_liveness: &'a [u8],
        deadframe: &'a [i64],
        deadframe_types: Option<&'a [majit_ir::Type]>,
        all_virtuals: Option<(Vec<i64>, Vec<i64>)>,
        allocator: &'a dyn BlackholeAllocator,
    ) -> Self {
        // resume.py:915-922 _init
        let mut resumecodereader = Reader::new(rd_numb);
        let items_resume_section = resumecodereader.next_item();
        let count = resumecodereader.next_item();

        // resume.py:1368-1376
        let (resume_after_guard_not_forced, virtuals_cache) =
            if let Some((ptrs, ints)) = all_virtuals {
                // resume.py:1373-1374: special case for GUARD_NOT_FORCED
                (2, VirtualCache::from_caches(ptrs, ints))
            } else {
                (0, VirtualCache::new())
            };

        ResumeDataDirectReader {
            resumecodereader,
            items_resume_section,
            count,
            consts: rd_consts,
            deadframe,
            deadframe_types,
            resume_after_guard_not_forced,
            rd_virtuals: None,
            virtuals_cache,
            allocator,
            all_liveness,
            virtualizable_ptr: 0,
        }
    }

    /// resume.py:924 _prepare — init virtuals and pending fields.
    pub fn prepare(
        &mut self,
        rd_virtuals: Option<&'a [VirtualInfo]>,
        rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    ) {
        // resume.py:925
        self.prepare_virtuals(rd_virtuals);
        // resume.py:926
        if let Some(guard_pf) = rd_guard_pendingfields {
            self.prepare_guard_pendingfields(guard_pf);
        }
    }

    /// resume.py:993 _prepare_pendingfields — variant for GuardPendingFieldEntry.
    ///
    /// RPython encodes pendingfield target/value as tagged values (TAGBOX/
    /// TAGCONST/TAGVIRTUAL) via _gettagged in _add_pending_fields
    /// (resume.py:548-549). At restore time, decode_ref(num) resolves
    /// the tagged value to a concrete pointer.
    ///
    /// `target_tagged` / `value_tagged` are always populated by the
    /// time pendingfields reach this method:
    ///
    ///   * The single production constructor at
    ///     `optimizeopt/optimizer.rs:3453` initializes both to
    ///     `UNASSIGNED`, but the entries are immediately fed through
    ///     `memo.finish()` (`optimizeopt/mod.rs:3390`) →
    ///     `_add_pending_fields` (`resume.rs:3554`), which writes
    ///     the tags from `_gettagged`.
    ///   * The sharing path (`_copy_resume_data_from`) routes resume
    ///     reads through `ResumeGuardCopiedDescr.prev` (compile.py:849
    ///     `get_resumestorage(): return prev`); readers reach the
    ///     donor's already-tagged entries via the descr's `prev`
    ///     pointer rather than touching the shared op directly.
    ///
    /// Hence the only branch is the RPython tagged path; an
    /// `UNASSIGNED` entry escaping into restore-time would be a
    /// soundness bug, so this method panics matching RPython's
    /// implicit invariant (`resume.py:1002` indexes the tagged
    /// number with `decode_ref`).
    fn prepare_guard_pendingfields(&mut self, pendingfields: &[majit_ir::GuardPendingFieldEntry]) {
        for pf in pendingfields {
            // resume.py:1000 PENDINGFIELDSTRUCT.lldescr parity:
            // derive (offset, size, type) from the descr (FieldDescr or
            // ArrayDescr) at consume time. RPython always carries
            // lldescr — pyre's producer at optimizer.rs:3389 mirrors
            // this by setting `pf.descr = pf_op.descr.clone()` for
            // every pending field (pf_op is always a Setfield_gc /
            // Setarrayitem_gc op with a descr).
            let descr = pf
                .descr
                .as_ref()
                .expect("resume.py:1000 PENDINGFIELDSTRUCT.lldescr must be set");
            let (field_offset, field_size, field_type) = if let Some(fd) = descr.as_field_descr() {
                (fd.offset(), fd.field_size(), fd.field_type())
            } else if let Some(ad) = descr.as_array_descr() {
                let item_idx = pf.item_index.max(0) as usize;
                let offset = ad.base_size() + item_idx * ad.item_size();
                (offset, ad.item_size(), ad.item_type())
            } else {
                panic!(
                    "pending field descr must be FieldDescr or ArrayDescr (descr_index={})",
                    pf.descr_index,
                );
            };
            // resume.py:1002-1007 tagged path. UNASSIGNED tags must
            // never reach this method; see doc comment above.
            assert!(
                pf.target_tagged != UNASSIGNED && pf.value_tagged != UNASSIGNED,
                "GuardPendingFieldEntry reached prepare_guard_pendingfields with \
                 UNASSIGNED tag (target_tagged={}, value_tagged={}, descr_index={}); \
                 _add_pending_fields must have run before restore time",
                pf.target_tagged,
                pf.value_tagged,
                pf.descr_index,
            );
            // resume.py:1002: struct = self.decode_ref(num)
            let struct_ptr = self.decode_ref(pf.target_tagged);
            // resume.py:1004-1007: setfield/setarrayitem dispatch by descr type
            let value = match field_type {
                majit_ir::Type::Ref => self.decode_ref(pf.value_tagged),
                majit_ir::Type::Float => self.decode_float(pf.value_tagged),
                _ => self.decode_int(pf.value_tagged),
            };

            if pf.item_index < 0 {
                // resume.py:1005: self.setfield(struct, fieldnum, descr)
                self.allocator.setfield_typed(
                    struct_ptr,
                    value,
                    pf.descr_index,
                    field_offset,
                    field_size,
                );
            } else {
                // resume.py:1007: self.setarrayitem(struct, itemindex, fieldnum, descr)
                self.allocator.setarrayitem_typed(
                    struct_ptr,
                    pf.item_index as usize,
                    value,
                    pf.descr_index,
                );
            }
        }
    }

    /// resume.py:1378 handling_async_forcing
    pub fn handling_async_forcing(&mut self) {
        self.resume_after_guard_not_forced = 1;
    }

    // ---- AbstractResumeDataReader methods (resume.py:928-1038) ----

    /// resume.py:928 read_jitcode_pos_pc
    pub fn read_jitcode_pos_pc(&mut self) -> (i32, i32) {
        let jitcode_pos = self.resumecodereader.next_item();
        let pc = self.resumecodereader.next_item();
        (jitcode_pos, pc)
    }

    /// resume.py:933 next_int
    pub fn next_int(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_int(tagged)
    }

    /// resume.py:936 next_ref
    pub fn next_ref(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_ref(tagged)
    }

    /// resume.py:939 next_float
    pub fn next_float(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_float(tagged)
    }

    /// resume.py:1410-1421 load_next_value_of_type
    pub fn next_value_of_type(&mut self, tp: majit_ir::Type) -> i64 {
        match tp {
            majit_ir::Type::Int => self.next_int(),
            majit_ir::Type::Ref => self.next_ref(),
            majit_ir::Type::Float => self.next_float(),
            other => panic!("load_next_value_of_type: unexpected type {other:?}"),
        }
    }

    /// resume.py:942 done_reading
    pub fn done_reading(&self) -> bool {
        self.resumecodereader.items_read >= self.items_resume_section as usize
    }

    /// resume.py:945 getvirtual_ptr
    ///
    /// Returns the index'th virtual, building it lazily if needed.
    /// Note that this may be called recursively; that's why the
    /// allocate() methods must fill in the cache as soon as they
    /// have the object, before they fill its fields.
    pub fn getvirtual_ptr(&mut self, index: usize) -> i64 {
        // resume.py:950: assert self.virtuals_cache is not None
        assert!(
            !self.virtuals_cache.is_empty(),
            "getvirtual_ptr: virtuals_cache is empty (rd_virtuals not prepared)"
        );
        // resume.py:951-952
        let v = self.virtuals_cache.get_ptr(index);
        if v != 0 {
            return v;
        }
        // resume.py:953-955: lazy allocation
        assert!(self.rd_virtuals.is_some(), "rd_virtuals is None");
        // Safety: rd_virtuals is an immutable slice reference that we need to
        // read while mutating virtuals_cache through self. The slice data is
        // never modified by allocate(), only the cache vectors are written.
        let rd_virtuals_ptr = self.rd_virtuals.unwrap().as_ptr();
        let rd_virtuals_len = self.rd_virtuals.unwrap().len();
        let vinfo = unsafe { &*rd_virtuals_ptr.add(index) };
        debug_assert!(index < rd_virtuals_len);
        let allocator = self.allocator as *const dyn BlackholeAllocator;
        let v = vinfo.allocate(self, index, unsafe { &*allocator });
        debug_assert_eq!(
            v,
            self.virtuals_cache.get_ptr(index),
            "resume.py: bad cache"
        );
        v
    }

    /// resume.py:958 getvirtual_int
    pub fn getvirtual_int(&mut self, index: usize) -> i64 {
        // resume.py:959: assert self.virtuals_cache is not None
        assert!(
            !self.virtuals_cache.is_empty(),
            "getvirtual_int: virtuals_cache is empty (rd_virtuals not prepared)"
        );
        // resume.py:960-961
        let v = self.virtuals_cache.get_int(index);
        if v != 0 {
            return v;
        }
        // resume.py:962-966
        assert!(self.rd_virtuals.is_some(), "rd_virtuals is None");
        let rd_virtuals_ptr = self.rd_virtuals.unwrap().as_ptr();
        let vinfo = unsafe { &*rd_virtuals_ptr.add(index) };
        assert!(vinfo.is_about_raw(), "getvirtual_int: not a raw virtual");
        let allocator = self.allocator as *const dyn BlackholeAllocator;
        let v = vinfo.allocate_int(self, index, unsafe { &*allocator });
        debug_assert_eq!(
            v,
            self.virtuals_cache.get_int(index),
            "resume.py: bad cache"
        );
        v
    }

    /// resume.py:969 force_all_virtuals
    pub fn force_all_virtuals(&mut self) -> (&[i64], &[i64]) {
        if let Some(rd_virtuals) = self.rd_virtuals {
            for i in 0..rd_virtuals.len() {
                let rd_virtual = &rd_virtuals[i];
                if rd_virtual.is_about_raw() {
                    // resume.py:977: kind == INT
                    self.getvirtual_int(i);
                } else {
                    // resume.py:976: kind == REF
                    self.getvirtual_ptr(i);
                }
            }
        }
        (
            &self.virtuals_cache.virtuals_ptr_cache,
            &self.virtuals_cache.virtuals_int_cache,
        )
    }

    /// resume.py:983 _prepare_virtuals
    fn prepare_virtuals(&mut self, virtuals: Option<&'a [VirtualInfo]>) {
        if let Some(v) = virtuals {
            self.rd_virtuals = Some(v);
            // resume.py:990-991
            self.virtuals_cache = VirtualCache::from_caches(vec![0; v.len()], vec![0; v.len()]);
        }
    }

    // ---- ResumeDataDirectReader methods (resume.py:1380-1601) ----

    /// resume.py:1381-1384 `consume_one_section(self, blackholeinterp)`.
    ///
    /// ```python
    /// def consume_one_section(self, blackholeinterp):
    ///     self.blackholeinterp = blackholeinterp
    ///     info = blackholeinterp.get_current_position_info()
    ///     self._prepare_next_section(info)
    /// ```
    pub fn consume_one_section(&mut self, bh: &mut BlackholeInterpreter) {
        // resume.py:1383
        let info = bh.get_current_position_info();
        // resume.py:1384
        self._prepare_next_section(info, bh);
    }

    /// resume.py:1017-1026 `_prepare_next_section(self, info)`.
    ///
    /// ```python
    /// def _prepare_next_section(self, info):
    ///     from rpython.jit.codewriter.jitcode import enumerate_vars
    ///     enumerate_vars(info,
    ///             self.metainterp_sd.liveness_info,
    ///             self._callback_i,
    ///             self._callback_r,
    ///             self._callback_f,
    ///             self.unique_id)
    /// ```
    ///
    /// `self.all_liveness` shadows `self.metainterp_sd.liveness_info` —
    /// the shared packed buffer that `enumerate_vars` indexes with
    /// `info`. The three callbacks still call `next_int`/`next_ref`/
    /// `next_float` on this reader (resume.py:1028-1038), matching
    /// `_callback_i/_callback_r/_callback_f` plus `write_an_int/write_a_ref/
    /// write_a_float` (resume.py:1590-1597).
    fn _prepare_next_section(&mut self, info: usize, bh: &mut BlackholeInterpreter) {
        use majit_translate::liveness::LivenessIterator;

        let all_liveness: &[u8] = self.all_liveness;

        // jitcode.py:149-151 — three length bytes.
        let length_i = all_liveness[info] as u32;
        let length_r = all_liveness[info + 1] as u32;
        let length_f = all_liveness[info + 2] as u32;
        // jitcode.py:152
        let mut offset = info + 3;

        // resume.py:1028-1030 `_callback_i` / jitcode.py:153-157.
        if length_i != 0 {
            let mut it = LivenessIterator::new(offset, length_i, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_int();
                // resume.py:1590-1591 `write_an_int`.
                bh.setarg_i(reg_idx as usize, value);
            }
            offset = it.offset;
        }
        // resume.py:1032-1034 `_callback_r` / jitcode.py:158-162.
        if length_r != 0 {
            let mut it = LivenessIterator::new(offset, length_r, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_ref();
                // resume.py:1593-1594 `write_a_ref`.
                bh.setarg_r(reg_idx as usize, value);
            }
            offset = it.offset;
        }
        // resume.py:1036-1038 `_callback_f` / jitcode.py:163-166.
        if length_f != 0 {
            let mut it = LivenessIterator::new(offset, length_f, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_float();
                // resume.py:1596-1597 `write_a_float`.
                bh.setarg_f(reg_idx as usize, value);
            }
            // `offset` is the end of the float section; no further use.
            let _ = offset;
        }
    }

    /// resume.py:1386 consume_virtualref_info
    pub fn consume_virtualref_info(&mut self, vrefinfo: Option<&dyn VRefInfo>) {
        // resume.py:1389
        let size = self.resumecodereader.next_item();
        // resume.py:1390-1391
        if vrefinfo.is_none() || size == 0 {
            // resume.py:1391: assert size == 0
            assert!(
                size == 0,
                "consume_virtualref_info: vrefinfo is None but size={size} != 0"
            );
            return;
        }
        let vrefinfo = vrefinfo.unwrap();
        // resume.py:1393-1397
        for _i in 0..size {
            let virtual_val = self.next_ref();
            let vref = self.next_ref();
            // resume.py:1397
            vrefinfo.continue_tracing(vref, virtual_val);
        }
    }

    /// resume.py:1399 consume_vable_info
    pub fn consume_vable_info(&mut self, vinfo: &dyn VirtualizableInfo, vable_size: i32) {
        // resume.py:1403
        assert!(vable_size > 0);
        // resume.py:1404
        let virtualizable = self.next_ref();
        self.virtualizable_ptr = virtualizable;
        // resume.py:1406
        let expected = vinfo.get_total_size(virtualizable) as i32;
        // resume.py:1406: assert vinfo.get_total_size(virtualizable) == vable_size - 1
        assert!(
            expected == vable_size - 1,
            "consume_vable_info: vinfo.get_total_size(0x{:x}) = {} != vable_size - 1 = {}",
            virtualizable,
            expected,
            vable_size - 1
        );
        // resume.py:1407
        vinfo.reset_token_gcref(virtualizable);
        // resume.py:1408
        vinfo.write_from_resume_data_partial(virtualizable, self);
    }

    /// resume.py:1424 consume_vref_and_vable
    pub fn consume_vref_and_vable(
        &mut self,
        vrefinfo: Option<&dyn VRefInfo>,
        vinfo: Option<&dyn VirtualizableInfo>,
        ginfo: Option<&dyn GreenfieldInfo>,
    ) {
        // resume.py:1425
        let vable_size = self.resumecodereader.next_item();

        if self.resume_after_guard_not_forced != 2 {
            // resume.py:1427-1428
            if let Some(vi) = vinfo {
                self.consume_vable_info(vi, vable_size);
            }
            // resume.py:1429-1430
            if ginfo.is_some() {
                let _ginfo_item = self.resumecodereader.next_item();
            }
            // resume.py:1431
            self.consume_virtualref_info(vrefinfo);
        } else {
            // resume.py:1433-1435
            self.resumecodereader.jump(vable_size as usize);
            let vref_size = self.resumecodereader.next_item();
            self.resumecodereader.jump(vref_size as usize * 2);
        }
    }

    /// resume.py:1552 decode_int
    pub fn decode_int(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1555 — ConstInt.getint(): return the i64 value.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx].getint()
            }
            TAGINT => {
                // resume.py:1557
                num as i64
            }
            TAGVIRTUAL => {
                // resume.py:1559
                self.getvirtual_int(num as usize)
            }
            TAGBOX => {
                // resume.py:1561-1564
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                self.deadframe[idx as usize]
            }
            _ => unreachable!("bad tag: {tag}"),
        }
    }

    /// resume.py:1566 decode_ref
    pub fn decode_ref(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1569-1571
                if tagged_eq(tagged, NULLREF) {
                    return 0; // ConstPtr.value (null pointer)
                }
                // resume.py:1571
                let idx = (num - TAG_CONST_OFFSET) as usize;
                // history.py:316 ConstPtr.getref_base() returns the GCREF value.
                self.consts[idx].getref_base().as_usize() as i64
            }
            TAGVIRTUAL => {
                // resume.py:1573
                self.getvirtual_ptr(num as usize)
            }
            TAGBOX => {
                // resume.py:1575-1578
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                let value = self.deadframe[idx as usize];
                match self
                    .deadframe_types
                    .and_then(|tys| tys.get(idx as usize))
                    .copied()
                    .unwrap_or(majit_ir::Type::Ref)
                {
                    majit_ir::Type::Ref => value,
                    // RPython: decode_ref + TAGBOX always returns a GC
                    // pointer via cpu.get_ref_value(). These Int/Float
                    // branches are needed because the optimizer may
                    // unbox Ref→Int in deadframe slots.
                    majit_ir::Type::Int => self.allocator.box_int(value),
                    majit_ir::Type::Float => self.allocator.box_float(value),
                    majit_ir::Type::Void => value,
                }
            }
            TAGINT => {
                // RPython: decode_ref never sees TAGINT (assert tag == TAGBOX).
                // Fires when optimizer produces TAGINT in ref-register snapshot.
                self.allocator.box_int(num as i64)
            }
            _ => {
                panic!("decode_ref: unexpected tag {tag}")
            }
        }
    }

    /// resume.py:1580 decode_float
    pub fn decode_float(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1583 — ConstFloat.getfloatstorage(): i64 bits.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx].getfloatstorage()
            }
            TAGBOX => {
                // resume.py:1585-1588
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                self.deadframe[idx as usize]
            }
            _ => {
                // resume.py:1580 — only TAGCONST and TAGBOX valid for floats
                panic!("decode_float: unexpected tag {tag}")
            }
        }
    }

    /// Decode a VirtualFieldSource as a REF value (resume.py:1566 decode_ref).
    ///
    /// Virtual sources go through getvirtual_ptr (REF virtuals).
    pub fn decode_field_source(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe[*index],
            // resume.py:1568 ConstPtr.getref_base() — the Const carries its type.
            ResumeValueSource::Constant(c) => c.getref_base().as_usize() as i64,
            ResumeValueSource::Virtual(index) => self.getvirtual_ptr(*index),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// Decode a VirtualFieldSource as an INT value (resume.py:1552 decode_int).
    ///
    /// Virtual sources go through getvirtual_int (INT/raw virtuals).
    pub fn decode_field_source_int(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe[*index],
            // resume.py:1555 ConstInt.getint().
            ResumeValueSource::Constant(c) => c.getint(),
            ResumeValueSource::Virtual(index) => self.getvirtual_int(*index),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// Decode a VirtualFieldSource as a FLOAT value (resume.py:1554 decode_float).
    ///
    /// Floats are stored as raw i64 bits. TAGVIRTUAL is invalid for
    /// float fields — virtual floats would route through a different
    /// VirtualInfo variant.
    pub fn decode_field_source_float(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe[*index],
            // resume.py:1583 ConstFloat.getfloatstorage().
            ResumeValueSource::Constant(c) => c.getfloatstorage(),
            ResumeValueSource::Virtual(_) => {
                panic!("decode_field_source_float: TAGVIRTUAL not valid for float field")
            }
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// resume.py:1520-1529 setinteriorfield(index, array, fieldnum, descr)
    ///
    /// Dispatches by descr.is_pointer_field() / is_float_field() / else.
    pub fn setinteriorfield(
        &mut self,
        index: usize,
        array: i64,
        source: &VirtualFieldSource,
        descr: &majit_ir::DescrRef,
        allocator: &dyn BlackholeAllocator,
    ) {
        let is_pointer = descr
            .as_interior_field_descr()
            .map_or(false, |ifd| ifd.field_descr().is_pointer_field());
        let is_float = descr
            .as_interior_field_descr()
            .map_or(false, |ifd| ifd.field_descr().is_float_field());
        if is_pointer {
            let value = self.decode_field_source(source);
            allocator.setinteriorfield_gc_r(array, index, value, descr);
        } else if is_float {
            let value = self.decode_field_source_float(source);
            allocator.setinteriorfield_gc_f(array, index, value, descr);
        } else {
            let value = self.decode_field_source_int(source);
            allocator.setinteriorfield_gc_i(array, index, value, descr);
        }
    }

    /// resume.py:1599 int_add_const
    pub fn int_add_const(&self, base: i64, offset: i64) -> i64 {
        base + offset
    }
}

/// resume.py:1312 blackhole_from_resumedata
///
/// Build a chain of BlackholeInterpreters from encoded resume data.
/// Returns the topmost (innermost) interpreter.
///
/// `resolve_jitcode` corresponds to RPython's `jitcodes[jitcode_pos]` lookup
/// (`resume.py:1339`). Matches upstream's `(jitcode, pc)` tuple result.
pub struct ResolvedJitCode {
    pub jitcode: std::sync::Arc<crate::jitcode::JitCode>,
    pub pc: usize,
    pub virtualizable_stack_base: Option<usize>,
}

impl ResolvedJitCode {
    pub fn new(jitcode: std::sync::Arc<crate::jitcode::JitCode>, pc: usize) -> Self {
        Self {
            jitcode,
            pc,
            virtualizable_stack_base: None,
        }
    }

    pub fn with_virtualizable_stack_base(mut self, stack_base: usize) -> Self {
        self.virtualizable_stack_base = Some(stack_base);
        self
    }
}

pub fn blackhole_from_resumedata<'a>(
    builder: &mut crate::blackhole::BlackholeInterpBuilder,
    resolve_jitcode: &dyn Fn(i32, i32) -> Option<ResolvedJitCode>,
    rd_numb: &'a [u8],
    rd_consts: &'a [majit_ir::Const],
    all_liveness: &'a [u8],
    deadframe: &'a [i64],
    deadframe_types: Option<&'a [majit_ir::Type]>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    vrefinfo: Option<&dyn VRefInfo>,
    vinfo: Option<&dyn VirtualizableInfo>,
    ginfo: Option<&dyn GreenfieldInfo>,
    allocator: &'a dyn BlackholeAllocator,
) -> Option<(BlackholeInterpreter, i64)> {
    // resume.py:1315-1327 The initialization is stack-critical code: it
    // must not be interrupted by StackOverflow, otherwise the
    // jit_virtual_refs are left in a dangling state.
    //
    // RPython wraps the body in try/finally so _stop() runs on every
    // exit path. The RAII CriticalCodeGuard gives us Drop-based
    // guarantee — ordinary returns, `?` propagation, AND panic unwind
    // all re-enable the report_error flag.
    let _cc_guard = crate::CriticalCodeGuard::enter();
    // resume.py:1317-1321
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        all_liveness,
        deadframe,
        deadframe_types,
        None,
        allocator,
    );

    // resume.py:1324
    resumereader.prepare(rd_virtuals, rd_guard_pendingfields);

    // resume.py:1325
    resumereader.consume_vref_and_vable(vrefinfo, vinfo, ginfo);
    drop(_cc_guard);

    // resume.py:1404: virtualizable pointer read by consume_vable_info.
    let virtualizable_ptr = resumereader.virtualizable_ptr;

    // resume.py:1332-1343
    // Build chain bottom-up: first frame acquired is the outermost.
    let mut curbh: Option<Box<BlackholeInterpreter>> = None;

    while !resumereader.done_reading() {
        // resume.py:1334-1336
        let mut nextbh = builder.acquire_interp();
        nextbh.nextblackholeinterp = curbh;

        // resume.py:1338-1340
        let (jitcode_pos, pc) = resumereader.read_jitcode_pos_pc();
        // resume.py:1339-1340: jitcode = jitcodes[jitcode_pos]; curbh.setposition(jitcode, pc)
        let resolved = resolve_jitcode(jitcode_pos, pc)?;
        nextbh.setposition(resolved.jitcode.clone(), resolved.pc);
        if let Some(stack_base) = resolved.virtualizable_stack_base {
            nextbh.virtualizable_stack_base = stack_base;
        }

        // resume.py:1341
        resumereader.consume_one_section(&mut nextbh);

        // resume.py:1342
        nextbh.handle_rvmprof_enter();

        curbh = Some(Box::new(nextbh));
    }

    curbh.map(|b| (*b, virtualizable_ptr))
}

/// resume.py:1345 force_from_resumedata
///
/// Force all virtuals from resume data without running a blackhole.
/// Used for GUARD_NOT_FORCED handling.
/// Returns (virtuals_cache_ptr, virtuals_cache_int) — RPython VirtualCache parity.
pub fn force_from_resumedata<'a>(
    rd_numb: &'a [u8],
    rd_consts: &'a [majit_ir::Const],
    all_liveness: &'a [u8],
    deadframe: &'a [i64],
    deadframe_types: Option<&'a [majit_ir::Type]>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    vrefinfo: Option<&dyn VRefInfo>,
    vinfo: Option<&dyn VirtualizableInfo>,
    ginfo: Option<&dyn GreenfieldInfo>,
    allocator: &'a dyn BlackholeAllocator,
) -> (Vec<i64>, Vec<i64>) {
    // resume.py:1347-1348
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        all_liveness,
        deadframe,
        deadframe_types,
        None,
        allocator,
    );
    // resume.py:1371 common-case __init__ calls self._prepare(storage)
    // before handling_async_forcing() flips the GUARD_NOT_FORCED state.
    resumereader.prepare(rd_virtuals, rd_guard_pendingfields);
    resumereader.handling_async_forcing();
    // resume.py:1350
    resumereader.consume_vref_and_vable(vrefinfo, vinfo, ginfo);
    // resume.py:1351: return resumereader.force_all_virtuals()
    let (ptrs, ints) = resumereader.force_all_virtuals();
    (ptrs.to_vec(), ints.to_vec())
}
