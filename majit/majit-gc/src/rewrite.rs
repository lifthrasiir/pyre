/// GC rewriter — transforms high-level allocation and store operations
/// into GC-aware lower-level IR before code generation.
///
/// Converts:
/// - NEW / NEW_WITH_VTABLE -> CALL_MALLOC_NURSERY + tid initialization
///   (consecutive fixed-size allocations are batched into a single
///    CALL_MALLOC_NURSERY with NURSERY_PTR_INCREMENT for subsequent objects)
/// - NEW_ARRAY / NEW_ARRAY_CLEAR -> CALL_MALLOC_NURSERY_VARSIZE
/// - SETFIELD_GC with a Ref-typed value -> COND_CALL_GC_WB + SETFIELD_GC
///
/// Reference: rpython/jit/backend/llsupport/rewrite.py GcRewriterAssembler.
use std::collections::{BTreeMap, HashMap, HashSet};

use majit_ir::Type;
use majit_ir::descr::{DescrRef, FieldDescr, SizeDescr};
use majit_ir::resoperation::{Op, OpCode, OpRef};

use crate::{GcRewriter, WriteBarrierDescr};

/// Alignment for nursery allocations (8 bytes).
const NURSERY_ALIGN: usize = 8;

/// Align `size` up to `NURSERY_ALIGN`.
fn round_up(size: usize) -> usize {
    (size + NURSERY_ALIGN - 1) & !(NURSERY_ALIGN - 1)
}

/// `get_array_token(rstr.STR, ...)` / `get_array_token(rstr.UNICODE, ...)`
/// as consumed by rewrite.py:295-318.  Reads `(itemsize, basesize)` from
/// the injected ArrayDescr and applies the `basesize -= 1` correction
/// for STR (the extra_item_after_alloc null).
fn strgetsetitem_token(op: &Op, is_str: bool) -> (i64, i64) {
    let descr = op
        .descr
        .as_ref()
        .expect("STR/UNICODE getitem/setitem op must carry an ArrayDescr");
    let ad = descr
        .as_array_descr()
        .expect("STR/UNICODE getitem/setitem descr must be an ArrayDescr");
    let itemsize = ad.item_size() as i64;
    let mut basesize = ad.base_size() as i64;
    if is_str {
        // rewrite.py:298 assert itemsize == 1 for STR.
        assert_eq!(
            itemsize, 1,
            "rewrite.py:298 STR getitem/setitem itemsize must be 1"
        );
        basesize -= 1; // rewrite.py:299 — skip extra null character
    }
    (itemsize, basesize)
}

/// resoperation.py:1524-1531 `OpHelpers.get_gc_load`.
/// Select GC_LOAD_I / GC_LOAD_R / GC_LOAD_F by result type.
fn get_gc_load(tp: Type) -> OpCode {
    match tp {
        Type::Int => OpCode::GcLoadI,
        Type::Ref => OpCode::GcLoadR,
        Type::Float => OpCode::GcLoadF,
        Type::Void => panic!("get_gc_load: cannot lower a Void-typed load"),
    }
}

/// resoperation.py:1533-1541 `OpHelpers.get_gc_load_indexed`.
/// Select GC_LOAD_INDEXED_I / GC_LOAD_INDEXED_R / GC_LOAD_INDEXED_F by
/// result type.
fn get_gc_load_indexed(tp: Type) -> OpCode {
    match tp {
        Type::Int => OpCode::GcLoadIndexedI,
        Type::Ref => OpCode::GcLoadIndexedR,
        Type::Float => OpCode::GcLoadIndexedF,
        Type::Void => panic!("get_gc_load_indexed: cannot lower a Void-typed load"),
    }
}

/// regalloc.py:871 — compiled_loop_token._ll_initial_locs + frame_info.
///
/// Per-callee metadata needed by handle_call_assembler to allocate and
/// fill the callee jitframe.
#[derive(Clone, Debug)]
pub struct CallAssemblerCalleeLocs {
    /// regalloc.py:869 — byte offsets of each inputarg within jf_frame,
    /// relative to BASEITEMOFS (i.e. index_list[i] = loc.value - base_ofs).
    pub _ll_initial_locs: Vec<i32>,
    /// Total jitframe depth (in slots).
    /// regalloc.py:861 — needed for gen_malloc_frame allocation size.
    pub frame_depth: usize,
    /// rewrite.py:669 — ptr2int(loop_token.compiled_loop_token.frame_info).
    /// Raw address of the callee's JitFrameInfo struct.
    pub frame_info_ptr: usize,
    /// pyjitpl.py:3605 — jd.index_of_virtualizable.
    /// Index into the original arglist of the virtualizable box.
    /// -1 if no virtualizable.
    pub index_of_virtualizable: i32,
}

/// GC rewriter implementation.
///
/// Walks a list of IR operations and rewrites allocation / store ops
/// into backend-friendly forms (CALL_MALLOC_NURSERY, COND_CALL_GC_WB, etc.).
pub struct GcRewriterImpl {
    /// Nursery free pointer address (for inline allocation).
    pub nursery_free_addr: usize,
    /// Nursery top pointer address.
    pub nursery_top_addr: usize,
    /// Maximum object size for nursery allocation.
    pub max_nursery_size: usize,
    /// Write barrier descriptor.
    pub wb_descr: WriteBarrierDescr,
    /// JitFrame info for call_assembler rewriting.
    /// rewrite.py:665 — handle_call_assembler needs frame layout info.
    pub jitframe_info: Option<JitFrameDescrs>,
    /// Type annotations for constant OpRefs. RPython's ConstPtr/ConstInt
    /// boxes carry `.type` intrinsically; here we pass constant types
    /// from the optimizer so the rewriter can check `v.type == 'r'` on
    /// constant values (rewrite.py:930).
    pub constant_types: HashMap<u32, Type>,
    /// rewrite.py:673 — lookup compiled_loop_token._ll_initial_locs
    /// by target token number. Provided by the backend.
    pub call_assembler_callee_locs:
        Option<Box<dyn Fn(u64) -> Option<CallAssemblerCalleeLocs> + Send>>,
    /// llmodel.py:39 `load_supported_factors = (1,)` — the default for
    /// CPUs whose addressing mode only scales by one. x86 overrides this
    /// at `rpython/jit/backend/x86/runner.py:31` with `(1, 2, 4, 8)`.
    /// Consumed by `cpu_simplify_scale` (rewrite.py:1124) to decide
    /// whether a non-constant index's factor can be folded into the
    /// backend's addressing mode or must be pre-scaled in IR.
    pub load_supported_factors: &'static [i64],
    /// model.py:22 `supports_load_effective_address = False` (base) /
    /// x86/runner.py:22 + arm/runner.py:26 override to `True`.
    /// Consumed by `emit_load_effective_address` (rewrite.py:1083): when
    /// `True` the helper emits a single `LOAD_EFFECTIVE_ADDRESS`; when
    /// `False` it expands to `(INT_LSHIFT?) + INT_ADD + INT_ADD`.
    pub supports_load_effective_address: bool,
    /// llsupport/gc.py:30-34 `malloc_zero_filled` parity.
    ///
    /// `true` when the allocator zero-fills payload bytes on
    /// allocation.  pyre's `Nursery` uses `alloc_zeroed` (nursery.rs:68)
    /// and `reset()` memsets to zero on recycle (nursery.rs:105-110),
    /// so production is always `true`.  Gates `clear_gc_fields` per
    /// rewrite.py:499-500; a future non-zero-fill allocator path would
    /// flip this to `false` and let the existing plumbing emit
    /// explicit NULL-pointer stores at flush time
    /// (rewrite.py:761-766).
    pub malloc_zero_filled: bool,
    /// llsupport/gc.py:39 `self.memcpy_fn = memcpy_fn` cast to a Signed
    /// integer via `cast_ptr_to_adr` + `cast_adr_to_int`
    /// (rewrite.py:1046-1047). Embedded as a ConstInt into the lowered
    /// `CALL_N(memcpy_fn, dst, src, n)` emitted by
    /// `rewrite_copy_str_content` for COPYSTRCONTENT / COPYUNICODECONTENT.
    pub memcpy_fn: i64,
    /// llsupport/gc.py:40-43 `self.memcpy_descr = get_call_descr(...)`.
    /// CallDescr stamped onto the lowered `CALL_N(memcpy_fn, ...)` —
    /// `[Signed, Signed, Signed] -> Void`, `EF_CANNOT_RAISE`,
    /// `can_collect=False`. Single instance shared across rewrites
    /// (`make_memcpy_calldescr` singleton, descr.rs).
    pub memcpy_descr: DescrRef,
    /// llsupport/gc.py:46 `self.str_descr = get_array_descr(self, rstr.STR)`.
    /// Provides `basesize` / `itemsize=1` for `rewrite_copy_str_content`
    /// COPYSTRCONTENT lowering — basesize is offset by `-1` at use to
    /// skip the `extra_item_after_alloc=True` null terminator
    /// (rstr.py:1226; rewrite.py:1051-1053).
    pub str_descr: DescrRef,
    /// llsupport/gc.py:47 `self.unicode_descr = get_array_descr(self, rstr.UNICODE)`.
    /// Provides `basesize` / `itemsize` (2 or 4 bytes per UCS) for
    /// `rewrite_copy_str_content` COPYUNICODECONTENT lowering;
    /// `itemscale = log2(itemsize)` (rewrite.py:1057-1063).
    pub unicode_descr: DescrRef,
    /// llsupport/gc.py:48 `self.str_hash_descr = get_field_descr(self, rstr.STR, 'hash')`.
    /// FieldDescr for the `hash` field of `rstr.STR`. Consumed by
    /// `clear_varsize_gc_fields` at NEWSTR allocation when
    /// `malloc_zero_filled=false` (rewrite.py:529-530), where upstream
    /// emits `emit_setfield(result, ConstInt(0), descr=hash_descr)`.
    pub str_hash_descr: DescrRef,
    /// llsupport/gc.py:49 `self.unicode_hash_descr = get_field_descr(self, rstr.UNICODE, 'hash')`.
    /// Same role as `str_hash_descr` for NEWUNICODE
    /// (rewrite.py:531-535).
    pub unicode_hash_descr: DescrRef,
    /// llsupport/gc.py:33-37 `self.fielddescr_vtable = get_field_descr(
    /// self, rclass.OBJECT, 'typeptr')` or `None` under
    /// `gcremovetypeptr=True`.  Consumed by `handle_new`'s
    /// NEW_WITH_VTABLE branch (rewrite.py:482-484) to stamp the vtable
    /// onto the freshly-allocated object's typeptr slot.  `None`
    /// disables the vtable store entirely (matching upstream's
    /// `gcremovetypeptr` build configuration).
    pub fielddescr_vtable: Option<DescrRef>,
    /// llsupport/gc.py:157 `fielddescr_tid = None` (Boehm) /
    /// llsupport/gc.py:394 `self.fielddescr_tid = get_field_descr(self,
    /// self.GCClass.HDR, 'tid')` (framework GC).  Consumed by
    /// `rewrite.py:914-918` `gen_initialize_tid` to stamp the type id
    /// onto the freshly-allocated object's header word.  `None` makes
    /// `gen_initialize_tid` a no-op (Boehm path).
    ///
    /// pyre's HDR sits *before* the object pointer (vs RPython's HDR at
    /// the object pointer); `gen_initialize_tid` translates the descr's
    /// offset by `-HDR_SIZE` so the GC_STORE addresses the header word.
    pub fielddescr_tid: Option<DescrRef>,
    /// gc.py:422-431 `generate_function('malloc_array', ...)` function addr.
    pub malloc_array_fn: i64,
    /// gc.py:433-444 `generate_function('malloc_array_nonstandard', ...)`
    /// function addr.
    pub malloc_array_nonstandard_fn: i64,
    /// gc.py:453-458 `generate_function('malloc_str', ...)` function addr.
    pub malloc_str_fn: i64,
    /// gc.py:460-465 `generate_function('malloc_unicode', ...)` function addr.
    pub malloc_unicode_fn: i64,
    /// gc.py:481-490 `generate_function('malloc_big_fixedsize', ...)` function addr.
    /// Consumed by `rewrite.py:778-796 gen_malloc_fixedsize` framework-GC arm
    /// when a fixed-size NEW exceeds the nursery threshold.
    pub malloc_big_fixedsize_fn: i64,
    /// gc.py:45 `self.malloc_array_descr = get_call_descr(...)`.
    pub malloc_array_descr: DescrRef,
    /// gc.py:45 `self.malloc_array_nonstandard_descr = get_call_descr(...)`.
    pub malloc_array_nonstandard_descr: DescrRef,
    /// gc.py:45 `self.malloc_str_descr = get_call_descr(...)`.
    pub malloc_str_descr: DescrRef,
    /// gc.py:45 `self.malloc_unicode_descr = get_call_descr(...)`.
    pub malloc_unicode_descr: DescrRef,
    /// gc.py:45 `self.malloc_big_fixedsize_descr = get_call_descr(...)`.
    pub malloc_big_fixedsize_descr: DescrRef,
    /// gc.py:396 `self.standard_array_basesize`.
    pub standard_array_basesize: usize,
    /// gc.py:397 `self.standard_array_length_ofs`.
    pub standard_array_length_ofs: usize,
}

/// JitFrame field descriptors for handle_call_assembler.
///
/// rewrite.py:666 — `descrs = self.gc_ll_descr.getframedescrs(self.cpu)`
#[derive(Clone)]
pub struct JitFrameDescrs {
    /// GC type id for JitFrame (from gc.register_type).
    pub jitframe_tid: u32,
    /// JitFrame fixed header size (bytes).
    pub jitframe_fixed_size: usize,
    /// Byte offsets of JitFrame fields (from majit_metainterp::jitframe).
    pub jf_frame_info_ofs: i32,
    pub jf_descr_ofs: i32,
    pub jf_force_descr_ofs: i32,
    pub jf_savedata_ofs: i32,
    pub jf_guard_exc_ofs: i32,
    pub jf_forward_ofs: i32,
    /// Offset from JitFrame start to jf_frame length field.
    pub jf_frame_ofs: usize,
    /// unpack_arraydescr(arraydescr): basesize, measured from JitFrame start.
    pub jf_frame_baseitemofs: usize,
    /// descrs.arraydescr.lendescr.offset, measured from JitFrame start.
    pub jf_frame_lengthofs: usize,
    /// SIGN_SIZE: size of one jf_frame slot.
    pub sign_size: usize,
}

impl JitFrameDescrs {
    /// llmodel.py:80-90 + llmodel.py:97-104 — itemsize of the per-arg-type
    /// frame arraydescr (signedarraydescr / refarraydescr / floatarraydescr),
    /// read via getarraydescr_for_frame + unpack_arraydescr_size.
    ///
    /// Upstream builds:
    ///   signedarraydescr = ad                        (itemsize = WORD)
    ///   refarraydescr    = ArrayDescr(.., ad.itemsize, ..)
    ///   floatarraydescr  = ArrayDescr(..,
    ///         ad.itemsize * 2 if WORD == 4 else ad.itemsize, ..)
    /// so on 32-bit builds a FLOAT slot spans two Signed words.
    fn frame_itemsize(&self, ty: Type) -> i64 {
        match ty {
            Type::Int | Type::Ref => self.sign_size as i64,
            Type::Float => {
                if self.sign_size == 4 {
                    (self.sign_size * 2) as i64
                } else {
                    self.sign_size as i64
                }
            }
            Type::Void => panic!("CALL_ASSEMBLER arg must have a concrete type"),
        }
    }
}

/// rewrite.py:1117-1132 `cpu_simplify_scale` return value.
///
/// Encodes the three semantic outcomes that upstream signals via the
/// `(index_box, emit)` tuple:
///   - `Const`        — index folded into offset; caller must emit the
///                      non-indexed GC_LOAD / GC_STORE form (upstream's
///                      `index_box is None` check at rewrite.py:148, :199).
///   - `Passthrough`  — original index re-used unchanged; the CPU's
///                      addressing mode supports the requested factor.
///   - `PreScale(op)` — the helper constructed an unemitted INT_LSHIFT /
///                      INT_MUL; `_emit_mul_if_factor_offset_not_supported`
///                      (or vector_ext callers) emits it before reading the
///                      OpRef back as the final indexed-op `index` arg.
enum ScaledIndex {
    Const,
    Passthrough(OpRef),
    PreScale(Op),
}

/// rewrite.py:719-758 last_zero_arrays parity.
///
/// A ZERO_ARRAY op already emitted into `out` whose start/length will
/// be tightened (or zeroed out) at flush time based on which indices
/// were covered by subsequent SETARRAYITEM ops.  Mirrors RPython's
/// in-place setarg pattern — one ZERO_ARRAY per array, trimmed from
/// both ends; never split into multiple runs.
struct PendingZero {
    /// Index in `out` of the already-emitted ZERO_ARRAY op.
    out_index: usize,
    /// The OpRef of the array being zeroed.
    array_ref: OpRef,
    /// Initial length (number of items) before any trimming.
    length: usize,
    /// Per-item byte size from the array descriptor (`scale` in
    /// rewrite.py:727).
    scale: i64,
}

/// Per-rewrite mutable state (not stored on the struct so that
/// `rewrite_for_gc` can take `&self`).
struct RewriteState {
    /// Output operation list.
    out: Vec<Op>,
    /// Next position index for emitted result ops that do not have an
    /// explicit source position to preserve.
    next_pos: u32,
    /// Constant pool (from optimizer) — maps OpRef key → i64 value.
    constants: HashMap<u32, i64>,
    /// rewrite.py:930 parity — structural equivalent of Box.type.
    /// Maps OpRef.0 → Type for all known values (both op results and
    /// constants). In RPython each Box carries its own `.type` attribute;
    /// here we mirror that with an explicit lookup table because OpRef
    /// is a plain u32 without embedded type information.
    result_types: HashMap<u32, Type>,

    // ── Nursery batching ──
    /// The index in `out` of the current CALL_MALLOC_NURSERY op, if any.
    /// We try to merge consecutive small allocations into one bump.
    pending_malloc_idx: Option<usize>,
    /// Total size accumulated in the current CALL_MALLOC_NURSERY.
    pending_malloc_total: usize,
    /// Size of the *previous* individual allocation (used for
    /// NURSERY_PTR_INCREMENT offset).
    previous_size: usize,
    /// OpRef of the last object produced by the current nursery batch.
    last_malloced_ref: OpRef,

    // ── Write barrier tracking ──
    /// rewrite.py:41-45: _write_barrier_applied — set of OpRef indices
    /// whose write barrier has already been emitted (freshly allocated
    /// objects, or objects we already issued a WB for). Cleared whenever
    /// we emit an operation that can trigger a collection or on LABEL.
    wb_applied: HashSet<OpRef>,
    /// Forwarding map from original result OpRefs to rewritten result OpRefs.
    forwarding: HashMap<OpRef, OpRef>,

    // ── Array length tracking (rewrite.py:59 _known_lengths) ──
    /// Maps array OpRef → known length. Populated when NEW_ARRAY has a
    /// constant length operand (rewrite.py:551). Cleared on LABEL
    /// (rewrite.py:1005) and emitting_an_operation_that_can_collect.
    known_lengths: HashMap<OpRef, usize>,

    // ── Pending zero tracking ──
    /// Deferred ZERO_ARRAY ops that may be optimized away if subsequent
    /// SETARRAYITEM writes cover the entire range.
    pending_zeros: Vec<PendingZero>,
    /// Tracks which array indices have been explicitly SET since the
    /// pending zero was recorded. Keyed by array OpRef index.
    initialized_indices: HashMap<OpRef, HashSet<usize>>,
    /// rewrite.py:61 `_delayed_zero_setfields = {}`.
    ///
    /// Map from base OpRef → {byte-offset: ()} of zero-init
    /// SETFIELD_GC stores deferred by `clear_gc_fields`.  An explicit
    /// SETFIELD_GC that overwrites the same offset removes the entry
    /// via `consider_setfield_gc` (rewrite.py:506-512); anything still
    /// pending at the next can-collect / flush point is emitted as
    /// `GC_STORE(ptr, ofs, 0, WORD)` by `emit_pending_zeros`
    /// (rewrite.py:761-766).
    _delayed_zero_setfields: HashMap<OpRef, BTreeMap<i64, ()>>,

    // ── INT_ADD/INT_SUB constant-fold tracking (rewrite.py:64) ──
    /// `_constant_additions[box]` = `(older_box, constant_add)` for an
    /// int_add/int_sub whose constant operand can be folded into a
    /// downstream GC_STORE_INDEXED / GC_LOAD_INDEXED offset.  See
    /// rewrite.py:1008 record_int_add_or_sub and rewrite.py:173
    /// _try_use_older_box.
    ///
    /// pyre's emit_setarrayitem path does not currently lower to
    /// GC_STORE_INDEXED, so this map is populated but its
    /// _try_use_older_box consumer is parked until that lowering is
    /// ported.  The parity skeleton is kept here so the structural
    /// presence matches upstream and the consumer can be wired without
    /// re-introducing the field.
    _constant_additions: HashMap<OpRef, (OpRef, i64)>,
    /// Next constant index for `OpRef::from_const`. Shares the
    /// constant-namespace with the tracer's ConstantPool; initialized in
    /// `with_constants` from the passed-in map so newly emitted constants
    /// do not collide with tracer entries.
    next_const_idx: u32,

    /// rewrite.py:470-471 `_changed_op` / `_changed_op_to` parity.
    ///
    /// When `remove_tested_failarg` rewrites an upcoming guard's failargs
    /// to substitute the tested box with a fresh `SAME_AS_I`, the
    /// rewritten guard is stashed here keyed by its position in the
    /// input op list. The main dispatch loop checks for a substitution
    /// at iteration `i` and swaps the rewritten op in place of the
    /// original (rewrite.py:366-367).
    changed_ops: HashMap<usize, Op>,

    /// rewrite.py:96-99 `get_box_replacement` — source→replacement mapping
    /// for ops that `transform_to_gc_load` has forwarded to a lowered
    /// form (GC_LOAD / GC_LOAD_INDEXED / GC_STORE / GC_STORE_INDEXED).
    /// Upstream sets this via `op.set_forwarded(newload)` and
    /// `emit_op` follows the forwarding at emission time.  In this Rust
    /// port we operate on owned `Op` values, so the replacement is
    /// keyed by the main-loop iteration index (stashed in
    /// `current_i`) and consumed by `emit_maybe_forwarded` when the
    /// outer dispatch reaches the op's emission site.
    forwarded_ops: HashMap<usize, Op>,
    /// Current main-loop iteration index, set by the outer dispatch
    /// before invoking `transform_to_gc_load` / `handle_*` helpers.
    /// Read by `set_forwarded` / `emit_maybe_forwarded` to key the
    /// `forwarded_ops` map.
    current_i: usize,
}

impl RewriteState {
    fn new(hint: usize, next_pos: u32) -> Self {
        RewriteState {
            out: Vec::with_capacity(hint + hint / 4),
            next_pos,
            constants: HashMap::new(),
            result_types: HashMap::new(),
            pending_malloc_idx: None,
            pending_malloc_total: 0,
            previous_size: 0,
            last_malloced_ref: OpRef::NONE,
            wb_applied: HashSet::new(),
            forwarding: HashMap::new(),
            known_lengths: HashMap::new(),
            pending_zeros: Vec::new(),
            initialized_indices: HashMap::new(),
            _delayed_zero_setfields: HashMap::new(),
            _constant_additions: HashMap::new(),
            next_const_idx: 0,
            changed_ops: HashMap::new(),
            forwarded_ops: HashMap::new(),
            current_i: 0,
        }
    }

    fn with_constants(hint: usize, next_pos: u32, constants: HashMap<u32, i64>) -> Self {
        // P3 category E — `constants` is an index-keyed constant pool
        // (raw u32 key), not a Box-identity dict.  Bit-helpers replace
        // the `OpRef::from_raw(k).is_constant()` round-trip that would
        // land on the to-be-retired `OpRef::Untyped` variant.
        let next_const_idx = constants
            .keys()
            .filter(|&&k| OpRef::raw_is_constant(k))
            .map(|&k| OpRef::raw_const_index(k))
            .max()
            .map_or(0, |m| m + 1);
        let mut s = Self::new(hint, next_pos);
        s.constants = constants;
        s.next_const_idx = next_const_idx;
        s
    }

    /// Resolve a constant value from the constant pool.
    fn resolve_constant(&self, key: u32) -> Option<i64> {
        self.constants.get(&key).copied()
    }

    /// Emit a fresh constant OpRef for `value`.
    ///
    /// rewrite.py:149/671/682 parity: RPython constructs a new
    /// `ConstInt(value)` at each call site without caching. Mirrors that
    /// — every call grows the constant pool by one entry in the shared
    /// high-bit constant namespace (same as the tracer's ConstantPool).
    /// Mints `OpRef::ConstInt` (history.py:220 `ConstInt.type = 'i'`)
    /// rather than the type-erased `from_const` Untyped variant.
    fn const_int(&mut self, value: i64) -> OpRef {
        let opref = OpRef::const_int(self.next_const_idx);
        self.next_const_idx += 1;
        self.constants.insert(opref.raw(), value);
        opref
    }

    /// Emit an op. Void ops do not consume a result id.
    ///
    /// For non-Void results this also registers `op.result_type()` in
    /// `result_types`.  RPython Boxes carry `.type` intrinsically
    /// (rewrite.py:930 `v.type`); the Rust port uses `result_types` as
    /// the structural equivalent, so any newly emitted result op must
    /// populate the table, not only the input-trace ops copied in by
    /// `run_with_constants` (line 2137-2142).
    fn emit(&mut self, mut op: Op) -> OpRef {
        let rt = op.result_type();
        let pos = if rt == Type::Void {
            OpRef::NONE
        } else {
            let pos = match rt {
                Type::Int => OpRef::int_op(self.next_pos),
                Type::Float => OpRef::float_op(self.next_pos),
                Type::Ref => OpRef::ref_op(self.next_pos),
                Type::Void => unreachable!("Void filtered above"),
            };
            self.next_pos += 1;
            self.result_types.insert(pos.raw(), rt);
            pos
        };
        op.pos = pos;
        self.out.push(op);
        pos
    }

    /// Emit a result-producing op, preserving the provided position when the
    /// source trace already assigned one.
    ///
    /// Registers `op.result_type()` in `result_types` for both the
    /// fresh-position and preserved-position cases — see `emit()` doc.
    /// For a preserved position the table already holds an entry from
    /// the input-trace scan, but re-registering is either a no-op
    /// (matching type) or an explicit overwrite when the caller built
    /// a rewritten op with a different type at the same position.
    fn emit_result(&mut self, mut op: Op, preferred_pos: OpRef) -> OpRef {
        let rt = op.result_type();
        let pos = if preferred_pos.is_none() {
            let pos = OpRef::op_typed(self.next_pos, rt);
            self.next_pos += 1;
            pos
        } else {
            preferred_pos
        };
        self.result_types.insert(pos.raw(), rt);
        op.pos = pos;
        self.out.push(op);
        pos
    }

    /// rewrite.py:699-711 emitting_an_operation_that_can_collect
    fn emitting_an_operation_that_can_collect(&mut self) {
        self.pending_malloc_idx = None;
        self.wb_applied.clear();
        self.emit_pending_zeros();
        // rewrite.py:708-711: clear _constant_additions here, not only
        // in emit_label, to avoid keeping the older boxes alive across
        // a potentially-collecting op.
        self._constant_additions.clear();
    }

    /// rewrite.py:1008 record_int_add_or_sub.
    ///
    /// When `op` is `INT_ADD/INT_ADD_OVF/INT_SUB/INT_SUB_OVF` whose
    /// non-result operand is a `ConstInt`, remember the (older box,
    /// constant) pair so a downstream GC_STORE_INDEXED / GC_LOAD_INDEXED
    /// emit can fold the constant into its `offset` argument via
    /// `_try_use_older_box`.
    ///
    /// `is_subtraction` distinguishes INT_SUB (negate the constant
    /// before storing) from INT_ADD.  Mirrors rewrite.py:1015.
    fn record_int_add_or_sub(&mut self, op: &Op, is_subtraction: bool) {
        let v_arg0 = op.arg(0);
        let v_arg1 = op.arg(1);
        let (box_arg, mut constant) = if let Some(c) = self.resolve_constant(v_arg1.raw()) {
            let signed = if is_subtraction { -c } else { c };
            (v_arg0, signed)
        } else if !is_subtraction {
            // rewrite.py:1019-1024: int_add only — try arg0 as constant.
            let Some(c) = self.resolve_constant(v_arg0.raw()) else {
                return;
            };
            (v_arg1, c)
        } else {
            return;
        };
        // rewrite.py:1026-1030 invariant: if box itself is a recorded
        // sum, fold its constant in and chain to the older origin.
        let box_arg = if let Some(&(older, extra)) = self._constant_additions.get(&box_arg) {
            constant += extra;
            older
        } else {
            box_arg
        };
        self._constant_additions.insert(op.pos, (box_arg, constant));
    }

    /// rewrite.py:173-182 _try_use_older_box.
    ///
    /// If `index_box` is a recorded `_constant_additions` entry, replace
    /// it with the older box and add `factor * extra_offset` to
    /// `offset`.
    fn _try_use_older_box(&self, index_box: OpRef, factor: i64, offset: i64) -> (OpRef, i64) {
        if let Some(&(older, extra)) = self._constant_additions.get(&index_box) {
            return (older, offset + factor * extra);
        }
        (index_box, offset)
    }

    fn remember_wb(&mut self, r: OpRef) {
        self.wb_applied.insert(r);
    }

    /// rewrite.py:66-67: remember_known_length
    fn remember_known_length(&mut self, op: OpRef, length: usize) {
        self.known_lengths.insert(op, length);
    }

    /// rewrite.py:81-82: known_length(op, default)
    fn known_length(&self, op: OpRef, default: usize) -> usize {
        self.known_lengths.get(&op).copied().unwrap_or(default)
    }

    /// rewrite.py:714: write_barrier_applied(op)
    fn wb_already_applied(&self, r: OpRef) -> bool {
        self.wb_applied.contains(&r)
    }

    /// rewrite.py:930 parity: `v.type` — get the type of an OpRef.
    fn result_type_of(&self, r: OpRef) -> Option<Type> {
        self.result_types.get(&r.raw()).copied()
    }

    /// rewrite.py:930 parity: `isinstance(v, ConstPtr) and not needs_write_barrier(v.value)`.
    /// A null ConstPtr never needs a write barrier.
    fn is_null_constant(&self, r: OpRef) -> bool {
        if let Some(&val) = self.constants.get(&r.raw()) {
            val == 0
        } else {
            false
        }
    }

    fn resolve(&self, r: OpRef) -> OpRef {
        if r.is_none() {
            return r;
        }
        self.forwarding.get(&r).copied().unwrap_or(r)
    }

    fn rewrite_op(&self, op: &Op) -> Op {
        let mut rewritten = op.clone();
        for arg in rewritten.args.iter_mut() {
            *arg = self.resolve(*arg);
        }
        if let Some(fail_args) = rewritten.fail_args.as_mut() {
            for arg in fail_args.iter_mut() {
                *arg = self.resolve(*arg);
            }
        }
        rewritten.pos = OpRef::NONE;
        rewritten
    }

    fn record_result_mapping(&mut self, old_pos: OpRef, new_pos: OpRef) {
        if !old_pos.is_none() {
            self.forwarding.insert(old_pos, new_pos);
        }
    }

    fn emit_rewritten_from(&mut self, original: &Op, rewritten: Op) -> OpRef {
        let result = if original.result_type() == Type::Void {
            self.emit(rewritten)
        } else {
            self.emit_result(rewritten, original.pos)
        };
        if original.result_type() != Type::Void {
            self.record_result_mapping(original.pos, result);
        }
        result
    }

    /// rewrite.py:128-130 `replace_op_with(op, newop)` — stash `lowered`
    /// as the replacement for the op at the current main-loop iteration.
    /// A subsequent `emit_maybe_forwarded` call for the same iteration
    /// will emit the stashed replacement.
    fn set_forwarded(&mut self, lowered: Op) {
        self.forwarded_ops.insert(self.current_i, lowered);
    }

    /// rewrite.py:100-126 `emit_op` — emits either the replacement
    /// previously stashed via `set_forwarded` (if any) or the rewritten
    /// original.  Preserves the original's position mapping so downstream
    /// uses of the original's `OpRef` resolve to the lowered op's result.
    fn emit_maybe_forwarded(&mut self, original: &Op) -> OpRef {
        if let Some(lowered) = self.forwarded_ops.remove(&self.current_i) {
            let result = if original.result_type() == Type::Void {
                self.emit(lowered)
            } else {
                self.emit_result(lowered, original.pos)
            };
            if original.result_type() != Type::Void {
                self.record_result_mapping(original.pos, result);
            }
            result
        } else {
            let rewritten = self.rewrite_op(original);
            self.emit_rewritten_from(original, rewritten)
        }
    }

    /// rewrite.py:84-91 `delayed_zero_setfields(op)` — get-or-create the
    /// per-base byte-offset set, resolving `r` through the forwarding
    /// map first (RPython calls `get_box_replacement(op)` here).
    fn delayed_zero_setfields(&mut self, r: OpRef) -> &mut BTreeMap<i64, ()> {
        let key = self.resolve(r);
        self._delayed_zero_setfields.entry(key).or_default()
    }

    /// Record that a SETARRAYITEM wrote to `array_ref[index]`,
    /// so the pending zero for that slot can be skipped.
    fn record_setarrayitem_index(&mut self, array_ref: OpRef, index: usize) {
        if self
            .pending_zeros
            .iter()
            .any(|pz| pz.array_ref == array_ref)
        {
            self.initialized_indices
                .entry(array_ref)
                .or_default()
                .insert(index);
        }
    }

    /// rewrite.py:719-758 emit_pending_zeros.
    ///
    /// Mutates each previously-emitted ZERO_ARRAY in place: trim from
    /// both ends past any indices that subsequent SETARRAYITEM writes
    /// covered, then rewrite arg(1)/arg(2) as byte offset / byte
    /// length (multiplied by the array's `scale`) and arg(3)/arg(4)
    /// to ConstInt(1) so the backend treats arg(1)/arg(2) as raw
    /// bytes.  Length 0 leaves the op as a no-op for the backend
    /// (matches rewrite.py:754 "may be ConstInt(0)").
    fn emit_pending_zeros(&mut self) {
        let pending = std::mem::take(&mut self.pending_zeros);
        let inited = std::mem::take(&mut self.initialized_indices);

        for pz in pending {
            let written = inited.get(&pz.array_ref);

            // rewrite.py:744-753 trim-from-front / trim-from-back.
            let mut start: usize = 0;
            while start < pz.length && written.is_some_and(|s| s.contains(&start)) {
                start += 1;
            }
            let mut stop: usize = pz.length;
            while stop > start && written.is_some_and(|s| s.contains(&(stop - 1))) {
                stop -= 1;
            }
            let scaled_start = self.const_int(start as i64 * pz.scale);
            let scaled_len = self.const_int((stop - start) as i64 * pz.scale);
            let one = self.const_int(1);

            let op = &mut self.out[pz.out_index];
            op.args[1] = scaled_start;
            op.args[2] = scaled_len;
            op.args[3] = one;
            op.args[4] = one;
        }

        // rewrite.py:760-766 — NULL-pointer writes still pending for
        // any zero-init fields not covered by a subsequent explicit
        // SETFIELD_GC.  RPython uses `WORD` (architecture pointer
        // size); pyre targets 64-bit exclusively so WORD == 8.
        //
        // The constant path inside `emit_gc_store_or_indexed`
        // (rewrite.py:148-150) collapses (ConstInt(ofs), factor=1,
        // offset=0) to a plain `GC_STORE(ptr, ConstInt(ofs),
        // ConstInt(0), ConstInt(WORD))`, which is what we emit here
        // directly.
        let pending_zsf = std::mem::take(&mut self._delayed_zero_setfields);
        for (ptr, entries) in pending_zsf {
            for ofs in entries.keys().copied() {
                let ofs_ref = self.const_int(ofs);
                let zero_ref = self.const_int(0);
                let word_ref = self.const_int(8);
                let store = Op::new(OpCode::GcStore, &[ptr, ofs_ref, zero_ref, word_ref]);
                self.emit(store);
            }
        }
    }
}

impl GcRewriterImpl {
    /// Can we use the nursery for this allocation size?
    fn can_use_nursery(&self, size: usize) -> bool {
        size <= self.max_nursery_size
    }

    /// rewrite.py:431-448 `could_merge_with_next_guard` parity.
    ///
    /// Returns true when `op` should be kept adjacent to the next guard,
    /// triggering a `emit_pending_zeros` flush at the top of the iteration
    /// (rewrite.py:376-377). Two cases:
    ///   * `op` is an overflow-producing arithmetic op (INT_*_OVF),
    ///     which pairs with a following GUARD_NO_OVERFLOW / GUARD_OVERFLOW.
    ///   * `op` is a comparison whose boolean result is tested by the
    ///     immediately following GUARD_TRUE/GUARD_FALSE/COND_CALL. In that
    ///     case the tested value appearing in the guard's failargs is
    ///     hoisted out into a dedicated `SAME_AS_I(0/1)` via
    ///     `remove_tested_failarg`.
    fn could_merge_with_next_guard(
        &self,
        op: &Op,
        i: usize,
        ops: &[Op],
        st: &mut RewriteState,
    ) -> bool {
        if !op.opcode.is_comparison() {
            // rewrite.py:436 fallback: int_xxx_ovf + guard_{,no_}overflow
            return op.opcode.is_ovf();
        }
        if i + 1 >= ops.len() {
            return false;
        }
        let next_op = &ops[i + 1];
        // rewrite.py:441-443 — merge only with a directly-consuming guard/cond_call.
        // RPython's `rop.COND_CALL` is the void-result variant, matching
        // pyre's `CondCallN`.
        if !matches!(
            next_op.opcode,
            OpCode::GuardTrue | OpCode::GuardFalse | OpCode::CondCallN
        ) {
            return false;
        }
        // rewrite.py:445 `next_op.getarg(0) is not op` — in pyre OpRef
        // carries the same identity role as RPython's box object.
        if next_op.arg(0) != op.pos {
            return false;
        }
        self.remove_tested_failarg(next_op, i + 1, st);
        true
    }

    /// rewrite.py:450-471 `remove_tested_failarg` parity.
    ///
    /// When a GUARD_TRUE/GUARD_FALSE's tested value is also present in the
    /// guard's failargs, emit a `SAME_AS_I(value)` (where `value = 0` for
    /// GUARD_TRUE / `1` for GUARD_FALSE — the constant the tested box would
    /// hold on the failure path) and rewrite the failargs list so the
    /// guard points at that SAME_AS_I instead of the boolean. The rewritten
    /// guard is stashed in `st.changed_ops` keyed by its index so the main
    /// dispatch loop substitutes it on the next iteration.
    fn remove_tested_failarg(&self, op: &Op, op_idx: usize, st: &mut RewriteState) {
        // rewrite.py:452-453: no-op for non-GUARD_{TRUE,FALSE} (e.g. COND_CALL
        // is merge-eligible via could_merge_with_next_guard but does not
        // carry failargs in the RPython sense).
        if !matches!(op.opcode, OpCode::GuardTrue | OpCode::GuardFalse) {
            return;
        }
        let fail_args = match op.fail_args.as_ref() {
            Some(fa) if !fa.is_empty() => fa,
            _ => return,
        };
        let target = op.arg(0);
        // rewrite.py:456-459: guard's failargs contain the tested box?
        let Some(idx) = fail_args.iter().position(|&a| a == target) else {
            return;
        };
        // rewrite.py:463 `value = int(opnum == rop.GUARD_FALSE)`
        let value: i64 = i64::from(op.opcode == OpCode::GuardFalse);
        let const_ref = st.const_int(value);
        let same = Op::new(OpCode::SameAsI, &[const_ref]);
        let same_pos = st.emit_result(same, OpRef::NONE);

        // rewrite.py:466-469 — rewrite failargs + stash the copy-and-changed
        // guard for the next iteration to pick up.
        let mut new_fail = fail_args.clone();
        new_fail[idx] = same_pos;
        let mut new_guard = op.clone();
        new_guard.fail_args = Some(new_fail);
        // pos is reassigned when emit/emit_result runs on the substituted op.
        new_guard.pos = OpRef::NONE;
        st.changed_ops.insert(op_idx, new_guard);
    }

    // ────────────────────────────────────────────────────────
    // NEW / NEW_WITH_VTABLE  → CALL_MALLOC_NURSERY + tid init
    // ────────────────────────────────────────────────────────

    fn handle_new(&self, op: &Op, st: &mut RewriteState) {
        let descr = op
            .descr
            .as_ref()
            .expect("NEW must have a SizeDescr")
            .as_size_descr()
            .expect("NEW descr must be SizeDescr");

        // rewrite.py:474-484 handle_malloc_operation parity:
        // descr.size in RPython already includes the GC header (the
        // OBJECT type is built with `size = sizeof(header) + sizeof(fields)`).
        // pyre's PyreSizeDescr reports `obj_size` as the bare struct size
        // (e.g. `size_of::<W_IntObject>() == 16`) WITHOUT the GC header,
        // so we add it here so that CallMallocNursery sees the same
        // "object-with-header" total that the cranelift backend expects
        // (it strips the header back off before passing to the alloc
        // shim's payload size).
        let size = round_up(descr.size() + crate::header::GcHeader::SIZE);
        let type_id = descr.type_id();

        // rewrite.py:540-543 — `if gen_malloc_nursery(size, op):
        //                          gen_initialize_tid(op, descr.tid)
        //                       else:
        //                          gen_malloc_fixedsize(size, descr.tid, op)`.
        // The fast path stamps the tid header inline; the slow path's
        // CALL_R helper (`malloc_big_fixedsize`) does it inside the
        // helper so `gen_malloc_fixedsize` does NOT call
        // `gen_initialize_tid` after the fact.
        let obj_ref = match self.gen_malloc_nursery(size, op.pos, st) {
            Some(r) => {
                self.gen_initialize_tid(r, type_id, st);
                r
            }
            None => self.gen_malloc_fixedsize(size, type_id, op.pos, st),
        };
        st.record_result_mapping(op.pos, obj_ref);

        // rewrite.py:479-484 handle_malloc_operation parity:
        //   elif opnum == rop.NEW_WITH_VTABLE:
        //       ...
        //       if self.gc_ll_descr.fielddescr_vtable is not None:
        //           self.emit_setfield(op, ConstInt(descr.get_vtable()),
        //                              descr=self.gc_ll_descr.fielddescr_vtable)
        //
        // Emit the vtable setfield SYNCHRONOUSLY (matching RPython). A
        // previous pyre-only deferral stored (obj, vtable) pairs in
        // `pending_vtable_inits` and flushed them on the next non-setfield
        // op; that left the object's ob_type slot uninitialized across
        // intermediate setfields and, when a guard fired in that window,
        // fail_args captured a partially-initialized nursery pointer whose
        // ob_type=NULL eventually crashed the blackhole's binary_op_fn
        // path (memory: phase5_super_lift_bisect_2026_04_17.md).
        if op.opcode == OpCode::NewWithVtable {
            // rewrite.py:482 `if self.gc_ll_descr.fielddescr_vtable is not None`.
            if let Some(vtable_fd_ref) = self.fielddescr_vtable.as_ref() {
                let vtable = descr.vtable();
                // Defensive — pyre's NEW_WITH_VTABLE descrs in production
                // always carry a non-zero vtable; some test fixtures
                // synthesize 0, in which case skip the store rather than
                // emit a NULL typeptr.
                if vtable != 0 {
                    self.gen_initialize_vtable(obj_ref, vtable, vtable_fd_ref, st);
                }
            }
        }

        // rewrite.py:544 `self.clear_gc_fields(descr, op)` — record every
        // GC-pointer field's byte offset so a pending NULL store is
        // emitted at the next flush point, unless cleared first by an
        // explicit SETFIELD_GC (rewrite.py:506-512).  No-op under pyre's
        // default zero-fill nursery (see `malloc_zero_filled`).
        self.clear_gc_fields(descr, obj_ref, st);
    }

    /// rewrite.py:498-504 `clear_gc_fields`.
    ///
    /// For every GC-pointer field on the fresh allocation, remember
    /// that a NULL-pointer store is needed unless a subsequent
    /// SETFIELD_GC overwrites it first.  Early-returns when the
    /// allocator already zero-fills payload bytes
    /// (`self.malloc_zero_filled`, rewrite.py:499-500).
    fn clear_gc_fields(&self, descr: &dyn SizeDescr, result: OpRef, st: &mut RewriteState) {
        if self.malloc_zero_filled {
            return;
        }
        // rewrite.py:501-504 — populate `delayed_zero_setfields[result][ofs] = None`
        // per GC-pointer field (`descr.gc_fielddescrs` / unpack_fielddescr).
        let entries = st.delayed_zero_setfields(result);
        for fd in descr.gc_fielddescrs() {
            entries.insert(fd.offset() as i64, ());
        }
    }

    // ────────────────────────────────────────────────────────
    // NEW_ARRAY / NEW_ARRAY_CLEAR  → CALL_MALLOC_NURSERY_VARSIZE / CALL_R
    // ────────────────────────────────────────────────────────

    /// rewrite.py:546-586 handle_new_array parity.
    ///
    /// kind: FLAG_ARRAY=0, FLAG_STR=1, FLAG_UNICODE=2.
    ///
    /// `descr_ref` is the ArrayDescr to use for size / length-field /
    /// per-item layout queries.  Upstream rewrite.py:489-494 passes
    /// `self.gc_ll_descr.{str,unicode}_descr` for NEWSTR/NEWUNICODE and
    /// `op.getdescr()` for NEW_ARRAY; the dispatcher in
    /// `rewrite_for_gc_with_constants` is what threads the right
    /// instance through this signature.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre still lacks the Boehm branch
    /// (`gen_boehm_malloc_array`).  Framework-GC path #4
    /// (`gen_malloc_array` / `gen_malloc_str` / `gen_malloc_unicode`)
    /// is ported below and emits CALL_R + CHECK_MEMORY_ERROR like
    /// rewrite.py:768-846.
    fn handle_new_array(&self, descr_ref: DescrRef, op: &Op, st: &mut RewriteState, kind: i64) {
        let descr = descr_ref
            .as_array_descr()
            .expect("handle_new_array descr must be ArrayDescr");

        let item_size = descr.item_size();
        let v_length = st.resolve(op.arg(0)); // the length operand
        let length_const = st.resolve_constant(v_length.raw());

        // rewrite.py:548-558 — total_size for the constant-size /
        // zero-itemsize fast path.  Stays at -1 when v_length is a
        // ConstInt that overflows `basesize + itemsize * num_elem`,
        // matching upstream's `OverflowError: pass`.
        let mut total_size: i64 = -1;
        if let Some(num_elem) = length_const {
            if num_elem >= 0 {
                if let Some(var_size) = (item_size as i64).checked_mul(num_elem) {
                    if let Some(t) = (descr.base_size() as i64).checked_add(var_size) {
                        total_size = t;
                    }
                }
            }
        } else if item_size == 0 {
            // rewrite.py:557-558 — non-const length but zero itemsize
            // means no variable payload; fold to fixed-size basesize.
            total_size = descr.base_size() as i64;
        } else if self.can_use_nursery(1) {
            // rewrite.py:559-568 path #1 — varsize nursery fast path.
            // Returns None when the descr is non-standard FLAG_ARRAY,
            // in which case we fall through to path #4
            // (`gen_malloc_array_nonstandard`) per rewrite.py:853-856.
            if let Some(r) =
                self.gen_malloc_nursery_varsize(descr_ref.clone(), kind, v_length, op.pos, st)
            {
                st.record_result_mapping(op.pos, r);
                if let Some(len_descr) = descr.len_descr() {
                    self.gen_initialize_len(r, v_length, len_descr, st);
                }
                self.clear_varsize_gc_fields(
                    kind,
                    descr_ref.clone(),
                    item_size as i64,
                    r,
                    v_length,
                    op.opcode,
                    st,
                );
                return;
            }
        }

        // rewrite.py:569-584 paths #2 / #4.
        let result = if total_size >= 0 {
            // pyre layout note: gen_malloc_nursery expects HDR + payload
            // bytes (handle_new_fixedsize line 836); upstream's basesize
            // already includes the header offset.  Add HDR_SIZE here so
            // the bump-pointer alloc covers the same span.
            let s = crate::header::GcHeader::SIZE + total_size as usize;
            if let Some(r) = self.gen_malloc_nursery(s, op.pos, st) {
                // rewrite.py:569-572 path #2 — constant-size nursery.
                st.record_result_mapping(op.pos, r);
                self.gen_initialize_tid(r, descr.type_id(), st);
                if let Some(len_descr) = descr.len_descr() {
                    self.gen_initialize_len(r, v_length, len_descr, st);
                }
                r
            } else {
                // rewrite.py:573-584 path #4 — typed slow malloc helpers.
                let r = match op.opcode {
                    OpCode::NewArray | OpCode::NewArrayClear => {
                        self.gen_malloc_array(descr_ref.clone(), v_length, op.pos, st)
                    }
                    OpCode::Newstr => self.gen_malloc_str(v_length, op.pos, st),
                    OpCode::Newunicode => self.gen_malloc_unicode(v_length, op.pos, st),
                    _ => panic!("unexpected varsize alloc opcode: {:?}", op.opcode),
                };
                st.record_result_mapping(op.pos, r);
                r
            }
        } else {
            let r = match op.opcode {
                OpCode::NewArray | OpCode::NewArrayClear => {
                    self.gen_malloc_array(descr_ref.clone(), v_length, op.pos, st)
                }
                OpCode::Newstr => self.gen_malloc_str(v_length, op.pos, st),
                OpCode::Newunicode => self.gen_malloc_unicode(v_length, op.pos, st),
                _ => panic!("unexpected varsize alloc opcode: {:?}", op.opcode),
            };
            st.record_result_mapping(op.pos, r);
            r
        };

        // rewrite.py:566-567 (path #1 inline) / rewrite.py:585-586
        // (paths #2/#3/#4 tail) clear_varsize_gc_fields.  Emits
        // ZERO_ARRAY for NEW_ARRAY_CLEAR and a hash-field zeroing
        // store for NEWSTR / NEWUNICODE, gated on !malloc_zero_filled.
        self.clear_varsize_gc_fields(
            kind,
            descr_ref.clone(),
            item_size as i64,
            result,
            v_length,
            op.opcode,
            st,
        );

        // rewrite.py:551: if isinstance(v_length, ConstInt):
        //     self.remember_known_length(op, v_length.getint())
        // Upstream calls this BEFORE total_size computation, but the key
        // is the OpRef of the alloc result, so the call is safely
        // hoisted to here without changing the semantic outcome.
        if let Some(num_elem) = length_const {
            st.remember_known_length(result, num_elem as usize);
        }
    }

    /// rewrite.py:520-535 `clear_varsize_gc_fields`.
    ///
    /// Short-circuits on `malloc_zero_filled=true` — pyre's production
    /// nursery zero-fills payload bytes, so callers already observe a
    /// zeroed array / hash field.  Under `malloc_zero_filled=false`
    /// this fans out per `kind`:
    ///   * FLAG_ARRAY + NEW_ARRAY_CLEAR → `handle_clear_array_contents`
    ///   * FLAG_STR / FLAG_UNICODE → zero the `hash` field at offset 0
    ///
    /// The upstream hash-field store comes from
    /// `gc_ll_descr.{str,unicode}_hash_descr`; rstr.STR and
    /// rstr.UNICODE both keep `hash` as the first Signed field
    /// (rstr.py:1226-1238), so a `GC_STORE(result, 0, 0, WORD)`
    /// matches the upstream `emit_setfield(result, c_zero,
    /// descr=hash_descr)` contract.
    #[allow(clippy::too_many_arguments)]
    fn clear_varsize_gc_fields(
        &self,
        kind: i64,
        arraydescr: DescrRef,
        ad_itemsize: i64,
        result: OpRef,
        v_length: OpRef,
        opnum: OpCode,
        st: &mut RewriteState,
    ) {
        if self.malloc_zero_filled {
            return;
        }
        // rewrite.py:523-528 FLAG_ARRAY path.
        if kind == 0 {
            if opnum == OpCode::NewArrayClear {
                self.handle_clear_array_contents(arraydescr, ad_itemsize, result, v_length, st);
            }
            return;
        }
        // rewrite.py:529-535 FLAG_STR / FLAG_UNICODE: zero the hash
        // field via emit_setfield(result, ConstInt(0), descr=hash_descr).
        // Offset / size come from gc_ll_descr.{str,unicode}_hash_descr
        // (gc.py:48-49) — both rstr.STR and rstr.UNICODE keep `hash` at
        // offset 0 with `Signed` size, but reading it through the
        // descr keeps the layout assumption explicit.
        if kind == 1 || kind == 2 {
            let hash_descr_ref = if kind == 1 {
                &self.str_hash_descr
            } else {
                &self.unicode_hash_descr
            };
            let hash_fd = hash_descr_ref
                .as_field_descr()
                .expect("gc_ll_descr.{str,unicode}_hash_descr must be a FieldDescr");
            let zero = st.const_int(0);
            self.emit_setfield(result, zero, hash_fd, st);
        }
    }

    /// rewrite.py:588-611 `handle_clear_array_contents`.
    ///
    /// Emits a `ZERO_ARRAY` covering the entire array, registering the
    /// op in `pending_zeros` when `v_length` is a constant so
    /// `emit_pending_zeros` can trim the range against subsequent
    /// SETARRAYITEM_GC writes (rewrite.py:610-611).
    fn handle_clear_array_contents(
        &self,
        arraydescr: DescrRef,
        ad_itemsize: i64,
        v_arr: OpRef,
        v_length: OpRef,
        st: &mut RewriteState,
    ) {
        // rewrite.py:589 assert v_length is not None.
        if v_length.is_none() {
            return;
        }
        // rewrite.py:590-591 constant zero-length short-circuit.
        let length_const = st.resolve_constant(v_length.raw());
        if matches!(length_const, Some(0)) {
            return;
        }
        // rewrite.py:600-602 — only pre-scale for non-ConstInt v_length.
        // Upstream gates the helper call on `not isinstance(v_length,
        // ConstInt)`; for ConstInt the scale stays in args[3]/args[4]
        // and `emit_pending_zeros` handles it later.
        let mut scale = ad_itemsize;
        let mut v_length_scaled = v_length;
        if length_const.is_none() {
            let (new_scale, _new_offset, scaled_opt) =
                self._emit_mul_if_factor_offset_not_supported(v_length, scale, 0, st);
            if let Some(s) = scaled_opt {
                v_length_scaled = s;
                scale = new_scale;
            }
        }
        // rewrite.py:603-609 emit ZERO_ARRAY with scale doubled into
        // args[3] and args[4] (upstream puts both to `ConstInt(scale)`;
        // emit_pending_zeros later rewrites both to 1 after byte-level
        // trim for ConstInt lengths).
        let c_zero = st.const_int(0);
        let c_scale_a = st.const_int(scale);
        let c_scale_b = st.const_int(scale);
        let mut zero_op = Op::new(
            OpCode::ZeroArray,
            &[v_arr, c_zero, v_length_scaled, c_scale_a, c_scale_b],
        );
        zero_op.descr = Some(arraydescr);
        let out_index = st.out.len();
        st.emit(zero_op);
        // rewrite.py:610-611 — register in last_zero_arrays only for
        // ConstInt lengths so emit_pending_zeros can optimize the range.
        if let Some(n) = length_const {
            st.pending_zeros.push(PendingZero {
                out_index,
                array_ref: v_arr,
                length: n as usize,
                scale,
            });
        }
    }

    // ────────────────────────────────────────────────────────
    // CALL_MALLOC_NURSERY_VARSIZE / slow malloc helpers
    // ────────────────────────────────────────────────────────

    /// rewrite.py:848-866 `gen_malloc_nursery_varsize`.
    ///
    /// Returns `Some(opref)` when the nursery fast path emits the
    /// CALL_MALLOC_NURSERY_VARSIZE op, `None` when the upstream
    /// `kind == FLAG_ARRAY` non-standard-shape guard
    /// (rewrite.py:853-856) rejects the descr — in that case the
    /// caller falls through to `gen_malloc_array` /
    /// `gen_boehm_malloc_array` like upstream's path #4.
    fn gen_malloc_nursery_varsize(
        &self,
        arraydescr: DescrRef,
        kind: i64,
        v_length: OpRef,
        result_pos: OpRef,
        st: &mut RewriteState,
    ) -> Option<OpRef> {
        let ad = arraydescr
            .as_array_descr()
            .expect("gen_malloc_nursery_varsize descr must be ArrayDescr");
        // rewrite.py:853-856 — standard-shape gate: only `FLAG_ARRAY`
        // (kind == 0) is constrained; `FLAG_STR` (1) / `FLAG_UNICODE`
        // (2) always proceed because their descrs are by definition
        // non-standard-shaped but the nursery layout still accepts
        // them.
        const FLAG_ARRAY: i64 = 0;
        if kind == FLAG_ARRAY {
            let length_ofs = ad
                .len_descr()
                .map_or(self.standard_array_length_ofs, |fd| fd.offset());
            if ad.base_size() != self.standard_array_basesize
                || length_ofs != self.standard_array_length_ofs
            {
                return None;
            }
        }
        st.emitting_an_operation_that_can_collect();
        let kind_ref = st.const_int(kind);
        let itemsize_ref = st.const_int(ad.item_size() as i64);
        let mut varsize_op = Op::new(
            OpCode::CallMallocNurseryVarsize,
            &[kind_ref, itemsize_ref, v_length],
        );
        varsize_op.descr = Some(arraydescr);
        Some(st.emit_result(varsize_op, result_pos))
    }

    /// rewrite.py:768-776 `_gen_call_malloc_gc`.
    fn gen_call_malloc_gc(
        &self,
        args: &[OpRef],
        result_pos: OpRef,
        calldescr: DescrRef,
        st: &mut RewriteState,
    ) -> OpRef {
        st.emitting_an_operation_that_can_collect();
        let mut call_op = Op::new(OpCode::CallR, args);
        call_op.descr = Some(calldescr);
        let result = st.emit_result(call_op, result_pos);
        st.emit(Op::new(OpCode::CheckMemoryError, &[result]));
        result
    }

    /// rewrite.py:809-834 `gen_malloc_array`.
    fn gen_malloc_array(
        &self,
        arraydescr: DescrRef,
        v_num_elem: OpRef,
        result_pos: OpRef,
        st: &mut RewriteState,
    ) -> OpRef {
        let ad = arraydescr
            .as_array_descr()
            .expect("gen_malloc_array descr must be ArrayDescr");
        let len_descr = ad.len_descr();
        let length_ofs = len_descr.map_or(self.standard_array_length_ofs, |fd| fd.offset());
        let is_standard = ad.base_size() == self.standard_array_basesize
            && len_descr.is_some_and(|fd| fd.offset() == self.standard_array_length_ofs);
        if is_standard {
            let fn_ref = st.const_int(self.malloc_array_fn);
            let itemsize_ref = st.const_int(ad.item_size() as i64);
            let typeid_ref = st.const_int(ad.type_id() as i64);
            self.gen_call_malloc_gc(
                &[fn_ref, itemsize_ref, typeid_ref, v_num_elem],
                result_pos,
                self.malloc_array_descr.clone(),
                st,
            )
        } else {
            let fn_ref = st.const_int(self.malloc_array_nonstandard_fn);
            let basesize_ref = st.const_int(ad.base_size() as i64);
            let itemsize_ref = st.const_int(ad.item_size() as i64);
            let lengthofs_ref = st.const_int(length_ofs as i64);
            let typeid_ref = st.const_int(ad.type_id() as i64);
            self.gen_call_malloc_gc(
                &[
                    fn_ref,
                    basesize_ref,
                    itemsize_ref,
                    lengthofs_ref,
                    typeid_ref,
                    v_num_elem,
                ],
                result_pos,
                self.malloc_array_nonstandard_descr.clone(),
                st,
            )
        }
    }

    /// rewrite.py:778-796 `gen_malloc_fixedsize` (framework GC arm).
    ///
    /// Emits `CALL_R(malloc_big_fixedsize_fn, size, typeid)` followed
    /// by `CHECK_MEMORY_ERROR` (via `gen_call_malloc_gc`) and stamps
    /// `remember_wb` on the result so the freshly-malloc'd object
    /// skips the write barrier (rewrite.py:794-796 — fixed-size
    /// objects are zero-initialized by `clear_gc_fields`, so any
    /// store into a gc field needs no WB).
    ///
    /// pyre is framework-GC only; the Boehm `else` arm
    /// (`malloc_fixedsize_fn`) is intentionally not ported.  The
    /// helper itself stamps the tid into the GC header (matching
    /// upstream `malloc_big_fixedsize` at gc.py:481-490 which goes
    /// through `do_malloc_fixedsize_clear` with type_id), so callers
    /// MUST NOT emit a separate `gen_initialize_tid`.
    fn gen_malloc_fixedsize(
        &self,
        size: usize,
        type_id: u32,
        result_pos: OpRef,
        st: &mut RewriteState,
    ) -> OpRef {
        debug_assert_eq!(
            size & (std::mem::size_of::<usize>() - 1),
            0,
            "rewrite.py:785 `assert (size & (WORD-1)) == 0` — size must be word-aligned"
        );
        let fn_ref = st.const_int(self.malloc_big_fixedsize_fn);
        let size_ref = st.const_int(size as i64);
        let typeid_ref = st.const_int(type_id as i64);
        let result = self.gen_call_malloc_gc(
            &[fn_ref, size_ref, typeid_ref],
            result_pos,
            self.malloc_big_fixedsize_descr.clone(),
            st,
        );
        st.remember_wb(result);
        result
    }

    /// rewrite.py:836-840 `gen_malloc_str`.
    ///
    /// PRE-EXISTING-ADAPTATION: upstream's `malloc_str` helper closure
    /// captures `str_type_id = self.str_descr.tid` (gc.py:451) at
    /// generate-time.  Rust `extern "C" fn` can't lexically capture, so
    /// the type id is threaded as an explicit CALL arg sourced from
    /// `gc_ll_descr.str_descr.type_id()`.  Matches the
    /// `malloc_str_calldescr` Signed/Signed signature documented there.
    fn gen_malloc_str(&self, v_num_elem: OpRef, result_pos: OpRef, st: &mut RewriteState) -> OpRef {
        let fn_ref = st.const_int(self.malloc_str_fn);
        let type_id = self
            .str_descr
            .as_array_descr()
            .expect("gc_ll_descr.str_descr must be an ArrayDescr")
            .type_id();
        let type_id_ref = st.const_int(type_id as i64);
        self.gen_call_malloc_gc(
            &[fn_ref, type_id_ref, v_num_elem],
            result_pos,
            self.malloc_str_descr.clone(),
            st,
        )
    }

    /// rewrite.py:842-846 `gen_malloc_unicode`.
    ///
    /// PRE-EXISTING-ADAPTATION: see `gen_malloc_str`.  Type id sourced
    /// from `gc_ll_descr.unicode_descr.type_id()` (gc.py:455
    /// `unicode_type_id = self.unicode_descr.tid`).
    fn gen_malloc_unicode(
        &self,
        v_num_elem: OpRef,
        result_pos: OpRef,
        st: &mut RewriteState,
    ) -> OpRef {
        let fn_ref = st.const_int(self.malloc_unicode_fn);
        let type_id = self
            .unicode_descr
            .as_array_descr()
            .expect("gc_ll_descr.unicode_descr must be an ArrayDescr")
            .type_id();
        let type_id_ref = st.const_int(type_id as i64);
        self.gen_call_malloc_gc(
            &[fn_ref, type_id_ref, v_num_elem],
            result_pos,
            self.malloc_unicode_descr.clone(),
            st,
        )
    }

    // ────────────────────────────────────────────────────────
    // COPYSTRCONTENT / COPYUNICODECONTENT → memcpy CALL_N
    // ────────────────────────────────────────────────────────

    /// rewrite.py:1045-1080 `rewrite_copy_str_content`.
    ///
    /// Lowers `COPYSTRCONTENT(src, dst, src_start, dst_start, length)` (and
    /// the UNICODE variant) to:
    ///
    /// ```text
    /// i1 = LOAD_EFFECTIVE_ADDRESS(src_gcptr, src_start, basesize, shift)
    /// i2 = LOAD_EFFECTIVE_ADDRESS(dst_gcptr, dst_start, basesize, shift)
    /// CALL_N(memcpy_fn, i2, i1, count, descr=memcpy_descr)
    /// ```
    ///
    /// For UNICODE, `count` is `length << shift` (byte count); for STR the
    /// basesize is additionally offset by `-1` to skip the STR
    /// `extra_item_after_alloc` null terminator (rewrite.py:1051-1053;
    /// rstr.py:1226 `extra_item_after_alloc=True`).
    fn rewrite_copy_str_content(&self, op: &Op, st: &mut RewriteState) {
        // rewrite.py:1046-1048 — pull memcpy_fn / memcpy_descr off the
        // gc_ll_descr instance fields (gc.py:39-43).
        let memcpy_fn = self.memcpy_fn;
        let memcpy_descr = self.memcpy_descr.clone();

        // rewrite.py:1049-1064 — basesize/itemscale come from the
        // canonical str_descr / unicode_descr held on gc_ll_descr
        // (gc.py:46-47), NOT from the op itself; upstream's
        // COPY{STR,UNICODE}CONTENT carries no arraydescr.
        let (mut basesize, itemscale) = if op.opcode == OpCode::Copystrcontent {
            let ad = self
                .str_descr
                .as_array_descr()
                .expect("gc_ll_descr.str_descr must be an ArrayDescr");
            // rewrite.py:1054 `assert self.gc_ll_descr.str_descr.itemsize == 1`.
            assert_eq!(
                ad.item_size(),
                1,
                "rewrite.py:1054 str_descr.itemsize must be 1"
            );
            (ad.base_size() as i64, 0i64)
        } else {
            let ad = self
                .unicode_descr
                .as_array_descr()
                .expect("gc_ll_descr.unicode_descr must be an ArrayDescr");
            let itemsize = ad.item_size() as i64;
            // rewrite.py:1059-1063 — itemscale = log2(itemsize) for 2/4.
            let itemscale = match itemsize {
                2 => 1,
                4 => 2,
                _ => {
                    panic!("rewrite.py:1064 unknown unicode itemsize {itemsize} — expected 2 or 4")
                }
            };
            (ad.base_size() as i64, itemscale)
        };
        if op.opcode == OpCode::Copystrcontent {
            // rewrite.py:1051-1053 — one extra item after the string buffer
            // (rstr.py:1226 `extra_item_after_alloc=True`), so the `chars`
            // array starts at `str_descr.basesize - 1`.
            basesize -= 1;
        }

        // rewrite.py:1065-1068 — effective source / destination addresses.
        let src_gcptr = st.resolve(op.arg(0));
        let dst_gcptr = st.resolve(op.arg(1));
        let src_index = st.resolve(op.arg(2));
        let dst_index = st.resolve(op.arg(3));

        let i1 = self.emit_load_effective_address(src_gcptr, src_index, basesize, itemscale, st);
        let i2 = self.emit_load_effective_address(dst_gcptr, dst_index, basesize, itemscale, st);

        // rewrite.py:1069-1078 — byte count.
        //   STR:     arg = op.getarg(4)                         (itemscale=0)
        //   UNICODE: arg = ConstInt(op.getarg(4).getint() << itemscale)
        //            or INT_LSHIFT(op.getarg(4), ConstInt(itemscale))
        let arg = if op.opcode == OpCode::Copystrcontent {
            st.resolve(op.arg(4))
        } else {
            let v_length = st.resolve(op.arg(4));
            if let Some(c) = st.resolve_constant(v_length.raw()) {
                // rewrite.py:1073-1074 — constant-fold the shift.
                st.const_int(c << itemscale)
            } else {
                // rewrite.py:1075-1078 — emit INT_LSHIFT.
                let shift_ref = st.const_int(itemscale);
                let lshift = Op::new(OpCode::IntLshift, &[v_length, shift_ref]);
                st.emit_result(lshift, OpRef::NONE)
            }
        };

        // rewrite.py:1079-1080 — CALL_N(memcpy_fn, i2, i1, arg, descr=memcpy_descr).
        let memcpy_fn_const = st.const_int(memcpy_fn);
        let mut call_op = Op::new(OpCode::CallN, &[memcpy_fn_const, i2, i1, arg]);
        call_op.descr = Some(memcpy_descr);
        st.emit(call_op);
    }

    /// rewrite.py:1082-1098 `emit_load_effective_address`.
    ///
    /// CPUs with `supports_load_effective_address = True` (x86, aarch64)
    /// emit a single LEA op; CPUs without it (model.py:22 base default)
    /// expand to `(INT_LSHIFT? + INT_ADD + INT_ADD)`.  The LEA arg order
    /// is `[v_gcptr, v_index, c_baseofs, c_shift]` per
    /// resoperation.py:1052-1054.
    fn emit_load_effective_address(
        &self,
        v_gcptr: OpRef,
        v_index: OpRef,
        base: i64,
        itemscale: i64,
        st: &mut RewriteState,
    ) -> OpRef {
        if self.supports_load_effective_address {
            // rewrite.py:1083-1088 — single LEA op.
            let base_ref = st.const_int(base);
            let shift_ref = st.const_int(itemscale);
            let lea = Op::new(
                OpCode::LoadEffectiveAddress,
                &[v_gcptr, v_index, base_ref, shift_ref],
            );
            st.emit_result(lea, OpRef::NONE)
        } else {
            // rewrite.py:1089-1098 — fallback expansion.
            //   if itemscale > 0:
            //       v_index = INT_LSHIFT(v_index, ConstInt(itemscale))
            //   i1b = INT_ADD(v_gcptr, v_index)
            //   i1  = INT_ADD(i1b, ConstInt(base))
            let scaled = if itemscale > 0 {
                let shift_ref = st.const_int(itemscale);
                let lshift = Op::new(OpCode::IntLshift, &[v_index, shift_ref]);
                st.emit_result(lshift, OpRef::NONE)
            } else {
                v_index
            };
            let add1 = Op::new(OpCode::IntAdd, &[v_gcptr, scaled]);
            let i1b = st.emit_result(add1, OpRef::NONE);
            let base_ref = st.const_int(base);
            let add2 = Op::new(OpCode::IntAdd, &[i1b, base_ref]);
            st.emit_result(add2, OpRef::NONE)
        }
    }

    // ────────────────────────────────────────────────────────
    // SETFIELD_GC  → maybe COND_CALL_GC_WB + SETFIELD_GC
    // ────────────────────────────────────────────────────────

    /// rewrite.py:926-934 `handle_write_barrier_setfield`.
    /// Emits a write barrier before the store when the stored value is a
    /// non-null reference into a pointer-bearing field AND the base has
    /// not already been WB'd.  Does *not* emit the store itself — the
    /// caller is expected to follow up with `emit_maybe_forwarded` so
    /// the lowered GC_STORE (forwarded by `transform_to_gc_load`) lands
    /// after the WB.
    fn handle_write_barrier_setfield(&self, op: &Op, st: &mut RewriteState) {
        let obj = st.resolve(op.arg(0));
        if st.wb_already_applied(obj) {
            return;
        }
        // rewrite.py:930-931: check the stored VALUE's type.
        //   v = op.getarg(1)
        //   if (v.type == 'r' and (not isinstance(v, ConstPtr) or
        //       rgc.needs_write_barrier(v.value))):
        //
        // Gate on field descriptor: if the field is not a pointer field,
        // the GC won't trace it, so no WB is needed regardless of value
        // type.  In RPython val.type=='r' implies the field is GCREF;
        // here ForceToken (Ref) stores to an Int-typed field (offset 128),
        // a pyre-specific divergence.
        let field_is_ptr = op
            .descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .map(|fd| fd.is_pointer_field())
            .unwrap_or(false);
        let val = st.resolve(op.arg(1));
        let val_is_ref = if field_is_ptr {
            match st.result_type_of(val) {
                Some(tp) => tp == Type::Ref,
                None => true, // field is ptr → assume value is Ref
            }
        } else {
            false
        };
        if !val_is_ref || st.is_null_constant(val) {
            return;
        }
        self.gen_write_barrier(obj, st);
    }

    /// rewrite.py:948-953 `gen_write_barrier`.
    fn gen_write_barrier(&self, v_base: OpRef, st: &mut RewriteState) {
        let wb_op = Op::new(OpCode::CondCallGcWb, &[v_base]);
        st.emit(wb_op);
        st.remember_wb(v_base);
    }

    /// rewrite.py:506-512 `consider_setfield_gc`.
    ///
    /// Drops the `(base, offset)` entry from `_delayed_zero_setfields`
    /// so the pending-zero flush at `emit_pending_zeros`
    /// (rewrite.py:761-766) does not re-zero a slot that this explicit
    /// SETFIELD_GC is about to overwrite.
    ///
    /// Under pyre's default zero-fill nursery configuration
    /// (`malloc_zero_filled = true`), `clear_gc_fields` skips its
    /// insertion path, so this is effectively a no-op.  The body is
    /// wired for parity so that a non-zero-fill allocator automatically
    /// activates the delayed-zero tracking without further callsite
    /// changes.
    fn consider_setfield_gc(&self, op: &Op, st: &mut RewriteState) {
        let Some(fd) = op.descr.as_ref().and_then(|d| d.as_field_descr()) else {
            return;
        };
        let offset = fd.offset() as i64;
        let base = st.resolve(op.arg(0));
        if let Some(entries) = st._delayed_zero_setfields.get_mut(&base) {
            entries.remove(&offset);
        }
    }

    // ────────────────────────────────────────────────────────
    // SETARRAYITEM_GC  → maybe COND_CALL_GC_WB{_ARRAY} + SETARRAYITEM_GC
    // rewrite.py:936-946 handle_write_barrier_setarrayitem
    // ────────────────────────────────────────────────────────

    /// rewrite.py:514-518 consider_setarrayitem_gc: record the constant
    /// index so emit_pending_zeros can skip this slot.
    ///
    /// ```text
    /// if not isinstance(array_box, ConstPtr) and index_box.is_constant():
    ///     self.remember_setarrayitem_occurred(array_box, index_box.getint())
    /// ```
    fn consider_setarrayitem_gc(&self, op: &Op, st: &mut RewriteState) {
        let array_ref = st.resolve(op.arg(0));
        let index_ref = op.arg(1);
        if st.resolve_constant(array_ref.raw()).is_some() {
            return;
        }
        let Some(idx_val) = st.resolve_constant(index_ref.raw()) else {
            return;
        };
        st.record_setarrayitem_index(array_ref, idx_val as usize);
    }

    /// rewrite.py:936-944: handle_write_barrier_setarrayitem.
    /// Emits CondCallGcWb / CondCallGcWbArray as needed; the SETARRAYITEM
    /// op itself is NOT emitted here — RPython forwards the op to
    /// GC_STORE_INDEXED inside transform_to_gc_load (rewrite.py:220-221)
    /// and then `self.emit_op(op)` follows the forwarding. We do the
    /// equivalent in the caller by invoking handle_setarrayitem after WB.
    fn handle_write_barrier_setarrayitem(&self, op: &Op, st: &mut RewriteState) {
        let val = st.resolve(op.arg(0));
        // rewrite.py:938-942
        if !st.wb_already_applied(val) {
            let v = st.resolve(op.arg(2));
            let val_is_ref = match st.result_type_of(v) {
                Some(tp) => tp == Type::Ref,
                None => op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .map(|ad| ad.is_array_of_pointers())
                    .unwrap_or(false),
            };
            if val_is_ref && !st.is_null_constant(v) {
                self.gen_write_barrier_array(val, st.resolve(op.arg(1)), st);
            }
        }
    }

    /// rewrite.py:132-138 handle_setarrayitem.
    /// Lowers SETARRAYITEM_GC / SETARRAYITEM_RAW into GC_STORE /
    /// GC_STORE_INDEXED via `emit_gc_store_or_indexed`, which forwards
    /// the original op to the lowered form (the emission happens later
    /// in the main loop via `emit_maybe_forwarded` for RAW and via the
    /// SETARRAYITEM_GC write-barrier arm for GC).
    fn handle_setarrayitem(&self, op: &Op, st: &mut RewriteState) {
        let descr = op.descr.as_ref().expect("SETARRAYITEM needs ArrayDescr");
        let ad = descr
            .as_array_descr()
            .expect("SETARRAYITEM descr must be ArrayDescr");
        let itemsize = ad.item_size() as i64;
        let basesize = ad.base_size() as i64;
        let ptr = st.resolve(op.arg(0));
        let index = st.resolve(op.arg(1));
        let value = st.resolve(op.arg(2));
        self.emit_gc_store_or_indexed(
            Some(op),
            ptr,
            index,
            value,
            itemsize,
            itemsize,
            basesize,
            st,
        );
    }

    /// rewrite.py:140-158 emit_gc_store_or_indexed (with cpu_simplify_scale
    /// inlined). `load_supported_factors` drives the non-constant branch:
    /// factors outside that set are pre-scaled in IR, factors inside it pass
    /// through to the backend's native addressing mode.
    ///
    /// When `original` is `Some`, the lowered GC_STORE / GC_STORE_INDEXED is
    /// *forwarded* as the replacement for the original op (upstream's
    /// `replace_op_with`); the main loop emits the forwarded op at the
    /// appropriate point in the output stream (after any write barrier).
    /// When `None`, the lowered op is emitted directly — used for
    /// internal stores synthesised by the rewriter that do not replace an
    /// input op (e.g. tid initialisation for fresh allocations).
    fn emit_gc_store_or_indexed(
        &self,
        original: Option<&Op>,
        ptr: OpRef,
        index: OpRef,
        value: OpRef,
        itemsize: i64,
        factor: i64,
        offset: i64,
        st: &mut RewriteState,
    ) {
        // rewrite.py:142-143
        let (index, offset) = st._try_use_older_box(index, factor, offset);

        // rewrite.py:144-146
        let (factor, offset, index_opt) =
            self._emit_mul_if_factor_offset_not_supported(index, factor, offset, st);

        // rewrite.py:148-154 — GC_STORE vs GC_STORE_INDEXED.
        let newload = match index_opt {
            None => {
                let offset_ref = st.const_int(offset);
                let itemsize_ref = st.const_int(itemsize);
                Op::new(OpCode::GcStore, &[ptr, offset_ref, value, itemsize_ref])
            }
            Some(idx) => {
                let factor_ref = st.const_int(factor);
                let offset_ref = st.const_int(offset);
                let itemsize_ref = st.const_int(itemsize);
                Op::new(
                    OpCode::GcStoreIndexed,
                    &[ptr, idx, value, factor_ref, offset_ref, itemsize_ref],
                )
            }
        };

        // rewrite.py:155-158
        if original.is_some() {
            st.set_forwarded(newload);
        } else {
            st.emit(newload);
        }
    }

    /// rewrite.py:166-171 `_emit_mul_if_factor_offset_not_supported`.
    ///
    /// Wrapper around `cpu_simplify_scale` that emits the constructed
    /// pre-scale op (when one was needed) and folds the result into the
    /// `(factor, offset, index)` triple consumed by the indexed GC
    /// load/store builders.  `Option<OpRef>` mirrors upstream's
    /// `index_box is None` ConstInt sentinel — `None` means the index
    /// folded into `offset` and the caller should emit the non-indexed
    /// GC_LOAD / GC_STORE form.
    fn _emit_mul_if_factor_offset_not_supported(
        &self,
        index_box: OpRef,
        factor: i64,
        offset: i64,
        st: &mut RewriteState,
    ) -> (i64, i64, Option<OpRef>) {
        let (factor, offset, scaled) = self.cpu_simplify_scale(index_box, factor, offset, st);
        let index_opt = match scaled {
            ScaledIndex::Const => None,
            ScaledIndex::Passthrough(idx) => Some(idx),
            // rewrite.py:169-170 — the pre-scale op is emitted here.
            ScaledIndex::PreScale(op) => Some(st.emit_result(op, OpRef::NONE)),
        };
        (factor, offset, index_opt)
    }

    /// rewrite.py:1117-1132 `cpu_simplify_scale`.
    ///
    /// Pure decision function: given the raw `(index, factor, offset)`
    /// triple supplied by an indexed GC load/store builder, decide
    /// whether to (a) fold a ConstInt index into `offset`, (b) pass the
    /// triple through unchanged when the CPU's addressing mode supports
    /// `factor`, or (c) construct (but not emit) an `INT_LSHIFT` /
    /// `INT_MUL` pre-scale op.  The returned `ScaledIndex` carries the
    /// constructed pre-scale op so `_emit_mul_if_factor_offset_not_
    /// supported` can emit it; vector_ext.py callers (rpython/jit/
    /// backend/llsupport/vector_ext.py:127, :157) emit it themselves.
    fn cpu_simplify_scale(
        &self,
        index_box: OpRef,
        factor: i64,
        offset: i64,
        st: &mut RewriteState,
    ) -> (i64, i64, ScaledIndex) {
        // rewrite.py:1118-1122 — ConstInt path.
        if let Some(index_val) = st.resolve_constant(index_box.raw()) {
            return (1, index_val * factor + offset, ScaledIndex::Const);
        }
        // rewrite.py:1124-1133 — non-constant path.
        if factor != 1 && !self.load_supported_factors.contains(&factor) {
            assert!(factor > 0, "cpu_simplify_scale: factor must be positive");
            // rewrite.py:1127-1131 — power-of-two factor → INT_LSHIFT,
            // else INT_MUL.
            let mul_op = if (factor & (factor - 1)) == 0 {
                let shift = (factor as u64).trailing_zeros() as i64;
                let shift_ref = st.const_int(shift);
                Op::new(OpCode::IntLshift, &[index_box, shift_ref])
            } else {
                let factor_ref = st.const_int(factor);
                Op::new(OpCode::IntMul, &[index_box, factor_ref])
            };
            return (1, offset, ScaledIndex::PreScale(mul_op));
        }
        // rewrite.py:1132 — pass-through.
        (factor, offset, ScaledIndex::Passthrough(index_box))
    }

    /// rewrite.py:160-164 handle_getarrayitem.
    /// Lowers GETARRAYITEM_{GC,RAW}_{I,R,F} (including the PURE variants,
    /// per rewrite.py:216-219) into GC_LOAD / GC_LOAD_INDEXED by
    /// forwarding the op through `emit_gc_load_or_indexed`.
    fn handle_getarrayitem(&self, op: &Op, st: &mut RewriteState) {
        let descr = op.descr.as_ref().expect("GETARRAYITEM needs ArrayDescr");
        let ad = descr
            .as_array_descr()
            .expect("GETARRAYITEM descr must be ArrayDescr");
        let itemsize = ad.item_size() as i64;
        let ofs = ad.base_size() as i64;
        let sign = ad.is_item_signed();
        let ptr = st.resolve(op.arg(0));
        let index = st.resolve(op.arg(1));
        self.emit_gc_load_or_indexed(op, ptr, index, itemsize, itemsize, ofs, sign, st);
    }

    /// rewrite.py:184-210 emit_gc_load_or_indexed (with cpu_simplify_scale
    /// inlined). Forwards `original` to either GC_LOAD_{I,R,F} (when the
    /// index resolves to a constant) or GC_LOAD_INDEXED_{I,R,F}.
    ///
    /// The caller is expected to supply the already-resolved
    /// `ptr` / `index` args and the raw (itemsize, factor, offset, sign)
    /// tuple from `unpack_arraydescr` / `unpack_fielddescr` /
    /// `unpack_interiorfielddescr` or from `get_array_token` /
    /// `get_field_token` for the string and unicode helpers.
    ///
    /// `sign` is encoded into the emitted `itemsize` arg by negating it
    /// (rewrite.py:192-194) — the backend decodes the sign back out of
    /// the sign bit on the nsize operand.
    fn emit_gc_load_or_indexed(
        &self,
        original: &Op,
        ptr: OpRef,
        index: OpRef,
        itemsize: i64,
        factor: i64,
        offset: i64,
        sign: bool,
        st: &mut RewriteState,
    ) {
        // rewrite.py:186-187
        let (index, offset) = st._try_use_older_box(index, factor, offset);

        // rewrite.py:188-190
        let (factor, offset, index_opt) =
            self._emit_mul_if_factor_offset_not_supported(index, factor, offset, st);

        // rewrite.py:192-194 — encode signed-ness into the itemsize value.
        let itemsize_enc = if sign { -itemsize } else { itemsize };

        // rewrite.py:196-198 — optype from op.type (result-kind of the
        // original load op determines the GC_LOAD_I / R / F variant).
        let optype = original.opcode.result_type();

        // rewrite.py:199-205 — GC_LOAD vs GC_LOAD_INDEXED.
        let newload = match index_opt {
            None => {
                let offset_ref = st.const_int(offset);
                let itemsize_ref = st.const_int(itemsize_enc);
                Op::new(get_gc_load(optype), &[ptr, offset_ref, itemsize_ref])
            }
            Some(idx) => {
                let factor_ref = st.const_int(factor);
                let offset_ref = st.const_int(offset);
                let itemsize_ref = st.const_int(itemsize_enc);
                Op::new(
                    get_gc_load_indexed(optype),
                    &[ptr, idx, factor_ref, offset_ref, itemsize_ref],
                )
            }
        };

        // rewrite.py:206-209 — pyre callers always pass `op`, so we
        // always replace_op_with (set_forwarded).
        st.set_forwarded(newload);
    }

    /// rewrite.py:660-663 `emit_setfield`.
    ///
    /// Synthetic field store helper: emits `GC_STORE(ptr, ConstInt(0),
    /// value, ConstInt(size))` via `emit_gc_store_or_indexed` (which
    /// folds the ConstInt(0)*1 + ofs into a non-indexed GC_STORE per
    /// `cpu_simplify_scale`'s ConstInt branch).  `op=None` so the
    /// lowered op is emitted directly rather than forwarded.
    fn emit_setfield(&self, ptr: OpRef, value: OpRef, fd: &dyn FieldDescr, st: &mut RewriteState) {
        self.emit_setfield_raw(ptr, value, fd.offset() as i64, fd.field_size() as i64, st);
    }

    /// Raw-offset variant of `emit_setfield` (Rust adaptation).
    ///
    /// pyre's `JitFrameDescrs` carries raw `i32` offsets and
    /// `sign_size: usize` rather than upstream's per-field `FieldDescr`
    /// objects (see rewrite.py:641-650 `emit_setfield(frame, c_null,
    /// descr=descrs.jf_*)`).  `handle_call_assembler` consumes those
    /// raw offsets directly; this helper deduplicates the
    /// `Op::new(GcStore, ...) + st.emit` boilerplate so the lowering
    /// still passes through `emit_gc_store_or_indexed` and
    /// `cpu_simplify_scale` like every other field store.  Convergence
    /// path: replace `JitFrameDescrs::jf_*_ofs` with `SimpleFieldDescr`
    /// instances, then have these callers use `emit_setfield(fd)`
    /// directly.
    fn emit_setfield_raw(
        &self,
        ptr: OpRef,
        value: OpRef,
        ofs: i64,
        size: i64,
        st: &mut RewriteState,
    ) {
        let zero = st.const_int(0);
        self.emit_gc_store_or_indexed(None, ptr, zero, value, size, 1, ofs, st);
    }

    /// rewrite.py:212-342 `transform_to_gc_load`.
    ///
    /// Central dispatcher that lowers high-level memory accessors to
    /// GC_LOAD / GC_LOAD_INDEXED / GC_STORE / GC_STORE_INDEXED. Each arm
    /// matches its upstream counterpart line-by-line; the emission uses
    /// `emit_gc_load_or_indexed` / `emit_gc_store_or_indexed`, which
    /// either forward the op (when `original` is `Some`) or emit directly.
    ///
    /// Returns `true` only for the `GETFIELD_GC_*` fast-path at
    /// rewrite.py:259-260, which flushes pending zeros, forwards, and
    /// emits the forwarded op itself — the caller (`rewrite`) then
    /// skips the rest of the main-loop body.  All other arms forward
    /// the op and return `false`, delegating emission to the main loop
    /// via `emit_maybe_forwarded` (or the write-barrier arms).
    fn transform_to_gc_load(&self, op: &Op, st: &mut RewriteState) -> bool {
        const NOT_SIGNED: bool = false;
        let opnum = op.opcode;

        // rewrite.py:216-218 `rop.is_getarrayitem(opnum) or opnum in
        // (GETARRAYITEM_RAW_I, GETARRAYITEM_RAW_F)`.  Upstream omits
        // GETARRAYITEM_RAW_R because codewriter rejects raw ref array
        // reads at `rpython/jit/codewriter/jtransform.py:775`
        // (`getarrayitem_raw_r not supported`); mirror that omission so
        // a rogue GETARRAYITEM_RAW_R does not get silently lowered into
        // GC_LOAD_INDEXED_R here.
        if opnum.is_getarrayitem()
            || matches!(opnum, OpCode::GetarrayitemRawI | OpCode::GetarrayitemRawF)
        {
            self.handle_getarrayitem(op, st);
            return false;
        }
        // rewrite.py:220-221
        if matches!(opnum, OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw) {
            self.handle_setarrayitem(op, st);
            return false;
        }
        // rewrite.py:222-227 RAW_STORE
        if matches!(opnum, OpCode::RawStore) {
            let descr = op.descr.as_ref().expect("RAW_STORE needs ArrayDescr");
            let ad = descr
                .as_array_descr()
                .expect("RAW_STORE descr must be ArrayDescr");
            let itemsize = ad.item_size() as i64;
            let ofs = ad.base_size() as i64;
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            let value = st.resolve(op.arg(2));
            self.emit_gc_store_or_indexed(Some(op), ptr, index, value, itemsize, 1, ofs, st);
            return false;
        }
        // rewrite.py:228-232 RAW_LOAD_{I,F}
        if matches!(opnum, OpCode::RawLoadI | OpCode::RawLoadF) {
            let descr = op.descr.as_ref().expect("RAW_LOAD needs ArrayDescr");
            let ad = descr
                .as_array_descr()
                .expect("RAW_LOAD descr must be ArrayDescr");
            let itemsize = ad.item_size() as i64;
            let ofs = ad.base_size() as i64;
            let sign = ad.is_item_signed();
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            self.emit_gc_load_or_indexed(op, ptr, index, itemsize, 1, ofs, sign, st);
            return false;
        }
        // rewrite.py:233-238 GETINTERIORFIELD_GC_{I,R,F}
        if matches!(
            opnum,
            OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR | OpCode::GetinteriorfieldGcF
        ) {
            let descr = op
                .descr
                .as_ref()
                .expect("GETINTERIORFIELD needs InteriorFieldDescr");
            let ifd = descr
                .as_interior_field_descr()
                .expect("GETINTERIORFIELD descr must be InteriorFieldDescr");
            let ad = ifd.array_descr();
            let fd = ifd.field_descr();
            let ofs = (ad.base_size() + fd.offset()) as i64;
            let itemsize = ad.item_size() as i64;
            let fieldsize = fd.field_size() as i64;
            let sign = fd.is_field_signed();
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            self.emit_gc_load_or_indexed(op, ptr, index, fieldsize, itemsize, ofs, sign, st);
            return false;
        }
        // rewrite.py:239-245 SETINTERIORFIELD_{RAW,GC}
        if matches!(
            opnum,
            OpCode::SetinteriorfieldRaw | OpCode::SetinteriorfieldGc
        ) {
            let descr = op
                .descr
                .as_ref()
                .expect("SETINTERIORFIELD needs InteriorFieldDescr");
            let ifd = descr
                .as_interior_field_descr()
                .expect("SETINTERIORFIELD descr must be InteriorFieldDescr");
            let ad = ifd.array_descr();
            let fd = ifd.field_descr();
            let ofs = (ad.base_size() + fd.offset()) as i64;
            let itemsize = ad.item_size() as i64;
            let fieldsize = fd.field_size() as i64;
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            let value = st.resolve(op.arg(2));
            self.emit_gc_store_or_indexed(
                Some(op),
                ptr,
                index,
                value,
                fieldsize,
                itemsize,
                ofs,
                st,
            );
            return false;
        }
        // rewrite.py:246-247 GETFIELD_{GC,RAW}_{I,R,F}.
        // Upstream excludes GETFIELD_GC_PURE_{I,R,F}: the pure variants
        // are `is_always_pure` at `resoperation.rs:1228` and must retain
        // their pure-op identity; lowering them to GC_LOAD_* would drop
        // purity and let the optimizer CSE-fold them differently from
        // their upstream siblings.  So the pure arm is intentionally
        // not handled here and falls through to the main loop's default
        // arm (which emits the op unchanged).
        if matches!(
            opnum,
            OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
        ) {
            let descr = op.descr.as_ref().expect("GETFIELD needs FieldDescr");
            let fd = descr
                .as_field_descr()
                .expect("GETFIELD descr must be FieldDescr");
            let ofs = fd.offset() as i64;
            let itemsize = fd.field_size() as i64;
            let sign = fd.is_field_signed();
            let ptr = st.resolve(op.arg(0));
            let cint_zero = st.const_int(0);
            let is_gc = matches!(
                opnum,
                OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF,
            );
            if is_gc {
                // rewrite.py:250-260 — flush pending zeros, forward, and
                // emit the forwarded op *here* so that the main loop
                // short-circuits (return True).
                st.emit_pending_zeros();
                self.emit_gc_load_or_indexed(op, ptr, cint_zero, itemsize, 1, ofs, sign, st);
                st.emit_maybe_forwarded(op);
                return true;
            }
            self.emit_gc_load_or_indexed(op, ptr, cint_zero, itemsize, 1, ofs, sign, st);
            return false;
        }
        // rewrite.py:262-266 SETFIELD_{GC,RAW}
        if matches!(opnum, OpCode::SetfieldGc | OpCode::SetfieldRaw) {
            let descr = op.descr.as_ref().expect("SETFIELD needs FieldDescr");
            let fd = descr
                .as_field_descr()
                .expect("SETFIELD descr must be FieldDescr");
            let ofs = fd.offset() as i64;
            let itemsize = fd.field_size() as i64;
            let ptr = st.resolve(op.arg(0));
            let value = st.resolve(op.arg(1));
            let cint_zero = st.const_int(0);
            self.emit_gc_store_or_indexed(Some(op), ptr, cint_zero, value, itemsize, 1, ofs, st);
            return false;
        }
        // rewrite.py:267-272 ARRAYLEN_GC
        if matches!(opnum, OpCode::ArraylenGc) {
            let descr = op.descr.as_ref().expect("ARRAYLEN_GC needs ArrayDescr");
            let ad = descr
                .as_array_descr()
                .expect("ARRAYLEN_GC descr must be ArrayDescr");
            let ofs = ad
                .len_descr()
                .expect("ARRAYLEN_GC descr must have lendescr")
                .offset() as i64;
            // rewrite.py:272 WORD itemsize, unsigned.
            let word = std::mem::size_of::<usize>() as i64;
            let ptr = st.resolve(op.arg(0));
            let cint_zero = st.const_int(0);
            self.emit_gc_load_or_indexed(op, ptr, cint_zero, word, 1, ofs, NOT_SIGNED, st);
            return false;
        }
        // rewrite.py:273-282 STRLEN / UNICODELEN — load length field
        // via `get_array_token(...).ofs_length`, which lives on the
        // ArrayDescr as `lendescr.offset`.  Upstream reads a WORD,
        // unsigned.
        if matches!(opnum, OpCode::Strlen | OpCode::Unicodelen) {
            let word = std::mem::size_of::<usize>() as i64;
            let descr = op
                .descr
                .as_ref()
                .expect("STRLEN/UNICODELEN op must carry an ArrayDescr");
            let ad = descr
                .as_array_descr()
                .expect("STRLEN/UNICODELEN descr must be an ArrayDescr");
            let ld = ad
                .len_descr()
                .expect("STR/UNICODE ArrayDescr must carry lendescr");
            let ofs = ld.offset() as i64;
            let ptr = st.resolve(op.arg(0));
            let cint_zero = st.const_int(0);
            self.emit_gc_load_or_indexed(op, ptr, cint_zero, word, 1, ofs, NOT_SIGNED, st);
            return false;
        }
        // rewrite.py:283-294 STRHASH / UNICODEHASH — `get_field_token(
        // rstr.STR/UNICODE, 'hash', ...)` with `sign=True` and
        // `assert size == WORD`.  The upstream call returns
        // (offset, size); pyre injects a FieldDescr that carries both.
        if matches!(opnum, OpCode::Strhash | OpCode::Unicodehash) {
            let word = std::mem::size_of::<usize>() as i64;
            let descr = op
                .descr
                .as_ref()
                .expect("STRHASH/UNICODEHASH op must carry a FieldDescr");
            let fd = descr
                .as_field_descr()
                .expect("STRHASH/UNICODEHASH descr must be a FieldDescr");
            assert_eq!(fd.field_size() as i64, word, "rewrite.py:286/292 assert");
            let ofs = fd.offset() as i64;
            let ptr = st.resolve(op.arg(0));
            let cint_zero = st.const_int(0);
            self.emit_gc_load_or_indexed(op, ptr, cint_zero, word, 1, ofs, true, st);
            return false;
        }
        // rewrite.py:295-301 STRGETITEM — `basesize -= 1` skips the
        // `extra_item_after_alloc` null terminator carried by
        // `rstr.STR.chars` (`rstr.py:1226-1228`).  `itemsize == 1` is
        // asserted upstream at rewrite.py:298.
        if matches!(opnum, OpCode::Strgetitem) {
            let (itemsize, basesize) = strgetsetitem_token(op, /*is_str=*/ true);
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            self.emit_gc_load_or_indexed(
                op, ptr, index, itemsize, itemsize, basesize, NOT_SIGNED, st,
            );
            return false;
        }
        // rewrite.py:302-306 UNICODEGETITEM — UNICODE has no
        // extra_item_after_alloc, so basesize is used as-is.
        if matches!(opnum, OpCode::Unicodegetitem) {
            let (itemsize, basesize) = strgetsetitem_token(op, /*is_str=*/ false);
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            self.emit_gc_load_or_indexed(
                op, ptr, index, itemsize, itemsize, basesize, NOT_SIGNED, st,
            );
            return false;
        }
        // rewrite.py:307-313 STRSETITEM.
        if matches!(opnum, OpCode::Strsetitem) {
            let (itemsize, basesize) = strgetsetitem_token(op, /*is_str=*/ true);
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            let value = st.resolve(op.arg(2));
            self.emit_gc_store_or_indexed(
                Some(op),
                ptr,
                index,
                value,
                itemsize,
                itemsize,
                basesize,
                st,
            );
            return false;
        }
        // rewrite.py:314-318 UNICODESETITEM.
        if matches!(opnum, OpCode::Unicodesetitem) {
            let (itemsize, basesize) = strgetsetitem_token(op, /*is_str=*/ false);
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            let value = st.resolve(op.arg(2));
            self.emit_gc_store_or_indexed(
                Some(op),
                ptr,
                index,
                value,
                itemsize,
                itemsize,
                basesize,
                st,
            );
            return false;
        }
        // rewrite.py:319-330 GC_LOAD_INDEXED_{I,R,F} normalisation.
        if matches!(
            opnum,
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF
        ) {
            let scale = st
                .resolve_constant(op.arg(2).raw())
                .expect("GC_LOAD_INDEXED scale must be ConstInt");
            let offset = st
                .resolve_constant(op.arg(3).raw())
                .expect("GC_LOAD_INDEXED offset must be ConstInt");
            let size = st
                .resolve_constant(op.arg(4).raw())
                .expect("GC_LOAD_INDEXED size must be ConstInt");
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            self.emit_gc_load_or_indexed(op, ptr, index, size.abs(), scale, offset, size < 0, st);
            return false;
        }
        if matches!(opnum, OpCode::GcStoreIndexed) {
            let scale = st
                .resolve_constant(op.arg(3).raw())
                .expect("GC_STORE_INDEXED scale must be ConstInt");
            let offset = st
                .resolve_constant(op.arg(4).raw())
                .expect("GC_STORE_INDEXED offset must be ConstInt");
            let size = st
                .resolve_constant(op.arg(5).raw())
                .expect("GC_STORE_INDEXED size must be ConstInt");
            let ptr = st.resolve(op.arg(0));
            let index = st.resolve(op.arg(1));
            let value = st.resolve(op.arg(2));
            // rewrite.py:338: use abs(size) for safety even though store
            // size is expected to be positive.
            self.emit_gc_store_or_indexed(
                Some(op),
                ptr,
                index,
                value,
                size.abs(),
                scale,
                offset,
                st,
            );
            return false;
        }
        // rewrite.py:342
        false
    }

    // ────────────────────────────────────────────────────────
    // rewrite.py:955-973 gen_write_barrier_array
    // ────────────────────────────────────────────────────────

    fn gen_write_barrier_array(&self, v_base: OpRef, v_index: OpRef, st: &mut RewriteState) {
        if self.wb_descr.jit_wb_cards_set != 0 {
            // If we know statically the length of 'v_base', and it is not
            // too big, then produce a regular write_barrier. If it's
            // unknown or too big, produce a write_barrier_from_array.
            const LARGE: usize = 130;
            let length = st.known_length(v_base, LARGE);
            if length >= LARGE {
                // Unknown or too big: produce COND_CALL_GC_WB_ARRAY.
                let wb_op = Op::new(OpCode::CondCallGcWbArray, &[v_base, v_index]);
                st.emit(wb_op);
                // rewrite.py:970: a WB_ARRAY is not enough to prevent
                // any future write barriers, so don't remember_wb!
                return;
            }
        }
        // Fall-back: produce a regular write_barrier.
        let wb_op = Op::new(OpCode::CondCallGcWb, &[v_base]);
        st.emit(wb_op);
        st.remember_wb(v_base);
    }

    // ────────────────────────────────────────────────────────
    // gen_malloc_nursery: batched bump-pointer allocation
    // ────────────────────────────────────────────────────────

    /// rewrite.py:879-912 `gen_malloc_nursery` parity.
    ///
    /// Try to emit (or extend) a CALL_MALLOC_NURSERY for `size` bytes.
    /// Returns `Some(result)` on success; you still need to write the
    /// tid (rewrite.py:881-882).  Returns `None` when the requested
    /// size exceeds `can_use_nursery_malloc` — upstream's caller then
    /// falls back to `gen_malloc_fixedsize` /
    /// `gen_malloc_array` / `gen_malloc_str` / `gen_malloc_unicode`,
    /// whose `_gen_call_malloc_gc` helper does NOT mark the result as
    /// wb_applied (rewrite.py:775-776) because the slow malloc path
    /// may return an oldgen object.
    fn gen_malloc_nursery(
        &self,
        size: usize,
        result_pos: OpRef,
        st: &mut RewriteState,
    ) -> Option<OpRef> {
        let size = round_up(size);

        // rewrite.py:884-886 — caller picks a slow path when nursery
        // can't accommodate the size.
        if !self.can_use_nursery(size) {
            return None;
        }

        // rewrite.py:893-898 merge with previous CALL_MALLOC_NURSERY
        if let Some(prev_idx) = st.pending_malloc_idx {
            let new_total = st.pending_malloc_total + size;
            if self.can_use_nursery(new_total) {
                let new_total_ref = st.const_int(new_total as i64);
                st.out[prev_idx].args[0] = new_total_ref;
                st.pending_malloc_total = new_total;

                // rewrite.py:896: NURSERY_PTR_INCREMENT(last, ConstInt(previous_size))
                let prev_size_ref = st.const_int(st.previous_size as i64);
                let incr_op = Op::new(
                    OpCode::NurseryPtrIncrement,
                    &[st.last_malloced_ref, prev_size_ref],
                );
                let r = st.emit_result(incr_op, result_pos);
                st.previous_size = size;
                st.last_malloced_ref = r;
                st.remember_wb(r);
                return Some(r);
            }
        }

        // rewrite.py:903: CALL_MALLOC_NURSERY(ConstInt(size))
        st.emitting_an_operation_that_can_collect();
        let size_ref = st.const_int(size as i64);
        let op = Op::new(OpCode::CallMallocNursery, &[size_ref]);
        let r = st.emit_result(op, result_pos);
        st.pending_malloc_idx = Some(st.out.len() - 1);
        st.pending_malloc_total = size;
        st.previous_size = size;
        st.last_malloced_ref = r;
        st.remember_wb(r);
        Some(r)
    }

    // ────────────────────────────────────────────────────────
    // Helpers for header initialisation
    // ────────────────────────────────────────────────────────

    /// rewrite.py:914-918 gen_initialize_tid parity.
    ///
    /// RPython:
    /// ```python
    /// def gen_initialize_tid(self, v_newgcobj, tid):
    ///     if self.gc_ll_descr.fielddescr_tid is not None:
    ///         self.emit_setfield(v_newgcobj, ConstInt(tid),
    ///                            descr=self.gc_ll_descr.fielddescr_tid)
    /// ```
    /// `emit_setfield` lowers to `GC_STORE(ptr, ConstInt(offset),
    /// ConstInt(tid), ConstInt(size))` via `emit_gc_store_or_indexed`.
    ///
    /// pyre layout note: HDR sits at `obj_ptr - HDR_SIZE` (vs RPython's
    /// HDR at `obj_ptr + 0`).  `fielddescr_tid.offset` is the offset of
    /// `tid` within the header struct (0 for pyre's single-word HDR);
    /// the actual store address is `obj_ptr + (-HDR_SIZE +
    /// descr.offset())`.  None disables the store (Boehm parity).
    ///
    /// `fielddescr_tid.field_size` is 4 bytes (descr.rs
    /// `make_tid_field_descr`): pyre's HDR packs type id into the lower
    /// 32 bits and gc flags (TRACK_YOUNG_PTRS / VISITED / …) into the
    /// upper 32 bits, and the slow `dynasm_nursery_slowpath` /
    /// cranelift-side malloc helpers may promote large or
    /// post-collection allocations to the old gen, where
    /// `collector.rs:449 alloc_in_oldgen` pre-stamps `TRACK_YOUNG_PTRS`
    /// in those upper bits.  A full-word store from this helper would
    /// wipe that bit and leave a fresh oldgen object invisible to the
    /// remembered-set machinery, dropping any subsequent young pointer
    /// written into it.  Restricting the GC_STORE width to
    /// `field_size = 4` keeps the upper half intact.
    fn gen_initialize_tid(&self, obj: OpRef, tid: u32, st: &mut RewriteState) {
        let Some(tid_fd_ref) = self.fielddescr_tid.as_ref() else {
            return;
        };
        let tid_fd = tid_fd_ref
            .as_field_descr()
            .expect("gc_ll_descr.fielddescr_tid must be a FieldDescr");
        let ofs = st.const_int(-(crate::header::GcHeader::SIZE as i64) + tid_fd.offset() as i64);
        let tid_val = st.const_int(tid as i64);
        let size = st.const_int(tid_fd.field_size() as i64);
        let store = Op::new(OpCode::GcStore, &[obj, ofs, tid_val, size]);
        st.emit(store);
    }

    /// rewrite.py:479-484 gen_initialize_vtable parity.
    ///
    /// RPython: emit_setfield(obj, ConstInt(vtable), descr=fielddescr_vtable)
    /// — the typeptr field of `rclass.OBJECT`. Offset / size come from
    /// the supplied `fielddescr_vtable` (gc.py:36 `get_field_descr(self,
    /// rclass.OBJECT, 'typeptr')`); for the canonical layout the
    /// vtable pointer sits at offset 0 with `Signed` size.
    fn gen_initialize_vtable(
        &self,
        obj: OpRef,
        vtable: usize,
        vtable_fd_ref: &DescrRef,
        st: &mut RewriteState,
    ) {
        let vtable_fd = vtable_fd_ref
            .as_field_descr()
            .expect("gc_ll_descr.fielddescr_vtable must be a FieldDescr");
        let vtable_ref = st.const_int(vtable as i64);
        self.emit_setfield(obj, vtable_ref, vtable_fd, st);
    }

    /// rewrite.py:920-922 gen_initialize_len parity.
    ///
    /// RPython: `emit_setfield(v_newgcobj, v_length, descr=arraylen_descr)`.
    /// Routes through the local `emit_setfield` helper so the lowered
    /// store passes through `cpu_simplify_scale` (rewrite.py:1118-1122)
    /// for the ConstInt(0) index fold, matching upstream's emission
    /// path.
    fn gen_initialize_len(
        &self,
        obj: OpRef,
        length: OpRef,
        len_descr: &dyn FieldDescr,
        st: &mut RewriteState,
    ) {
        self.emit_setfield(obj, length, len_descr, st);
    }

    /// rewrite.py:665-695 handle_call_assembler:
    ///   1. gen_malloc_frame — allocate callee jitframe from nursery
    ///   2. gen_initialize_tid + zero GC fields
    ///   3. store each arg at _ll_initial_locs[i] offset
    ///   4. replace multi-arg CALL_ASSEMBLER with single-arg [frame]
    ///
    /// Dispatched from the `CallAssembler{I,R,F,N}` arm of `rewrite_loop`
    /// (rewrite.py:413-416 parity).
    fn handle_call_assembler(&self, op: &Op, st: &mut RewriteState) {
        let descrs = self.jitframe_info.as_ref().unwrap();
        let lookup = self.call_assembler_callee_locs.as_ref().unwrap();

        // rewrite.py:667-668 — loop_token = op.getdescr(); JitCellToken
        let loop_token_descr = op
            .descr
            .as_ref()
            .and_then(|d| d.as_loop_token_descr())
            .expect("CallAssembler op must carry a loop-token descriptor");
        let token = loop_token_descr.loop_token_number();

        // rewrite.py:673 — index_list = loop_token.compiled_loop_token._ll_initial_locs
        // RPython: compiled_loop_token is pre-allocated with the token;
        // frame_info pointer is stable. Self-recursive calls go through
        // register_pending_call_assembler_target() BEFORE tracing emits
        // any CALL_ASSEMBLER op referencing this token.
        let callee_locs = lookup(token)
            .expect("pending CALL_ASSEMBLER target must be registered before rewriter runs");
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[gc-rewrite][call-assembler] token={} frame_info_ptr=0x{:x} ll_initial_locs={:?} frame_depth={} index_of_virtualizable={}",
                token,
                callee_locs.frame_info_ptr,
                callee_locs._ll_initial_locs,
                callee_locs.frame_depth,
                callee_locs.index_of_virtualizable,
            );
        }

        // rewrite.py:627-653 — gen_malloc_frame(llfi)
        // RPython reads jfi_frame_size from frame_info AT RUNTIME so
        // the allocation size is correct even for self-recursive calls
        // (where frame_info is pre-allocated with [0,0] and updated
        // after compilation).
        let llfi = st.const_int(callee_locs.frame_info_ptr as i64);
        // jitframe.py:30-36 — JITFRAMEINFO.jfi_frame_depth and
        // jfi_frame_size are both lltype.Signed, so the unpack_fielddescr
        // size read by emit_getfield is sign_size (the Signed word width).
        let signed_size = st.const_int(descrs.sign_size as i64);

        // rewrite.py:628-632 — GC_LOAD_I(frame_info, jfi_frame_size_ofs,
        // sign_size) where (ofs, sign_size, sign) = unpack_fielddescr(
        // descrs.jfi_frame_size).
        let jfi_frame_size_ofs = st.const_int(std::mem::size_of::<isize>() as i64);
        let size = st.emit(Op::new(
            OpCode::GcLoadI,
            &[llfi, jfi_frame_size_ofs, signed_size],
        ));
        // rewrite.py:634 — gen_malloc_nursery_varsize_frame(size)
        st.emitting_an_operation_that_can_collect();
        let malloc_op = Op::new(OpCode::CallMallocNurseryVarsizeFrame, &[size]);
        let frame = st.emit_result(malloc_op, OpRef::NONE);
        st.remember_wb(frame);

        // rewrite.py:635 — gen_initialize_tid(frame, descrs.arraydescr.tid)
        self.gen_initialize_tid(frame, descrs.jitframe_tid, st);

        // rewrite.py:641-650 — emit_setfield(frame, c_null, descr=jf_*)
        // with (_, size, _) = unpack_fielddescr(descr). jitframe.py:63-81
        // every zeroed field (jf_descr / jf_force_descr / jf_savedata /
        // jf_guard_exc / jf_forward) is a GCREF or Ptr, i.e. pointer-
        // sized. majit's homogeneous JitFrame layout keeps all six at
        // sign_size; route through sign_size to mirror the per-descr
        // read.
        let zero = st.const_int(0);
        let signed_size_val = descrs.sign_size as i64;
        for &ofs in &[
            descrs.jf_descr_ofs,
            descrs.jf_force_descr_ofs,
            descrs.jf_savedata_ofs,
            descrs.jf_guard_exc_ofs,
            descrs.jf_forward_ofs,
        ] {
            self.emit_setfield_raw(frame, zero, ofs as i64, signed_size_val, st);
        }

        // rewrite.py:639-640 — emit_getfield(frame_info, descrs.jfi_frame_depth),
        // rewrite.py:651-652 — gen_initialize_len(frame, length, ...).
        // Both read/write lltype.Signed values (jfi_frame_depth and the
        // jf_frame length field).
        let jfi_frame_depth_ofs = st.const_int(0);
        let length = st.emit(Op::new(
            OpCode::GcLoadI,
            &[llfi, jfi_frame_depth_ofs, signed_size],
        ));
        self.emit_setfield_raw(
            frame,
            length,
            descrs.jf_frame_lengthofs as i64,
            signed_size_val,
            st,
        );

        // rewrite.py:671 — emit_setfield(frame, ConstInt(llfi),
        // descr=descrs.jf_frame_info). jf_frame_info is Ptr(JITFRAMEINFO)
        // (jitframe.py:63) so the field size is the pointer width, which
        // in majit's layout coincides with sign_size.
        self.emit_setfield_raw(
            frame,
            llfi,
            descrs.jf_frame_info_ofs as i64,
            signed_size_val,
            st,
        );

        // rewrite.py:672-683 — store each arg at _ll_initial_locs[i] with
        // per-arg itemsize from getarraydescr_for_frame(arg.type).
        let arglist: Vec<OpRef> = op.args.iter().map(|&a| st.resolve(a)).collect();
        let index_list = &callee_locs._ll_initial_locs;
        for (i, &arg) in arglist.iter().enumerate() {
            // rewrite.py:675-677 — descr = cpu.getarraydescr_for_frame(arg.type);
            //                      _, itemsize, _ = unpack_arraydescr_size(descr)
            let arg_ty = st
                .result_type_of(arg)
                .expect("CALL_ASSEMBLER arg lacks a typed producer");
            let itemsize = descrs.frame_itemsize(arg_ty);
            let itemsize_ref = st.const_int(itemsize);
            // rewrite.py:678-681 — array_offset = index_list[i] (bytes);
            //                      _, basesize, _ = unpack_arraydescr(descr);
            //                      offset = basesize + array_offset.
            let offset = descrs.jf_frame_baseitemofs as i32 + index_list[i];
            let ofs_ref = st.const_int(offset as i64);
            st.emit(Op::new(
                OpCode::GcStore,
                &[frame, ofs_ref, arg, itemsize_ref],
            ));
        }

        // rewrite.py:685-695 — replace multi-arg with [frame] or
        // [frame, arglist[index_of_virtualizable]]
        let new_args = if callee_locs.index_of_virtualizable >= 0 {
            let vable_idx = callee_locs.index_of_virtualizable as usize;
            vec![frame, arglist[vable_idx]]
        } else {
            vec![frame]
        };
        let mut call_asm = Op::new(op.opcode, &new_args);
        call_asm.descr = op.descr.clone();
        call_asm.fail_args = op
            .fail_args
            .as_ref()
            .map(|fa| fa.iter().map(|&a| st.resolve(a)).collect());
        st.emit_rewritten_from(op, call_asm);
    }
}

impl GcRewriterImpl {
    /// rewrite.py:988-1001 remove_bridge_exception: check a common
    /// case where SaveExcClass + SaveException + RestoreException
    /// appear at the start of a bridge and are unused. Strip them.
    fn remove_bridge_exception(ops: &[Op]) -> Vec<Op> {
        let mut start = 0;
        if ops
            .first()
            .map_or(false, |op| op.opcode == OpCode::IncrementDebugCounter)
        {
            start = 1;
        }
        if ops.len() >= start + 3
            && ops[start].opcode == OpCode::SaveExcClass
            && ops[start + 1].opcode == OpCode::SaveException
            && ops[start + 2].opcode == OpCode::RestoreException
        {
            let mut result = Vec::with_capacity(ops.len() - 3);
            result.extend_from_slice(&ops[..start]);
            result.extend_from_slice(&ops[start + 3..]);
            return result;
        }
        ops.to_vec()
    }
}

impl GcRewriter for GcRewriterImpl {
    fn rewrite_for_gc(&self, ops: &[Op]) -> Vec<Op> {
        self.rewrite_for_gc_with_constants(ops, &HashMap::new()).0
    }

    fn rewrite_for_gc_with_constants(
        &self,
        ops: &[Op],
        constants: &HashMap<u32, i64>,
    ) -> (Vec<Op>, HashMap<u32, i64>) {
        // rewrite.py:988-1001 remove_bridge_exception: strip a
        // SaveExcClass+SaveException+RestoreException prefix that is
        // a no-op (common in bridges).
        let ops = Self::remove_bridge_exception(ops);

        let next_pos = ops
            .iter()
            .filter_map(|op| (!op.pos.is_none()).then_some(op.pos.raw()))
            .max()
            .map_or(0, |max_pos| max_pos.saturating_add(1));
        let mut st = RewriteState::with_constants(ops.len(), next_pos, constants.clone());
        // Build result_types map from input ops — structural equivalent
        // of RPython's Box.type attribute (rewrite.py:930 `v.type`).
        for op in &ops {
            let rt = op.result_type();
            if rt != Type::Void && !op.pos.is_none() {
                st.result_types.insert(op.pos.raw(), rt);
            }
        }
        // Merge constant types — RPython's ConstPtr/ConstInt carry type
        // intrinsically; here we inject them into the same map.
        for (&k, &tp) in &self.constant_types {
            st.result_types.entry(k).or_insert(tp);
        }
        for (i, orig_op) in ops.iter().enumerate() {
            // rewrite.py:366-367 — if `remove_tested_failarg` rewrote this
            // op on a previous iteration, use the stashed replacement.
            let owned = st.changed_ops.remove(&i);
            let op: &Op = owned.as_ref().unwrap_or(orig_op);
            st.current_i = i;

            // rewrite.py:376-378 — is_guard OR could_merge_with_next_guard
            // triggers emit_pending_zeros at the top of the iteration.
            // could_merge_with_next_guard may also emit a SAME_AS_I and
            // stash a rewritten guard via remove_tested_failarg, so it
            // must be called regardless of whether the flush path is
            // taken — the flush only fires when one of the two branches
            // returns true.
            let merges = self.could_merge_with_next_guard(op, i, &ops, &mut st);
            if op.opcode.is_guard() || merges {
                st.emit_pending_zeros();
            }

            // rewrite.py:368-370 — transform_to_gc_load forwards memory
            // accessors to GC_LOAD / GC_STORE forms.  Returns true only
            // for the GETFIELD_GC fast-path, which also emits the
            // forwarded op itself.
            if self.transform_to_gc_load(op, &mut st) {
                continue;
            }

            match op.opcode {
                // Skip debug merge points (they carry no semantics).
                OpCode::DebugMergePoint => continue,

                // rewrite.py:1003-1006 emit_label
                OpCode::Label => {
                    st.emitting_an_operation_that_can_collect();
                    st.known_lengths.clear();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── Allocation ──
                OpCode::New | OpCode::NewWithVtable => {
                    self.handle_new(op, &mut st);
                }
                // rewrite.py:485-494 — descr source per opcode mirrors
                // upstream: NEW_ARRAY threads op.getdescr(); NEWSTR /
                // NEWUNICODE thread self.gc_ll_descr.{str,unicode}_descr.
                OpCode::NewArray | OpCode::NewArrayClear => {
                    let descr_ref = op
                        .descr
                        .clone()
                        .expect("NEW_ARRAY must carry an ArrayDescr");
                    self.handle_new_array(descr_ref, op, &mut st, 0); // FLAG_ARRAY
                }
                OpCode::Newstr => {
                    // rewrite.py:489-491 `handle_new_array(self.gc_ll_descr.str_descr, op, FLAG_STR)`.
                    self.handle_new_array(self.str_descr.clone(), op, &mut st, 1);
                }
                OpCode::Newunicode => {
                    // rewrite.py:492-494 `handle_new_array(self.gc_ll_descr.unicode_descr, op, FLAG_UNICODE)`.
                    self.handle_new_array(self.unicode_descr.clone(), op, &mut st, 2);
                }

                // ── COPYSTRCONTENT / COPYUNICODECONTENT → memcpy CALL_N ──
                // rewrite.py:388-391 `rewrite_copy_str_content` replaces
                // the copy op with LOAD_EFFECTIVE_ADDRESS × 2 + CALL_N.
                OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                    self.rewrite_copy_str_content(op, &mut st);
                }

                // ── Stores that may need a write barrier ──
                //
                // rewrite.py:392-404 — the write-barrier section runs AFTER
                // `transform_to_gc_load` has forwarded the store op to
                // GC_STORE / GC_STORE_INDEXED.  `emit_maybe_forwarded`
                // follows the forward and emits the lowered op.
                OpCode::SetfieldGc => {
                    // rewrite.py:393-395 — consider_setfield_gc clears the
                    // pending zero-init entry before WB emission.
                    self.consider_setfield_gc(op, &mut st);
                    self.handle_write_barrier_setfield(op, &mut st);
                    st.emit_maybe_forwarded(op);
                    continue;
                }
                OpCode::SetinteriorfieldGc => {
                    // rewrite.py:946 `handle_write_barrier_setinteriorfield
                    // = handle_write_barrier_setarrayitem`.
                    self.handle_write_barrier_setarrayitem(op, &mut st);
                    st.emit_maybe_forwarded(op);
                    continue;
                }
                OpCode::SetarrayitemGc => {
                    // rewrite.py:401-404
                    self.consider_setarrayitem_gc(op, &mut st);
                    self.handle_write_barrier_setarrayitem(op, &mut st);
                    st.emit_maybe_forwarded(op);
                    continue;
                }

                // ── call_assembler: rewrite.py:414 handle_call_assembler ──
                OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN => {
                    self.handle_call_assembler(op, &mut st);
                    continue;
                }

                // ── Operations that can trigger GC ──
                _ if op.opcode.can_malloc() => {
                    // rewrite.py:379-380 — emitting_an_operation_that_can_collect
                    // already flushes pending zeros (rewrite.py:707).
                    st.emitting_an_operation_that_can_collect();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── GUARD_ALWAYS_FAILS lowering (rewrite.py:419-426) ──
                // Upstream turns an always-failing guard into
                //   SAME_AS_I(0)
                //   GUARD_VALUE(same_as, ConstInt(1))
                // so the backend can share its GUARD_VALUE emission path.
                // failargs are carried over via copy_and_change.
                OpCode::GuardAlwaysFails => {
                    let zero = st.const_int(0);
                    let one = st.const_int(1);
                    let same = Op::new(OpCode::SameAsI, &[zero]);
                    let same_pos = st.emit_result(same, OpRef::NONE);
                    let newop =
                        op.copy_and_change(OpCode::GuardValue, Some(&[same_pos, one]), None);
                    let rewritten = st.rewrite_op(&newop);
                    st.emit(rewritten);
                }

                // ── Guards: emit_pending_zeros was already called at the
                // top of the iteration per rewrite.py:376-378; here we only
                // need to emit the (forwarded) guard op itself. Guards do
                // not clear wb_applied — only emitting_an_operation_that_
                // can_collect does that (rewrite.py:699-711).
                _ if op.opcode.is_guard() => {
                    let rewritten = st.rewrite_op(op);
                    st.emit(rewritten);
                }

                // ── Everything else: pass through unchanged. ──
                OpCode::CondCallGcWb => {
                    let rewritten = st.rewrite_op(op);
                    let obj = rewritten.arg(0);
                    st.emit(rewritten);
                    st.remember_wb(obj);
                }
                OpCode::CondCallGcWbArray => {
                    // rewrite.py:970: WB_ARRAY does not mark the base as
                    // barrier-applied; future setarrayitems still need
                    // their own barrier (no remember_wb call).
                    let rewritten = st.rewrite_op(op);
                    st.emit(rewritten);
                }
                // ── Final ops (Jump, Finish) flush pending zeros before emit. ──
                _ if op.opcode.is_final() => {
                    st.emit_pending_zeros();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // rewrite.py:383-387 — record INT_ADD/INT_SUB whose
                // constant operand is later folded into a GC_STORE_INDEXED
                // / GC_LOAD_INDEXED offset via `_try_use_older_box`.
                OpCode::IntAdd | OpCode::IntAddOvf => {
                    let rewritten = st.rewrite_op(op);
                    st.record_int_add_or_sub(&rewritten, false);
                    st.emit_rewritten_from(op, rewritten);
                }
                OpCode::IntSub | OpCode::IntSubOvf => {
                    let rewritten = st.rewrite_op(op);
                    st.record_int_add_or_sub(&rewritten, true);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── Everything else: follow forwarding if `transform_to_gc_load`
                // has forwarded this op (GETARRAYITEM, GETFIELD_RAW,
                // SETFIELD_RAW, SETARRAYITEM_RAW, SETINTERIORFIELD_RAW,
                // ARRAYLEN_GC, RAW_LOAD, RAW_STORE, GC_LOAD_INDEXED,
                // GC_STORE_INDEXED …); otherwise pass through unchanged.
                _ => {
                    st.emit_maybe_forwarded(op);
                }
            }
        }

        // Flush any remaining pending zeros at end of trace.
        st.emit_pending_zeros();

        (st.out, st.constants)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use majit_ir::descr::{ArrayDescr, Descr, DescrRef, SizeDescr};
    use majit_ir::value::Type;

    const TEST_STANDARD_ARRAY_BASESIZE: usize = std::mem::size_of::<usize>();
    const TEST_STANDARD_ARRAY_LENGTH_OFS: usize = 0;
    const TEST_MALLOC_ARRAY_FN: i64 = 0x1111;
    const TEST_MALLOC_ARRAY_NONSTANDARD_FN: i64 = 0x2222;
    const TEST_MALLOC_STR_FN: i64 = 0x3333;
    const TEST_MALLOC_UNICODE_FN: i64 = 0x4444;
    const TEST_MALLOC_BIG_FIXEDSIZE_FN: i64 = 0x5555;

    // ── Minimal concrete descriptor implementations for testing ──

    #[derive(Debug)]
    struct TestSizeDescr {
        size: usize,
        type_id: u32,
        vtable: usize,
        gc_fields: Vec<Arc<dyn FieldDescr>>,
    }

    impl Descr for TestSizeDescr {
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            self.size
        }
        fn type_id(&self) -> u32 {
            self.type_id
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            self.vtable != 0
        }
        fn vtable(&self) -> usize {
            self.vtable
        }
        fn gc_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
            &self.gc_fields
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        offset: usize,
        field_size: usize,
        field_type: Type,
    }

    impl Descr for TestFieldDescr {
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }
        fn field_size(&self) -> usize {
            self.field_size
        }
        fn field_type(&self) -> Type {
            self.field_type
        }
    }

    #[derive(Debug)]
    struct TestArrayDescr {
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
        len_descr: Option<Arc<TestFieldDescr>>,
    }

    impl Descr for TestArrayDescr {
        fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
            Some(self)
        }
    }

    impl ArrayDescr for TestArrayDescr {
        fn base_size(&self) -> usize {
            self.base_size
        }
        fn item_size(&self) -> usize {
            self.item_size
        }
        fn type_id(&self) -> u32 {
            self.type_id
        }
        fn item_type(&self) -> Type {
            self.item_type
        }
        fn len_descr(&self) -> Option<&dyn FieldDescr> {
            self.len_descr
                .as_ref()
                .map(|fd| fd.as_ref() as &dyn FieldDescr)
        }
    }

    fn make_rewriter() -> GcRewriterImpl {
        GcRewriterImpl {
            nursery_free_addr: 0x1000,
            nursery_top_addr: 0x2000,
            max_nursery_size: 4096,
            wb_descr: WriteBarrierDescr {
                jit_wb_if_flag: 1,
                jit_wb_if_flag_byteofs: 0,
                jit_wb_if_flag_singlebyte: 1,
                jit_wb_cards_set: 0,
                jit_wb_card_page_shift: 0,
                jit_wb_cards_set_byteofs: 0,
                jit_wb_cards_set_singlebyte: 0,
            },
            jitframe_info: None,
            constant_types: HashMap::new(),
            call_assembler_callee_locs: None,
            // llmodel.py:39 default keeps existing pre-scale-everything behavior
            // in tests written against it; per-backend overrides have dedicated
            // tests below.
            load_supported_factors: &[1],
            // x86/runner.py:22 + arm/runner.py:26 — production parity;
            // dedicated `_without_load_effective_address` tests flip this
            // to `false` to exercise the fallback (rewrite.py:1089-1098).
            supports_load_effective_address: true,
            // Match the production nursery's zero-fill behavior; the
            // `clear_gc_fields` / `_delayed_zero_setfields` tests flip
            // this to `false` per-test.
            malloc_zero_filled: true,
            // gc.py:39-43, gc.py:46-49 fields.
            memcpy_fn: majit_ir::memcpy_fn_addr(),
            memcpy_descr: majit_ir::make_memcpy_calldescr(),
            str_descr: str_array_descr(),
            unicode_descr: unicode_array_descr(),
            str_hash_descr: hash_field_descr(),
            unicode_hash_descr: hash_field_descr(),
            // gc.py:33-37 `fielddescr_vtable`. Test fixtures always
            // install a Some so the existing test_new_with_vtable
            // continues to exercise the typeptr stamping path.
            fielddescr_vtable: Some(majit_ir::make_vtable_field_descr()),
            // gc.py:394 `fielddescr_tid`. Test fixtures always install a
            // Some so existing handle_new tests continue to exercise the
            // tid header stamping path (matches framework-GC mode).
            fielddescr_tid: Some(majit_ir::make_tid_field_descr()),
            malloc_array_fn: TEST_MALLOC_ARRAY_FN,
            malloc_array_nonstandard_fn: TEST_MALLOC_ARRAY_NONSTANDARD_FN,
            malloc_str_fn: TEST_MALLOC_STR_FN,
            malloc_unicode_fn: TEST_MALLOC_UNICODE_FN,
            malloc_big_fixedsize_fn: TEST_MALLOC_BIG_FIXEDSIZE_FN,
            malloc_array_descr: majit_ir::make_malloc_array_calldescr(),
            malloc_array_nonstandard_descr: majit_ir::make_malloc_array_nonstandard_calldescr(),
            malloc_str_descr: majit_ir::make_malloc_str_calldescr(),
            malloc_unicode_descr: majit_ir::make_malloc_unicode_calldescr(),
            malloc_big_fixedsize_descr: majit_ir::make_malloc_big_fixedsize_calldescr(),
            standard_array_basesize: TEST_STANDARD_ARRAY_BASESIZE,
            standard_array_length_ofs: TEST_STANDARD_ARRAY_LENGTH_OFS,
        }
    }

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef::op_typed(pos, opcode.result_type());
        op
    }

    fn mk_op_with_descr(opcode: OpCode, args: &[OpRef], pos: u32, descr: DescrRef) -> Op {
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = OpRef::op_typed(pos, opcode.result_type());
        op
    }

    fn size_descr(size: usize, type_id: u32) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
            gc_fields: Vec::new(),
        })
    }

    fn size_descr_with_gc_fields(
        size: usize,
        type_id: u32,
        gc_fields: Vec<Arc<dyn FieldDescr>>,
    ) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
            gc_fields,
        })
    }

    fn vtable_descr(size: usize, type_id: u32, vtable: usize) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable,
            gc_fields: Vec::new(),
        })
    }

    fn ref_field_descr() -> DescrRef {
        Arc::new(TestFieldDescr {
            offset: 0,
            field_size: 8,
            field_type: Type::Ref,
        })
    }

    fn int_field_descr() -> DescrRef {
        Arc::new(TestFieldDescr {
            offset: 8,
            field_size: 8,
            field_type: Type::Int,
        })
    }

    fn array_len_field_descr() -> Arc<TestFieldDescr> {
        Arc::new(TestFieldDescr {
            offset: TEST_STANDARD_ARRAY_LENGTH_OFS,
            field_size: std::mem::size_of::<usize>(),
            field_type: Type::Int,
        })
    }

    fn array_descr_ref() -> DescrRef {
        Arc::new(TestArrayDescr {
            base_size: TEST_STANDARD_ARRAY_BASESIZE,
            item_size: 8,
            type_id: 5,
            item_type: Type::Ref,
            len_descr: Some(array_len_field_descr()),
        })
    }

    fn array_descr_int() -> DescrRef {
        Arc::new(TestArrayDescr {
            base_size: TEST_STANDARD_ARRAY_BASESIZE,
            item_size: 4,
            type_id: 6,
            item_type: Type::Int,
            len_descr: Some(array_len_field_descr()),
        })
    }

    // ── Test 1: NEW → CALL_MALLOC_NURSERY + tid init ──

    #[test]
    fn test_new_rewrite() {
        let rw = make_rewriter();
        let ops = vec![Op::with_descr(OpCode::New, &[], size_descr(32, 7))];

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // Expect: CallMallocNursery, GcStore (tid)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallMallocNursery);
        // rewrite.py:474-484 parity: size arg is ConstInt(descr.size + GcHeader::SIZE).
        let size_val = constants[&result[0].args[0].raw()];
        assert_eq!(size_val, (32 + crate::header::GcHeader::SIZE) as i64);
        assert_eq!(result[1].opcode, OpCode::GcStore); // tid init
        let tid_val = constants[&result[1].args[2].raw()];
        assert_eq!(tid_val, 7); // type_id = 7
        // GcHeader packs type id (lower 32 bits) and gc flags (upper 32
        // bits) into a single u64.  gen_initialize_tid must emit a
        // 4-byte store so that the runtime-set flags
        // (collector.rs:449 alloc_in_oldgen ORs in TRACK_YOUNG_PTRS for
        // oldgen-promoted allocs) survive the type id stamp.
        let store_size = constants[&result[1].args[3].raw()];
        assert_eq!(
            store_size, 4,
            "gen_initialize_tid must emit a 4-byte store (type id half) so \
             oldgen TRACK_YOUNG_PTRS in the upper 32 bits is preserved"
        );
    }

    // ── Test 2: NEW_ARRAY → CALL_MALLOC_NURSERY_VARSIZE ──

    #[test]
    fn test_new_array_rewrite() {
        let rw = make_rewriter();
        let length_ref = OpRef::int_op(100); // some prior op producing the length
        let ops = vec![Op::with_descr(
            OpCode::NewArray,
            &[length_ref],
            array_descr_int(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        // Expect: CallMallocNurseryVarsize
        assert!(
            result
                .iter()
                .any(|o| o.opcode == OpCode::CallMallocNurseryVarsize)
        );
        let varsize = result
            .iter()
            .find(|o| o.opcode == OpCode::CallMallocNurseryVarsize)
            .unwrap();
        // rewrite.py:858: [ConstInt(kind), ConstInt(itemsize), v_length]
        assert_eq!(varsize.args[2], length_ref);
    }

    /// Constant-length oversized arrays: rewrite.py:573-584 routes these
    /// through `gen_malloc_array`, not CALL_MALLOC_NURSERY_VARSIZE.
    /// Verify pyre now emits CALL_R(malloc_array_fn, ...) plus
    /// CHECK_MEMORY_ERROR, with the typed slow helper receiving the
    /// descriptor's type id directly.
    #[test]
    fn test_new_array_const_oversize_uses_malloc_array_helper() {
        let rw = make_rewriter(); // max_nursery_size = 4096
        let len_ref = OpRef::int_op(10_000);
        // array_descr_ref: base_size=8, item_size=8 →
        //   total = 8 + 8*512 = 4104; gen_malloc_nursery sees
        //   round_up(GcHeader::SIZE + 4104) = 4112 > 4096 → returns None.
        let mut constants = HashMap::new();
        constants.insert(10_000, 512_i64);
        let mut new_array = Op::with_descr(OpCode::NewArray, &[len_ref], array_descr_ref());
        new_array.pos = OpRef::ref_op(0);
        let ops = vec![new_array, Op::new(OpCode::Finish, &[])];

        let (result, consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        assert!(
            !result
                .iter()
                .any(|o| o.opcode == OpCode::CallMallocNurseryVarsize),
            "constant-length oversize must not fall back to \
             CALL_MALLOC_NURSERY_VARSIZE anymore"
        );
        let call_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::CallR)
            .expect("constant-length oversize must emit CALL_R slow helper");
        let call = &result[call_idx];
        assert_eq!(consts[&call.args[0].raw()], TEST_MALLOC_ARRAY_FN);
        assert_eq!(consts[&call.args[1].raw()], 8);
        assert_eq!(consts[&call.args[2].raw()], 5);
        assert_eq!(call.args[3], len_ref);
        assert!(
            result
                .get(call_idx + 1)
                .is_some_and(|o| o.opcode == OpCode::CheckMemoryError),
            "CALL_R slow helper must be followed by CHECK_MEMORY_ERROR"
        );
    }

    // ── Test 3: SETFIELD_GC with Ref value → write barrier inserted ──

    #[test]
    fn test_setfield_gc_ref_needs_wb() {
        // rewrite.py:262-266 + 401-404: transform_to_gc_load forwards
        // SETFIELD_GC to GC_STORE; the write-barrier arm emits WB then
        // emit_maybe_forwarded follows the forward.
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let val = OpRef::ref_op(1);
        let ops = vec![Op::with_descr(
            OpCode::SetfieldGc,
            &[obj, val],
            ref_field_descr(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        // Expect: CondCallGcWb(obj), GcStore(obj, 0, val, itemsize)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CondCallGcWb);
        assert_eq!(result[0].args[0], obj);
        assert_eq!(result[1].opcode, OpCode::GcStore);
    }

    // ── Parity guards against `transform_to_gc_load` over-reaching ──

    #[test]
    fn test_getfield_gc_pure_not_lowered() {
        // rewrite.py:246-247 excludes GETFIELD_GC_PURE_{I,R,F} from the
        // lowering arm — upstream only handles GETFIELD_GC_{I,R,F} and
        // GETFIELD_RAW_{I,R,F}.  The pure variant is `is_always_pure`
        // at `resoperation.rs:1228` and must retain that identity; a
        // stray lowering to GC_LOAD_R would drop purity semantics.
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let ops = vec![Op::with_descr(
            OpCode::GetfieldGcPureR,
            &[obj],
            ref_field_descr(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcPureR);
        assert!(
            !result.iter().any(|op| matches!(
                op.opcode,
                OpCode::GcLoadR | OpCode::GcLoadI | OpCode::GcLoadF
            )),
            "GETFIELD_GC_PURE_R must not be lowered to GC_LOAD_*"
        );
    }

    #[test]
    fn test_getarrayitem_raw_r_not_lowered() {
        // rewrite.py:216-218 only pulls GETARRAYITEM_RAW_I and
        // GETARRAYITEM_RAW_F into the lowering arm; GETARRAYITEM_RAW_R
        // is intentionally missing because `jtransform.py:775`
        // (`getarrayitem_raw_r not supported`) rejects raw ref array
        // reads earlier at codewriter time.  If one somehow reaches
        // the rewriter here it must pass through — otherwise we would
        // be enabling a code path upstream explicitly disallows.
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let idx = OpRef::int_op(1);
        let ops = vec![Op::with_descr(
            OpCode::GetarrayitemRawR,
            &[obj, idx],
            array_descr_ref(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GetarrayitemRawR);
        assert!(
            !result
                .iter()
                .any(|op| matches!(op.opcode, OpCode::GcLoadIndexedR | OpCode::GcLoadR)),
            "GETARRAYITEM_RAW_R must not be lowered to GC_LOAD_INDEXED_R / GC_LOAD_R"
        );
    }

    // ── Test 4: SETFIELD_GC with Int value → no write barrier ──

    #[test]
    fn test_setfield_gc_int_no_wb() {
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let val = OpRef::ref_op(1);
        let ops = vec![Op::with_descr(
            OpCode::SetfieldGc,
            &[obj, val],
            int_field_descr(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        // Only the lowered GC_STORE — no WB for non-ref fields.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GcStore);
    }

    // ── Tests 4a-c: delayed_zero_setfields (rewrite.py:498-512, 761-766) ──

    fn ref_field_descr_at(offset: usize) -> Arc<dyn FieldDescr> {
        Arc::new(TestFieldDescr {
            offset,
            field_size: 8,
            field_type: Type::Ref,
        })
    }

    fn ref_field_descr_ref_at(offset: usize) -> DescrRef {
        Arc::new(TestFieldDescr {
            offset,
            field_size: 8,
            field_type: Type::Ref,
        })
    }

    /// rewrite.py:499-500 + rewrite.py:761-766 — malloc_zero_filled=true
    /// short-circuits `clear_gc_fields`, so NEW emits no pending NULL
    /// stores at the next flush point.  Mirrors pyre's production
    /// nursery (which `alloc_zeroed`s).
    #[test]
    fn test_clear_gc_fields_zero_filled_skips() {
        let rw = make_rewriter(); // malloc_zero_filled = true
        let gc_fields = vec![ref_field_descr_at(24), ref_field_descr_at(32)];
        let descr = size_descr_with_gc_fields(48, 42, gc_fields);
        let ops = vec![
            Op::with_descr(OpCode::New, &[], descr),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, _consts) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // Allocation header stores only (CallMallocNursery + tid GcStore) + Jump.
        // No delayed-zero NULL-pointer stores must be emitted because
        // the allocator already zero-fills.
        let gc_stores: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .collect();
        assert_eq!(
            gc_stores.len(),
            1,
            "malloc_zero_filled=true must emit only the tid init store, got {:?}",
            result
        );
    }

    /// rewrite.py:498-504 + rewrite.py:761-766 — when the allocator does
    /// not zero-fill, every GC field's byte offset is remembered and
    /// flushed as `GC_STORE(ptr, ofs, 0, 8)` at the next can-collect /
    /// flush point.
    #[test]
    fn test_emit_pending_zeros_flushes_delayed_setfields() {
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let gc_fields = vec![ref_field_descr_at(24), ref_field_descr_at(32)];
        let descr = size_descr_with_gc_fields(48, 42, gc_fields);
        let ops = vec![
            Op::with_descr(OpCode::New, &[], descr),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, consts) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // Collect the NULL-pointer stores emitted by the pending-zero flush.
        let mut seen_offsets: Vec<i64> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            // skip the tid header store (ofs=0, value=type_id, itemsize=4).
            .filter(|o| consts.get(&o.args[2].raw()).copied() == Some(0))
            .map(|o| consts[&o.args[1].raw()])
            .collect();
        seen_offsets.sort();
        assert_eq!(
            seen_offsets,
            vec![24, 32],
            "pending-zero flush must emit one NULL store per zero-init GC field"
        );
    }

    /// rewrite.py:506-512 — an explicit SETFIELD_GC at offset `ofs`
    /// removes `ofs` from `_delayed_zero_setfields`, so the flush does
    /// not re-zero the slot.
    #[test]
    fn test_consider_setfield_gc_drops_overwritten_offset() {
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let gc_fields = vec![ref_field_descr_at(24), ref_field_descr_at(32)];
        let descr = size_descr_with_gc_fields(48, 42, gc_fields);
        let val = OpRef::const_ptr(100);
        let mut constants = HashMap::new();
        constants.insert(val.raw(), 0x1234);
        let ops = vec![
            Op::with_descr(OpCode::New, &[], descr),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::ref_op(0), val],
                ref_field_descr_ref_at(24),
            ),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        let null_offsets: Vec<i64> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .filter(|o| consts.get(&o.args[2].raw()).copied() == Some(0))
            .map(|o| consts[&o.args[1].raw()])
            .collect();
        assert_eq!(
            null_offsets,
            vec![32],
            "SETFIELD_GC at ofs=24 must drop the pending-zero at ofs=24; only ofs=32 remains"
        );
    }

    // ── Test 5: Non-GC ops pass through unchanged ──

    #[test]
    fn test_passthrough() {
        let rw = make_rewriter();
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(2)]),
            Op::new(OpCode::Jump, &[]),
        ];

        let result = rw.rewrite_for_gc(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 6: Multiple allocations are batched ──

    #[test]
    fn test_batched_allocations() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(24, 1)),
            Op::with_descr(OpCode::New, &[], size_descr(32, 2)),
        ];

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        assert!(result.iter().any(|o| o.opcode == OpCode::CallMallocNursery));
        assert!(
            result
                .iter()
                .any(|o| o.opcode == OpCode::NurseryPtrIncrement)
        );

        let malloc = result
            .iter()
            .find(|o| o.opcode == OpCode::CallMallocNursery)
            .unwrap();
        // rewrite.py:893-895: combined size = round_up(24+8) + round_up(32+8) = 32 + 40 = 72
        let header = crate::header::GcHeader::SIZE as usize;
        let expected_size = round_up(24 + header) as i64 + round_up(32 + header) as i64;
        assert_eq!(constants[&malloc.args[0].raw()], expected_size);

        let incr = result
            .iter()
            .find(|o| o.opcode == OpCode::NurseryPtrIncrement)
            .unwrap();
        // rewrite.py:898: ConstInt(previous_size) = round_up(24 + GcHeader::SIZE) = 32
        assert_eq!(constants[&incr.args[1].raw()], round_up(24 + header) as i64);

        // Both should have tid initialisation.
        let tid_stores: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .collect();
        assert_eq!(tid_stores.len(), 2);
        assert_eq!(constants[&tid_stores[0].args[2].raw()], 1); // first type_id
        assert_eq!(constants[&tid_stores[1].args[2].raw()], 2); // second type_id
    }

    // ── Test 7: A collecting operation between two NEWs prevents batching ──

    #[test]
    fn test_call_breaks_batch() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(24, 1)),
            Op::new(OpCode::CallN, &[OpRef::ref_op(99)]),
            Op::with_descr(OpCode::New, &[], size_descr(24, 2)),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // There should be two separate CallMallocNursery ops
        // (the CallN in between flushes the batch).
        let malloc_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CallMallocNursery)
            .count();
        assert_eq!(malloc_count, 2);
    }

    // ── Test 8: WB not duplicated for same object ──

    #[test]
    fn test_wb_not_duplicated() {
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let val1 = OpRef::ref_op(1);
        let val2 = OpRef::ref_op(2);
        let ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[obj, val1], ref_field_descr()),
            Op::with_descr(OpCode::SetfieldGc, &[obj, val2], ref_field_descr()),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // Only one CondCallGcWb, then two lowered GC_STORE.
        let wb_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 1);
        assert_eq!(
            result
                .iter()
                .filter(|o| o.opcode == OpCode::GcStore)
                .count(),
            2
        );
    }

    // ── Test 9: Freshly allocated object skips WB ──

    #[test]
    fn test_fresh_alloc_skips_wb() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(32, 1)),
            // The freshly allocated object (pos 0) is used as the target of a store.
            // We build the SetfieldGc referencing pos=0 from the CallMallocNursery result.
        ];

        rw.rewrite_for_gc(&ops);

        // Now rewrite a SetfieldGc that stores a ref into the new object.
        let ops2 = vec![
            Op::with_descr(OpCode::New, &[], size_descr(32, 1)),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef::ref_op(0), OpRef::ref_op(99)], // arg(0) = pos of the alloc = 0
                ref_field_descr(),
            ),
        ];

        let result2 = rw.rewrite_for_gc(&ops2);

        // The CallMallocNursery result at pos=0 is in wb_applied,
        // so the SetfieldGc at arg(0)=OpRef::ref_op(0) should NOT get a WB.
        // Expected: CallMallocNursery, GcStore(tid), SetfieldGc
        // No CondCallGcWb because OpRef::ref_op(0) was remembered.
        let wb_count = result2
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 0);
    }

    // ── Test 10: NEW_WITH_VTABLE also writes vtable ──

    #[test]
    fn test_new_with_vtable() {
        let rw = make_rewriter();
        let ops = vec![Op::with_descr(
            OpCode::NewWithVtable,
            &[],
            vtable_descr(48, 3, 0xDEAD),
        )];

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // CallMallocNursery + GcStore(tid) + GcStore(vtable)
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::CallMallocNursery);
        assert_eq!(result[1].opcode, OpCode::GcStore);
        let tid_ref = result[1].args[2];
        assert_eq!(constants[&tid_ref.raw()], 3);
        assert_eq!(result[2].opcode, OpCode::GcStore);
        let vtable_ref = result[2].args[2];
        assert_eq!(constants[&vtable_ref.raw()], 0xDEAD_i64);
    }

    // ── Test 11: SETARRAYITEM_GC with Ref — no card marking → regular WB ──

    #[test]
    fn test_setarrayitem_gc_ref_wb() {
        // make_rewriter() has jit_wb_cards_set = 0 (card marking disabled).
        // rewrite.py:955-973: without card marking, gen_write_barrier_array
        // falls back to gen_write_barrier → COND_CALL_GC_WB.
        // rewrite.py:132 + 1124-1130: non-constant index, itemsize=8 is
        // power-of-2 and not in load_supported_factors=[1] → pre-scale
        // via INT_LSHIFT before GC_STORE_INDEXED.
        //
        // Emission order is [pre-scale, WB, lowered store] because
        // `transform_to_gc_load` runs *before* the write-barrier arm
        // (rewrite.py:368-370, 401-404) — the pre-scale INT_LSHIFT is
        // emitted inline inside emit_gc_store_or_indexed while the
        // GC_STORE_INDEXED itself is forwarded and emitted only after
        // the WB via `emit_maybe_forwarded`.
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let idx = OpRef::int_op(1);
        let val = OpRef::ref_op(2);
        let ops = vec![Op::with_descr(
            OpCode::SetarrayitemGc,
            &[obj, idx, val],
            array_descr_ref(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::IntLshift);
        assert_eq!(result[1].opcode, OpCode::CondCallGcWb);
        assert_eq!(result[1].args[0], obj);
        assert_eq!(result[2].opcode, OpCode::GcStoreIndexed);
    }

    // ── Test 12: Collecting op clears WB memoisation ──

    #[test]
    fn test_collecting_op_clears_wb() {
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let val = OpRef::ref_op(1);
        let ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[obj, val], ref_field_descr()),
            // This call can collect, clearing the WB set.
            Op::new(OpCode::CallN, &[OpRef::ref_op(99)]),
            Op::with_descr(OpCode::SetfieldGc, &[obj, val], ref_field_descr()),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // Two CondCallGcWb — the second one is needed because the CallN cleared the set.
        let wb_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 2);
    }

    #[test]
    fn test_explicit_result_positions_are_preserved_through_rewrite() {
        let rw = make_rewriter();
        // `OpCode::New` produces a Ref result (resoperation.py:469
        // `RefOp` mixin), so the test mints typed `RefOp` pos rather
        // than the default `Untyped` minted by `mk_op_with_descr`.
        let mut new_a = Op::with_descr(OpCode::New, &[], size_descr(24, 1));
        new_a.pos = OpRef::ref_op(2);
        let mut new_b = Op::with_descr(OpCode::New, &[], size_descr(16, 2));
        new_b.pos = OpRef::ref_op(3);
        let ops = vec![new_a, new_b, mk_op(OpCode::Finish, &[OpRef::ref_op(3)], 4)];

        let result = rw.rewrite_for_gc(&ops);

        let first_alloc = result
            .iter()
            .find(|op| op.opcode == OpCode::CallMallocNursery)
            .unwrap();
        let second_alloc = result
            .iter()
            .find(|op| op.opcode == OpCode::NurseryPtrIncrement)
            .unwrap();
        let finish = result.last().unwrap();

        assert_eq!(first_alloc.pos, OpRef::ref_op(2));
        assert_eq!(second_alloc.pos, OpRef::ref_op(3));
        assert_eq!(finish.opcode, OpCode::Finish);
        assert_eq!(finish.args[0], OpRef::ref_op(3));
        assert!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::GcStore)
                .all(|op| op.pos.is_none())
        );
    }

    #[test]
    fn test_rewrite_preserves_incoming_wb_and_adds_its_own() {
        // rewrite.py:955-973 gen_write_barrier_array does NOT call
        // remember_wb(), so consecutive SETARRAYITEM_GC with Ref values
        // each emit their own WB. Running the rewriter over a trace that
        // already contains a COND_CALL_GC_WB_ARRAY + SETARRAYITEM pair
        // preserves the incoming WB and emits a new one for the
        // SETARRAYITEM — matching upstream rather than an idempotence
        // shortcut.
        let rw = make_rewriter();
        let once = vec![
            mk_op(
                OpCode::CondCallGcWbArray,
                &[OpRef::ref_op(5), OpRef::int_op(1)],
                7,
            ),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(5), OpRef::int_op(1), OpRef::ref_op(6)],
                8,
                array_descr_ref(),
            ),
        ];

        let twice = rw.rewrite_for_gc(&once);

        let wb_count = twice
            .iter()
            .filter(|op| op.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert!(
            wb_count >= 1,
            "at least the pre-existing WB_ARRAY is preserved"
        );
        // rewrite.py:132 + 220-221: SETARRAYITEM_GC is lowered to
        // GC_STORE_INDEXED via handle_setarrayitem → emit_gc_store_or_indexed.
        assert_eq!(
            twice
                .iter()
                .filter(|op| op.opcode == OpCode::SetarrayitemGc)
                .count(),
            0
        );
        assert_eq!(
            twice
                .iter()
                .filter(|op| op.opcode == OpCode::GcStoreIndexed)
                .count(),
            1
        );
    }

    // ── could_merge_with_next_guard / remove_tested_failarg tests ──

    #[test]
    fn test_comparison_guard_true_hoists_tested_failarg() {
        // rewrite.py:431-471 parity: INT_LT followed by GUARD_TRUE(INT_LT)
        // with the comparison's result appearing in the guard's failargs.
        // The rewriter must emit SAME_AS_I(0) before the comparison and
        // rewrite the guard's failargs to reference the SAME_AS_I output.
        let rw = make_rewriter();

        let mut int_lt = Op::new(OpCode::IntLt, &[OpRef::int_op(0), OpRef::int_op(1)]);
        int_lt.pos = OpRef::int_op(2);
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef::int_op(2)]);
        guard.store_final_boxes(vec![OpRef::int_op(0), OpRef::int_op(2), OpRef::int_op(1)]);
        let ops = vec![int_lt, guard, Op::new(OpCode::Finish, &[])];

        let result = rw.rewrite_for_gc(&ops);

        // Expect the rewriter to have emitted SAME_AS_I BEFORE the IntLt.
        let same_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::SameAsI)
            .expect("SAME_AS_I must be emitted");
        let lt_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::IntLt)
            .expect("IntLt survives");
        let guard_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::GuardTrue)
            .expect("GuardTrue survives");
        assert!(
            same_idx < lt_idx,
            "SAME_AS_I must be emitted before the comparison"
        );
        assert!(lt_idx < guard_idx, "GuardTrue must follow the comparison");

        // The guard's failargs must now reference the SAME_AS_I output
        // at the position where OpRef::ref_op(2) (the IntLt result) used to appear.
        let same_pos = result[same_idx].pos;
        let guard_fa = result[guard_idx]
            .fail_args
            .as_ref()
            .expect("guard keeps failargs");
        assert_eq!(
            guard_fa.as_slice(),
            &[OpRef::int_op(0), same_pos, OpRef::int_op(1)],
            "OpRef::ref_op(2) → SAME_AS_I substitution"
        );
    }

    #[test]
    fn test_comparison_guard_false_hoists_with_one_constant() {
        // GUARD_FALSE: rewrite.py:463 `value = int(opnum == GUARD_FALSE)` ⇒ 1.
        let rw = make_rewriter();
        let mut int_eq = Op::new(OpCode::IntEq, &[OpRef::int_op(0), OpRef::int_op(1)]);
        int_eq.pos = OpRef::int_op(2);
        let mut guard = Op::new(OpCode::GuardFalse, &[OpRef::int_op(2)]);
        guard.store_final_boxes(vec![OpRef::int_op(2)]);
        let ops = vec![int_eq, guard, Op::new(OpCode::Finish, &[])];

        let (result, consts) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        let same = result
            .iter()
            .find(|o| o.opcode == OpCode::SameAsI)
            .expect("SAME_AS_I must be emitted for GUARD_FALSE merge");
        let const_ref = same.args[0];
        assert_eq!(
            consts[&const_ref.raw()],
            1,
            "GUARD_FALSE hoists SAME_AS_I(1) per rewrite.py:463",
        );
    }

    #[test]
    fn test_guard_always_fails_lowers_to_same_as_guard_value() {
        // rewrite.py:419-425: GUARD_ALWAYS_FAILS ⇒ SAME_AS_I(0) +
        // GUARD_VALUE(same_as, 1). Failargs are propagated via
        // copy_and_change.
        let rw = make_rewriter();
        let mut guard = Op::new(OpCode::GuardAlwaysFails, &[]);
        guard.store_final_boxes(vec![OpRef::int_op(10), OpRef::int_op(11)]);
        let ops = vec![guard, Op::new(OpCode::Finish, &[])];

        let (result, consts) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        assert!(
            result.iter().all(|o| o.opcode != OpCode::GuardAlwaysFails),
            "GUARD_ALWAYS_FAILS is lowered"
        );
        let same = result
            .iter()
            .find(|o| o.opcode == OpCode::SameAsI)
            .expect("SAME_AS_I is emitted");
        let gv = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardValue)
            .expect("GuardValue replaces GuardAlwaysFails");
        assert_eq!(gv.args[0], same.pos);
        assert_eq!(
            consts[&gv.args[1].raw()],
            1,
            "GuardValue checks against ConstInt(1)",
        );
        assert_eq!(
            consts[&same.args[0].raw()],
            0,
            "SAME_AS_I uses ConstInt(0) per rewrite.py:421",
        );
        let gv_fa = gv.fail_args.as_ref().expect("GuardValue inherits failargs");
        assert_eq!(gv_fa.as_slice(), &[OpRef::int_op(10), OpRef::int_op(11)]);
    }

    #[test]
    fn test_comparison_guard_mismatch_is_passthrough() {
        // Guard that does NOT test the previous op's result: merge does
        // not fire, no SAME_AS_I is emitted.
        let rw = make_rewriter();
        let mut int_lt = Op::new(OpCode::IntLt, &[OpRef::int_op(0), OpRef::int_op(1)]);
        int_lt.pos = OpRef::int_op(2);
        // GuardTrue reads some unrelated OpRef::ref_op(5), not OpRef::ref_op(2).
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef::int_op(5)]);
        guard.store_final_boxes(vec![OpRef::int_op(0), OpRef::int_op(1)]);
        let ops = vec![int_lt, guard, Op::new(OpCode::Finish, &[])];

        let result = rw.rewrite_for_gc(&ops);
        assert!(
            result.iter().all(|o| o.opcode != OpCode::SameAsI),
            "no merge → no hoisted SAME_AS_I"
        );
    }

    // ── Pending zero flush tests ──

    /// Helper: build a constants map mapping `key` → `value` for tests
    /// that need the rewriter's resolve_constant to find a length.
    fn const_pool(entries: &[(u32, i64)]) -> HashMap<u32, i64> {
        entries.iter().copied().collect()
    }

    #[test]
    fn test_pending_zero_fully_initialized() {
        // NEW_ARRAY_CLEAR(3) + SET[0] + SET[1] + SET[2] → ZERO_ARRAY emitted
        // with length=0 (RPython rewrite.py:754 "may be ConstInt(0)").
        // rewrite.py:514-518 consider_setarrayitem_gc requires the index
        // to be `ConstInt` (`index_box.is_constant()` / `getint()`); the
        // pyre equivalent is an entry in the constant pool. OpRefs 10/11/12
        // hold the literal indices 0/1/2 so `resolve_constant` returns the
        // item number.
        //
        // malloc_zero_filled=false exercises the `clear_varsize_gc_fields`
        // path (rewrite.py:521) that actually emits ZERO_ARRAY.
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let mut new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[OpRef::int_op(3)],
            array_descr_int(),
        );
        new_array.pos = OpRef::ref_op(0);
        let constants = const_pool(&[(3, 3), (10, 0), (11, 1), (12, 2)]);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(10), OpRef::int_op(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(11), OpRef::int_op(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(12), OpRef::int_op(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, out_consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        // All indices were SET, so the in-place ZERO_ARRAY is rewritten
        // to byte_length 0 — backend treats it as a no-op.
        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "ZERO_ARRAY stays in place per parity");
        assert_eq!(
            out_consts[&zeros[0].args[2].raw()],
            0,
            "byte length must be 0"
        );
    }

    #[test]
    fn test_pending_zero_partially_initialized() {
        // NEW_ARRAY_CLEAR(4) + SET[0] + SET[1] → ZERO_ARRAY trimmed to
        // start=2 items, length=2 items → byte_start=8, byte_len=8.
        // Index OpRefs 10/11 are ConstInt 0/1 per rewrite.py:514-518.
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let mut new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[OpRef::int_op(4)],
            array_descr_int(),
        );
        new_array.pos = OpRef::ref_op(0);
        let constants = const_pool(&[(4, 4), (10, 0), (11, 1)]);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(10), OpRef::int_op(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(11), OpRef::int_op(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, out_consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "should emit exactly one ZERO_ARRAY");
        // item_size = 4 (array_descr_int), so start_items=2 → 8 bytes,
        // length_items=2 → 8 bytes.
        assert_eq!(out_consts[&zeros[0].args[1].raw()], 8, "byte start");
        assert_eq!(out_consts[&zeros[0].args[2].raw()], 8, "byte length");
        assert_eq!(out_consts[&zeros[0].args[3].raw()], 1, "scale arg(3) is 1");
        assert_eq!(out_consts[&zeros[0].args[4].raw()], 1, "scale arg(4) is 1");
    }

    #[test]
    fn test_pending_zero_flushed_at_guard() {
        // Guard forces pending zero flush even if no indices were SET.
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let mut new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[OpRef::int_op(3)],
            array_descr_int(),
        );
        new_array.pos = OpRef::ref_op(0);
        let constants = const_pool(&[(3, 3)]);

        let ops = vec![
            new_array,
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(50)]),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, out_consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        // The ZERO_ARRAY should appear before the guard.
        let zero_idx = result.iter().position(|o| o.opcode == OpCode::ZeroArray);
        let guard_idx = result.iter().position(|o| o.opcode == OpCode::GuardTrue);
        assert!(zero_idx.is_some(), "ZERO_ARRAY should be emitted");
        assert!(guard_idx.is_some(), "GuardTrue should be present");
        assert!(
            zero_idx.unwrap() < guard_idx.unwrap(),
            "ZERO_ARRAY should come before GuardTrue"
        );

        let zero = result
            .iter()
            .find(|o| o.opcode == OpCode::ZeroArray)
            .unwrap();
        // No SETs, length=3 items × 4 bytes/item = 12 bytes.
        assert_eq!(out_consts[&zero.args[1].raw()], 0, "byte start");
        assert_eq!(out_consts[&zero.args[2].raw()], 12, "byte length");
    }

    #[test]
    fn test_pending_zero_gap_in_middle() {
        // NEW_ARRAY_CLEAR(5) + SET[0] + SET[2] + SET[4] — RPython does
        // trim-from-both-ends only (no middle splitting): start=1 (skip
        // index 0), stop=4 (skip index 4), length=3.  Index 2 falls
        // inside the zero range and is re-zeroed before the SET.
        // Index OpRefs 10/12/14 hold ConstInt 0/2/4 per rewrite.py:514-518.
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let mut new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[OpRef::int_op(5)],
            array_descr_int(),
        );
        new_array.pos = OpRef::ref_op(0);
        let constants = const_pool(&[(5, 5), (10, 0), (12, 2), (14, 4)]);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(10), OpRef::int_op(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(12), OpRef::int_op(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(14), OpRef::int_op(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, out_consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "rewrite.py:719 emits one ZERO_ARRAY");
        // start_items=1 → 4 bytes, length_items=3 → 12 bytes.
        assert_eq!(out_consts[&zeros[0].args[1].raw()], 4, "byte start");
        assert_eq!(out_consts[&zeros[0].args[2].raw()], 12, "byte length");
    }

    #[test]
    fn test_pending_zero_no_clear() {
        // Plain NEW_ARRAY (not CLEAR) should NOT produce any ZERO_ARRAY.
        let rw = make_rewriter();
        let mut new_array =
            Op::with_descr(OpCode::NewArray, &[OpRef::int_op(3)], array_descr_int());
        new_array.pos = OpRef::ref_op(0);

        let ops = vec![new_array, Op::new(OpCode::Finish, &[])];

        let result = rw.rewrite_for_gc(&ops);

        let zero_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .count();
        assert_eq!(
            zero_count, 0,
            "plain NEW_ARRAY should not produce ZERO_ARRAY"
        );
    }

    // ── Test: gen_write_barrier_array LARGE threshold ──

    fn make_rewriter_with_cards() -> GcRewriterImpl {
        GcRewriterImpl {
            nursery_free_addr: 0x1000,
            nursery_top_addr: 0x2000,
            max_nursery_size: 4096,
            wb_descr: WriteBarrierDescr {
                jit_wb_if_flag: 1,
                jit_wb_if_flag_byteofs: 0,
                jit_wb_if_flag_singlebyte: 1,
                jit_wb_cards_set: 0x40,
                jit_wb_card_page_shift: 7,
                jit_wb_cards_set_byteofs: 0,
                jit_wb_cards_set_singlebyte: 0x40,
            },
            jitframe_info: None,
            constant_types: HashMap::new(),
            call_assembler_callee_locs: None,
            load_supported_factors: &[1],
            supports_load_effective_address: true,
            malloc_zero_filled: true,
            memcpy_fn: majit_ir::memcpy_fn_addr(),
            memcpy_descr: majit_ir::make_memcpy_calldescr(),
            str_descr: str_array_descr(),
            unicode_descr: unicode_array_descr(),
            str_hash_descr: hash_field_descr(),
            unicode_hash_descr: hash_field_descr(),
            // gc.py:33-37 `fielddescr_vtable`. Test fixtures always
            // install a Some so the existing test_new_with_vtable
            // continues to exercise the typeptr stamping path.
            fielddescr_vtable: Some(majit_ir::make_vtable_field_descr()),
            // gc.py:394 `fielddescr_tid`. Test fixtures install a Some
            // (framework-GC mode) so handle_new tests exercise the tid
            // header stamping path.
            fielddescr_tid: Some(majit_ir::make_tid_field_descr()),
            malloc_array_fn: TEST_MALLOC_ARRAY_FN,
            malloc_array_nonstandard_fn: TEST_MALLOC_ARRAY_NONSTANDARD_FN,
            malloc_str_fn: TEST_MALLOC_STR_FN,
            malloc_unicode_fn: TEST_MALLOC_UNICODE_FN,
            malloc_big_fixedsize_fn: TEST_MALLOC_BIG_FIXEDSIZE_FN,
            malloc_array_descr: majit_ir::make_malloc_array_calldescr(),
            malloc_array_nonstandard_descr: majit_ir::make_malloc_array_nonstandard_calldescr(),
            malloc_str_descr: majit_ir::make_malloc_str_calldescr(),
            malloc_unicode_descr: majit_ir::make_malloc_unicode_calldescr(),
            malloc_big_fixedsize_descr: majit_ir::make_malloc_big_fixedsize_calldescr(),
            standard_array_basesize: TEST_STANDARD_ARRAY_BASESIZE,
            standard_array_length_ofs: TEST_STANDARD_ARRAY_LENGTH_OFS,
        }
    }

    #[test]
    fn test_setarrayitem_gc_after_const_alloc_no_wb() {
        // rewrite.py:910-911 — gen_malloc_nursery's tail
        // `remember_write_barrier(op)` records the fresh nursery alloc
        // in wb_applied; rewrite.py:937-938 `if not write_barrier_applied
        // (val): ...` then short-circuits the WB on the immediate
        // SETARRAYITEM_GC.  Both length=10 (< LARGE) and length=200
        // (>= LARGE) take handle_new_array path #2 (constant-size
        // nursery), so neither gen_write_barrier_array branch fires
        // (LARGE threshold logic stays gated behind path #4 fallback,
        // not yet ported in pyre).
        for &num_elem in &[10_i64, 200_i64] {
            let rw = make_rewriter_with_cards();
            let len_ref = OpRef::int_op(10_000);
            let mut constants = HashMap::new();
            constants.insert(10_000, num_elem);
            let mut new_array =
                Op::with_descr(OpCode::NewArrayClear, &[len_ref], array_descr_ref());
            new_array.pos = OpRef::ref_op(0);
            let ops = vec![
                new_array,
                Op::with_descr(
                    OpCode::SetarrayitemGc,
                    &[OpRef::ref_op(0), OpRef::int_op(1), OpRef::ref_op(2)],
                    array_descr_ref(),
                ),
                Op::new(OpCode::Finish, &[]),
            ];
            let (result, _) = rw.rewrite_for_gc_with_constants(&ops, &constants);
            let wb = result
                .iter()
                .filter(|o| o.opcode == OpCode::CondCallGcWb)
                .count();
            let wb_arr = result
                .iter()
                .filter(|o| o.opcode == OpCode::CondCallGcWbArray)
                .count();
            assert_eq!(
                wb, 0,
                "fresh alloc is wb_applied → no regular WB (num_elem={num_elem})"
            );
            assert_eq!(
                wb_arr, 0,
                "fresh alloc is wb_applied → no WB_ARRAY (num_elem={num_elem})"
            );
        }
    }

    #[test]
    fn test_setarrayitem_gc_unknown_length_uses_wb_array() {
        // rewrite.py:962: unknown length defaults to LARGE → WB_ARRAY.
        let rw = make_rewriter_with_cards();
        // No NEW_ARRAY, so v_base has unknown length.
        let ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(1), OpRef::ref_op(2)],
                array_descr_ref(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = rw.rewrite_for_gc(&ops);
        let wb_arr = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(wb_arr, 1, "unknown length should get WB_ARRAY");
    }

    #[test]
    fn test_setarrayitem_gc_int_value_no_wb() {
        // rewrite.py:940: v.type != 'r' → no write barrier at all.
        let rw = make_rewriter_with_cards();
        let ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(0), OpRef::int_op(1), OpRef::ref_op(2)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = rw.rewrite_for_gc(&ops);
        let any_wb = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb || o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(any_wb, 0, "int store should not produce any WB");
    }

    fn str_array_descr() -> DescrRef {
        // rstr.py:1226 `extra_item_after_alloc=True` → base_size = 17
        // (16-byte GC header surrogate + 1 trailing null), itemsize = 1.
        Arc::new(TestArrayDescr {
            base_size: 17,
            item_size: 1,
            type_id: 7,
            item_type: Type::Int,
            len_descr: Some(Arc::new(TestFieldDescr {
                offset: std::mem::size_of::<usize>(),
                field_size: std::mem::size_of::<usize>(),
                field_type: Type::Int,
            })),
        })
    }

    fn unicode_array_descr() -> DescrRef {
        // rstr.py UNICODE has no extra_item_after_alloc; itemsize = 4.
        Arc::new(TestArrayDescr {
            base_size: 16,
            item_size: 4,
            type_id: 8,
            item_type: Type::Int,
            len_descr: Some(Arc::new(TestFieldDescr {
                offset: std::mem::size_of::<usize>(),
                field_size: std::mem::size_of::<usize>(),
                field_type: Type::Int,
            })),
        })
    }

    /// Test stand-in for `gc_ll_descr.{str,unicode}_hash_descr`
    /// (gc.py:48-49): the `hash` field of rstr.STR / rstr.UNICODE
    /// lives at offset 0 with `Signed` size.
    fn hash_field_descr() -> DescrRef {
        Arc::new(TestFieldDescr {
            offset: 0,
            field_size: std::mem::size_of::<usize>(),
            field_type: Type::Int,
        })
    }

    // ── COPYSTRCONTENT → LEA × 2 + CALL_N(memcpy) ──
    //
    // rpython/jit/backend/llsupport/test/test_rewrite.py:1460-1469
    // `test_rewrite_copystrcontents`.
    #[test]
    fn test_rewrite_copystrcontents() {
        let rw = make_rewriter();
        // [p0, p1, i0, i1, i_len]
        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        let i0 = OpRef::int_op(2);
        let i1 = OpRef::int_op(3);
        let i_len = OpRef::int_op(4);
        let ops = vec![Op::with_descr(
            OpCode::Copystrcontent,
            &[p0, p1, i0, i1, i_len],
            str_array_descr(),
        )];

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        assert_eq!(
            result.len(),
            3,
            "expected LEA + LEA + CALL_N, got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        // i_src = load_effective_address(p0, i0, basesize-1=16, shift=0)
        assert_eq!(result[0].opcode, OpCode::LoadEffectiveAddress);
        assert_eq!(result[0].arg(0), p0);
        assert_eq!(result[0].arg(1), i0);
        assert_eq!(constants[&result[0].arg(2).raw()], 16);
        assert_eq!(constants[&result[0].arg(3).raw()], 0);
        // i_dst = load_effective_address(p1, i1, 16, 0)
        assert_eq!(result[1].opcode, OpCode::LoadEffectiveAddress);
        assert_eq!(result[1].arg(0), p1);
        assert_eq!(result[1].arg(1), i1);
        assert_eq!(constants[&result[1].arg(2).raw()], 16);
        assert_eq!(constants[&result[1].arg(3).raw()], 0);
        // call_n(memcpy_fn, i_dst, i_src, i_len)
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[2].arg(1), result[1].pos); // dst
        assert_eq!(result[2].arg(2), result[0].pos); // src
        assert_eq!(result[2].arg(3), i_len);
        assert!(result[2].descr.is_some(), "CALL_N must carry memcpy_descr");
    }

    // ── COPYUNICODECONTENT with non-constant length → LEA × 2 + LSHIFT + CALL_N ──
    #[test]
    fn test_rewrite_copyunicodecontents_dynamic_length() {
        let rw = make_rewriter();
        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        let i0 = OpRef::int_op(2);
        let i1 = OpRef::int_op(3);
        let i_len = OpRef::int_op(4);
        let ops = vec![Op::with_descr(
            OpCode::Copyunicodecontent,
            &[p0, p1, i0, i1, i_len],
            unicode_array_descr(),
        )];

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // Expect: LEA, LEA, INT_LSHIFT(i_len, 2), CALL_N
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::LoadEffectiveAddress);
        // basesize=16, shift=2 (itemsize=4 → itemscale=2)
        assert_eq!(constants[&result[0].arg(2).raw()], 16);
        assert_eq!(constants[&result[0].arg(3).raw()], 2);
        assert_eq!(result[1].opcode, OpCode::LoadEffectiveAddress);
        assert_eq!(result[2].opcode, OpCode::IntLshift);
        assert_eq!(result[2].arg(0), i_len);
        assert_eq!(constants[&result[2].arg(1).raw()], 2);
        assert_eq!(result[3].opcode, OpCode::CallN);
        assert_eq!(result[3].arg(3), result[2].pos);
    }

    // ── COPYSTRCONTENT without LEA → INT_ADD × 4 + CALL_N ──
    //
    // rpython/jit/backend/llsupport/test/test_rewrite.py:1471-1483
    // `test_rewrite_copystrcontents_without_load_effective_address`.
    #[test]
    fn test_rewrite_copystrcontents_without_load_effective_address() {
        let mut rw = make_rewriter();
        rw.supports_load_effective_address = false;
        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        let i0 = OpRef::int_op(2);
        let i1 = OpRef::int_op(3);
        let i_len = OpRef::int_op(4);
        let ops = vec![Op::with_descr(
            OpCode::Copystrcontent,
            &[p0, p1, i0, i1, i_len],
            str_array_descr(),
        )];

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // Expect (itemscale=0 so no INT_LSHIFT):
        //   i2b = int_add(p0, i0)
        //   i2  = int_add(i2b, basesize-1)
        //   i3b = int_add(p1, i1)
        //   i3  = int_add(i3b, basesize-1)
        //   call_n(memcpy_fn, i3, i2, i_len, descr=memcpy_descr)
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[0].arg(0), p0);
        assert_eq!(result[0].arg(1), i0);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[1].arg(0), result[0].pos);
        assert_eq!(constants[&result[1].arg(1).raw()], 16); // str_basesize - 1
        assert_eq!(result[2].opcode, OpCode::IntAdd);
        assert_eq!(result[2].arg(0), p1);
        assert_eq!(result[2].arg(1), i1);
        assert_eq!(result[3].opcode, OpCode::IntAdd);
        assert_eq!(result[3].arg(0), result[2].pos);
        assert_eq!(constants[&result[3].arg(1).raw()], 16);
        assert_eq!(result[4].opcode, OpCode::CallN);
        assert_eq!(result[4].arg(1), result[3].pos); // dst
        assert_eq!(result[4].arg(2), result[1].pos); // src
        assert_eq!(result[4].arg(3), i_len);
        assert!(result[4].descr.is_some(), "CALL_N must carry memcpy_descr");
    }

    // ── COPYUNICODECONTENT without LEA → LSHIFT + INT_ADD + INT_ADD per side + LSHIFT(len) + CALL_N ──
    //
    // rpython/jit/backend/llsupport/test/test_rewrite.py:1497-1512
    // `test_rewrite_copyunicodecontents_without_load_effective_address`.
    #[test]
    fn test_rewrite_copyunicodecontents_without_load_effective_address() {
        let mut rw = make_rewriter();
        rw.supports_load_effective_address = false;
        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        let i0 = OpRef::int_op(2);
        let i1 = OpRef::int_op(3);
        let i_len = OpRef::int_op(4);
        let ops = vec![Op::with_descr(
            OpCode::Copyunicodecontent,
            &[p0, p1, i0, i1, i_len],
            unicode_array_descr(),
        )];

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // Expect (itemscale=2):
        //   i0s = int_lshift(i0, 2)
        //   i2b = int_add(p0, i0s)
        //   i2  = int_add(i2b, basesize)
        //   i1s = int_lshift(i1, 2)
        //   i3b = int_add(p1, i1s)
        //   i3  = int_add(i3b, basesize)
        //   i4  = int_lshift(i_len, 2)
        //   call_n(memcpy_fn, i3, i2, i4, descr=memcpy_descr)
        assert_eq!(result.len(), 8);
        assert_eq!(result[0].opcode, OpCode::IntLshift);
        assert_eq!(result[0].arg(0), i0);
        assert_eq!(constants[&result[0].arg(1).raw()], 2);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[1].arg(0), p0);
        assert_eq!(result[1].arg(1), result[0].pos);
        assert_eq!(result[2].opcode, OpCode::IntAdd);
        assert_eq!(result[2].arg(0), result[1].pos);
        assert_eq!(constants[&result[2].arg(1).raw()], 16);
        assert_eq!(result[3].opcode, OpCode::IntLshift);
        assert_eq!(result[3].arg(0), i1);
        assert_eq!(constants[&result[3].arg(1).raw()], 2);
        assert_eq!(result[4].opcode, OpCode::IntAdd);
        assert_eq!(result[4].arg(0), p1);
        assert_eq!(result[4].arg(1), result[3].pos);
        assert_eq!(result[5].opcode, OpCode::IntAdd);
        assert_eq!(result[5].arg(0), result[4].pos);
        assert_eq!(constants[&result[5].arg(1).raw()], 16);
        assert_eq!(result[6].opcode, OpCode::IntLshift);
        assert_eq!(result[6].arg(0), i_len);
        assert_eq!(constants[&result[6].arg(1).raw()], 2);
        assert_eq!(result[7].opcode, OpCode::CallN);
        assert_eq!(result[7].arg(1), result[5].pos); // dst
        assert_eq!(result[7].arg(2), result[2].pos); // src
        assert_eq!(result[7].arg(3), result[6].pos);
    }
}
