/// JIT optimization pipeline.
///
/// Translated from rpython/jit/metainterp/optimizeopt/.
///
/// The optimizer chains multiple passes, each implementing the Optimization trait.
/// Operations flow through the chain: IntBounds → Rewrite → Virtualize → String →
/// Pure → Guard → Simplify → Heap (configurable).
pub mod bridgeopt;
pub mod dependency;
pub mod earlyforce;
pub mod guard;
pub mod heap;
pub mod info;
pub mod intbounds;
pub mod intdiv;
pub mod intutils;
// optimize module is at crate::optimize (RPython: metainterp/optimize.py)
pub mod optimizer;
pub mod pure;
pub mod renamer;
pub mod rewrite;
pub mod schedule;
pub mod shortpreamble;
pub mod simplify;
pub mod unroll;
pub mod vector;
pub mod version;
pub mod virtualize;
pub mod virtualstate;
pub mod vstring;
// walkvirtual moved to crate::walkvirtual (RPython: metainterp/walkvirtual.py)

use std::collections::{HashMap, HashSet};

use crate::optimizeopt::intutils::IntBound;
use crate::resume::SnapshotBox;
use info::{EnsuredPtrInfo, PtrInfo};
use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Type, Value};
use std::collections::VecDeque;

pub type SnapshotBoxes = Vec<Option<Vec<SnapshotBox>>>;
pub type SnapshotFrameSizes = Vec<Option<Vec<usize>>>;
pub type SnapshotFramePcs = Vec<Option<Vec<(i32, i32)>>>;

pub(crate) fn snapshot_get<T>(store: &[Option<T>], pos: i32) -> Option<&T> {
    if pos < 0 {
        return None;
    }
    store.get(pos as usize).and_then(Option::as_ref)
}

pub(crate) fn snapshot_contains<T>(store: &[Option<T>], pos: i32) -> bool {
    snapshot_get(store, pos).is_some()
}

pub(crate) fn snapshot_insert<T>(store: &mut Vec<Option<T>>, pos: i32, value: T) {
    assert!(pos >= 0, "snapshot position must be non-negative");
    let idx = pos as usize;
    if store.len() <= idx {
        store.resize_with(idx + 1, || None);
    }
    store[idx] = Some(value);
}

pub(crate) fn next_snapshot_pos<T>(store: &[Option<T>]) -> i32 {
    store.len() as i32
}

pub(crate) fn majit_log_enabled() -> bool {
    std::env::var_os("MAJIT_LOG").is_some()
}

/// info.py:13-15 INFO_NULL / INFO_NONNULL / INFO_UNKNOWN constants.
///
/// Used by `PtrInfo::getnullness` and `IntBound::getnullness` to
/// report whether a slot is known null, known non-null, or unknown.
/// Matches RPython's integer enum values exactly so that majit code
/// can be ported line-by-line from upstream `optimizer.py:127` /
/// `rewrite.py:496-503` `_optimize_nullness` switches.
pub const INFO_NULL: i8 = 0;
pub const INFO_NONNULL: i8 = 1;
pub const INFO_UNKNOWN: i8 = 2;

/// Create a ResumeAtPositionDescr for optimizer-generated guards.
///
/// Delegates to compile::make_resume_at_position_descr which wraps a
/// real ResumeGuardDescr — clone_descr() preserves resume data (RPython
/// ResumeAtPositionDescr is a plain subclass of ResumeGuardDescr).
pub fn make_resume_at_position_descr() -> DescrRef {
    crate::compile::make_resume_at_position_descr()
}

/// optimizer.py:47-54 OptimizationResult: result of an optimization pass.
#[derive(Debug)]
pub enum OptimizationResult {
    /// Emit this operation (possibly modified).
    Emit(Op),
    /// Replace with a different operation; continue with the next pass.
    Replace(Op),
    /// optimizer.py:567 `send_extra_operation(newop, opt=None)` — re-dispatch
    /// the new op from the first optimization, dropping the original.
    /// autogenintrules.py:54-55 uses this pattern for every rewrite-style
    /// rule so that chained OptIntBounds rules (add_zero, int_is_zero, …)
    /// fire on the rewritten op.
    Restart(Op),
    /// Remove the operation entirely.
    Remove,
    /// Pass the operation to the next pass unchanged.
    PassOn,
    /// rewrite.py:406 — a guard was proven to always fail; abort the trace.
    /// RPython raises `InvalidLoop`; the optimizer catches it and discards
    /// the loop or bridge.
    InvalidLoop,
}

/// optimizer.py:47-54: deferred postprocess for GUARD_CLASS/GUARD_NONNULL_CLASS.
/// RPython's postprocess_GUARD_CLASS runs after the guard is emitted to
/// _newoperations. In majit, recorded here by rewrite and executed by
/// emit_operation.
#[derive(Debug)]
pub struct PendingGuardClassPostprocess {
    pub obj: majit_ir::OpRef,
    pub class_val: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ImportedShortPureArg {
    OpRef(OpRef),
    /// Const arg with source OpRef for matching in force_preamble_op.
    /// RPython: Const Box has identity; get_box_replacement returns itself.
    Const(Value, OpRef),
}

#[derive(Clone, Debug)]
pub struct ImportedShortPureOp {
    pub opcode: OpCode,
    pub descr: Option<DescrRef>,
    pub args: Vec<ImportedShortPureArg>,
    pub result: OpRef,
    /// RPython: PreambleOp stored in pure cache. Used by force_op_from_preamble.
    pub pop: crate::optimizeopt::info::PreambleOp,
}

impl ImportedShortPureOp {
    /// Construct with auto-generated PreambleOp from fields.
    pub fn new(
        opcode: OpCode,
        descr: Option<DescrRef>,
        args: Vec<ImportedShortPureArg>,
        result: OpRef,
        source: OpRef,
        invented_name: bool,
    ) -> Self {
        let replay_args: Vec<OpRef> = args
            .iter()
            .map(|a| match a {
                ImportedShortPureArg::OpRef(r) => *r,
                ImportedShortPureArg::Const(_, src) => *src,
            })
            .collect();
        let mut replay = majit_ir::Op::new(opcode, &replay_args);
        // shortpreamble.py:112-126 PureOp.produce_op constructs TWO distinct
        // RPython Op objects:
        //
        //   * `op` — the alt identifier. For invented_name it is
        //     `self.orig_op.copy_and_change(...).set_forwarded(self.res)`,
        //     a freshly-allocated Op with its own `_forwarded` slot.
        //   * `preamble_op` — the replay Op passed in from `add_op_to_short`.
        //     Its `_forwarded` slot is seeded with the exported info by
        //     `ShortPreambleBuilder.__init__` (shortpreamble.py:425).
        //
        // The two `_forwarded` slots live on two different RPython Op
        // objects, so `op.set_forwarded(self.res)` and
        // `preamble_op.set_forwarded(info)` never collide.
        //
        // pyre's flat-OpRef model has only one slot per OpRef. To preserve
        // PyPy parity we must allocate TWO different OpRefs for the two
        // identities when they would collide:
        //
        //   * non-invented Pure: `op = self.res`. PyPy does NOT install a
        //     forwarding on `op`, so the slot at `source` is free to hold
        //     `info`. We can leave `replay.pos = source` — both objects
        //     point at one slot, which is allowed because only `info`
        //     occupies it.
        //   * invented Pure: PyPy installs `op.set_forwarded(self.res)` on
        //     the alt. In pyre, `produce_pure` calls
        //     `replace_op(source, canonical)` (shortpreamble.rs:1279) which
        //     overwrites the source box's `_forwarded` slot with
        //     `Forwarded::Box(canonical_box)`.
        //     If `replay.pos` also pointed at `source`, the alt's
        //     replacement chain and the replay's info would share one slot
        //     and the info would be lost. We move `replay.pos` to the
        //     pre-allocated body-visible OpRef (`result`) so it has its
        //     own slot.
        replay.pos = if invented_name { result } else { source };
        replay.descr = descr.clone();
        // shortpreamble.py:116-120: pop.op = self.orig_op.copy_and_change(...)
        // for invented (the alt identifier) or self.res for non-invented.
        // pyre's `source` IS the alt identifier for invented (the synthetic
        // alias allocated by the compound-dedup pass at
        // shortpreamble.rs:478-491) and IS self.res for non-invented.
        let pop_op = source;
        ImportedShortPureOp {
            opcode,
            descr,
            args,
            result,
            pop: crate::optimizeopt::info::PreambleOp {
                op: pop_op,
                invented_name,
                preamble_op: replay,
            },
        }
    }
}

impl PartialEq for ImportedShortPureOp {
    fn eq(&self, other: &Self) -> bool {
        self.opcode == other.opcode
            && self.descr.as_ref().map(|d| d.index()) == other.descr.as_ref().map(|d| d.index())
            && self.args == other.args
            && self.result == other.result
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedShortAlias {
    pub result: OpRef,
    pub same_as_source: OpRef,
    pub same_as_opcode: OpCode,
}

/// optimizer.py:787-789: constant_fold — allocate an immutable object at
/// compile time when all fields are constants. The callback receives the
/// SizeDescr size_bytes, and returns a raw pointer (GcRef) to freshly
/// allocated memory. The optimizer writes field values directly.
pub type ConstantFoldAllocFn = Box<dyn Fn(usize) -> majit_ir::GcRef>;

/// Re-export of `info::StringLengthResolver` for callers that import
/// `optimizeopt::StringLengthResolver`. The runtime hook signature is
/// `Arc<dyn Fn(GcRef, u8) -> Option<i64> + Send + Sync>`. See
/// `info::EnsuredPtrInfo::getlenbound` for the consumer side.
pub use crate::optimizeopt::info::StringLengthResolver;
pub use crate::optimizeopt::info::{StringConstantAllocator, StringContentResolver};

/// Context provided to optimization passes.
///
/// Holds the shared state that passes read from and write to.
pub struct OptContext {
    /// The output operation list being built.
    pub new_operations: Vec<Op>,
    /// Constants known at optimization time for operation-namespace OpRefs.
    pub constants: Vec<Option<Value>>,
    /// Constants for constant-namespace OpRefs, keyed by const_index().
    pub const_pool: HashMap<u32, Value>,
    /// RPython: mapping dict in inline_short_preamble — separate from _forwarded.
    /// Maps Phase 1 source OpRefs to Phase 2 short arg OpRefs.
    /// Number of input arguments, used to offset emitted op positions
    /// so that variable indices don't collide with input arg indices.
    num_inputs: u32,
    /// opencoder.py:259-267 inputarg base in the OpRef namespace.
    ///
    /// RPython lets each TraceIterator allocate fresh inputarg boxes whose
    /// Python identity (`is`) distinguishes them from any other phase's
    /// boxes; majit needs to encode that as a numeric offset because OpRef
    /// IS the identity. Phase 1 uses `inputarg_base = 0` (legacy positional
    /// layout); Phase 2/bridges shift inputarg OpRefs above the parent
    /// trace's high water mark by setting `inputarg_base = parent_high_water`.
    /// `reserve_pos` floors `next_pos` at `inputarg_base + num_inputs +
    /// new_operations.len()` so freshly emitted ops never collide with
    /// inputargs or imported high-water marks.
    pub(crate) inputarg_base: u32,
    /// Next unique op position for newly emitted or queued extra operations.
    pub(crate) next_pos: u32,
    /// Zero-based counter for constant-namespace OpRef allocation.
    pub(crate) next_const_idx: u32,
    /// RPython emit_extra(op, emit=False) parity: ops queued to be
    /// processed starting from a specific pass index (skipping earlier passes).
    /// Used by heap's force_lazy_set to route ops through remaining passes
    /// without re-entering the heap pass itself.
    pub(crate) extra_operations_after: VecDeque<(usize, Op)>,
    /// optimizer.py:47-54: deferred postprocess for GUARD_CLASS.
    /// Set by rewrite pass, executed by emit_operation after the guard
    /// is added to new_operations (matching RPython's callback pattern).
    pub(crate) pending_guard_class_postprocess: Option<PendingGuardClassPostprocess>,
    /// rewrite.py:282: postprocess_GUARD_NONNULL → mark_last_guard.
    /// Deferred until emit adds the guard to new_operations.
    pub(crate) pending_mark_last_guard: Option<OpRef>,
    /// virtualize.py:84-90 postprocess_FINISH queues the stashed
    /// GUARD_NOT_FORCED_2 here so the outer optimizer can insert it at
    /// `len(_newoperations) - 1` with full `store_final_boxes_in_guard`
    /// semantics.
    pub(crate) pending_finish_guard_postprocess: Option<Op>,
    // ptr_info merged into forwarded (Forwarded::Info variant)
    //
    // RPython parity: per-OpRef IntBound storage lives ENTIRELY on
    // `box._forwarded` (Forwarded::IntBound), accessed via getintbound /
    // setintbound. The previous `int_lower_bounds` (heap.py array length
    // hint), `int_bounds` (per-pass snapshot), and `imported_int_bounds`
    // (preamble import) maps were a majit-only divergence from RPython's
    // single source of truth. They've been merged into Forwarded::IntBound
    // at write time so reads naturally see all sources via getintbound.
    /// RPython shortpreamble.py / pure.py: imported pure-operation results from
    /// the preamble. Phase 2 uses these as cross-iteration CSE facts.
    pub imported_short_pure_ops: Vec<ImportedShortPureOp>,
    /// (base_len, short_args): virtual field values start at base_len
    /// within short_args. Used by install_imported_virtuals.
    pub imported_virtual_args: Option<(usize, Vec<OpRef>)>,
    /// RPython shortpreamble.py / rewrite.py: imported CALL_LOOPINVARIANT
    /// results keyed by constant function pointer.
    pub imported_loop_invariant_results: HashMap<i64, OpRef>,
    /// Phase 2 imported virtuals (from Phase 1 export). Used by
    /// store_final_boxes_in_guard to resolve NONE positions
    /// inherited from Phase 1 virtualization.
    pub imported_virtuals: Vec<crate::optimizeopt::optimizer::ImportedVirtual>,
    /// Phase 2 imported label args (OpRefs in Phase 2 namespace).
    pub imported_label_args: Option<Vec<OpRef>>,
    /// RPython shortpreamble.py: active phase-2 short preamble builder.
    /// Tracks which imported short facts are actually consumed by the body.
    pub imported_short_preamble_builder:
        Option<crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    /// RPython optimizer.py: quasi_immutable_deps — collected during optimization.
    /// (object_ptr, field_index) pairs identifying specific quasi-immutable
    /// slots the trace depends on. After compilation, per-slot watchers
    /// are registered.
    pub quasi_immutable_deps: HashSet<(u64, u32)>,
    /// info.py:716-721: ConstPtrInfo._get_info — const_infos stores
    /// StructPtrInfo for constant GC objects, keyed by pointer address.
    /// RPython: optheap.const_infos[ref] = StructPtrInfo(descr)
    pub const_infos: HashMap<usize, crate::optimizeopt::info::PtrInfo>,
    /// Dedup imported short fact uses so the builder stays in first-use order.
    imported_short_preamble_used: HashSet<OpRef>,
    /// unroll.py:37 / optimizer.py:354: potential_extra_ops[op] = preamble_op
    /// Populated by force_op_from_preamble, consumed by force_box.
    pub(crate) potential_extra_ops: HashMap<OpRef, crate::optimizeopt::info::PreambleOp>,
    /// RPython unroll.py: live ExtendedShortPreambleBuilder while replaying an
    /// existing target token's short preamble.
    active_short_preamble_producer:
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    /// RPython shortpreamble.py: pass-collected preamble producers aligned to
    /// the exported loop-header inputargs.
    pub exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// optimizer.py: `can_replace_guards` — disable guard replacement during
    /// bridge compilation. Defaults to true for preamble.
    pub can_replace_guards: bool,
    /// RPython optimizer.py: `patchguardop` — the last GUARD_FUTURE_CONDITION op.
    /// Used by unroll to attach resume data to extra guards from short preamble.
    pub patchguardop: Option<Op>,
    /// RPython optimizer.py: end_args after force_at_the_end_of_preamble().
    /// export_state() prefers this over a raw get_replacement() snapshot.
    pub preamble_end_args: Option<Vec<OpRef>>,
    /// Phase-2 loop-body mode from optimizer.skip_flush.
    /// RPython unroll.py relies on this distinction so virtualize can keep
    /// body-side allocations concrete when guard recovery cannot rebuild them.
    pub skip_flush_mode: bool,
    /// Index of the pass currently executing propagate_forward.
    /// Used by passes to call send_extra_operation_after(self_idx, ..)
    /// matching RPython's emit_extra(op, emit=False) which routes to
    /// self.next_optimization.
    pub current_pass_idx: usize,
    /// earlyforce.py:32: self.optimizer.optearlyforce = self
    /// Index of the OptEarlyForce pass in the pass chain.
    /// Used by force_at_the_end_of_preamble and force_box to route
    /// forced operations starting from earlyforce.next (= heap).
    pub optearlyforce_idx: usize,
    /// optimizer.py: pendingfields — deferred SetfieldGc/SetarrayitemGc ops
    /// where the stored value is virtual. Set by OptHeap.emitting_operation()
    /// before a guard, consumed by emit_operation() to encode into
    /// the guard's rd_pendingfields.
    pub pending_for_guard: Vec<Op>,
    /// optimizer.py: pure_from_args1 parity — reverse-pure relationships
    /// registered by rewrite pass (CAST_*, CONVERT_*) and consumed by pure pass.
    /// Each entry: (opcode, arg0, result) meaning pure(opcode, arg0) = result.
    pub pending_pure_from_args: Vec<(OpCode, OpRef, OpRef)>,
    /// optimizer.py: pure_from_args2 parity — binary reverse-pure relationships
    /// registered by rewrite pass (INSTANCE_PTR_EQ/NE swapped-args). Consumed
    /// by OptPure. Each entry: (opcode, arg0, arg1, result) meaning
    /// pure(opcode, arg0, arg1) = result.
    pub pending_pure_from_args2: Vec<(OpCode, OpRef, OpRef, OpRef)>,
    /// optimizer.py:787: constant_fold allocator callback.
    /// When set, the optimizer can fold immutable virtuals filled with
    /// constants into compile-time constant pointers (info.py:140-145).
    pub constant_fold_alloc: Option<ConstantFoldAllocFn>,
    /// info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` — runtime hook for
    /// constant byte-string / unicode-string length lookup. Set by the
    /// host runtime (pyre etc.) at OptContext construction time. When
    /// `None`, `EnsuredPtrInfo::getlenbound(Some(_))` falls back to
    /// `IntBound::nonnegative()`.
    pub string_length_resolver: Option<StringLengthResolver>,
    /// info.py:788-790 `ConstPtrInfo._unpack_str(mode)` — runtime hook for
    /// extracting character data from a constant string GcRef.
    pub string_content_resolver: Option<StringContentResolver>,
    /// history.py:377 `get_const_ptr_for_string(s)` — runtime hook for
    /// creating a constant string GcRef from char values (used by
    /// force_box constant-folding path, vstring.py:79-90).
    pub string_constant_alloc: Option<StringConstantAllocator>,
    /// True while optimizer.py:_emit_operation equivalent is forcing args
    /// just before final emission. In this phase, virtual forcing must emit
    /// directly into new_operations instead of re-entering the pass chain.
    pub in_final_emission: bool,
    /// effectinfo.py: CallInfoCollection — maps oopspec indices to
    /// (calldescr, func_ptr) pairs. Used by generate_modified_call
    /// (vstring.py:853) to emit specialized string comparison calls.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// resume.py parity: per-guard snapshot boxes from tracing time.
    /// Used by emit() to call store_final_boxes_in_guard inline (RPython
    /// calls this during optimization, not post-assembly).
    pub snapshot_boxes: SnapshotBoxes,
    /// Per-frame box counts for multi-frame snapshots.
    /// opencoder.py:819 capture_resumedata encodes multiple frames;
    /// this tracks the boundary between callee and caller sections.
    pub snapshot_frame_sizes: SnapshotFrameSizes,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: SnapshotBoxes,
    /// Per-guard virtualref boxes from tracing-time snapshots.
    /// resume.py:243-247 _number_boxes consumes vref_array as a section
    /// after vable_array. opencoder.py:767 records vref_boxes here.
    pub snapshot_vref_boxes: SnapshotBoxes,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: SnapshotFramePcs,
    /// ConstantPool type map for BoxEnv.is_const() during inline numbering.
    pub constant_types_for_numbering: HashMap<u32, majit_ir::Type>,
    /// compile.rs value_types parity: OpRef → Type for all defined values
    /// (inputargs + operation results). Used by store_final_boxes_in_guard
    /// to infer fail_arg types correctly for OpRefs from earlier phases.
    pub value_types: HashMap<u32, majit_ir::Type>,
    /// Inputarg types indexed by slot, mirroring `Optimizer.trace_inputarg_types`
    /// (recorder side `InputArg{Int,Ref,Float}.tp`). Slot `i` corresponds to
    /// `OpRef(inputarg_base + i)`. Populated in `setup_optimizations` so
    /// readers can resolve inputarg `box.type` (history.py:220 parity)
    /// without going through the `value_types` HashMap. Slice 0.5 prep —
    /// once all inputarg-type readers route through here, the inputarg
    /// fan-in into `value_types` (optimizer.rs:1816-1818) goes away.
    pub inputarg_types: Vec<majit_ir::Type>,
    /// Phase 1 emit ops carried into Phase 2's lookup surface.
    ///
    /// In RPython, a Box referenced cross-phase keeps its `.type` attribute
    /// because the Box object survives both phases (history.py:220 parity).
    /// Pyre's flat `OpRef(u32)` requires explicit carry: Phase 1's emit ops
    /// — at positions in `[num_inputs, phase2_inputarg_base)` — appear as
    /// `imported_label_args`, fail-arg sources, and `record_same_as` source
    /// arguments inside Phase 2, but never inside Phase 2's own `new_operations`.
    /// `op_at` falls back to this slice so `op.type_` stays the single source
    /// of truth for Phase 1 emit OpRef types.
    pub phase1_emit_ops: Vec<Op>,
    /// Epic H H-3.0b: per-position BoxRef pool for the input ops being
    /// optimized. `box_pool[i]` mirrors the `AbstractValue` for the
    /// trace position whose `OpRef::raw() == i` (recorder issuance order:
    /// inputargs first, then ops). Empty when the optimizer is invoked
    /// without an upstream recorder pool (tests, retrace helpers).
    ///
    /// RPython parity: PyPy's `_forwarded` slot lives on the `Box`
    /// object itself, so the optimizer just calls
    /// `box.set_forwarded(target)` directly. The Rust pool gives the
    /// optimizer the same per-position object access that RPython gets
    /// for free from object identity. PtrInfo / IntBound / forwarding
    /// state is carried by the BoxRef `_forwarded` slot.
    pub box_pool: crate::r#box::BoxPool,
    /// optimizer.py:644,679 _last_guard_op — index of the last guard in
    /// new_operations that had full resume data built. Consecutive guards
    /// share resume data via _copy_resume_data_from (ResumeGuardCopiedDescr).
    ///
    /// Production runs through `Optimizer::emit_operation`, which owns the
    /// guard chain via its own `last_guard_op_idx` (optimizer.rs:3584).
    /// This field tracks the chain for the standalone OptContext entry
    /// (unit tests that drive `OptContext::emit` without an `Optimizer`);
    /// `OptContext::emit` gates its guard handling on `!in_final_emission`
    /// to avoid duplicating Optimizer's bookkeeping.
    last_guard_idx: Option<usize>,
    /// Last rd_resume_position with a valid snapshot. Used as fallback
    /// for optimizer-created guards that can't share from a previous guard.
    /// resume.py parity: RPython guards always get a snapshot via
    /// capture_resumedata; pyre tracks the nearest valid position.
    pub last_seen_snapshot_pos: Option<i32>,
}

/// heaptracker.py:66: `if name == 'typeptr': continue`
/// Uses FieldDescr.is_typeptr() which checks `field_name() == "typeptr"`,
/// matching RPython's name-based filtering.
#[inline(always)]
pub(crate) fn is_typeptr_field(
    field_idx: u32,
    field_descrs: &[majit_ir::DescrRef],
    _descr: &majit_ir::DescrRef,
) -> bool {
    field_descrs
        .get(field_idx as usize)
        .and_then(|d| d.as_field_descr())
        .map(|fd| fd.is_typeptr())
        .unwrap_or(false)
}

/// resume.py:192-226 parity — BoxEnv for optimizer context.
///
/// Wraps an immutable reference to OptContext, implementing the BoxEnv
/// trait so that ResumeDataLoopMemo.number() can tag boxes during
/// store_final_boxes_in_guard.
pub struct OptBoxEnv<'a> {
    pub ctx: &'a OptContext,
}

impl<'a> majit_ir::BoxEnv for OptBoxEnv<'a> {
    fn get_box_replacement(&self, opref: OpRef) -> OpRef {
        self.ctx.get_box_replacement(opref)
    }

    fn get_box_replacement_not_const(&self, opref: OpRef) -> OpRef {
        self.ctx.get_box_replacement_not_const(opref)
    }

    fn is_const(&self, opref: OpRef) -> bool {
        // RPython resume.py:204: isinstance(box, Const)
        // True Const = constant-namespace OpRef or PtrInfo::Constant.
        // NOT optimizer-known values from make_constant() on operation results.
        if opref.is_constant() {
            return true;
        }
        // make_constant mirrors optimizer.py:432 as
        // `Forwarded::Box(constbox)`. Direct slot access also keeps legacy
        // `Info(Constant)` fixtures readable.
        let idx = opref.raw() as usize;
        if let Some(b) = self.ctx.box_pool.get(idx) {
            use crate::optimizeopt::info::OpInfo;
            match &*b.get_forwarded() {
                crate::r#box::Forwarded::Box(target) if target.is_constant() => return true,
                crate::r#box::Forwarded::Info(OpInfo::Constant(_) | OpInfo::FloatConst(_)) => {
                    return true;
                }
                _ => {}
            }
        }
        // info.py: ConstPtrInfo.is_constant() → True
        matches!(
            self.ctx.peek_ptr_info_via_box(opref),
            Some(crate::optimizeopt::info::PtrInfo::Constant(_))
        )
    }

    fn get_const(&self, opref: OpRef) -> (i64, majit_ir::Type) {
        // make_constant mirrors optimizer.py:432 as
        // `Forwarded::Box(constbox)`. Direct slot access also keeps legacy
        // `Info(Constant)` fixtures readable.
        let idx = opref.raw() as usize;
        let const_value: Option<Value> = if let Some(b) = self.ctx.box_pool.get(idx) {
            use crate::optimizeopt::info::OpInfo;
            match &*b.get_forwarded() {
                crate::r#box::Forwarded::Box(target) => target.const_value(),
                crate::r#box::Forwarded::Info(OpInfo::Constant(v)) => Some(*v),
                crate::r#box::Forwarded::Info(OpInfo::FloatConst(f)) => Some(Value::Float(*f)),
                _ => None,
            }
        } else {
            None
        };
        if let Some(val) = const_value {
            let (raw, tp) = match val {
                Value::Int(v) => (v, majit_ir::Type::Int),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                Value::Void => (0, majit_ir::Type::Int),
            };
            let tp = self
                .ctx
                .constant_types_for_numbering
                .get(&opref.raw())
                .copied()
                .unwrap_or(tp);
            return (raw, tp);
        }
        // RPython ConstPtr parity: check numbering type overrides first.
        let type_override = self
            .ctx
            .constant_types_for_numbering
            .get(&opref.raw())
            .copied();
        match self.ctx.get_constant(opref) {
            Some(Value::Int(v)) => (v, type_override.unwrap_or(majit_ir::Type::Int)),
            Some(Value::Float(f)) => (f.to_bits() as i64, majit_ir::Type::Float),
            Some(Value::Ref(r)) => (r.0 as i64, majit_ir::Type::Ref),
            _ => {
                // info.py: ConstPtrInfo — GcRef constant stored in PtrInfo
                if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) =
                    self.ctx.peek_ptr_info_via_box(opref)
                {
                    (gcref.0 as i64, majit_ir::Type::Ref)
                } else {
                    (0, majit_ir::Type::Int)
                }
            }
        }
    }

    fn get_type(&self, opref: OpRef) -> majit_ir::Type {
        // RPython parity: resume.py:201 calls `box = box.get_box_replacement()`
        // before any Const / type judgement. Same-type invariant on
        // `replace_op` makes this usually equivalent to reading `opref`
        // directly, but doing replacement first matches upstream order
        // and is robust against forwarding chains.
        let resolved = self.ctx.get_box_replacement(opref);
        // Phase 1-5 OpRef enum carries box.type in the variant tag for
        // typed positions; reading the tag on the resolved box is the
        // line-by-line equivalent of upstream `box.type`
        // (resoperation.py:29 / history.py:220). Untyped/None fall
        // through to the legacy chain during the Slice 0.5 transition.
        if let Some(tp) = resolved.ty() {
            if tp != majit_ir::Type::Void {
                return tp;
            }
        }
        // Constant type (history.py:220 ConstInt.type parity).
        // The `constant_types_for_numbering` override is a raw-pointer
        // marker on `'i'`-typed Boxes (RPython `getrawptrinfo` ConstInt
        // path); it does NOT change `box.type` from `'i'` to `'r'`, so
        // we never read it here as a type source.  This matches
        // `OptContext::opref_type`'s documented invariant.
        if let Some(val) = self.ctx.get_constant(resolved) {
            return val.get_type();
        }
        // Producing op intrinsic type (resoperation.py:1693
        // `opclasses[opnum].type` parity).
        if let Some(op) = self.ctx.op_at(resolved) {
            if op.type_ != majit_ir::Type::Void {
                return op.type_;
            }
        }
        // value_types aggregates the four upstream sources (original trace
        // ops, prev-phase carry, inputarg types, transformed trace result
        // types) seeded in `Optimizer::make_ctx`. Removed in Slice 0.5
        // once writers feed Op.type_ exclusively.
        if let Some(&tp) = self.ctx.value_types.get(&resolved.raw()) {
            if tp != majit_ir::Type::Void {
                return tp;
            }
        }
        // Inputarg slot (history.py:220 `InputArg{Int,Ref,Float}.type`
        // parity). Slice 0.5 prep — placed last so legacy `value_types`
        // writers keep priority during the transition; once Slice 0.5
        // drops the inputarg fan-in this becomes the authoritative
        // inputarg-type source.
        if let Some(tp) = self.ctx.inputarg_type(resolved) {
            return tp;
        }
        // PtrInfo presence → Ref type (for non-emitted ops like input args)
        if self.ctx.has_ptr_info_via_box(resolved) {
            return majit_ir::Type::Ref;
        }
        majit_ir::Type::Int
    }

    fn is_virtual_ref(&self, opref: OpRef) -> bool {
        // info.py:880-886 getptrinfo(op) first applies get_box_replacement(op)
        // before reading PtrInfo. Guard resume numbering walks ORIGINAL
        // snapshot boxes, so virtual classification must follow the same
        // replacement chain or forwarded virtual boxes get mis-tagged as
        // ordinary liveboxes.
        let resolved = self.ctx.get_box_replacement(opref);
        self.ctx
            .peek_ptr_info_via_box(resolved)
            .is_some_and(|info| info.is_virtual())
    }

    fn is_virtual_raw(&self, _opref: OpRef) -> bool {
        // pyre doesn't have Int-typed virtual objects
        false
    }

    fn has_known_class(&self, opref: OpRef) -> bool {
        // bridgeopt.py:79-80: getptrinfo(box).get_known_class(cpu) is not None
        let resolved = self.ctx.get_box_replacement(opref);
        self.ctx
            .peek_ptr_info_via_box(resolved)
            .and_then(|info| info.get_known_class())
            .is_some()
    }

    fn get_virtual_fields(&self, opref: OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        let resolved = self.ctx.get_box_replacement(opref);
        let info = self.ctx.peek_ptr_info_via_box(resolved)?;
        match info {
            PtrInfo::Virtual(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: vi.known_class,
                // info.py:243-247 `_visitor_walk_recursive` registers the
                // full `_fields` list in descriptor order, leaving unfilled
                // slots as `None`. Preserve that shape so `fieldnums` aligns
                // 1:1 with `descr.get_all_fielddescrs()` for `_cached_vinfo`
                // reuse at resume.py:307-315.
                field_oprefs: vi
                    .field_descrs
                    .iter()
                    .enumerate()
                    .map(|(fi, _)| {
                        vi.fields
                            .iter()
                            .find(|(field_idx, _)| *field_idx == fi as u32)
                            .map(|(_, vref)| self.ctx.get_box_replacement(*vref))
                            .unwrap_or(OpRef::NONE)
                    })
                    .collect(),
            }),
            PtrInfo::VirtualStruct(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .field_descrs
                    .iter()
                    .enumerate()
                    .map(|(fi, _)| {
                        vi.fields
                            .iter()
                            .find(|(field_idx, _)| *field_idx == fi as u32)
                            .map(|(_, vref)| self.ctx.get_box_replacement(*vref))
                            .unwrap_or(OpRef::NONE)
                    })
                    .collect(),
            }),
            PtrInfo::VirtualArray(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .items
                    .iter()
                    .map(|vref| self.ctx.get_box_replacement(*vref))
                    .collect(),
            }),
            PtrInfo::VirtualArrayStruct(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .element_fields
                    .iter()
                    .flat_map(|ef| {
                        vi.fielddescrs.iter().enumerate().map(|(fi, _)| {
                            ef.iter()
                                .find(|(field_idx, _)| *field_idx == fi as u32)
                                .map(|(_, vref)| self.ctx.get_box_replacement(*vref))
                                .unwrap_or(OpRef::NONE)
                        })
                    })
                    .collect(),
            }),
            PtrInfo::VirtualRawBuffer(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: None,
                known_class: None,
                field_oprefs: vi
                    .values
                    .iter()
                    .map(|vref| self.ctx.get_box_replacement(*vref))
                    .collect(),
            }),
            // `info.py:478-482` `RawSlicePtrInfo._visitor_walk_recursive`:
            //
            // ```python
            // def _visitor_walk_recursive(self, op, visitor):
            //     source_op = get_box_replacement(op.getarg(0))
            //     visitor.register_virtual_fields(op, [source_op])
            //     if self.parent.is_virtual():
            //         self.parent.visitor_walk_recursive(source_op, visitor)
            // ```
            //
            // pyre's consumer (`resume.rs::encode_*` worklist at
            // `resume.rs:3517`) drives the recursion off `get_virtual_fields`
            // — registering the parent OpRef here lets the worklist enqueue
            // the parent and re-enter `get_virtual_fields` on it, which
            // matches RPython's `parent.visitor_walk_recursive(source_op,
            // visitor)` follow-up.  Only fires while the slice is still
            // virtual (`slice.parent` non-NONE); after `force_box_impl`
            // materializes the slice the gate in `is_virtual` flips False
            // and the caller drops the entry.
            PtrInfo::VirtualRawSlice(vi) if !vi.parent.is_none() => {
                Some(majit_ir::VirtualFieldsInfo {
                    descr: None,
                    known_class: None,
                    field_oprefs: vec![self.ctx.get_box_replacement(vi.parent)],
                })
            }
            // vstring.py:207-208 VStringPlainInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, self._chars)`
            //
            // vstring.py:255-260 VStringSliceInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, [self.s, self.start, self.lgtop])`
            //   (then recurses into the parent string if it is itself virtual).
            //
            // vstring.py:319-325 VStringConcatInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, [self.vleft, self.vright])`
            //
            // Only virtual StrPtrInfo variants register fields; a non-virtual
            // `VStringVariant::Ptr` skips this arm (falls through to None).
            PtrInfo::Str(sinfo) if sinfo.is_virtual() => {
                use crate::optimizeopt::info::VStringVariant;
                let field_oprefs: Vec<OpRef> = match &sinfo.variant {
                    // vstring.py:207-208: self._chars. `None` slots represent
                    // unfilled positions — the resume encoder later tags those
                    // with UNINITIALIZED (matching how RPython treats missing
                    // char boxes, since STRSETITEM may not have run for every
                    // index yet at guard time).
                    VStringVariant::Plain(p) => p
                        ._chars
                        .iter()
                        .map(|slot| {
                            slot.map(|r| self.ctx.get_box_replacement(r))
                                .unwrap_or(OpRef::NONE)
                        })
                        .collect(),
                    // vstring.py:255-257: [self.s, self.start, self.lgtop].
                    VStringVariant::Slice(s) => vec![
                        self.ctx.get_box_replacement(s.s),
                        self.ctx.get_box_replacement(s.start),
                        self.ctx.get_box_replacement(s.lgtop),
                    ],
                    // vstring.py:319-324: [self.vleft, self.vright].
                    VStringVariant::Concat(c) => vec![
                        self.ctx.get_box_replacement(c.vleft),
                        self.ctx.get_box_replacement(c.vright),
                    ],
                    // Non-virtual `VStringVariant::Ptr` would not reach
                    // here because of the `is_virtual()` guard above.
                    VStringVariant::Ptr => Vec::new(),
                };
                Some(majit_ir::VirtualFieldsInfo {
                    descr: None,
                    known_class: None,
                    field_oprefs,
                })
            }
            _ => None,
        }
    }

    fn make_virtual_info(
        &self,
        opref: OpRef,
        fieldnums: Vec<i16>,
    ) -> Option<std::rc::Rc<majit_ir::RdVirtualInfo>> {
        let resolved = self.ctx.get_box_replacement(opref);
        let info = self.ctx.peek_ptr_info_via_box(resolved)?;
        // resume.py:307-315 `ResumeDataVirtualAdder.make_virtual_info`:
        //
        //     vinfo = info._cached_vinfo
        //     if vinfo is not None and vinfo.equals(fieldnums):
        //         return vinfo
        //     vinfo = info.visitor_dispatch_virtual_type(self)
        //     vinfo.set_content(fieldnums)
        //     info._cached_vinfo = vinfo
        //     return vinfo
        //
        // The cache stores an `Rc<RdVirtualInfo>` so that cache hits return
        // the same shared handle (matching RPython's Python object identity
        // on cache hit, info.py:124-128). Downstream storage in
        // `storage.rd_virtuals` keeps the shared handle so two guards that
        // reference the same virtual with the same fieldnums end up pointing
        // at the same `RdVirtualInfo` object.
        if let Some(cache) = info.cached_vinfo() {
            if let Some(vinfo) = cache.borrow().as_ref() {
                if vinfo.equals(&fieldnums) {
                    return Some(std::rc::Rc::clone(vinfo));
                }
            }
        }
        let mut builder = RdVirtualInfoBuilder;
        let mut vinfo = info.visitor_dispatch_virtual_type(&mut builder)??;
        // resume.py:313: vinfo.set_content(fieldnums)
        vinfo.set_content(fieldnums);
        let shared = std::rc::Rc::new(vinfo);
        // resume.py:314: info._cached_vinfo = vinfo — store the shared handle
        // so a later equals-hit returns the SAME object.
        if let Some(cache) = info.cached_vinfo() {
            *cache.borrow_mut() = Some(std::rc::Rc::clone(&shared));
            // `info` is a clone returned from `peek_ptr_info_via_box`;
            // mutating its independent `cached_vinfo` RefCell does not feed
            // back into the BoxRef canonical slot. Project the cached Rc
            // handle directly onto the BoxRef PtrInfo so subsequent
            // BoxRef-routing readers (`virtual_info_would_be_reused`)
            // observe the cached vinfo.
            if let Some(b) = self.ctx.box_pool.get(resolved.raw() as usize).cloned() {
                if let Some(pi) = b.ptr_info_mut() {
                    if let Some(c) = pi.cached_vinfo() {
                        *c.borrow_mut() = Some(std::rc::Rc::clone(&shared));
                    }
                }
            }
        }
        Some(shared)
    }

    fn virtual_info_would_be_reused(&self, opref: OpRef, fieldnums: &[i16]) -> bool {
        let resolved = self.ctx.get_box_replacement(opref);
        // BoxRef-routing reader; cached_vinfo's RefCell clones shallowly so the
        // inner Rc<RdVirtualInfo> is shared with the canonical PtrInfo — read of
        // .borrow() yields the same content as the original cache.
        let Some(info) = self.ctx.peek_ptr_info_via_box(resolved) else {
            return false;
        };
        let Some(cache) = info.cached_vinfo() else {
            return false;
        };
        cache
            .borrow()
            .as_ref()
            .is_some_and(|vinfo| vinfo.equals(fieldnums))
    }
}

/// resume.py:298-357 `ResumeDataVirtualAdder` in its role as a
/// `VirtualVisitor` — each `visit_*` builds a fresh `RdVirtualInfo`
/// subclass without fieldnums (the caller attaches those via
/// `set_content`). In pyre, `make_virtual_info` lives on the
/// `BoxEnv` impl (not on `ResumeDataVirtualAdder`) because PtrInfo
/// lookup is the optimizer's responsibility, so the visitor adapter
/// is this zero-sized helper instead of `ResumeDataVirtualAdder`
/// itself.
struct RdVirtualInfoBuilder;

impl crate::walkvirtual::VirtualVisitor for RdVirtualInfoBuilder {
    type VInfo = Option<majit_ir::RdVirtualInfo>;

    fn visit_not_virtual(&mut self, _value: OpRef) -> Self::VInfo {
        // resume.py:317-318 `visit_not_virtual` asserts unreachable.
        debug_assert!(false, "visit_not_virtual reached via virtual dispatch");
        None
    }

    // resume.py:320-321 visit_virtual → VirtualInfo(descr, fielddescrs)
    fn visit_virtual(
        &mut self,
        descr: &majit_ir::DescrRef,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // `FieldDescrInfo.index` must carry the stable descriptor index
        // (tagged offset) so resume-data readers can identify the field by
        // byte offset. Previously stored `fi as u32` (iteration counter),
        // which made `extract_pyre_field_offset` always fail for virtuals
        // being materialized on bridge entry.
        let built_fielddescrs: Vec<majit_ir::FieldDescrInfo> = fielddescrs
            .iter()
            .map(|descr| {
                let fd = descr.as_field_descr();
                majit_ir::FieldDescrInfo {
                    index: descr.index(),
                    offset: fd.map(|f| f.offset()).unwrap_or(0),
                    field_type: fd.map(|f| f.field_type()).unwrap_or(majit_ir::Type::Int),
                    field_size: fd.map(|f| f.field_size()).unwrap_or(8),
                }
            })
            .collect();
        let sd = descr.as_size_descr();
        Some(majit_ir::RdVirtualInfo::VirtualInfo {
            descr: Some(descr.clone()),
            type_id: sd.map(|s| s.type_id()).unwrap_or(0),
            descr_index: descr.index(),
            // resume.py:619 allocate_with_vtable(descr=self.descr) — the
            // vtable is derived from descr; majit mirrors by reading it
            // off the SizeDescr when the descr carries class info.
            known_class: sd.map(|s| s.vtable() as i64).filter(|&v| v != 0),
            fielddescrs: built_fielddescrs,
            fieldnums: Vec::new(),
            descr_size: sd.map(|s| s.size()).unwrap_or(0),
        })
    }

    // resume.py:323-324 visit_vstruct → VStructInfo(typedescr, fielddescrs)
    fn visit_vstruct(
        &mut self,
        typedescr: &majit_ir::DescrRef,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // See `visit_virtual` — index must be the stable descriptor index,
        // not the iteration counter.
        let built_fielddescrs: Vec<majit_ir::FieldDescrInfo> = fielddescrs
            .iter()
            .map(|descr| {
                let fd = descr.as_field_descr();
                majit_ir::FieldDescrInfo {
                    index: descr.index(),
                    offset: fd.map(|f| f.offset()).unwrap_or(0),
                    field_type: fd.map(|f| f.field_type()).unwrap_or(majit_ir::Type::Int),
                    field_size: fd.map(|f| f.field_size()).unwrap_or(8),
                }
            })
            .collect();
        let sd = typedescr.as_size_descr();
        Some(majit_ir::RdVirtualInfo::VStructInfo {
            typedescr: Some(typedescr.clone()),
            type_id: sd.map(|s| s.type_id()).unwrap_or(0),
            descr_index: typedescr.index(),
            fielddescrs: built_fielddescrs,
            fieldnums: Vec::new(),
            descr_size: sd.map(|s| s.size()).unwrap_or(0),
        })
    }

    // resume.py:326-330 visit_varray → VArrayInfoClear / VArrayInfoNotClear
    fn visit_varray(&mut self, arraydescr: &majit_ir::DescrRef, clear: bool) -> Self::VInfo {
        let kind = arraydescr
            .as_array_descr()
            .map(|ad| match ad.item_type() {
                majit_ir::Type::Float => 2u8,
                majit_ir::Type::Int => 1u8,
                _ => 0u8,
            })
            .unwrap_or(0);
        let ad = Some(arraydescr.clone());
        let descr_index = arraydescr.index();
        Some(if clear {
            majit_ir::RdVirtualInfo::VArrayInfoClear {
                arraydescr: ad,
                descr_index,
                kind,
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                arraydescr: ad,
                descr_index,
                kind,
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py:332-333 visit_varraystruct → VArrayStructInfo
    fn visit_varraystruct(
        &mut self,
        arraydescr: &majit_ir::DescrRef,
        length: usize,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // info.py:701-704: visitor_dispatch_virtual_type always hands
        // down the canonical get_all_interiorfielddescrs() list; fall
        // back to the variant's cached fielddescrs when descr lacks it.
        let canonical_fielddescrs: Vec<majit_ir::DescrRef> = arraydescr
            .as_array_descr()
            .and_then(|ad| ad.get_all_interiorfielddescrs())
            .map(|fds| fds.to_vec())
            .unwrap_or_else(|| fielddescrs.to_vec());
        let mut fo = Vec::new();
        let mut fs = Vec::new();
        let mut ft = Vec::new();
        for fd in &canonical_fielddescrs {
            if let Some(ifd) = fd.as_interior_field_descr() {
                let fld = ifd.field_descr();
                fo.push(fld.offset());
                fs.push(fld.field_size());
                ft.push(match fld.field_type() {
                    majit_ir::Type::Float => 2u8,
                    majit_ir::Type::Int => 1u8,
                    _ => 0u8,
                });
            } else {
                fo.push(fo.len() * 8);
                fs.push(8);
                ft.push(0);
            }
        }
        if ft.is_empty() {
            ft = vec![0u8; canonical_fielddescrs.len()];
        }
        let ad = arraydescr.as_array_descr();
        let fielddescr_count = canonical_fielddescrs.len();
        Some(majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr: Some(arraydescr.clone()),
            descr_index: arraydescr.index(),
            size: length,
            fielddescrs: canonical_fielddescrs,
            fielddescr_indices: (0..fielddescr_count).map(|i| i as u32).collect(),
            field_types: ft,
            base_size: ad.map(|a| a.base_size()).unwrap_or(0),
            item_size: ad.map(|a| a.item_size()).unwrap_or(0),
            field_offsets: fo,
            field_sizes: fs,
            fieldnums: Vec::new(),
        })
    }

    // resume.py:335-336 visit_vrawbuffer → VRawBufferInfo
    fn visit_vrawbuffer(
        &mut self,
        func: i64,
        size: usize,
        offsets: &[usize],
        descrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        let descr_infos: Vec<majit_ir::ArrayDescrInfo> = descrs
            .iter()
            .map(|d| {
                let ad = d.as_array_descr();
                majit_ir::ArrayDescrInfo {
                    index: d.index(),
                    base_size: ad.map_or(0, |a| a.base_size()),
                    item_size: ad.map_or(8, |a| a.item_size()),
                    item_type: ad.map_or(1, |a| {
                        if a.is_array_of_pointers() {
                            0
                        } else if a.is_array_of_floats() {
                            2
                        } else {
                            1
                        }
                    }),
                    is_signed: ad.map_or(true, |a| a.is_item_signed()),
                }
            })
            .collect();
        Some(majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets: offsets.to_vec(),
            descrs: descr_infos,
            fieldnums: Vec::new(),
        })
    }

    // resume.py:338-339 visit_vrawslice → VRawSliceInfo
    fn visit_vrawslice(&mut self, offset: usize) -> Self::VInfo {
        Some(majit_ir::RdVirtualInfo::VRawSliceInfo {
            offset,
            fieldnums: Vec::new(),
        })
    }

    // resume.py:341-345 visit_vstrplain → VStrPlainInfo / VUniPlainInfo
    fn visit_vstrplain(&mut self, is_unicode: bool) -> Self::VInfo {
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniPlainInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrPlainInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py:347-351 visit_vstrconcat → VStrConcatInfo / VUniConcatInfo
    fn visit_vstrconcat(&mut self, is_unicode: bool) -> Self::VInfo {
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniConcatInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrConcatInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py:353-357 visit_vstrslice → VStrSliceInfo / VUniSliceInfo
    fn visit_vstrslice(&mut self, is_unicode: bool) -> Self::VInfo {
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniSliceInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrSliceInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    fn register_virtual_fields(&mut self, _virtualbox: OpRef, _fieldboxes: &[OpRef]) {
        // resume.py:359-368 — field registration happens elsewhere in pyre
        // (via resume.rs worklist + get_virtual_fields), not through the
        // visitor. This adapter is only a builder for RdVirtualInfo.
    }

    fn already_seen_virtual(&mut self, _virtualbox: OpRef) -> bool {
        // resume.py:380-386 — same split as register_virtual_fields; the
        // pyre worklist tracks visited boxes directly.
        false
    }
}

impl OptContext {
    /// RPython optimizer.py: add to quasi_immutable_deps
    pub fn add_quasi_immutable_dep(&mut self, dep: (u64, u32)) {
        self.quasi_immutable_deps.insert(dep);
    }

    /// RPython Box.type invariant: each OpRef's type is fixed at creation
    /// (IntFrontendOp / RefFrontendOp / FloatFrontendOp are separate classes
    /// and `box.type` is a class property, immutable for the box's lifetime).
    /// pyre carries the type externally in `value_types`, so enforce the
    /// invariant here: inserting an OpRef that already has a *different*
    /// type is the only form of divergence from RPython. The assertion is
    /// debug-only so release builds are unaffected; it flips on the
    /// violation so we can locate the callsite and emit an explicit re-box
    /// (wrapint / wrapfloat / unbox) instead of silently retyping.
    ///
    /// Void (no result) is rejected because value_types keeps only
    /// producing ops; callers gate on `op.result_type() != Void` before
    /// calling in.
    pub fn register_value_type(&mut self, opref: majit_ir::OpRef, tp: majit_ir::Type) {
        debug_assert_ne!(
            tp,
            majit_ir::Type::Void,
            "register_value_type: Void has no box value (opref={:?})",
            opref,
        );
        // Slice 0.5: when `opref` already encodes its type via the Phase
        // 1-5 OpRef enum variant tag (resoperation.py:29 AbstractValue.type
        // parity), the side-table write is redundant — readers consult
        // `opref.ty()` at priority 0 in `opref_type` /
        // `OptBoxEnv::get_type`. Verify the variant-tag and registered
        // `tp` agree (debug-only) and skip the HashMap write so producers
        // that allocate typed positions never grow `value_types`.
        if let Some(variant_tp) = opref.ty() {
            debug_assert_eq!(
                variant_tp, tp,
                "register_value_type: OpRef variant tag ({:?}) disagrees \
                 with registered tp ({:?}) at {:?} — fix the producer \
                 (typed factory mismatch with op.result_type())",
                variant_tp, tp, opref,
            );
            return;
        }
        if let Some(&existing) = self.value_types.get(&opref.raw()) {
            debug_assert_eq!(
                existing, tp,
                "register_value_type: OpRef {:?} retyped from {:?} to {:?} \
                 (RPython Box.type invariant violation; emit explicit \
                 wrapint/wrapfloat/unbox at the producer instead of \
                 retyping in place)",
                opref, existing, tp,
            );
        }
        // Phase 0 dual-source check: when the producing Op already lives
        // in new_operations its intrinsic `type_` field (resoperation.py:1693
        // `opclasses[opnum].type` parity) must agree with the registered
        // `tp`. Caught here during development; once all readers migrate
        // to `op.type_` (Slice 0.4) the side-table itself disappears.
        #[cfg(debug_assertions)]
        if let Some(producer) = self.new_operations.iter().rev().find(|o| o.pos == opref) {
            debug_assert_eq!(
                producer.type_, tp,
                "register_value_type: Op.type_ ({:?}) disagrees with \
                 registered tp ({:?}) at {:?} (opcode={:?}) — Box.type \
                 parity violation",
                producer.type_, tp, opref, producer.opcode,
            );
        }
        self.value_types.insert(opref.raw(), tp);
    }

    pub fn new(estimated_ops: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            const_pool: HashMap::new(),
            num_inputs: 0,
            inputarg_base: 0,
            next_pos: 0,
            next_const_idx: 0,
            extra_operations_after: VecDeque::new(),
            pending_guard_class_postprocess: None,
            pending_mark_last_guard: None,
            pending_finish_guard_postprocess: None,
            imported_short_pure_ops: Vec::new(),
            imported_virtual_args: None,
            imported_loop_invariant_results: HashMap::new(),
            imported_short_preamble_builder: None,
            const_infos: HashMap::new(),
            imported_short_preamble_used: HashSet::new(),

            potential_extra_ops: HashMap::new(),
            active_short_preamble_producer: None,
            exported_short_boxes: Vec::new(),

            imported_virtuals: Vec::new(),
            imported_label_args: None,
            can_replace_guards: true,
            patchguardop: None,
            preamble_end_args: None,
            skip_flush_mode: false,
            current_pass_idx: 0,
            optearlyforce_idx: 0,

            in_final_emission: false,
            callinfocollection: None,
            pending_for_guard: Vec::new(),
            pending_pure_from_args: Vec::new(),
            pending_pure_from_args2: Vec::new(),
            constant_fold_alloc: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            quasi_immutable_deps: HashSet::new(),
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: Vec::new(),
            snapshot_vable_boxes: Vec::new(),
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: Vec::new(),

            constant_types_for_numbering: HashMap::new(),
            value_types: HashMap::new(),
            inputarg_types: Vec::new(),
            phase1_emit_ops: Vec::new(),
            last_guard_idx: None,
            last_seen_snapshot_pos: None,
            box_pool: crate::r#box::BoxPool::new(),
        }
    }

    /// Test-only inputarg-free constructor. Production paths must always go
    /// through [`Self::with_inputarg_types`] so every inputarg slot lands a
    /// typed `BoxRef` matching `opencoder.py:259 inputarg_from_tp(arg.type)`;
    /// passing `num_inputs > 0` here would silently drop the type tag and
    /// produce `Type::Void` reads. Sealed under `#[cfg(test)]` to make the
    /// rule structural rather than discipline-only.
    #[cfg(test)]
    pub fn with_num_inputs(estimated_ops: usize, num_inputs: usize) -> Self {
        debug_assert_eq!(
            num_inputs, 0,
            "with_num_inputs(_, {num_inputs}) — non-zero M requires typed seeding; \
             use `with_inputarg_types(estimated_ops, &[Type; M])` instead \
             (opencoder.py:259 inputarg_from_tp parity)",
        );
        Self::with_inputarg_types(estimated_ops, &[])
    }

    /// Construct an `OptContext` and seed `box_pool` with one
    /// `BoxRef::new_inputarg(tp, Some(i))` per entry of `inputarg_types`.
    ///
    /// Mirrors `TraceIterator::new` (`opencoder.rs:373-426`, parity with
    /// `opencoder.py:259-262` `inputarg_from_tp(arg.type)`). Test fixtures
    /// that construct via this helper exercise the optimizer's BoxRef-direct
    /// routing — the production path — instead of the legacy empty-pool
    /// Vec fallback.
    ///
    /// `inputarg_types` carries the type tags needed to round-trip
    /// `OpRef::input_arg_typed(i, tp)` on read; `with_num_inputs` is the
    /// untyped delegate that defaults every slot to `Type::Void`.
    pub fn with_inputarg_types(estimated_ops: usize, inputarg_types: &[majit_ir::Type]) -> Self {
        let num_inputs = inputarg_types.len();
        let mut ctx =
            Self::with_num_inputs_and_start_pos(estimated_ops, num_inputs, 0, num_inputs as u32);
        let seed: Vec<crate::r#box::BoxRef> = inputarg_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| crate::r#box::BoxRef::new_inputarg(tp, Some(i as u32)))
            .collect();
        ctx.box_pool = seed.into();
        // Mirror the production setup at `optimizer.rs:1859`
        // `ctx.inputarg_types = self.trace_inputarg_types.clone()` — the
        // typed-Box parity contract requires the field to be populated
        // alongside the BoxRef pool so `inputarg_type_at(i)` returns
        // `Some(tp)` matching `box.type` for slot i. Direct test callers
        // bypass the Optimizer, so this helper must seed the field itself.
        ctx.inputarg_types = inputarg_types.to_vec();
        ctx
    }

    /// Construct an `OptContext` whose inputarg / fresh-OpRef numbering is
    /// shifted to start above a parent trace's high water mark.
    ///
    /// `inputarg_base` corresponds to RPython's `start = trace._start` for
    /// `TraceIterator`: it is the smallest OpRef this iteration may use, and
    /// inputargs occupy `[inputarg_base, inputarg_base + num_inputs)`.
    /// `start_next_pos` is the value of `_index` after the inputargs were
    /// pre-allocated, i.e. the first fresh OpRef the optimizer will assign
    /// to a non-void op result. Phase 1 / standalone passes use
    /// `inputarg_base = 0`, `start_next_pos = num_inputs`; Phase 2 / bridges
    /// pass `inputarg_base = parent_high_water`,
    /// `start_next_pos = parent_high_water + num_inputs`.
    pub fn with_num_inputs_and_start_pos(
        estimated_ops: usize,
        num_inputs: usize,
        inputarg_base: u32,
        start_next_pos: u32,
    ) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            const_pool: HashMap::new(),
            num_inputs: num_inputs as u32,
            inputarg_base,
            next_pos: start_next_pos,
            next_const_idx: 0,
            extra_operations_after: VecDeque::new(),
            pending_guard_class_postprocess: None,
            pending_mark_last_guard: None,
            pending_finish_guard_postprocess: None,
            imported_short_pure_ops: Vec::new(),
            imported_virtual_args: None,
            imported_loop_invariant_results: HashMap::new(),
            imported_short_preamble_builder: None,
            const_infos: HashMap::new(),
            imported_short_preamble_used: HashSet::new(),

            potential_extra_ops: HashMap::new(),
            active_short_preamble_producer: None,
            exported_short_boxes: Vec::new(),

            imported_virtuals: Vec::new(),
            imported_label_args: None,
            can_replace_guards: true,
            patchguardop: None,
            preamble_end_args: None,
            skip_flush_mode: false,
            current_pass_idx: 0,
            optearlyforce_idx: 0,

            in_final_emission: false,
            callinfocollection: None,
            pending_for_guard: Vec::new(),
            pending_pure_from_args: Vec::new(),
            pending_pure_from_args2: Vec::new(),
            constant_fold_alloc: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            quasi_immutable_deps: HashSet::new(),
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: Vec::new(),
            snapshot_vable_boxes: Vec::new(),
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: Vec::new(),

            constant_types_for_numbering: HashMap::new(),
            value_types: HashMap::new(),
            inputarg_types: Vec::new(),
            phase1_emit_ops: Vec::new(),
            last_guard_idx: None,
            last_seen_snapshot_pos: None,
            box_pool: crate::r#box::BoxPool::new(),
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    /// Allocate a fresh OpRef position (for imported virtual heads).
    pub fn alloc_op_position(&mut self) -> OpRef {
        self.reserve_pos()
    }

    /// Allocate a fresh OpRef position with the producer's result type
    /// stamped on the variant tag (Slice 0.5 follow-up). Mirrors
    /// `alloc_op_position` for callers that already know `box.type` —
    /// the resulting OpRef is recognized at priority 0 by `opref_type` /
    /// `OptBoxEnv::get_type`, and `register_value_type` no longer needs
    /// to grow the side-table for these positions.
    pub fn alloc_op_position_typed(&mut self, tp: majit_ir::Type) -> OpRef {
        self.reserve_pos_typed(tp)
    }

    /// Allocate a fresh OpRef in the constant namespace, tagged with the
    /// constant's type so the variant matches RPython's
    /// ConstInt/ConstFloat/ConstPtr split (history.py:220/261/307,
    /// opencoder.py:321-333 `_untag` direct dispatch).
    pub(crate) fn reserve_const_ref(&mut self, tp: Type) -> OpRef {
        let opref = OpRef::const_typed(self.next_const_idx, tp);
        self.next_const_idx += 1;
        opref
    }

    pub(crate) fn const_ref_for_value(const_idx: u32, value: &Value) -> OpRef {
        match value {
            Value::Int(_) => OpRef::const_int(const_idx),
            Value::Float(_) => OpRef::const_float(const_idx),
            Value::Ref(_) => OpRef::const_ptr(const_idx),
            Value::Void => panic!("constant pool cannot contain a ConstVoid"),
        }
    }

    /// Dispatch on a `Value`'s type tag and produce a typed `*Op` OpRef
    /// at the given position. Counterpart to `const_ref_for_value` for
    /// the op-position namespace (resoperation.py:564-638
    /// IntOp/FloatOp/RefOp/VoidOp mixins).
    pub(crate) fn op_ref_for_value(pos: u32, value: &Value) -> OpRef {
        OpRef::op_typed(pos, value.get_type())
    }

    /// info.py:148,226 emit during force_box: routes through emit_extra
    /// normally, or direct emit when in_final_emission is true.
    pub fn emit_for_force(&mut self, op: Op) -> OpRef {
        if self.in_final_emission {
            self.emit(op)
        } else {
            self.emit_extra(self.current_pass_idx, op)
        }
    }

    /// Emit a boxed integer constant through the optimizer pipeline and return
    /// the resulting OpRef.
    pub fn emit_constant_int(&mut self, value: i64) -> OpRef {
        let pos_ref = self.reserve_pos_typed(Type::Int);
        let mut op = Op::new(OpCode::SameAsI, &[pos_ref]);
        op.pos = pos_ref;
        let opref = self.emit_extra(self.current_pass_idx, op);
        self.make_constant(opref, Value::Int(value));
        opref
    }

    /// Emit a boxed reference constant through the optimizer pipeline and
    /// return the resulting OpRef.
    pub fn emit_constant_ref(&mut self, value: GcRef) -> OpRef {
        let pos_ref = self.reserve_pos_typed(Type::Ref);
        let mut op = Op::new(OpCode::SameAsR, &[pos_ref]);
        op.pos = pos_ref;
        let opref = self.emit_extra(self.current_pass_idx, op);
        self.make_constant(opref, Value::Ref(value));
        opref
    }

    /// Emit a boxed float constant through the optimizer pipeline and return
    /// the resulting OpRef.
    pub fn emit_constant_float(&mut self, value: f64) -> OpRef {
        let pos_ref = self.reserve_pos_typed(Type::Float);
        let mut op = Op::new(OpCode::SameAsF, &[pos_ref]);
        op.pos = pos_ref;
        let opref = self.emit_extra(self.current_pass_idx, op);
        self.make_constant(opref, Value::Float(value));
        opref
    }

    /// optimizer.py:509-515 new_const_item(arraydescr) — default value for
    /// the given item type. Uses emit_extra (downstream-only) so this is
    /// safe to call during force_box / force_virtual.
    pub fn new_const_item(&mut self, item_type: Type) -> OpRef {
        match item_type {
            Type::Int | Type::Void => self.emit_constant_int(0),
            Type::Ref => self.emit_constant_ref(GcRef::NULL),
            Type::Float => self.emit_constant_float(0.0),
        }
    }

    /// vstring.py:110-119 / 171-175 / 251-253 / 281-295
    /// Per-subclass getstrlen() dispatch — returns a cached lgtop OpRef if
    /// available, or computes/emits the length and caches in StrPtrInfo.lgtop.
    /// Always returns a box (OpRef), never an i64 summary.
    ///
    /// Delegates to `getstrlen_for(opref, opref, mode)`.
    pub fn getstrlen_opref(&mut self, opref: OpRef, mode: u8) -> OpRef {
        self.getstrlen_for(opref, opref, mode)
    }

    /// vstring.py:110-119 StrPtrInfo.getstrlen(op, optstring, mode)
    ///
    /// Matches RPython's method dispatch where `self` (info) and `op` may
    /// differ: info lookup comes from `info_opref`, but the fallback STRLEN
    /// emission uses `op_opref`.  Cached lgtop is stored on `info_opref`'s
    /// PtrInfo.
    ///
    /// When both are the same, use `getstrlen_opref(opref, mode)` instead.
    pub fn getstrlen_for(&mut self, info_opref: OpRef, op_opref: OpRef, mode: u8) -> OpRef {
        let resolved = self.get_box_replacement(info_opref);
        // vstring.py:112/283: if self.lgtop is not None: return self.lgtop
        if let Some(info) = self.getptrinfo_via_box(resolved) {
            if let Some(lgtop) = info.get_cached_lgtop() {
                return lgtop;
            }
        }
        // vstring.py:174/253: constant or structurally-known length
        let known_len = self
            .getptrinfo_via_box(resolved)
            .and_then(|info| info.get_known_str_length(self, mode));
        if let Some(len) = known_len {
            let len_opref = self.make_constant_int(len);
            self.set_str_lgtop(resolved, len_opref);
            return len_opref;
        }
        // vstring.py:281-295: VStringConcatInfo.getstrlen — recursive
        // dispatch: getstrlen on each child, then _int_add.
        // Borrow-checker adaptation: extract vleft/vright before &mut self calls.
        let concat_children = self.getptrinfo_via_box(resolved).and_then(|info| {
            use crate::optimizeopt::info::VStringVariant;
            if let PtrInfo::Str(sinfo) = info {
                if let VStringVariant::Concat(c) = sinfo.variant {
                    return Some((c.vleft, c.vright));
                }
            }
            None
        });
        if let Some((vleft, vright)) = concat_children {
            // vstring.py:286-293
            let left_len = self.getstrlen_for(vleft, vleft, mode);
            let right_len = self.getstrlen_for(vright, vright, mode);
            let result = crate::optimizeopt::vstring::_int_add(left_len, right_len, self);
            // vstring.py:293: self.lgtop = _int_add(optstring, len1box, len2box)
            self.set_str_lgtop(resolved, result);
            return result;
        }
        // vstring.py:115-118: base StrPtrInfo — emit STRLEN/UNICODELEN
        // RPython: lengthop = ResOperation(mode.STRLEN, [op])
        // `op` comes from op_opref (the first arg to getstrlen in RPython).
        let op_resolved = self.get_box_replacement(op_opref);
        let strlen_opcode = if mode != 0 {
            majit_ir::OpCode::Unicodelen
        } else {
            majit_ir::OpCode::Strlen
        };
        let strlen_op = majit_ir::Op::new(strlen_opcode, &[op_resolved]);
        let result = self.emit_extra(self.current_pass_idx, strlen_op);
        // vstring.py:116: lengthop.set_forwarded(self.getlenbound(mode))
        let lenbound = self.get_str_lenbound(resolved);
        if let Some(bound) = lenbound {
            self.setintbound(result, &bound);
        }
        // vstring.py:117: self.lgtop = lengthop
        self.set_str_lgtop(resolved, result);
        result
    }

    /// vstring.py:117 — cache the length box in StrPtrInfo.lgtop.
    /// vstring.py:117/174/293 — `StrPtrInfo.lgtop = lgtop` direct field
    /// write. Routed through `with_ptr_info_mut` so the closure mutates
    /// `box_pool[idx].forwarded` PtrInfo in place for downstream
    /// BoxRef-routing readers.
    pub(crate) fn set_str_lgtop(&mut self, opref: OpRef, lgtop: OpRef) {
        let resolved = self.get_box_replacement(opref);
        if resolved.is_constant() {
            return;
        }
        self.with_ptr_info_mut(resolved, |info| {
            if let PtrInfo::Str(si) = info {
                si.lgtop = Some(lgtop);
            }
        });
    }

    /// vstring.py:62-70 StrPtrInfo.getlenbound(mode) — get lenbound from
    /// StrPtrInfo, lazily initializing it from self.length. Routed through
    /// `with_ptr_info_mut` (Epic H H-3.2c slice 61) for auto-mirror.
    fn get_str_lenbound(&mut self, opref: OpRef) -> Option<crate::optimizeopt::intutils::IntBound> {
        let resolved = self.get_box_replacement(opref);
        if resolved.is_constant() {
            return None;
        }
        self.with_ptr_info_mut(resolved, |info| {
            if let PtrInfo::Str(si) = info {
                // vstring.py:65-70
                if si.lenbound.is_none() {
                    si.lenbound = Some(if si.length == -1 {
                        crate::optimizeopt::intutils::IntBound::nonnegative()
                    } else {
                        crate::optimizeopt::intutils::IntBound::from_constant(si.length as i64)
                    });
                }
                si.lenbound.clone()
            } else {
                None
            }
        })
        .flatten()
    }

    pub(crate) fn reserve_pos(&mut self) -> OpRef {
        let raw = self.allocate_next_pos_raw();
        let idx = raw as usize;
        // Cat 1.3 fix: same `box_pool` eager extension as `reserve_pos_typed`.
        // PyPy single `_forwarded` slot model — every fresh OpRef must
        // correspond to a `BoxRef` so `get_box_replacement_box` returns a
        // valid Rc<Box> and `replace_op` mirror chains to the new target
        // without skipping. Untyped allocation falls back to `Type::Void`.
        if !self.box_pool.is_empty() {
            while self.box_pool.len() < idx {
                let position = self.box_pool.len() as u32;
                self.box_pool.push(crate::r#box::BoxRef::new_resop(
                    majit_ir::Type::Void,
                    position,
                ));
            }
            if self.box_pool.len() == idx {
                let position = self.box_pool.len() as u32;
                self.box_pool.push(crate::r#box::BoxRef::new_resop(
                    majit_ir::Type::Void,
                    position,
                ));
            }
        }
        OpRef::from_raw(raw)
    }

    /// Slice 0.5 follow-up: typed `reserve_pos`. Same allocation arithmetic
    /// as `reserve_pos`, but tags the resulting OpRef with the producer's
    /// result type so the Phase 1-5 variant tag (resoperation.py:29
    /// AbstractValue.type parity) is set at allocation time. Readers consult
    /// `opref.ty()` at priority 0 in `opref_type` / `OptBoxEnv::get_type`,
    /// so typed positions never grow `value_types`.
    pub(crate) fn reserve_pos_typed(&mut self, tp: majit_ir::Type) -> OpRef {
        let raw = self.allocate_next_pos_raw();
        let idx = raw as usize;
        // H-3.4 prerequisite (round-6 audit TODO B): eagerly extend
        // `box_pool` so fresh OpRefs from `emit` / `alloc_op_position_typed`
        // / `reserve_pos_typed` carry a BoxRef. Without this,
        // `get_box_replacement_box` returns None for these positions, and
        // `replace_op` mirror skips when the target is fresh (mod.rs:2927),
        // leaving the BoxRef forwarded chain incomplete.
        //
        // Empty `box_pool` (test / retrace baselines) stays empty — only
        // extend when the pool is plumbed by the recorder (production
        // paths post H-2.1/H-3.0b).
        //
        // Padding with `Void` placeholders covers raw positions skipped by
        // `allocate_next_pos_raw` (it advances past slots claimed by the
        // `constants` table).
        if !self.box_pool.is_empty() {
            while self.box_pool.len() < idx {
                let position = self.box_pool.len() as u32;
                self.box_pool.push(crate::r#box::BoxRef::new_resop(
                    majit_ir::Type::Void,
                    position,
                ));
            }
            if self.box_pool.len() == idx {
                let position = self.box_pool.len() as u32;
                self.box_pool
                    .push(crate::r#box::BoxRef::new_resop(tp, position));
            }
            // else: `box_pool[idx]` already exists (recorder pre-populated).
        }
        OpRef::op_typed(raw, tp)
    }

    /// opencoder.py:271 `_index` parity: floor at the iteration's inputarg
    /// base + num_inputs + emitted-op count, so a freshly allocated raw
    /// position never lands inside the inputarg slice or below the parent
    /// trace's high water mark when called from a Phase 2 / bridge
    /// OptContext.
    fn allocate_next_pos_raw(&mut self) -> u32 {
        self.next_pos = self
            .next_pos
            .max(self.inputarg_base + self.num_inputs + self.new_operations.len() as u32);
        while self
            .constants
            .get(self.next_pos as usize)
            .is_some_and(|value| value.is_some())
        {
            self.next_pos += 1;
        }
        debug_assert!(
            !OpRef::from_raw(self.next_pos).is_constant(),
            "reserve_pos overflowed into constant namespace: {}",
            self.next_pos
        );
        let raw = self.next_pos;
        self.next_pos += 1;
        raw
    }

    /// Emit an operation to the output.
    ///
    /// If the op has no pos assigned (NONE), sets it to `num_inputs + idx`
    /// so the backend's variable numbering stays consistent.
    pub fn emit(&mut self, mut op: Op) -> OpRef {
        if op.pos.is_none() || op.pos.is_constant() {
            // Slice 0.5 follow-up: tag the freshly allocated position with
            // the producer op's result type so the variant-tag readers
            // (`opref_type`/`OptBoxEnv::get_type`) resolve at priority 0
            // (resoperation.py:1693 `opclasses[opnum].type` parity).
            op.pos = self.reserve_pos_typed(op.result_type());
        } else {
            // Step 2 Commit D1/D2 invariants (Box identity plan, Step 7):
            //
            // (a) Phase 2 runs through a fresh TraceIterator whose
            //     `_index` starts at `next_global_opref`, so Phase 2 op
            //     results live in a disjoint `[next_global_opref..)`
            //     range that no prior `emit` has touched.
            //
            // (b) Phase 1 / standalone runs start `next_pos` at
            //     `max(num_inputs, max_raw_pos + 1)`, and `reserve_pos`
            //     is monotonic, so fresh positions are always above any
            //     raw trace op.pos the trace carries.
            //
            // (c) `import_state` only creates `Forwarded::Box` chains on
            //     inputarg slots (in `[inputarg_base..inputarg_base +
            //     num_inputs)`) — never on op-result positions that a
            //     later `emit` would try to use.
            //
            // Together these guarantee that:
            //   - `new_operations` never contains two ops at the same pos
            //   - an op being emitted whose pos is a non-void result does
            //     not already have a `Forwarded::Box` redirect set
            //
            // Earlier majit revisions compensated for the broken invariant
            // with two reactive branches in `emit()` (a collision reassign
            // that called `reserve_pos` again, and a forwarding-redirect
            // that called `reserve_pos` + `replace_op(old, new)` to route
            // downstream readers to the fresh position). Both branches
            // are dead under the Commit D1/D2 layout — verified by
            // `MAJIT_LOG=1 cargo test -p majit-metainterp --lib` reporting
            // zero "band-aid" fires across 909 tests. Hard-assert the
            // invariants here so any regression is caught at the emit
            // site rather than at a downstream symptom.
            debug_assert!(
                !self.new_operations.iter().any(|e| e.pos == op.pos),
                "emit: OpRef collision at {:?} — new_operations already contains this position. \
                 Phase 2 should run through a fresh TraceIterator (Commit D1) and Phase 1's \
                 reserve_pos() should be monotonic above all raw trace positions.",
                op.pos,
            );
            debug_assert!(
                !(self.has_op_forwarding(op.pos) && op.result_type() != majit_ir::Type::Void),
                "emit: Forwarded::Box redirect set on non-void result position {:?} — \
                 import_state should only forward inputarg slots in \
                 [inputarg_base..inputarg_base + num_inputs), and Phase 2 op results \
                 live in a disjoint range [p2_high_water..) (Commit D1).",
                op.pos,
            );
            self.next_pos = self.next_pos.max(op.pos.raw().saturating_add(1));
        }
        let pos_ref = op.pos;
        // RPython parity: emit() does NOT clear forwarding.
        // In RPython, Box._forwarded is never cleared by emit — each Box
        // has unique identity. The forwarding set by import_box must
        // survive body op emission for consumer switchover to work.

        // RPython optimizer.py:652-686 emit_guard_operation — guard resume
        // data sharing via _copy_resume_data_from / ResumeGuardCopiedDescr.
        //
        // RPython has exactly one emit path (`Optimizer._emit_operation`,
        // optimizer.py:614).  Pyre's `Optimizer::emit_operation`
        // (optimizer.rs:3259) handles guard dispatch (force_box on args,
        // emit_guard_operation, force_box on fail_args, _maybe_replace_guard_value)
        // and then calls `ctx.emit(op.clone())` for the
        // `_newoperations.append(op)` step (optimizer.py:646).  The
        // OptContext-side emit_guard_operation is the standalone path
        // used by unit tests that drive `OptContext::emit` directly
        // without going through `Optimizer::emit_operation`.
        //
        // Skip the OptContext guard handling when the Optimizer is
        // mid-flight (`in_final_emission == true`) so production runs
        // through one emit_guard_operation (RPython parity).  The
        // OptContext path remains the sole guard handler for the
        // standalone test entry point.
        //
        // optimizer.py:639-644: side-effectful non-guard ops clear the
        // sharing chain.  Only relevant for the OptContext-managed
        // `last_guard_idx`; in_final_emission runs use
        // `Optimizer::last_guard_op_idx` instead.
        if op.opcode.is_guard() {
            if !self.in_final_emission {
                self.emit_guard_operation(&mut op);
            }
        } else if !self.in_final_emission {
            // optimizer.py:705-711: is_call_pure_pure_canraise — CallPure that
            // can_raise(ignore_memoryerror=True) counts as side-effectful even
            // though has_no_side_effect is true for call_pure opcodes.
            let dominated_by_side_effect = if (op.opcode.has_no_side_effect()
                || op.opcode.is_ovf()
                || op.opcode.is_jit_debug())
                && !Self::is_call_pure_pure_canraise(&op)
            {
                false
            } else {
                true
            };
            if dominated_by_side_effect {
                self.last_guard_idx = None;
            }
        }

        // Slice 0.5: dropping the redundant `register_value_type` write —
        // the op is about to be pushed into `new_operations`, after which
        // `op_at` resolves its intrinsic `op.type_` (resoperation.py:1693
        // parity) and `opref_type` returns it via the primary fast path.
        // The side-table entry would be dead code.
        self.new_operations.push(op);
        pos_ref
    }

    /// RPython emit_extra(op, emit=False) parity: queue an operation to
    /// be processed through passes AFTER the calling pass. Skips earlier
    /// passes (including the caller) to avoid re-absorption loops.
    /// `after_pass_idx`: index of the calling pass (op starts from idx+1).
    pub fn emit_extra(&mut self, after_pass_idx: usize, mut op: Op) -> OpRef {
        if op.pos.is_none() {
            // Slice 0.5 follow-up: typed allocation, same rationale as `emit`.
            op.pos = self.reserve_pos_typed(op.result_type());
        } else {
            self.next_pos = self.next_pos.max(op.pos.raw().saturating_add(1));
        }
        let pos_ref = op.pos;
        // Slice 0.5: queued ops carry their intrinsic `op.type_` (Slice
        // 0.1 / resoperation.py:1693 parity). Once the queued op flushes
        // through `propagate_one` into `new_operations`, `op_at` resolves
        // its type without the side-table detour.
        self.extra_operations_after
            .push_back((after_pass_idx + 1, op));
        pos_ref
    }

    pub fn initialize_imported_short_preamble_builder(
        &mut self,
        label_args: &[OpRef],
        short_inputargs: &[OpRef],
        exported_short_boxes: &[crate::optimizeopt::shortpreamble::PreambleOp],
    ) {
        let produced: Vec<(OpRef, crate::optimizeopt::shortpreamble::ProducedShortOp)> =
            exported_short_boxes
                .iter()
                .map(|entry| {
                    (
                        entry.op.pos,
                        crate::optimizeopt::shortpreamble::ProducedShortOp {
                            kind: entry.kind.clone(),
                            preamble_op: entry.op.clone(),
                            invented_name: entry.invented_name,
                            same_as_source: entry.same_as_source,
                        },
                    )
                })
                .collect();
        let mut builder = crate::optimizeopt::shortpreamble::ShortPreambleBuilder::new(
            label_args,
            &produced,
            short_inputargs,
        );
        for (&const_idx, value) in &self.const_pool {
            builder.note_known_constant(Self::const_ref_for_value(const_idx, value));
        }
        self.imported_short_preamble_builder = Some(builder);
        self.imported_short_preamble_used.clear();
    }

    /// Phase B.4.a: shortpreamble.py:409-430 ShortPreambleBuilder constructor parity.
    ///
    /// Reads `exported_state.short_boxes` (RPython `ExportedState.short_boxes`)
    /// directly, classifying each `ProducedShortOp.preamble_op.args` as
    /// Slot/Const/Produced at consume time and rebuilding the
    /// `ShortPreambleBuilder` keyed by Phase 2 OpRefs.
    ///
    /// Replaces the legacy `_from_exported_ops` function which read the
    /// `Vec<ExportedShortOp>` enum-serialization path. The new path
    /// matches RPython literally — no intermediate enum, polymorphism via
    /// `produce_op`-side data on `ProducedShortOp` itself.
    ///
    /// The Phase-2 result OpRef for each entry is read from `result_map`,
    /// computed before this constructor just as RPython already has the
    /// target `Box` identities before `ProducedShortOp.produce_op` runs.
    /// Args are resolved via local `produced_results` (source pos →
    /// resolved OpRef) and the caller-owned `imported_constants` map
    /// (source const OpRef → seeded fresh slot), which is reused by the
    /// following produce-op loop.
    pub fn initialize_imported_short_preamble_builder_from_short_boxes(
        &mut self,
        short_args: &[OpRef],
        short_inputargs: &[OpRef],
        short_boxes: &[(OpRef, crate::optimizeopt::shortpreamble::ProducedShortOp)],
        short_box_const_values: &HashMap<OpRef, majit_ir::Value>,
        result_map: &HashMap<OpRef, OpRef>,
        mut imported_constants: &mut HashMap<OpRef, OpRef>,
        exported_infos: &HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
    ) -> bool {
        use crate::optimizeopt::shortpreamble::{
            PreambleOpKind, ProducedShortOp, ShortPreambleBuilder,
        };

        // shortpreamble.py:414-426 ShortPreambleBuilder.__init__ parity:
        //
        //   for produced_op in short_boxes:
        //       op = produced_op.short_op.res
        //       preamble_op = produced_op.preamble_op
        //       if isinstance(op, Const):
        //           info = optimizer.getinfo(op)
        //       else:
        //           info = exported_infos.get(op, None)
        //           if info is None:
        //               info = empty_info
        //       preamble_op.set_forwarded(info)
        //
        // RPython sets `_forwarded = info` on each `preamble_op` so a
        // later `use_box` (≡ majit `force_op_from_preamble_op`) reads
        // the info via `get_forwarded()` and routes it through
        // `setinfo_from_preamble(box, info, None)` + `info.make_guards`.
        //
        // majit's `force_op_from_preamble_op` reads the equivalent via
        // `self.get_ptr_info(preamble_source)`; without this pre-seed,
        // info from `exported_infos` for sources outside
        // `next_iteration_args` (Pure / Heap / LoopInvariant short-op
        // sources) never reaches the use_box path.
        //
        // Guard against clobbering existing forwarding (set by an
        // earlier `replace_op` from `import_state` for sources that
        // happen to coincide with `next_iteration_args`): only seed
        // when no forwarding is recorded yet.
        //
        // Replay slot rule, matching `ImportedShortPureOp::new` (mod.rs:194)
        // and the producer-side `pop.preamble_op.pos` written by
        // `produce_pure` / `produce_heap_field` / `produce_heap_array_item` /
        // `produce_loop_invariant` in shortpreamble.rs.
        //
        // The rule reduces to: `replay slot = result_opref` iff the producer
        // installs `replace_op(source, result_opref)`, otherwise `source`.
        // PyPy `shortpreamble.py:401, 414` calls `preamble_op.set_forwarded(info)`
        // on the replay Op object — distinct from `PreambleOp.op = self.res`.
        // pyre's flat-OpRef model collapses the two onto one slot per OpRef,
        // so when `replace_op` is installed at `source`, the replay's
        // `_forwarded` slot must be moved to `result_opref` to avoid the
        // `Forwarded::Box(target)` chain clobbering the seeded info.
        //
        //   * invented Pure → result_opref. `produce_pure` installs
        //     `replace_op(source, result_opref)` (Fix #3).
        //   * Heap (field + array) → result_opref. `produce_heap_field` /
        //     `produce_heap_array_item` install `replace_op` (Cat-2.2 B/C).
        //   * LoopInvariant → result_opref. `produce_loop_invariant` installs
        //     `replace_op` (Cat-2.2 A); the synthetic `SameAsI` replay built
        //     by `optimize_CALL_LOOPINVARIANT_*` in `rewrite.rs` writes
        //     `replay.pos = ctx.get_box_replacement(source)` to land on the
        //     same slot.
        //   * non-invented Pure → source. PyPy `shortpreamble.py:120 op = self.res`
        //     with no forwarding installed; pyre keeps source's slot free for
        //     the seeded info.
        //
        // The same rule drives `seed_at` (`set_preamble_forwarded_info`),
        // the BUILDER's `replay.pos`, and the builder-side `produced_results`
        // dependency map so all four sources of replay identity stay in
        // lockstep with PyPy `shortpreamble.py:414-426` (`__init__`).
        let replay_pos = |source: OpRef, produced_op: &ProducedShortOp| -> OpRef {
            let installs_replace_op = match produced_op.kind {
                PreambleOpKind::Pure => produced_op.invented_name,
                PreambleOpKind::Heap | PreambleOpKind::LoopInvariant => true,
                PreambleOpKind::InputArg | PreambleOpKind::Guard => false,
            };
            if installs_replace_op {
                *result_map.get(&source).unwrap_or(&source)
            } else {
                source
            }
        };
        for (source, produced_op) in short_boxes {
            if let Some(info) = exported_infos.get(source) {
                self.set_preamble_forwarded_info(replay_pos(*source, produced_op), info);
            }
        }

        let mut produced: Vec<(OpRef, ProducedShortOp)> = Vec::with_capacity(short_boxes.len());
        let mut produced_results: HashMap<OpRef, OpRef> = HashMap::new();
        // shortpreamble.py:PreambleOp.add_op_to_short — Pure ops whose
        // opcode is a Call get rewritten to the CallPure* equivalent so
        // the short preamble can replay the cached call without
        // re-executing arbitrary side effects.
        let pure_call_opcode = |op: OpCode| -> OpCode {
            match op {
                OpCode::CallI => OpCode::CallPureI,
                OpCode::CallR => OpCode::CallPureR,
                OpCode::CallF => OpCode::CallPureF,
                OpCode::CallN => OpCode::CallPureN,
                other => other,
            }
        };
        // shortpreamble.py:PreambleOp.add_op_to_short — LoopInvariant ops
        // become CallLoopinvariant* so the short preamble re-executes the
        // call exactly once per loop iteration.
        let loop_invariant_opcode = |result_type: majit_ir::Type| -> OpCode {
            match result_type {
                majit_ir::Type::Int => OpCode::CallLoopinvariantI,
                majit_ir::Type::Ref => OpCode::CallLoopinvariantR,
                majit_ir::Type::Float => OpCode::CallLoopinvariantF,
                majit_ir::Type::Void => OpCode::CallLoopinvariantN,
            }
        };
        // shortpreamble.py:283 `ShortBoxes.produce_arg` — classify an arg
        // through the shared classifier, then collapse the Slot/Const/
        // Produced variants down to the Phase-2 OpRef the builder needs.
        // Sharing the classifier with `ProducedShortOp::produce_op` (which
        // also dispatches via `classify_short_arg`) keeps the two consume
        // sites locked to a single rule, mirroring RPython's single
        // `produce_arg` path.
        let resolve_arg = |arg: OpRef,
                           ctx: &mut Self,
                           produced_results: &HashMap<OpRef, OpRef>,
                           imported_constants: &mut HashMap<OpRef, OpRef>|
         -> Option<OpRef> {
            crate::optimizeopt::shortpreamble::classify_short_arg(
                ctx,
                arg,
                short_inputargs,
                short_args,
                produced_results,
                imported_constants,
                short_box_const_values,
            )
            .map(|cls| match cls {
                crate::optimizeopt::ImportedShortPureArg::OpRef(r) => r,
                crate::optimizeopt::ImportedShortPureArg::Const(_, r) => r,
            })
        };

        for (source, produced_op) in short_boxes {
            // Some ProducedShortOps (PreambleOpKind::Heap with non-getfield /
            // non-getarrayitem opcodes, or other non-emitting entries) have
            // no Phase-2 result. They are no-ops for the imported builder.
            let result_opref = match result_map.get(source).copied() {
                Some(r) => r,
                None => continue,
            };
            match produced_op.kind {
                PreambleOpKind::Pure => {
                    let mut resolved_args = Vec::with_capacity(produced_op.preamble_op.args.len());
                    for &arg in &produced_op.preamble_op.args {
                        let Some(resolved) =
                            resolve_arg(arg, self, &produced_results, &mut imported_constants)
                        else {
                            return false;
                        };
                        resolved_args.push(resolved);
                    }
                    let mut op = Op::new(
                        pure_call_opcode(produced_op.preamble_op.opcode),
                        &resolved_args,
                    );
                    op.pos = replay_pos(*source, produced_op);
                    op.descr = produced_op.preamble_op.descr.clone();
                    let new_pop = ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        preamble_op: op,
                        invented_name: produced_op.invented_name,
                        same_as_source: produced_op.same_as_source,
                    };
                    produced.push((*source, new_pop.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, new_pop));
                    }
                    produced_results.insert(*source, replay_pos(*source, produced_op));
                }
                PreambleOpKind::Heap => {
                    let result_type = produced_op.preamble_op.result_type();
                    let descr = match produced_op.preamble_op.descr.clone() {
                        Some(d) => d,
                        None => continue,
                    };
                    let object_arg = produced_op.preamble_op.arg(0);
                    let Some(obj) =
                        resolve_arg(object_arg, self, &produced_results, &mut imported_constants)
                    else {
                        return false;
                    };
                    let new_pop = match produced_op.preamble_op.opcode {
                        OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                            let opcode = match result_type {
                                majit_ir::Type::Int => OpCode::GetfieldGcI,
                                majit_ir::Type::Ref => OpCode::GetfieldGcR,
                                majit_ir::Type::Float => OpCode::GetfieldGcF,
                                majit_ir::Type::Void => return false,
                            };
                            let mut op = Op::new(opcode, &[obj]);
                            op.pos = replay_pos(*source, produced_op);
                            op.descr = Some(descr);
                            ProducedShortOp {
                                kind: PreambleOpKind::Heap,
                                preamble_op: op,
                                invented_name: produced_op.invented_name,
                                same_as_source: produced_op.same_as_source,
                            }
                        }
                        OpCode::GetarrayitemGcI
                        | OpCode::GetarrayitemGcR
                        | OpCode::GetarrayitemGcF => {
                            let opcode = match result_type {
                                majit_ir::Type::Int => OpCode::GetarrayitemGcI,
                                majit_ir::Type::Ref => OpCode::GetarrayitemGcR,
                                majit_ir::Type::Float => OpCode::GetarrayitemGcF,
                                majit_ir::Type::Void => return false,
                            };
                            // shortpreamble.py:81 `g.getarg(1).getint()`:
                            // pull the integer VALUE through the shared
                            // `classify_short_arg` rule, which checks
                            // `short_box_const_values` (producer snapshot)
                            // first then the consumer ctx const pool.
                            // `OpRef::raw()` is a tagged trace position —
                            // not the constant integer value.
                            let index_arg = produced_op.preamble_op.arg(1);
                            let index_opref = match resolve_arg(
                                index_arg,
                                self,
                                &produced_results,
                                imported_constants,
                            ) {
                                Some(r) => r,
                                None => return false,
                            };
                            let mut op = Op::new(opcode, &[obj, index_opref]);
                            op.pos = replay_pos(*source, produced_op);
                            op.descr = Some(descr);
                            ProducedShortOp {
                                kind: PreambleOpKind::Heap,
                                preamble_op: op,
                                invented_name: produced_op.invented_name,
                                same_as_source: produced_op.same_as_source,
                            }
                        }
                        _ => continue,
                    };
                    produced.push((*source, new_pop.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, new_pop));
                    }
                    produced_results.insert(*source, replay_pos(*source, produced_op));
                }
                PreambleOpKind::LoopInvariant => {
                    let result_type = produced_op.preamble_op.result_type();
                    let Some(func_opref) = resolve_arg(
                        produced_op.preamble_op.arg(0),
                        self,
                        &produced_results,
                        imported_constants,
                    ) else {
                        return false;
                    };
                    if self.get_constant_int(func_opref).is_none() {
                        return false;
                    }
                    let mut op = Op::new(loop_invariant_opcode(result_type), &[func_opref]);
                    op.pos = replay_pos(*source, produced_op);
                    let new_pop = ProducedShortOp {
                        kind: PreambleOpKind::LoopInvariant,
                        preamble_op: op,
                        invented_name: produced_op.invented_name,
                        same_as_source: produced_op.same_as_source,
                    };
                    produced.push((*source, new_pop.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, new_pop));
                    }
                    produced_results.insert(*source, replay_pos(*source, produced_op));
                }
                PreambleOpKind::InputArg | PreambleOpKind::Guard => {}
            }
        }

        let mut builder = ShortPreambleBuilder::new(short_args, &produced, short_inputargs);
        for (&const_idx, value) in &self.const_pool {
            builder.note_known_constant(Self::const_ref_for_value(const_idx, value));
        }
        for &opref in imported_constants.values() {
            builder.note_known_constant(opref);
        }
        self.imported_short_preamble_builder = Some(builder);
        self.imported_short_preamble_used.clear();
        true
    }

    /// unroll.py:26-39: force_op_from_preamble(preamble_op)
    ///
    /// RPython receives a PreambleOp with invented_name already set.
    /// Calls use_box then registers in potential_extra_ops.
    pub fn force_op_from_preamble_op(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
    ) -> OpRef {
        let preamble_source = preamble_op.op;
        // RPython `return preamble_op.op` returns the carried Box. In majit,
        // `pop.op` stores the Phase 1 source position; `replace_op(source,
        // body_visible)` is called by the producer for invented Pure / Heap /
        // LoopInvariant, so walking the forwarding chain reaches the
        // body-visible OpRef. Non-invented Pure has no forwarding installed,
        // so `get_box_replacement(source) == source` and the body references
        // source directly (RPython parity for non-invented `op = self.res`).
        let result = self.get_box_replacement(preamble_op.op);
        let result_type = preamble_op.preamble_op.result_type();
        let is_constant = self.get_constant(preamble_source).is_some();
        if self.imported_short_preamble_used.insert(preamble_source) {
            // unroll.py:32: use_box(op, preamble_op.preamble_op, self).
            // RPython passes the preamble_op directly — no lookup miss possible.
            // majit prefers the produced_short_boxes lookup (Phase-2 remapped pos)
            // with fallback to info::PreambleOp.preamble_op.
            let (arg_guards, result_guards) = self.collect_use_box_guards(&preamble_op.preamble_op);
            // unroll.py:28: assert self.short_preamble_producer is not None
            if let Some(mut builder) = self.active_short_preamble_producer.take() {
                builder.use_box(
                    preamble_source,
                    &preamble_op.preamble_op,
                    &arg_guards,
                    &result_guards,
                );
                self.active_short_preamble_producer = Some(builder);
            } else if let Some(mut builder) = self.imported_short_preamble_builder.take() {
                builder.use_box(
                    preamble_source,
                    &preamble_op.preamble_op,
                    &arg_guards,
                    &result_guards,
                );
                self.imported_short_preamble_builder = Some(builder);
            } else {
                unreachable!("force_op_from_preamble_op: no short_preamble_producer");
            }
            // shortpreamble.py:401-405: info = preamble_op.get_forwarded();
            // preamble_op.set_forwarded(None);
            // optimizer.setinfo_from_preamble(box, info, None)
            //
            // RPython reads `_forwarded` from the replay Op object
            // (`preamble_op`), NOT from `preamble_op.op` (= self.res or
            // the alt for invented). pyre's flat-OpRef equivalent is
            // `pop.preamble_op.pos` — the OpRef the replay Op was
            // constructed at by `ImportedShortPureOp::new` (mod.rs:144).
            // For invented Pure that OpRef differs from `pop.op` (the
            // alt) so the alt's `replace_op(...)` chain at
            // `forwarded[pop.op]` does not collide with the replay's
            // info at `forwarded[pop.preamble_op.pos]`.
            if let Some(info) = self.take_preamble_forwarded_opinfo(preamble_op.preamble_op.pos) {
                self.setinfo_from_preamble_item_option(result, &info, None);
            }
            // RPython PreambleOp carries Box.type intrinsically. majit's
            // imported replay uses a distinct `resolved` OpRef, so preserve
            // the replay result type on first force when the import path did
            // not already seed `value_types` (e.g. focused unit fixtures).
            self.value_types.entry(result.raw()).or_insert(result_type);
            // unroll.py:34-37: potential_extra_ops[op] = preamble_op
            if !is_constant {
                // unroll.py:35-36: invented_name → get_box_replacement(op)
                let key = if preamble_op.invented_name {
                    self.get_box_replacement(preamble_source)
                } else {
                    preamble_source
                };
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit] potential_extra_ops.insert key={key:?} source={preamble_source:?} result={result:?} invented={}",
                        preamble_op.invented_name
                    );
                }
                self.potential_extra_ops.insert(key, preamble_op.clone());
            }
        }
        // unroll.py:38 `return preamble_op.op`. RPython's `preamble_op.op`
        // equals `self.res` (shortpreamble.py:120 `op = self.res`); pyre's
        // Phase 1 source IS `self.res` for the imported short box. Return
        // it directly so non-invented Pure body references resolve through
        // the Phase 1 OpRef. The use-before-def pass at LABEL emission
        // (unroll.rs `assemble_peeled_trace_with_jump_args`) extends
        // `LABEL.arglist` with that Phase 1 OpRef when the body actually
        // uses it; the orthodox `force_box → potential_extra_ops.pop →
        // add_preamble_op` path (shortpreamble.py:432-440) handles
        // `used_boxes` / `short_preamble_jump` / `extra_same_as` for the
        // imported short box.
        let _ = result;
        preamble_source
    }

    /// shortpreamble.py:383-396,401-406: collect guards from the forwarded
    /// info of preamble_op's args and result. RPython's `info = arg.get_forwarded()`
    /// returns whatever is stored — PtrInfo *or* IntBound — and calls
    /// `info.make_guards(...)` uniformly.
    fn collect_use_box_guards(&mut self, preamble_op: &Op) -> (Vec<Op>, Vec<Op>) {
        // shortpreamble.py:383-396: guards for InputArg args only
        let short_inputargs: Vec<OpRef> = self
            .imported_short_preamble_builder
            .as_ref()
            .map(|b| b.short_inputargs().to_vec())
            .or_else(|| {
                self.active_short_preamble_producer
                    .as_ref()
                    .map(|b| b.short_inputargs().to_vec())
            })
            .unwrap_or_default();

        // shortpreamble.py:383-401 line-by-line:
        //
        //   for arg in preamble_op.getarglist():
        //       if isinstance(arg, Const):
        //           continue
        //       if isinstance(arg, AbstractInputArg):
        //           info = arg.get_forwarded()
        //           if info is not None and info is not empty_info:
        //               info.make_guards(arg, self.short, optimizer)
        //       elif arg.get_forwarded() is None:
        //           pass
        //       else:
        //           self.short.append(arg)
        //           info = arg.get_forwarded()
        //           if info is not empty_info:
        //               info.make_guards(arg, self.short, optimizer)
        //           arg.set_forwarded(None)
        //
        // RPython has three branches per arg:
        //   * Const → skip (pyre: `OpRef::is_constant()`).
        //   * AbstractInputArg with forwarded info → emit guards, do NOT
        //     clear (info lives across iterations on input args).
        //   * non-input non-Const with forwarded info → also append the
        //     arg op to `self.short` (handled by the builder's `use_box`
        //     dependency walk at shortpreamble.rs:1660-1688), emit
        //     guards, AND clear the slot to prevent double-emission.
        //
        // pyre's `take_preamble_forwarded_opinfo` is the take-clear
        // primitive matching `arg.set_forwarded(None)`. We use it for the
        // non-input branch only; input args use the read-only snapshot.
        enum ForwardedInfo {
            // info.py:600 PtrInfo + ConstPtrInfo (info.py:706). PtrInfo
            // dispatches further to ConstPtrInfo::make_guards when the
            // PtrInfo is a Constant variant.
            Ptr(PtrInfo),
            // intutils.py:1264 IntBound::make_guards. Constant ints come
            // through this arm via IntBound::is_constant().
            Int(crate::optimizeopt::intutils::IntBound),
            // info.py:851 FloatConstInfo carries a single ConstFloat;
            // make_guards emits a GUARD_VALUE pinning `op` to the
            // ConstFloat.
            FloatConst(f64),
        }
        let snapshot_forwarded = |ctx: &Self, arg: OpRef| -> Option<ForwardedInfo> {
            // shortpreamble.py:387 `info = arg.get_forwarded()` — PyPy
            // returns the AbstractInfo subtype stored in `_forwarded`.
            // Pyre's BoxRef carries:
            //   `Forwarded::Info(OpInfo::Ptr(_))` — info.py:600 PtrInfo
            //   `Forwarded::Info(OpInfo::IntBound(_))` — intutils.py
            //   `Forwarded::Info(OpInfo::FloatConst(_))` — info.py:851
            // Const-forwarded boxes (`Forwarded::Box(constbox)` from
            // optimizer.py:432, or legacy `OpInfo::Constant`) are not
            // reachable here in production: PyPy use_box never sees a
            // Const in `_forwarded` because make_constant is not applied
            // to short-preamble inputarg boxes, and pyre matches that
            // call shape. Returning None on those arms preserves literal
            // PyPy use_box structure.
            let b = ctx.box_pool.get(arg.raw() as usize)?;
            use crate::optimizeopt::info::OpInfo;
            match &*b.get_forwarded() {
                crate::r#box::Forwarded::Info(OpInfo::Ptr(info)) => {
                    Some(ForwardedInfo::Ptr(info.clone()))
                }
                crate::r#box::Forwarded::Info(OpInfo::IntBound(b)) => {
                    Some(ForwardedInfo::Int(b.clone()))
                }
                crate::r#box::Forwarded::Info(OpInfo::FloatConst(f)) => {
                    Some(ForwardedInfo::FloatConst(*f))
                }
                _ => None,
            }
        };
        // Phase 1 (read-only): classify each arg per the PyPy three-branch
        // shape and snapshot the info-bearing slot.
        struct ArgEntry {
            arg: OpRef,
            info: ForwardedInfo,
            is_input: bool,
        }
        let mut arg_entries: Vec<ArgEntry> = Vec::new();
        for &arg in preamble_op.args.iter() {
            // Branch 1: shortpreamble.py:384 `isinstance(arg, Const): continue`.
            if arg.is_constant() || arg.is_none() {
                continue;
            }
            let is_input = short_inputargs.contains(&arg);
            if let Some(info) = snapshot_forwarded(self, arg) {
                arg_entries.push(ArgEntry {
                    arg,
                    info,
                    is_input,
                });
            }
            // shortpreamble.py:391 `elif arg.get_forwarded() is None: pass`
            // is the no-info branch; falling out of `snapshot_forwarded`
            // returning None is the equivalent.
        }
        let result_info: Option<(OpRef, ForwardedInfo)> =
            snapshot_forwarded(self, preamble_op.pos).map(|info| (preamble_op.pos, info));

        // Phase 2 (mutable): clear non-input arg slots — PyPy
        // `arg.set_forwarded(None)` (shortpreamble.py:397). Branch 2 (input
        // args) keeps its info; branch 3 (non-input) clears.
        for entry in &arg_entries {
            if !entry.is_input {
                let _ = self.take_preamble_forwarded_opinfo(entry.arg);
            }
        }

        // Phase 3: generate guards — alloc_const directly allocates constant
        // OpRefs, matching RPython where ConstInt/ConstPtr are created inline.
        let mut arg_guards = Vec::new();
        let mut alloc_const = |value: Value| -> OpRef {
            let pos = self.reserve_const_ref(value.get_type());
            self.seed_constant(pos, value);
            pos
        };
        // info.py:861 FloatConstInfo.make_guards / ConstPtrInfo path —
        // single-value info classes emit a GUARD_VALUE that pins `op` to
        // the recorded constant.
        let emit_const_guard =
            |arg: OpRef,
             value: &Value,
             guards: &mut Vec<Op>,
             alloc: &mut dyn FnMut(Value) -> OpRef| {
                let c = alloc(value.clone());
                guards.push(Op::new(OpCode::GuardValue, &[arg, c]));
            };
        for entry in &arg_entries {
            match &entry.info {
                ForwardedInfo::Ptr(p) => {
                    p.make_guards(entry.arg, &mut arg_guards, &mut alloc_const)
                }
                ForwardedInfo::Int(b) => {
                    b.make_guards(entry.arg, &mut arg_guards, &mut alloc_const)
                }
                ForwardedInfo::FloatConst(f) => emit_const_guard(
                    entry.arg,
                    &Value::Float(*f),
                    &mut arg_guards,
                    &mut alloc_const,
                ),
            }
        }
        let mut result_guards = Vec::new();
        if let Some((result_ref, info)) = &result_info {
            match info {
                ForwardedInfo::Ptr(p) => {
                    p.make_guards(*result_ref, &mut result_guards, &mut alloc_const)
                }
                ForwardedInfo::Int(b) => {
                    b.make_guards(*result_ref, &mut result_guards, &mut alloc_const)
                }
                ForwardedInfo::FloatConst(f) => emit_const_guard(
                    *result_ref,
                    &Value::Float(*f),
                    &mut result_guards,
                    &mut alloc_const,
                ),
            }
        }
        (arg_guards, result_guards)
    }

    /// shortpreamble.py:425 `preamble_op.set_forwarded(info)` for imported
    /// short preamble ops. Store the same family of info values that RPython
    /// stores in `_forwarded`, without transforming them through
    /// `setinfo_from_preamble` yet.
    fn set_preamble_forwarded_info(
        &mut self,
        source: OpRef,
        info: &crate::optimizeopt::info::OpInfo,
    ) {
        use crate::optimizeopt::info::OpInfo;
        if source.is_constant() || self.has_forwarding(source) {
            return;
        }
        let idx = source.raw() as usize;
        // BoxRef-direct write — authoritative.
        let b = self.ensure_box_at(idx);
        match info {
            OpInfo::Unknown => b.clear_forwarded(),
            other => b.set_forwarded_info(other.clone()),
        }
    }

    /// shortpreamble.py:401-405 line-by-line:
    ///
    /// ```python
    /// info = preamble_op.get_forwarded()
    /// preamble_op.set_forwarded(None)
    /// if optimizer is not None:
    ///     optimizer.setinfo_from_preamble(box, info, None)
    /// ```
    ///
    /// RPython reads `_forwarded` from the `preamble_op` Op object directly
    /// — `get_box_replacement` is NOT applied to the slot. Box replacement
    /// only matters for the `box` argument that subsequently receives the
    /// info via `setinfo_from_preamble(box, info, None)`. Walking the
    /// replacement chain on the source side would point at the body-visible
    /// OpRef whose slot is empty (the seed at `forwarded[source]` was
    /// installed by `set_preamble_forwarded_info`).
    ///
    /// `set_forwarded(None)` clears the slot so a second `use_box` for the
    /// same preamble op never re-fires `info.make_guards`. In majit's flat
    /// OpRef model the slot is shared with the Box→Box replacement chain
    /// (`Forwarded::Box`), which other code follows via
    /// `get_box_replacement`; clearing that variant would silently break
    /// downstream replacement, so only the info-bearing variants
    /// (Info / IntBound / Const) take + clear, matching PyPy's clear
    /// semantics on the info-bearing branches.
    fn take_preamble_forwarded_opinfo(
        &mut self,
        source: OpRef,
    ) -> Option<crate::optimizeopt::info::OpInfo> {
        use crate::optimizeopt::info::OpInfo;
        let idx = source.raw() as usize;
        // BoxRef-authoritative read. PyPy stores the replay op's forwarded
        // info directly on `preamble_op._forwarded`; pyre stores the same
        // state in the BoxRef slot keyed by `source.raw()`. Non-constant
        // `Forwarded::Box(target)` is a replacement chain and is excluded.
        // Const targets can still appear from legacy bridge/fixture replay
        // paths; normalize them to the OpInfo shape consumed by
        // `setinfo_from_preamble_item_option`.
        let b = self.box_pool.get(idx).cloned()?;
        let result = {
            let fwd = b.get_forwarded();
            match &*fwd {
                crate::r#box::Forwarded::Info(OpInfo::Ptr(p)) => Some(OpInfo::Ptr(p.clone())),
                crate::r#box::Forwarded::Info(OpInfo::IntBound(ib)) => {
                    Some(OpInfo::IntBound(ib.clone()))
                }
                crate::r#box::Forwarded::Info(OpInfo::FloatConst(f)) => {
                    Some(OpInfo::FloatConst(*f))
                }
                crate::r#box::Forwarded::Info(OpInfo::Constant(v)) => {
                    Some(OpInfo::Constant(v.clone()))
                }
                crate::r#box::Forwarded::Box(target) if target.is_constant() => {
                    target.const_value().map(|v| match v {
                        Value::Float(f) => OpInfo::FloatConst(f),
                        v => OpInfo::Constant(v),
                    })
                }
                _ => None,
            }
        };
        if result.is_some() {
            // shortpreamble.py:401 preamble_op.set_forwarded(None).
            b.clear_forwarded();
        }
        result
    }

    /// unroll.py:53-98: setinfo_from_preamble(op, preamble_info, exported_infos)
    /// RPython uses sequential `if` (not elif) so multiple properties accumulate.
    /// `exported_infos`: None from use_box path (shortpreamble.py:404),
    /// Some from import_state path. When None, virtual branch does NOT recurse.
    fn setinfo_from_preamble(
        &mut self,
        op: OpRef,
        preamble_info: &PtrInfo,
        exported_infos: Option<&HashMap<OpRef, crate::optimizeopt::info::OpInfo>>,
    ) {
        let op = self.get_box_replacement(op);
        // unroll.py:55: if op.get_forwarded() is not None: return
        // (covers Op redirect + Info + IntBound + Const states uniformly,
        // matching the sibling setinfo_from_preamble_item pattern below.)
        if self.has_forwarding(op) {
            return;
        }
        // unroll.py:57: if op.is_constant(): return
        if self.is_constant(op) {
            return;
        }

        // unroll.py:60-64: virtual — set_forwarded + recurse, then return
        if preamble_info.is_virtual() {
            self.set_ptr_info(op, preamble_info.clone());
            if let Some(infos) = exported_infos {
                let items: Vec<OpRef> = match preamble_info {
                    PtrInfo::Virtual(v) => v.fields.iter().map(|(_, r)| *r).collect(),
                    PtrInfo::VirtualArray(a) => a.items.iter().copied().collect(),
                    PtrInfo::VirtualStruct(s) => s.fields.iter().map(|(_, r)| *r).collect(),
                    PtrInfo::VirtualArrayStruct(a) => a
                        .element_fields
                        .iter()
                        .flat_map(|row| row.iter().map(|(_, r)| *r))
                        .collect(),
                    PtrInfo::VirtualRawBuffer(r) => r.values.clone(),
                    _ => Vec::new(),
                };
                self.setinfo_from_preamble_list(&items, infos);
            }
            return;
        }

        // unroll.py:65-68: constant — return early
        if let PtrInfo::Constant(gcref) = preamble_info {
            self.make_constant(op, Value::Ref(*gcref));
            return;
        }

        // --- Sequential checks (RPython: NOT elif, all accumulate) ---

        // unroll.py:69-74: Struct/Instance with descr → set_forwarded
        if preamble_info.get_descr().is_some() {
            if let PtrInfo::Struct(sinfo) = preamble_info {
                self.set_ptr_info(op, PtrInfo::struct_ptr(sinfo.descr.clone()));
            }
            if let PtrInfo::Instance(iinfo) = preamble_info {
                self.set_ptr_info(op, PtrInfo::instance(iinfo.descr.clone(), None));
            }
        }

        // unroll.py:75-77: known_class → make_constant_class(op, class, False)
        if let Some(cls) = preamble_info.get_known_class() {
            crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                self,
                op,
                cls.0 as i64,
                false, // update_last_guard=False (unroll.py:77)
            );
        }

        // unroll.py:79-84: ArrayPtrInfo → set_forwarded(ArrayPtrInfo(descr, lenbound))
        if let PtrInfo::Array(ainfo) = preamble_info {
            self.set_ptr_info(
                op,
                PtrInfo::array(ainfo.descr.clone(), ainfo.lenbound.clone()),
            );
        }

        // unroll.py:85-89: StrPtrInfo — clone lenbound
        if let PtrInfo::Str(sinfo) = preamble_info {
            let mut new_info = crate::optimizeopt::info::StrPtrInfo {
                lenbound: sinfo.lenbound.clone(),
                lgtop: None,
                mode: sinfo.mode,
                length: -1,
                // unroll.py:86: StrPtrInfo(preamble_info.mode) — always
                // rebuild a plain non-virtual StrPtrInfo; never carry
                // the previous iteration's virtual variant across.
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            };
            if new_info.lenbound.is_none() {
                new_info.lenbound = Some(crate::optimizeopt::intutils::IntBound::nonnegative());
            }
            self.set_ptr_info(op, PtrInfo::Str(new_info));
            return;
        }

        // unroll.py:91-92: is_nonnull → make_nonnull
        if preamble_info.is_nonnull() {
            self.make_nonnull(op);
        }
    }

    /// unroll.py:41-51 setinfo_from_preamble_list(lst, infos):
    ///
    /// ```python
    /// def setinfo_from_preamble_list(self, lst, infos):
    ///     for item in lst:
    ///         if item is None:
    ///             continue
    ///         i = infos.get(item, None)
    ///         if i is not None:
    ///             self.setinfo_from_preamble(item, i, infos)
    ///         else:
    ///             item.set_forwarded(None)
    ///             # let's not inherit stuff we don't
    ///             # know anything about
    /// ```
    ///
    /// Every `infos.get(item) is not None` branch funnels through
    /// `setinfo_from_preamble`, which starts with the early-return checks
    /// at unroll.py:54-58 (`get_box_replacement` + `get_forwarded` +
    /// `is_constant`). A shortcut that applies IntBound / FloatConst /
    /// Constant without those checks overwrites already-forwarded boxes.
    /// `setinfo_from_preamble_item` below is the shared dispatcher: it
    /// does the checks once and then routes to the variant-specific
    /// logic, so this method becomes the literal unroll.py loop body.
    fn setinfo_from_preamble_list(
        &mut self,
        items: &[OpRef],
        exported_infos: &HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
    ) {
        for &item in items {
            if item.is_none() {
                continue;
            }
            // unroll.py:45-46: i = infos.get(item, None)
            match exported_infos.get(&item).cloned() {
                Some(info) => {
                    // unroll.py:47: self.setinfo_from_preamble(item, i, infos)
                    self.setinfo_from_preamble_item(item, &info, exported_infos);
                }
                None => {
                    // unroll.py:49: item.set_forwarded(None)
                    // "let's not inherit stuff we don't know anything about"
                    self.clear_forwarded(item);
                }
            }
        }
    }

    /// unroll.py:53-98 `setinfo_from_preamble(op, preamble_info, exported_infos)`.
    ///
    /// Shared dispatcher covering the `isinstance(preamble_info, ...)` chain
    /// at unroll.py:59, 93, 97 — used by both `setinfo_from_preamble_list`
    /// (mod.rs recursive virtual field walker) and `OptUnroll::import_state`
    /// (unroll.rs top-level import). Centralising the dispatch avoids
    /// diverging shortcuts that skip the early-return checks.
    fn setinfo_from_preamble_item(
        &mut self,
        op: OpRef,
        preamble_info: &crate::optimizeopt::info::OpInfo,
        exported_infos: &HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
    ) {
        use crate::optimizeopt::info::OpInfo;
        // unroll.py:53-54 `op = get_box_replacement(op)`
        let target = self.get_box_replacement(op);
        // unroll.py:55-56 `if op.get_forwarded() is not None: return`
        if self.has_forwarding(target) {
            return;
        }
        // unroll.py:57-58 `if op.is_constant(): return`
        if self.is_constant(target) {
            return;
        }
        match preamble_info {
            // unroll.py:65-68 ConstPtrInfo: set_forwarded(preamble_info.getconst())
            OpInfo::Ptr(crate::optimizeopt::info::PtrInfo::Constant(gcref)) => {
                self.make_constant(target, Value::Ref(*gcref));
            }
            // unroll.py:59-92 general PtrInfo dispatch.
            OpInfo::Ptr(ptr_info) => {
                self.setinfo_from_preamble(target, ptr_info, Some(exported_infos));
            }
            // unroll.py:93-96 IntBound with widen(): intersect unconditionally.
            OpInfo::IntBound(bound) => {
                let widened = bound.widen();
                self.with_intbound_mut(target, |bm| {
                    let _ = bm.intersect(&widened);
                });
            }
            // unroll.py:97-98 FloatConstInfo: op.set_forwarded(preamble_info._const)
            OpInfo::FloatConst(f) => {
                self.make_constant(target, Value::Float(*f));
            }
            // Direct `OpInfo::Constant(Value)` — covers Int/other constant
            // forwarding exported for constant OpRefs without a dedicated
            // IntBound/FloatConst entry.
            OpInfo::Constant(value) => {
                self.make_constant(target, value.clone());
            }
            // unroll.py:53-98 has no dispatch arm for "no info" — the
            // caller never stores an `Unknown` entry in `exported_infos`
            // (see `collect_exported_info`'s `None` return at
            // unroll.rs:2889 mirroring unroll.py:440 `if info:`).
            OpInfo::Unknown => unreachable!(
                "exported_infos must never contain OpInfo::Unknown; \
                 the absent-entry branch (clear_forwarded) handles that case"
            ),
        }
    }

    fn setinfo_from_preamble_item_option(
        &mut self,
        op: OpRef,
        preamble_info: &crate::optimizeopt::info::OpInfo,
        exported_infos: Option<&HashMap<OpRef, crate::optimizeopt::info::OpInfo>>,
    ) {
        use crate::optimizeopt::info::OpInfo;
        let target = self.get_box_replacement(op);
        if self.has_forwarding(target) || self.is_constant(target) {
            return;
        }
        match preamble_info {
            OpInfo::Ptr(ptr_info) => {
                self.setinfo_from_preamble(target, ptr_info, exported_infos);
            }
            OpInfo::IntBound(bound) => {
                let widened = bound.widen();
                self.with_intbound_mut(target, |bm| {
                    let _ = bm.intersect(&widened);
                });
            }
            OpInfo::FloatConst(f) => {
                self.make_constant(target, Value::Float(*f));
            }
            OpInfo::Constant(value) => {
                self.make_constant(target, value.clone());
            }
            OpInfo::Unknown => {}
        }
    }

    /// unroll.py:49: item.set_forwarded(None)
    fn clear_forwarded(&mut self, opref: OpRef) {
        let idx = opref.raw() as usize;
        if let Some(b) = self.box_pool.get(idx).cloned() {
            b.clear_forwarded();
        }
    }

    /// optimizer.py:354: potential_extra_ops.pop(op)
    pub fn take_potential_extra_op(
        &mut self,
        result: OpRef,
    ) -> Option<crate::optimizeopt::info::PreambleOp> {
        self.potential_extra_ops.remove(&result)
    }

    pub fn activate_short_preamble_producer(
        &mut self,
        builder: crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder,
    ) {
        self.active_short_preamble_producer = Some(builder);
    }

    pub fn active_short_preamble_producer_mut(
        &mut self,
    ) -> Option<&mut crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.as_mut()
    }

    pub fn build_active_short_preamble(
        &self,
    ) -> Option<crate::optimizeopt::shortpreamble::ShortPreamble> {
        self.active_short_preamble_producer.as_ref().map(|builder| {
            builder.build_short_preamble_struct(&HashMap::new(), &self.constant_types_for_numbering)
        })
    }

    pub fn take_active_short_preamble_producer(
        &mut self,
    ) -> Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.take()
    }

    pub fn build_imported_short_preamble(
        &self,
    ) -> Option<crate::optimizeopt::shortpreamble::ShortPreamble> {
        self.imported_short_preamble_builder
            .as_ref()
            .map(|builder| {
                // RPython parity: extract constant pool from OptContext.
                // In RPython, Const objects in short preamble ops survive
                // across compilations via GC tracing. In majit, we must
                // snapshot the constant pool so build_short_preamble_struct
                // can capture referenced constants.
                let value_to_raw = |val: &majit_ir::Value| -> i64 {
                    match val {
                        majit_ir::Value::Int(v) => *v,
                        majit_ir::Value::Float(f) => f.to_bits() as i64,
                        majit_ir::Value::Ref(r) => r.0 as i64,
                        majit_ir::Value::Void => 0,
                    }
                };
                let value_to_type = |val: &majit_ir::Value| -> majit_ir::Type {
                    match val {
                        majit_ir::Value::Int(_) => majit_ir::Type::Int,
                        majit_ir::Value::Float(_) => majit_ir::Type::Float,
                        majit_ir::Value::Ref(_) => majit_ir::Type::Ref,
                        majit_ir::Value::Void => majit_ir::Type::Void,
                    }
                };
                let mut loop_constants: HashMap<u32, i64> = self
                    .const_pool
                    .iter()
                    .map(|(&i, val)| (Self::const_ref_for_value(i, val).raw(), value_to_raw(val)))
                    .collect();
                // `initialize_imported_short_preamble_builder_from_exported_ops`
                // (mod.rs:1693) imports cross-trace constants by allocating a
                // fresh local OpRef via `alloc_op_position()` and seeding it
                // with `make_constant`, which stores the value in
                // `self.constants` (Vec) rather than `self.const_pool` (the
                // CONST_BIT namespace). Those positions are still constants
                // for the purposes of short-preamble inlining, so include
                // them in `loop_constants` so `build_short_preamble_struct`'s
                // arg scan can recognise the seeded value and the downstream
                // `ExtendedShortPreambleBuilder::setup` `constants_set` check
                // accepts them as resolved deps.
                for (idx, slot) in self.constants.iter().enumerate() {
                    if let Some(val) = slot {
                        loop_constants
                            .entry(idx as u32)
                            .or_insert(value_to_raw(val));
                    }
                }
                let mut loop_constant_types = self.constant_types_for_numbering.clone();
                for (&i, val) in &self.const_pool {
                    loop_constant_types
                        .entry(Self::const_ref_for_value(i, val).raw())
                        .or_insert(value_to_type(val));
                }
                for (idx, slot) in self.constants.iter().enumerate() {
                    if let Some(val) = slot {
                        loop_constant_types
                            .entry(idx as u32)
                            .or_insert(value_to_type(val));
                    }
                }
                builder.build_short_preamble_struct(&loop_constants, &loop_constant_types)
            })
    }

    pub fn used_imported_short_aliases(&self) -> Vec<ImportedShortAlias> {
        self.imported_short_preamble_builder
            .as_ref()
            .map(|builder| {
                builder
                    .extra_same_as()
                    .iter()
                    .map(|op| ImportedShortAlias {
                        result: op.pos,
                        same_as_source: op.arg(0),
                        same_as_opcode: op.opcode,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// optimizer.py: pure_from_args1 parity.
    /// Register reverse-pure: pure(opcode, result) = arg0.
    /// Consumed by OptPure at flush time.
    pub fn register_pure_from_args1(&mut self, opcode: OpCode, result: OpRef, arg0: OpRef) {
        self.pending_pure_from_args.push((opcode, result, arg0));
    }

    /// info.py:557 pure_from_args(ARRAYLEN_GC, [op], ConstInt(len))
    pub fn pure_from_args_arraylen(&mut self, array_ref: OpRef, length: i64) {
        let len_ref = self.emit_constant_int(length);
        self.register_pure_from_args1(OpCode::ArraylenGc, array_ref, len_ref);
    }

    /// optimizer.py: pure_from_args2 parity.
    /// Register binary reverse-pure: pure(opcode, arg0, arg1) = result.
    /// Consumed by OptPure at flush time.
    pub fn register_pure_from_args2(
        &mut self,
        opcode: OpCode,
        result: OpRef,
        arg0: OpRef,
        arg1: OpRef,
    ) {
        self.pending_pure_from_args2
            .push((opcode, arg0, arg1, result));
    }

    pub fn replace_op(&mut self, old: OpRef, new: OpRef) {
        if old == new || old.is_constant() {
            return;
        }
        // RPython Box.type invariant: `make_equal_to(box, other)` only
        // fires between boxes of matching type (IntFrontendOp only ever
        // forwards to something with `.type == 'i'`). A cross-type
        // forward would change `opref_type(old)`'s answer from its
        // original kind to `new`'s kind — the exact retyping the
        // `register_value_type` guard is designed to forbid. Assert here
        // so the wrong-kind forward surfaces at the replace site rather
        // than silently at the next downstream lookup.
        if !new.is_none() {
            if let (Some(old_tp), Some(new_tp)) = (self.opref_type(old), self.opref_type(new)) {
                debug_assert_eq!(
                    old_tp, new_tp,
                    "replace_op: cross-type forward {:?}:{:?} -> {:?}:{:?} \
                     (RPython Box.type invariant violation; emit explicit \
                     wrapint/wrapfloat/unbox before forwarding instead of \
                     retyping through replace_op)",
                    old, old_tp, new, new_tp,
                );
            }
        }

        // `optimizer.py:387 make_equal_to` walks `op` to terminal first via
        // `get_box_replacement(op)` so the info read and the subsequent
        // `_forwarded` write land on the chain's terminal box. Without this,
        // a chain-resident `old` writes to a non-terminal slot and the
        // terminal's accumulated info is invisible to the transfer.
        //
        // `new.is_none()` is the pyre-only clear-forwarding sentinel (PyPy
        // has no `make_equal_to(op, None)`); it operates on `old`'s slot
        // directly so the chain from `old` is broken at its root rather
        // than walked to the terminal.
        let old = if new.is_none() {
            old
        } else {
            self.get_box_replacement(old)
        };
        if old == new || old.is_constant() {
            return;
        }
        let idx = old.raw() as usize;
        // `optimizer.py:387 make_equal_to`:
        //     opinfo = op.get_forwarded()
        //     op.set_forwarded(newop)
        //     if opinfo is not None and not newop.is_constant():
        //         newop.set_forwarded(opinfo)
        //
        // Capture `old`'s `AbstractInfo` BEFORE overwriting so we can
        // transfer it to `new` per PyPy `optimizer.py:400`. The pyre
        // BoxRef stores three `AbstractInfo` variants: PtrInfo, IntBound,
        // FloatConst (`info.py:851 FloatConstInfo(AbstractInfo)`).
        // `Box(Const)` / `None` have nothing to transfer.
        use crate::optimizeopt::info::OpInfo;
        let b = self.ensure_box_at(idx);
        let info_to_transfer: Option<OpInfo> = match &*b.get_forwarded() {
            crate::r#box::Forwarded::Info(
                opinfo @ (OpInfo::Ptr(_) | OpInfo::IntBound(_) | OpInfo::FloatConst(_)),
            ) => Some(opinfo.clone()),
            _ => None,
        };
        // `optimizer.py:387 make_equal_to → set_forwarded(newop)`. For const
        // targets we build a fresh `BoxRef::new_const_with_index(value, idx)`
        // per call site (history.py:220 ConstInt no-dedup parity, with
        // const_index threaded so the chain walker can reconstruct
        // `OpRef::const_int/float/ptr`). Non-const targets reuse the
        // target's existing `Rc<Box>` so chain-walking observes identity.
        if new.is_none() {
            b.clear_forwarded();
        } else if new.is_constant() {
            let const_index = new.const_index();
            let value = self
                .const_pool
                .get(&const_index)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "replace_op: constant target {new:?} missing from const_pool — \
                         caller produced a Const OpRef that bypassed seeding"
                    )
                });
            // Carry the const_index so the chain walker can reconstruct
            // OpRef::Const{Int,Float,Ptr}(const_index) when advancing
            // past `Forwarded::Box(const_box)`.
            b.set_forwarded_box(crate::r#box::BoxRef::new_const_with_index(
                value,
                const_index,
            ));
        } else {
            let target = self.ensure_box_at(new.raw() as usize);
            self.box_pool[idx].set_forwarded_box(target);
        }
        // `optimizer.py:400`: transfer captured info to `new` when `new` is
        // not constant. Const targets carry their value directly via the
        // forwarding chain; they are not legitimate `_forwarded` carriers
        // for PtrInfo / IntBound. PyPy unconditionally overrides `new`'s
        // slot, so we mirror that — last writer wins.
        if let Some(opinfo) = info_to_transfer {
            if !new.is_none() && !new.is_constant() {
                let new_idx = new.raw() as usize;
                let nb = self.ensure_box_at(new_idx);
                nb.set_forwarded_info(opinfo);
            }
        }
    }

    /// RPython unroll.py: source.set_forwarded(target)
    /// Sets forwarding from Phase 2 source to Phase 1 export target.
    /// setinfo_from_preamble_recursive then sets PtrInfo on the TARGET
    /// (via get_replacement).
    /// RPython get_box_replacement: follow forwarding chain, stop when the
    /// NEXT position has ptr_info set (it's a terminal, like RPython's
    /// get_box_replacement stopping at _forwarded=Info).
    /// info.py:111-118: mark_last_guard — record the last guard position
    /// on the PtrInfo for an OpRef. RPython: opinfo.mark_last_guard(optimizer).
    /// info.py:111-118: mark_last_guard.
    /// last_guard_pos = len(_newoperations) - 1.
    pub fn mark_last_guard(&mut self, opref: OpRef) {
        let pos = match self.new_operations.last() {
            Some(op) if op.opcode.is_guard() => (self.new_operations.len() - 1) as i32,
            _ => return,
        };
        let resolved = self.get_box_replacement(opref);
        let idx = resolved.raw() as usize;
        // `info.py:111-118 mark_last_guard` mutates the single `_forwarded`
        // PtrInfo slot in place.
        if let Some(b) = self.box_pool.get(idx).cloned() {
            if let Some(mut info) = b.ptr_info_mut() {
                info.set_last_guard_pos(pos);
            }
        }
    }

    /// info.py:100-103: get_last_guard.
    /// _newoperations[last_guard_pos].
    pub fn get_last_guard(&self, opref: OpRef) -> Option<&Op> {
        let pos = self.last_guard_pos_via_box(opref)?;
        self.new_operations.get(pos)
    }

    /// resoperation.py:57-68 get_box_replacement: follow the forwarding
    /// chain (op._forwarded) until we reach a terminal. RPython: walks
    /// op → op._forwarded → ... until None or Info instance.
    ///
    /// RPython invariant: get_box_replacement NEVER returns None.
    /// `_forwarded = None` means "no forwarding" (terminal), NOT
    /// "forwarded to None".
    ///
    /// NEVER consults mapping dicts — RPython's get_box_replacement only
    /// follows the _forwarded chain on the box itself.
    ///
    /// `_forwarded` is a single slot per `BoxRef` (matching RPython's
    /// single Python slot per box). The walker advances through
    /// `Forwarded::Box(target)` and terminates at `None` /
    /// `Forwarded::Info(_)` / a Const target's reconstructed
    /// `OpRef::const_int/float/ptr`.
    fn get_box_replacement_impl(&self, mut opref: OpRef, not_const: bool) -> OpRef {
        use crate::r#box::Forwarded as BoxForwarded;

        loop {
            if opref.is_constant() || opref.is_none() {
                return opref;
            }
            let idx = opref.raw() as usize;
            let mut next: Option<OpRef> = None;
            if let Some(b) = self.box_pool.get(idx) {
                match &*b.get_forwarded() {
                    BoxForwarded::None => {
                        // No forwarding — terminal at this position.
                    }
                    BoxForwarded::Info(_) => {}
                    BoxForwarded::Box(target) => {
                        if target.is_constant() {
                            if not_const {
                                return opref;
                            }
                            // Slice 19: `replace_op(_, const_target)` writes
                            // `Box(BoxRef::new_const_with_index(value, idx))`
                            // carrying the original const_index. Reconstruct
                            // the constant-namespace OpRef so the walker
                            // advances natively without consulting Vec.
                            // Pre-slice-19 callers (`new_const(value)`)
                            // omit the const_index — fall through and terminate
                            // at the source OpRef for those legacy/test paths.
                            if let Some(idx_const) = target.const_index() {
                                let const_opref = match target.type_() {
                                    majit_ir::Type::Int => OpRef::const_int(idx_const),
                                    majit_ir::Type::Float => OpRef::const_float(idx_const),
                                    majit_ir::Type::Ref => OpRef::const_ptr(idx_const),
                                    majit_ir::Type::Void => opref,
                                };
                                if !matches!(target.type_(), majit_ir::Type::Void) {
                                    next = Some(const_opref);
                                }
                            }
                        } else if let Some(pos) = target.position() {
                            let tp = target.type_();
                            // `Type::Void` targets are lazy-allocated phantom
                            // placeholders (`ensure_box_at`); the placeholder
                            // carries no type information, so preserve the
                            // source variant via `with_raw` instead of
                            // promoting to `void_op` / `input_arg_typed(_, Void)`.
                            if matches!(tp, majit_ir::Type::Void) {
                                next = Some(opref.with_raw(pos));
                            } else if target.is_inputarg() {
                                next = Some(OpRef::input_arg_typed(pos, tp));
                            } else if target.is_resop() {
                                next = Some(OpRef::op_typed(pos, tp));
                            }
                        }
                    }
                }
            }
            if let Some(n) = next {
                opref = n;
                continue;
            }
            return opref;
        }
    }

    pub fn get_box_replacement(&self, opref: OpRef) -> OpRef {
        self.get_box_replacement_impl(opref, false)
    }

    /// resoperation.py:58 get_box_replacement(not_const=True). This is used
    /// for guard fail args / backend liveboxes where RPython stops before a
    /// Const target, preserving the runtime box while resume numbering carries
    /// constants as TAGCONST.
    pub fn get_box_replacement_not_const(&self, opref: OpRef) -> OpRef {
        self.get_box_replacement_impl(opref, true)
    }

    /// PyPy parity reader — returns the terminal `BoxRef` in the
    /// `_forwarded` chain rooted at the Box for `opref`.
    ///
    /// `resoperation.py:57-68 get_box_replacement(self, op)` walks
    /// `op._forwarded` until `None | AbstractInfo`, returning the terminal
    /// Box object. PyPy callers consume the Box directly. The OpRef-returning
    /// `get_box_replacement` above is the pyre-side adaptation that exists
    /// only because the rest of the optimizer still indexes by integer
    /// `OpRef`; this BoxRef-returning variant is the parity-faithful API.
    ///
    /// `BoxRef._forwarded` (`box.rs`) is the authoritative storage; both
    /// readers walk the same chain and agree by construction.
    ///
    /// Returns `None` when `opref` is sentinel/None or has no entry in
    /// `box_pool` / `const_pool` (the test / retrace baseline that runs
    /// without an upstream pool). Callers in that case fall back to the
    /// `OpRef`-returning walker.
    pub fn get_box_replacement_box(&self, opref: OpRef) -> Option<crate::r#box::BoxRef> {
        if opref.is_none() {
            return None;
        }
        let start = if opref.is_constant() {
            // RPython parity: ConstInt/Float/Ptr (`history.py:220, 261, 307`)
            // are constructed fresh per call site without dedup; we mirror by
            // building a fresh `BoxRef::new_const(value)` from `const_pool`.
            // BoxKind::Const has no _forwarded slot so identity does not matter
            // for chain semantics — see `same_constant` (`history.py:204`).
            let value = self.const_pool.get(&opref.const_index()).copied()?;
            crate::r#box::BoxRef::new_const(value)
        } else {
            let idx = opref.raw() as usize;
            self.box_pool.get(idx).cloned()?
        };
        Some(start.get_box_replacement(false))
    }

    /// `optimizer.py:1009 getptrinfo + info.is_virtual()` BoxRef-routing
    /// helper. Returns whether the box at `opref` (after chain walk)
    /// carries a `PtrInfo` whose `is_virtual()` is true. Reads via
    /// `BoxRef::ptr_info()` on the chain-walked terminal box; an absent
    /// `box_pool` slot (synthetic test paths) returns `false`.
    pub fn is_virtual_via_box(&self, opref: OpRef) -> bool {
        if let Some(b) = self.get_box_replacement_box(opref) {
            return b.ptr_info().map_or(false, |p| p.is_virtual());
        }
        false
    }

    /// `optimizer.py:1009 getptrinfo + info.is_nonnull()` BoxRef-routing
    /// helper. Same shape as `is_virtual_via_box`.
    pub fn is_nonnull_via_box(&self, opref: OpRef) -> bool {
        if let Some(b) = self.get_box_replacement_box(opref) {
            return b.ptr_info().map_or(false, |p| p.is_nonnull());
        }
        false
    }

    /// `optimizer.py:99-113 getintbound` BoxRef-routing reader. Returns
    /// an owned `IntBound` clone when the box at `opref` carries one in its
    /// `_forwarded` Info slot. BoxRef is the only authoritative storage.
    pub fn peek_intbound_via_box(
        &self,
        opref: OpRef,
    ) -> Option<crate::optimizeopt::intutils::IntBound> {
        let b = self.get_box_replacement_box(opref)?;
        b.int_bound().map(|ib| ib.clone())
    }

    /// `optimizer.py:99-113 getptrinfo` BoxRef-routing reader returning a
    /// cloned `PtrInfo` snapshot from the authoritative `_forwarded` slot.
    pub fn peek_ptr_info_via_box(&self, opref: OpRef) -> Option<crate::optimizeopt::info::PtrInfo> {
        let b = self.get_box_replacement_box(opref)?;
        b.ptr_info().map(|p| p.clone())
    }

    /// PyPy `box.is_constant()` (`history.py:188`) returns True for Const
    /// objects. The pyre-side
    /// equivalent must accept three forwarding shapes that all encode "the
    /// box at `opref` is a known constant":
    ///   (a) `opref` is itself a Const-namespace OpRef.
    ///   (b) BoxRef forwarded slot is `Forwarded::Box(target)` where
    ///       `target` is `BoxKind::Const` — orthodox `optimizer.py:432
    ///       set_forwarded(constbox)` shape.
    ///   (c) BoxRef forwarded slot is `Forwarded::Info(OpInfo::Constant(_)
    ///       | OpInfo::FloatConst(_))` — legacy fixtures and bridge snapshots.
    ///
    /// Used by write-side gates (`make_nonnull`, `make_nonnull_str`, ...)
    /// that must skip when the receiver is constant; the OpRef-only
    /// `resolved.is_constant()` test misses (b)/(c) under the current
    /// `make_constant` write shape and would let `set_ptr_info` clobber
    /// the const slot with `OpInfo::Ptr(...)`. Mirrors the
    /// `OptBoxEnv::is_const` impl that resume numbering already uses.
    pub(crate) fn resolved_is_constant_via_box(&self, resolved: OpRef) -> bool {
        if resolved.is_constant() {
            return true;
        }
        let idx = resolved.raw() as usize;
        let Some(b) = self.box_pool.get(idx) else {
            return false;
        };
        use crate::optimizeopt::info::OpInfo;
        match &*b.get_forwarded() {
            crate::r#box::Forwarded::Box(target) if target.is_constant() => true,
            crate::r#box::Forwarded::Info(OpInfo::Constant(_) | OpInfo::FloatConst(_)) => true,
            _ => false,
        }
    }

    /// info.py: getptrinfo(op) — mutable variant. Mutates the PtrInfo
    /// stored on `box_pool[idx].forwarded` (`BoxRef._forwarded`) via the
    /// closure. The BoxRef slot is the authoritative storage and no separate
    /// mirror step is needed.
    ///
    /// Closure semantics: returns `Some(f(info))` when a `PtrInfo` exists
    /// at `opref`, `None` otherwise (no closure invocation).
    pub fn with_ptr_info_mut<R>(
        &mut self,
        opref: OpRef,
        f: impl FnOnce(&mut PtrInfo) -> R,
    ) -> Option<R> {
        let r = self.get_box_replacement(opref);
        let idx = r.raw() as usize;
        let b = self.box_pool.get(idx).cloned()?;
        let mut pi = b.ptr_info_mut()?;
        let result = f(&mut *pi);
        Some(result)
    }

    /// Closure-style wrapper around [`Self::ensure_ptr_info_arg0`].
    ///
    /// Closure mutations through `EnsuredPtrInfo::as_mut()` land on the
    /// BoxRef's `RefCell<Forwarded>` directly — single-slot RPython parity
    /// with `optimizer.py:467 ensure_ptr_info_arg0`'s mutate-in-place
    /// behavior.
    pub fn with_ensured_ptr_info_arg0<R>(
        &mut self,
        op: &Op,
        f: impl FnOnce(crate::optimizeopt::info::EnsuredPtrInfo) -> R,
    ) -> R {
        f(self.ensure_ptr_info_arg0(op))
    }

    /// `info.py:91-103 PtrInfo.get_last_guard_pos` BoxRef-routing reader.
    /// Routes through `BoxRef::ptr_info()` when a BoxRef exists at the
    /// resolved position (single `_forwarded` slot parity:
    /// `resoperation.py:233`).
    pub fn last_guard_pos_via_box(&self, opref: OpRef) -> Option<usize> {
        let b = self.get_box_replacement_box(opref)?;
        b.ptr_info().and_then(|p| p.get_last_guard_pos())
    }

    /// `optimizer.py:1009 getptrinfo(box) is not None` parity — true
    /// when the box at `opref` carries any `PtrInfo` in its
    /// `_forwarded` Info slot. BoxRef-routing variant of
    /// `get_ptr_info(opref).is_some()`.
    pub fn has_ptr_info_via_box(&self, opref: OpRef) -> bool {
        if let Some(b) = self.get_box_replacement_box(opref) {
            return b.ptr_info().is_some();
        }
        false
    }

    /// `virtualize.py is_standard_virtualizable_ref` parity — true when
    /// the box at `opref` carries a `PtrInfo::Virtualizable` info.
    /// BoxRef-routing variant of `matches!(get_ptr_info(opref),
    /// Some(PtrInfo::Virtualizable(_)))`.
    pub fn is_virtualizable_via_box(&self, opref: OpRef) -> bool {
        use crate::optimizeopt::info::PtrInfo;
        if let Some(b) = self.get_box_replacement_box(opref) {
            return b
                .ptr_info()
                .map_or(false, |p| matches!(*p, PtrInfo::Virtualizable(_)));
        }
        false
    }

    /// resoperation.py: op.get_forwarded() is not None — check if OpRef
    /// has any forwarding entry (Op, Info, IntBound, Const).
    ///
    /// `Const.get_forwarded()` returns `None` in RPython
    /// (`resoperation.py:1162`); short-circuit on the const-namespace
    /// `OpRef` so the caller doesn't index `box_pool` with a CONST_BIT
    /// `raw()` — which would either return None (large-index miss) or
    /// alias an unrelated slot if the pool happens to be that long.
    pub fn has_forwarding(&self, opref: OpRef) -> bool {
        if opref.is_constant() {
            return false;
        }
        let idx = opref.raw() as usize;
        match self.box_pool.get(idx) {
            Some(b) => !matches!(*b.get_forwarded(), crate::r#box::Forwarded::None),
            None => false,
        }
    }

    /// True only when opref has a non-const forwarding redirect.
    ///
    /// `replace_op(_, non_const)` writes `Forwarded::Box(target)` where
    /// `target` is a non-Const box. Both `replace_op(_, const_target)`
    /// and `make_constant` write `Forwarded::Box(const_box)` per
    /// `optimizer.py:432 box.set_forwarded(constbox)`. Splitting on
    /// `target.is_constant()` excludes the const-target shape so this
    /// returns true only for the inputarg-redirect case used by
    /// `import_state`.
    ///
    /// `Const.get_forwarded()` returns `None` upstream
    /// (`resoperation.py:1162`); short-circuit on the const-namespace
    /// `OpRef` so the caller doesn't index `box_pool` with a CONST_BIT
    /// `raw()`.
    pub fn has_op_forwarding(&self, opref: OpRef) -> bool {
        if opref.is_constant() {
            return false;
        }
        let idx = opref.raw() as usize;
        match self.box_pool.get(idx) {
            Some(b) => matches!(
                &*b.get_forwarded(),
                crate::r#box::Forwarded::Box(target) if !target.is_constant()
            ),
            None => false,
        }
    }

    /// Store a constant value WITHOUT setting box forwarding.
    /// Used for pre-populating backend constants and call_pure_results.
    ///
    /// RPython parity: `ConstInt`, `ConstPtr`, `ConstFloat` are distinct
    /// Box subclasses (history.py:220/261/307); two Boxes at the same
    /// OpRef position MUST NOT disagree on type.  Seeding a typed
    /// constant over a slot that already holds a different-typed value
    /// is a bug (typical source: `Value::Ref(0)` reseeded where
    /// `Value::Int(0)` lives, causing `opref_type` to flip Int→Ref and
    /// downstream `getintbound` to panic during bridge optimization).
    /// Assert the invariant instead of silently overwriting.
    pub fn seed_constant(&mut self, opref: OpRef, value: Value) {
        if opref.is_constant() {
            if let Some(existing) = self.const_pool.get(&opref.const_index()) {
                assert_eq!(
                    existing.get_type(),
                    value.get_type(),
                    "seed_constant: type mismatch at {:?} (existing={:?}, new={:?}) — \
                     ConstInt/ConstPtr/ConstFloat must never alias",
                    opref,
                    existing,
                    value,
                );
            }
            self.const_pool.insert(opref.const_index(), value.clone());
        } else {
            let idx = opref.raw() as usize;
            if idx >= self.constants.len() {
                self.constants.resize(idx + 1, None);
            }
            if let Some(Some(existing)) = self.constants.get(idx) {
                assert_eq!(
                    existing.get_type(),
                    value.get_type(),
                    "seed_constant: type mismatch at {:?} (existing={:?}, new={:?}) — \
                     ConstInt/ConstPtr/ConstFloat must never alias",
                    opref,
                    existing,
                    value,
                );
            }
            self.constants[idx] = Some(value);
        }
    }

    /// Read-only variant of `getintbound` — returns the IntBound stored on
    /// `box._forwarded` without materializing an unbounded one on first
    /// access. Returns `None` for boxes that have no IntBound forwarding.
    /// Used by exporters that take `&OptContext` and cannot mutate.
    pub(crate) fn peek_intbound(
        &self,
        opref: OpRef,
    ) -> Option<crate::optimizeopt::intutils::IntBound> {
        // optimizer.py:99-100: assert op.type == 'i'
        // None is allowed for test fixtures that don't seed value_types.
        assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "peek_intbound: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let replaced = self.get_box_replacement(opref);
        if let Some(Value::Int(v)) = self.get_constant(replaced) {
            return Some(crate::optimizeopt::intutils::IntBound::from_constant(
                v as i64,
            ));
        }
        // optimizer.py:107 second `assert op.type == 'i'` — Box.type is
        // immutable in RPython, so the replaced op must still be int-typed.
        assert!(
            matches!(self.opref_type(replaced), Some(majit_ir::Type::Int) | None),
            "peek_intbound: replaced OpRef must be int-typed, got {:?}",
            self.opref_type(replaced)
        );
        if replaced.is_constant() {
            return None;
        }
        // BoxRef-authoritative reader. IntBound writers populate the
        // BoxRef via `ensure_box_at`.
        let b = self.get_box_replacement_box(replaced)?;
        b.int_bound().map(|ib| ib.clone())
    }

    /// optimizer.py:99-113: getintbound(op) — get or create IntBound for
    /// an int-typed box. Lazy: creates unbounded on first access and stores
    /// it on the BoxRef's `_forwarded` slot.
    pub(crate) fn getintbound(&mut self, opref: OpRef) -> crate::optimizeopt::intutils::IntBound {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:100: assert op.type == 'i'
        assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "getintbound: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let replaced = self.get_box_replacement(opref);
        // optimizer.py:102-103: if isinstance(op, ConstInt): return from_constant
        if let Some(Value::Int(v)) = self.get_constant(replaced) {
            return crate::optimizeopt::intutils::IntBound::from_constant(v as i64);
        }
        // optimizer.py:110 second `assert op.type == 'i'` — Box.type is
        // immutable, so the replaced op must still be int-typed.
        assert!(
            matches!(self.opref_type(replaced), Some(majit_ir::Type::Int) | None),
            "getintbound: replaced OpRef must be int-typed, got {:?}",
            self.opref_type(replaced)
        );
        if replaced.is_constant() {
            return crate::optimizeopt::intutils::IntBound::unbounded();
        }
        let idx = replaced.raw() as usize;
        // BoxRef-authoritative read. IntBound writers populate the BoxRef via
        // `ensure_box_at`; non-IntBound states match PyPy's rare-case branch
        // and return unbounded without overwriting the slot.
        let b = self.ensure_box_at(idx);
        match &*b.get_forwarded() {
            crate::r#box::Forwarded::Info(OpInfo::IntBound(b)) => return b.clone(),
            crate::r#box::Forwarded::None => {}
            _ => return crate::optimizeopt::intutils::IntBound::unbounded(),
        }
        // optimizer.py:110-112: fw is None → create unbounded and store
        let intbound = crate::optimizeopt::intutils::IntBound::unbounded();
        b.set_forwarded_info(OpInfo::IntBound(intbound.clone()));
        intbound
    }

    /// optimizer.py:115-125: setintbound(op, bound) — intersect existing
    /// bound with new bound, or set if none exists.
    pub fn setintbound(&mut self, opref: OpRef, bound: &crate::optimizeopt::intutils::IntBound) {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:116: assert op.type == 'i'
        assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "setintbound: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let replaced = self.get_box_replacement(opref);
        // optimizer.py:116 `assert op.type == 'i'` — Box.type is immutable
        // in RPython, so the replaced op must still be int-typed.
        assert!(
            matches!(self.opref_type(replaced), Some(majit_ir::Type::Int) | None),
            "setintbound: replaced OpRef must be int-typed, got {:?}",
            self.opref_type(replaced)
        );
        if replaced.is_constant() || self.get_constant(replaced).is_some() {
            return;
        }
        let idx = replaced.raw() as usize;
        // BoxRef-authoritative: lazy-allocate a placeholder if absent.
        let b = self.ensure_box_at(idx);
        let new_intbound = {
            use crate::r#box::Forwarded as BoxFwd;
            match &*b.get_forwarded() {
                BoxFwd::Info(OpInfo::IntBound(cur)) => {
                    let mut cur = cur.clone();
                    let _ = cur.intersect(bound);
                    Some(cur)
                }
                BoxFwd::None => Some(bound.clone()),
                _ => None,
            }
        };
        if let Some(intbound) = new_intbound {
            b.set_forwarded_info(OpInfo::IntBound(intbound));
        }
    }

    /// In-place mutation helper for the IntBound stored on `box._forwarded`.
    ///
    /// RPython pattern equivalence: where RPython writes
    /// `self.getintbound(box).<method>(...)` and the method mutates the
    /// `IntBound` returned from `box.get_forwarded()` directly, the Rust
    /// borrow checker forces us to materialize the bound, mutate it, and
    /// store it back. This helper performs that read-modify-write atomically
    /// and threads through any return value from the closure (e.g. the
    /// `Result<bool, InvalidLoop>` flag from `intersect`/`make_*`).
    ///
    /// For Constant boxes the bound is "fixed" — RPython's `getintbound`
    /// returns `IntBound.from_constant(...)` and any `intersect` is a
    /// no-op (the constant value is already in range or InvalidLoop). This
    /// helper mirrors that by running the closure on a temporary that is
    /// discarded after — the constant cannot be widened.
    ///
    /// For non-IntBound forwarded info (RawBufferPtrInfo etc.), RPython's
    /// `getintbound` falls through to "return IntBound.unbounded()" without
    /// overwriting forwarding. We mirror by running the closure on a
    /// temporary unbounded that is discarded.
    pub fn with_intbound_mut<F, R>(&mut self, opref: OpRef, f: F) -> R
    where
        F: FnOnce(&mut crate::optimizeopt::intutils::IntBound) -> R,
    {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:99: assert op.type == 'i'
        assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "with_intbound_mut: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let replaced = self.get_box_replacement(opref);
        if let Some(Value::Int(v)) = self.get_constant(replaced) {
            let mut tmp = crate::optimizeopt::intutils::IntBound::from_constant(v as i64);
            return f(&mut tmp);
        }
        // optimizer.py:107/110 `assert op.type == 'i'` — Box.type is
        // immutable, so the replaced op must still be int-typed.
        assert!(
            matches!(self.opref_type(replaced), Some(majit_ir::Type::Int) | None),
            "with_intbound_mut: replaced OpRef must be int-typed, got {:?}",
            self.opref_type(replaced)
        );
        if replaced.is_constant() {
            let mut tmp = crate::optimizeopt::intutils::IntBound::unbounded();
            return f(&mut tmp);
        }
        let idx = replaced.raw() as usize;
        // BoxRef-authoritative mutation. `ensure_box_at` lazily allocates a
        // placeholder if absent, so any IntBound at `idx` lives on the
        // BoxRef's `_forwarded` slot.
        let b = self.ensure_box_at(idx);
        use crate::r#box::Forwarded as BoxFwd;
        let needs_init = matches!(*b.get_forwarded(), BoxFwd::None);
        if needs_init {
            // optimizer.py:110-112 first-access: materialize unbounded.
            let mut new_bound = crate::optimizeopt::intutils::IntBound::unbounded();
            let result = f(&mut new_bound);
            b.set_forwarded_info(OpInfo::IntBound(new_bound));
            return result;
        }
        if let Some(mut bound) = b.int_bound_mut() {
            return f(&mut *bound);
        }
        // BoxRef has Forwarded::Box(_) | Info(Ptr(_)) — RPython's "rare
        // case" arm: return IntBound.unbounded() without overwriting.
        let mut tmp = crate::optimizeopt::intutils::IntBound::unbounded();
        f(&mut tmp)
    }

    /// optimizer.py:410-432 make_constant(box, constbox).
    ///
    /// Mirrors PyPy optimizer.py:432: `box.set_forwarded(constbox)`.
    /// The constant Box carries the fresh Const identity; reader-side helpers
    /// still accept the old `Info(Constant)` shape for legacy fixtures.
    pub fn make_constant(&mut self, opref: OpRef, value: Value) {
        // optimizer.py:412: box = get_box_replacement(box)
        let replaced = self.get_box_replacement(opref);
        // optimizer.py:415-426: safety check — if box.get_forwarded() is
        // IntBound and the constant is Int, validate contains() + make_eq_const().
        // RPython checks ONE authoritative source: box._forwarded.
        if let Value::Int(intval) = value {
            let ridx = replaced.raw() as usize;
            // BoxRef-authoritative read of IntBound for the contains() +
            // make_eq_const() in-place mutation. IntBound writers populate
            // the BoxRef via ensure_box_at.
            if let Some(b) = self.box_pool.get(ridx) {
                if let Some(mut bound) = b.int_bound_mut() {
                    if !bound.contains(intval as i64) {
                        std::panic::panic_any(crate::optimize::InvalidLoop(
                            "constant int is outside the range allowed for that box",
                        ));
                    }
                    // optimizer.py:424-426: info.make_eq_const(value)
                    let _ = bound.make_eq_const(intval as i64);
                }
            }
        }
        // optimizer.py:427: if box.is_constant(): return
        // BoxRef-routing short-circuit (Epic H H-3.2c slice 63):
        // `replaced` is already Vec chain-walked, so `box[replaced]`'s
        // forwarded is the terminal slot. If it's `Box(const_box)` (the
        // make_constant mirror), `const_value()` returns Some(value);
        // otherwise None and we proceed to the make_constant body.
        if replaced.is_constant()
            || self
                .get_box_replacement_box(replaced)
                .and_then(|b| b.const_value())
                .is_some()
        {
            return;
        }
        // optimizer.py:429-431 — when promoting a virtual to a constant,
        // call `copy_fields_to_const(constinfo, optheap)` so the cached
        // field/item state survives via the const_infos pool.
        if let Value::Ref(gcref) = value {
            self.copy_fields_to_const(replaced, gcref);
        }
        // Store in constants array for get_constant() callers — but only
        // for op-result positions. InputArg positions share the same
        // `OpRef::raw()` as same-indexed `IntOp/FloatOp/RefOp` positions
        // (resoperation.py:719/727/739 vs 564-638), so writing into
        // `self.constants[InputArg.raw()]` would propagate via
        // `merge_backend_constants_from_ctx` into `backend.constants` and
        // cause regalloc.rs:1207 to treat the InputArg as inline constant.
        // The InputArg's runtime value flows through the input slot — never
        // as an inline backend constant. Reads of the optimizer's "this
        // box is constant" knowledge go through the box._forwarded slot
        // below.
        let idx = replaced.raw() as usize;
        if !matches!(
            replaced,
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
        ) {
            if idx >= self.constants.len() {
                self.constants.resize(idx + 1, None);
            }
            self.constants[idx] = Some(value.clone());
        }
        // optimizer.py:432: box.set_forwarded(constbox)
        let new_const_idx = self.next_const_idx;
        self.next_const_idx += 1;
        self.const_pool.insert(new_const_idx, value);
        let b = self.ensure_box_at(idx);
        b.set_forwarded_box(crate::r#box::BoxRef::new_const_with_index(
            value,
            new_const_idx,
        ));
    }

    /// info.py:194-198 (AbstractStructPtrInfo) + info.py:533-538 (ArrayPtrInfo)
    /// `copy_fields_to_const(constinfo, optheap)`.
    ///
    /// ```text
    /// # AbstractStructPtrInfo
    /// def copy_fields_to_const(self, constinfo, optheap):
    ///     if self._fields is not None:
    ///         info = constinfo._get_info(self.descr, optheap)
    ///         assert isinstance(info, AbstractStructPtrInfo)
    ///         info._fields = self._fields[:]
    ///
    /// # ArrayPtrInfo
    /// def copy_fields_to_const(self, constinfo, optheap):
    ///     descr = self.descr
    ///     if self._items is not None:
    ///         info = constinfo._get_array_info(descr, optheap)
    ///         assert isinstance(info, ArrayPtrInfo)
    ///         info._items = self._items[:]
    /// ```
    ///
    /// majit folds both per-type entries into a single helper because the
    /// per-source dispatch happens via the PtrInfo enum match. The
    /// `_get_info`/`_get_array_info` half is `const_infos.entry(...)`
    /// (RPython: `optheap.const_infos[ref]`).
    fn copy_fields_to_const(&mut self, source: OpRef, gcref: majit_ir::GcRef) {
        use crate::optimizeopt::info::{ArrayPtrInfo, FieldEntry, PtrInfo, StructPtrInfo};
        // BoxRef-routing reader (H-3.2c slice 57). `source` is always
        // chain-walked by the caller (`make_constant`), so peek's chain
        // walk is a no-op — owned PtrInfo clone here matches the prior
        // `Forwarded::Info(info)` immediate-slot read.
        let Some(info) = self.peek_ptr_info_via_box(source) else {
            return;
        };
        let key = gcref.as_usize();
        match info {
            // info.py:194-198 AbstractStructPtrInfo.copy_fields_to_const →
            // constinfo._get_info(self.descr, optheap) → StructPtrInfo(descr).
            PtrInfo::Instance(v) if !v.fields.is_empty() => {
                let Some(descr) = v.descr.clone() else {
                    return;
                };
                let fields = v.fields.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: Vec::new(),
                        field_descrs: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::Struct(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields = v.fields.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: Vec::new(),
                        field_descrs: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::Virtual(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields: Vec<(u32, FieldEntry)> = v
                    .fields
                    .iter()
                    .map(|&(k, r)| (k, FieldEntry::Value(r)))
                    .collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: Vec::new(),
                        field_descrs: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::VirtualStruct(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields: Vec<(u32, FieldEntry)> = v
                    .fields
                    .iter()
                    .map(|&(k, r)| (k, FieldEntry::Value(r)))
                    .collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: Vec::new(),
                        field_descrs: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            // info.py:533-538 ArrayPtrInfo.copy_fields_to_const →
            // constinfo._get_array_info(descr, optheap) → ArrayPtrInfo(descr).
            PtrInfo::Array(v) if !v.items.is_empty() => {
                let descr = v.descr.clone();
                let lenbound = v.lenbound.clone();
                let items = v.items.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Array(ArrayPtrInfo {
                        descr,
                        lenbound,
                        items: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Array(a) = ci {
                    a.items = items;
                }
            }
            PtrInfo::VirtualArray(v) if !v.items.is_empty() => {
                let descr = v.descr.clone();
                let len = v.items.len() as i64;
                let items: Vec<FieldEntry> =
                    v.items.iter().map(|&r| FieldEntry::Value(r)).collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Array(ArrayPtrInfo {
                        descr,
                        lenbound: IntBound::from_constant(len),
                        items: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Array(a) = ci {
                    a.items = items;
                }
            }
            _ => {}
        }
    }

    /// resume.py:157 getconst parity for synthetic rd_numb encoding.
    /// Matches OptBoxEnv::get_const: checks constant_types_for_numbering
    /// override, PtrInfo::Constant, and constant pool.
    /// Returns None if opref is not a constant.
    pub fn getconst(&self, opref: OpRef) -> Option<(i64, majit_ir::Type)> {
        let type_override = self.constant_types_for_numbering.get(&opref.raw()).copied();
        // Check constant pool (through replacement chain).
        if let Some(val) = self.get_constant(opref) {
            let (raw, tp) = match val {
                Value::Int(v) => (v, type_override.unwrap_or(majit_ir::Type::Int)),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                _ => return None,
            };
            return Some((raw, tp));
        }
        // Check raw constants (before replacement).
        if let Some(val) = self
            .constants
            .get(opref.raw() as usize)
            .and_then(|v| v.as_ref())
        {
            let (raw, tp) = match val {
                Value::Int(v) => (*v, type_override.unwrap_or(majit_ir::Type::Int)),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                _ => return None,
            };
            return Some((raw, tp));
        }
        // info.py: ConstPtrInfo — GcRef constant stored in PtrInfo.
        if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) =
            self.peek_ptr_info_via_box(opref)
        {
            return Some((gcref.0 as i64, majit_ir::Type::Ref));
        }
        None
    }

    /// Get the constant value for an operation, if known.
    ///
    /// BoxRef-routing reader. The BoxRef carries the constant value in
    /// one of two shapes:
    /// - `Forwarded::Box(target)` where `target` is a `BoxKind::Const(v)`
    ///   — orthodox `optimizer.py:432 box.set_forwarded(constbox)` shape,
    ///   written by `make_constant` and `replace_op(_, const_target)`.
    /// - `Forwarded::Info(OpInfo::Constant(v))` / `OpInfo::FloatConst(f)`
    ///   — legacy fixtures and bridge snapshot replay paths; production
    ///   writers no longer produce this shape after the make_constant flip.
    pub fn get_constant(&self, opref: OpRef) -> Option<Value> {
        use crate::optimizeopt::info::OpInfo;
        let opref = self.get_box_replacement(opref);
        if opref.is_constant() {
            return self.const_pool.get(&opref.const_index()).copied();
        }
        let idx = opref.raw() as usize;
        if let Some(b) = self.box_pool.get(idx) {
            match &*b.get_forwarded() {
                crate::r#box::Forwarded::Info(OpInfo::Constant(v)) => return Some(*v),
                crate::r#box::Forwarded::Info(OpInfo::FloatConst(f)) => {
                    return Some(Value::Float(*f));
                }
                crate::r#box::Forwarded::Box(target) => {
                    if let Some(value) = target.const_value() {
                        return Some(value);
                    }
                }
                _ => {}
            }
        }
        self.constants.get(idx).and_then(|v| *v)
    }

    /// Whether `opref` has a known constant value.
    pub fn is_constant(&self, opref: OpRef) -> bool {
        self.get_constant(opref).is_some()
    }

    /// Get constant integer value, if known.
    pub fn get_constant_int(&self, opref: OpRef) -> Option<i64> {
        self.get_constant(opref).and_then(|v| match v {
            Value::Int(i) => Some(i),
            _ => None,
        })
    }

    /// vstring.py:237 `optstring.getintbound(box).is_constant()` pattern.
    /// Returns the constant value if known either from the constant pool
    /// or from IntBound analysis.
    pub fn get_constant_int_or_bound(&self, opref: OpRef) -> Option<i64> {
        let resolved = self.get_box_replacement(opref);
        self.get_constant_int(resolved).or_else(|| {
            self.get_int_bound(resolved)
                .filter(|b| b.is_constant())
                .map(|b| b.get_constant_int())
        })
    }

    /// history.py:361 CONST_NULL = ConstPtr(ConstPtr.value).
    /// True iff `opref` is a Ref-typed null constant, mirroring
    /// `CONST_NULL.same_constant(box)` in RPython.
    pub fn is_const_null(&self, opref: OpRef) -> bool {
        matches!(
            self.get_constant(opref),
            Some(Value::Ref(r)) if r.0 == 0
        )
    }

    /// optimizer.py:705-711: is_call_pure_pure_canraise — a CallPure op whose
    /// effectinfo says check_can_raise(ignore_memoryerror=True). These ops are
    /// formally side-effect-free (has_no_side_effect), but their potential to
    /// raise means they break guard resume-data sharing.
    fn is_call_pure_pure_canraise(op: &Op) -> bool {
        if !op.opcode.is_call_pure() {
            return false;
        }
        let Some(ref descr) = op.descr else {
            return false;
        };
        let Some(cd) = descr.as_call_descr() else {
            return false;
        };
        cd.get_extra_info().check_can_raise(true)
    }

    /// optimizer.py:652-686 emit_guard_operation — decide whether to share
    /// resume data from the previous guard (_copy_resume_data_from) or build
    /// new resume data (store_final_boxes_in_guard).
    fn emit_guard_operation(&mut self, op: &mut Op) {
        let opnum = op.opcode;

        // optimizer.py:655-664: GUARD_(NO_)EXCEPTION following a guard that
        // is NOT GUARD_NOT_FORCED — give up sharing.  GUARD_NOT_FORCED_2
        // is excluded for the same reason as in the Optimizer path:
        // pyjitpl.py:3236 emits it at finish() only, so no exception
        // guard can follow.
        if opnum == OpCode::GuardNoException || opnum == OpCode::GuardException {
            if let Some(idx) = self.last_guard_idx {
                if self.new_operations[idx].opcode != OpCode::GuardNotForced {
                    self.last_guard_idx = None;
                }
            }
        }

        // optimizer.py:665-670: GUARD_ALWAYS_FAILS must never share.
        if opnum == OpCode::GuardAlwaysFails {
            self.last_guard_idx = None;
        }

        // optimizer.py:672: `self._last_guard_op and guard_op.getdescr() is None`
        // getdescr() is None only for optimizer-created guards (no descr
        // from tracing, no ResumeAtPositionDescr from unroll).
        // compile.py:925-926: GUARD_NOT_FORCED* must never share —
        // invent_fail_descr_for_op asserts copied_from_descr is None.
        let can_share = self.last_guard_idx.is_some()
            && op.descr.is_none()
            && opnum != OpCode::GuardNotForced
            && opnum != OpCode::GuardNotForced2;

        if can_share {
            let idx = self.last_guard_idx.unwrap();
            // compile.py:832 ResumeGuardCopiedDescr(prev) parity: stamp
            // a `ResumeGuardCopiedDescr` whose `prev` references the
            // donor's descr.  Readers go through
            // `FailDescr::rd_*()` which chases `prev` automatically
            // (compile.py:849 `get_resumestorage(): return prev`).
            // GUARD_EXCEPTION / GUARD_NO_EXCEPTION mint the exc variant.
            //
            // optimizer.py:691 `assert isinstance(last_descr,
            // compile.ResumeGuardDescr)` — the donor must be a finalized
            // ResumeGuardDescr (or subclass).  RPython enforces this on
            // every sharing emit; pyre's standalone OptContext path
            // matched the production Optimizer in name only and used to
            // silently leave `op.descr = None` when the donor lacked a
            // descr.  Tighten to RPython parity.
            let donor_descr = self.new_operations[idx].descr.clone().expect(
                "optimizer.py:691 assert isinstance(last_descr, \
                     ResumeGuardDescr): donor guard has no descr",
            );
            assert!(
                donor_descr.is_resume_guard(),
                "optimizer.py:691 assert isinstance(last_descr, \
                 ResumeGuardDescr): donor descr_index={} is not a \
                 ResumeGuardDescr subclass",
                donor_descr.index()
            );
            op.descr = Some(match opnum {
                OpCode::GuardException | OpCode::GuardNoException => {
                    crate::compile::make_resume_guard_copied_exc_descr(donor_descr)
                }
                _ => crate::compile::make_resume_guard_copied_descr(donor_descr),
            });
            op.fail_args = self.new_operations[idx].fail_args.clone();
            // bridgeopt.py parity: fail_arg_types carry the types the
            // serializer used when writing the class-knowledge bitfield in
            // rd_numb (memo.finish() uses numb_state.livebox_types). A
            // shared guard's rd_numb encodes the donor's livebox type
            // layout, so the sharer must inherit fail_arg_types too —
            // otherwise `deserialize_optimizer_knowledge` (bridgeopt.rs:911)
            // reconstructs a different Ref-set and reads past the buffer.
            op.fail_arg_types = self.new_operations[idx].fail_arg_types.clone();
            // optimizer.py:698-699: _maybe_replace_guard_value after copy.
            if op.opcode == OpCode::GuardValue {
                self.maybe_replace_guard_value(op);
            }
            // Don't update last_guard_idx — copied guards don't become sources.
        } else {
            // optimizer.py:678: store_final_boxes_in_guard.  This is
            // the standalone OptContext path (used by tests and the
            // direct ctx.emit_guard hook); it has no `pending_for_guard`
            // staging, so pass an empty Vec for the descr-side
            // set_rd_pendingfields write.
            self.store_final_boxes_in_guard(op, None, Vec::new());
            self.last_guard_idx = Some(self.new_operations.len());
            // optimizer.py:680-683: force_box on fail_args for unrolling.
            // Mirrors Optimizer.force_box contract: resolve replacement,
            // handle tracked preamble ops, force virtuals.
            if let Some(ref fa) = op.fail_args {
                let fargs: Vec<OpRef> = fa.iter().copied().collect();
                for farg in fargs {
                    if !farg.is_none() {
                        // regalloc.py:1206: Const objects skip forcing.
                        // Constant OpRefs may collide with virtual positions;
                        // forcing would corrupt the virtual's PtrInfo.
                        let resolved = self.get_box_replacement(farg);
                        if !self.is_constant(resolved) {
                            self.force_box_inline(farg);
                        }
                    }
                }
            }
            // optimizer.py:750-751: _maybe_replace_guard_value after store.
            if op.opcode == OpCode::GuardValue {
                self.maybe_replace_guard_value(op);
            }
        }

        // optimizer.py:684-685: GUARD_EXCEPTION clears sharing.
        if opnum == OpCode::GuardException {
            self.last_guard_idx = None;
        }
    }

    /// optimizer.py:754-778 _maybe_replace_guard_value — turn
    /// guard_value(bool) into guard_true/guard_false.
    fn maybe_replace_guard_value(&self, op: &mut Op) {
        let arg0 = op.arg(0);
        // optimizer.py:755: if op.getarg(0).type == 'i'
        let arg0_resolved = self.get_box_replacement(arg0);
        if self.opref_type(arg0_resolved) != Some(majit_ir::Type::Int) {
            return;
        }
        // optimizer.py:756: b = self.getintbound(op.getarg(0))
        let Some(bound) = self.get_int_bound(arg0_resolved) else {
            return;
        };
        if !bound.is_bool() {
            return;
        }
        let arg1 = op.arg(1);
        let Some(constvalue) = self.get_constant_int(arg1) else {
            return;
        };
        let new_opcode = match constvalue {
            0 => OpCode::GuardFalse,
            1 => OpCode::GuardTrue,
            _ => return, // optimizer.py:775: strange code, just disable
        };
        op.opcode = new_opcode;
        op.args.clear();
        op.args.push(arg0);
    }

    /// optimizer.py:345-364 force_box — inline equivalent for
    /// emit_guard_operation's fail_arg forcing (optimizer.py:680-683).
    /// Mirrors Optimizer.force_box contract: handle tracked preamble ops,
    /// then force virtuals to concrete. Path B (B.6.7) routes body refs
    /// through Phase 1 source directly, so the prior reverse-lookup 3rd
    /// key is no longer needed.
    fn force_box_inline(&mut self, opref: OpRef) -> OpRef {
        let resolved = self.get_box_replacement(opref);
        let tracked = self
            .take_potential_extra_op(resolved)
            .or_else(|| self.take_potential_extra_op(opref));
        if let Some(preamble_op) = tracked {
            let resolved_for_pop = self.get_box_replacement(preamble_op.op);
            if let Some(builder) = self.active_short_preamble_producer_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
            } else if let Some(builder) = self.imported_short_preamble_builder.as_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
            }
        }
        if let Some(mut info) = self.peek_ptr_info_via_box(resolved) {
            if info.is_virtual() {
                let forced = info.force_box(resolved, self);
                return self.get_box_replacement(forced);
            }
        }
        resolved
    }

    /// RPython optimizer.py:722-752 store_final_boxes_in_guard inline.
    /// Called from emit() for every guard during optimization. Produces
    /// rd_numb via memo.number() using the CURRENT optimizer state
    /// (replacement chain, constants, virtual info).
    /// resume.py ResumeDataVirtualAdder.finish() parity:
    /// Generate rd_numb + rd_consts + rd_virtuals for a guard.
    /// Called from store_final_boxes_in_guard in optimizer.rs.
    /// Uses snapshot data (vable_boxes, frame_pcs, multi-frame) when available.
    pub fn finalize_guard_resume_data(
        &mut self,
        op: &mut Op,
        knowledge: Option<crate::resume::OptimizerKnowledgeForResume>,
        pending_setfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) {
        self.store_final_boxes_in_guard(op, knowledge, pending_setfields);
    }

    fn store_final_boxes_in_guard(
        &mut self,
        op: &mut Op,
        knowledge: Option<crate::resume::OptimizerKnowledgeForResume>,
        mut pending_setfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) {
        use crate::resume::{ResumeDataLoopMemo, Snapshot};

        // optimizer.py:722-730 store_final_boxes_in_guard parity:
        //   if op.getdescr() is not None:
        //       descr = op.getdescr()
        //       assert isinstance(descr, compile.ResumeGuardDescr)
        //   else:
        //       descr = compile.invent_fail_descr_for_op(op.getopnum(), self)
        //       op.setdescr(descr)
        //
        // RPython has exactly one emit path, so this function never
        // sees a `ResumeGuardCopiedDescr` (sibling, not subclass —
        // compile.py:832) nor an already-finalized `ResumeGuardDescr`
        // (resume.py:397 `assert not storage.rd_numb` ensures finish()
        // runs at most once per descr).
        //
        // OptContext::emit gates its emit_guard_operation on
        // `!in_final_emission` so production runs through
        // `Optimizer::emit_guard_operation` once; the OptContext path is
        // limited to the standalone test entry.  Either way only fresh
        // descrs reach this function.
        assert!(
            op.descr.as_ref().map_or(true, |d| d.is_resume_guard()),
            "optimizer.py:723 store_final_boxes_in_guard expects \
             ResumeGuardDescr, got non-resume descr (kind={:?}, copied={})",
            op.descr.as_ref().map(|d| d.index()),
            op.descr
                .as_ref()
                .map_or(false, |d| d.is_resume_guard_copied())
        );

        // resume.py:397 `assert not storage.rd_numb` — finish() runs at
        // most once per ResumeGuardDescr.  RPython makes this invariant
        // load-bearing: a second call would clobber an already-numbered
        // livebox set and break bridge attachment.  Promoted from
        // debug_assert! so release builds catch double-finish too.
        assert!(
            op.descr
                .as_ref()
                .and_then(|d| d.as_fail_descr())
                .and_then(|fd| fd.rd_numb())
                .is_none(),
            "resume.py:397 finish() invoked twice on the same ResumeGuardDescr"
        );

        // resume.py:396-397:
        //   resume_position = self.guard_op.rd_resume_position
        //   assert resume_position >= 0
        // RPython: every guard has a valid rd_resume_position set by either
        // capture_resumedata (tracer guards) or patchguardop copy
        // (unroll.py:336/409). No fallback — the position is always set
        // before store_final_boxes_in_guard runs.
        let resume_pos = op.rd_resume_position;
        let has_snapshot = snapshot_contains(&self.snapshot_boxes, resume_pos);
        // resume.py:396-397: `assert resume_position >= 0` —
        // RPython asserts the position is set before calling
        // store_final_boxes_in_guard. Every guard from the production
        // pyre tracer captures its own snapshot via generate_guard /
        // capture_resumedata, and `gen_store_back_in_vable` inherits
        // the previous guard's snapshot id so its GUARD_NOT_FORCED_2
        // also has valid resume data. Hard-assert the invariant when
        // production snapshot data is wired through; the empty
        // `snapshot_boxes` case marks an isolated optimizer unit test
        // that constructs synthetic guards without going through the
        // pyre snapshot path, where the silent drop is acceptable.
        if !has_snapshot {
            // unroll.py:336/409 parity: when unroll creates a new guard from
            // a short preamble / virtual state import, it copies
            // rd_resume_position from patchguardop. If the new guard arrives
            // here without a snapshot, it must come from a patchguardop
            // context — inherit the patchguardop's resume_position.
            // resume.py:396-397: RPython asserts resume_position >= 0.
            let fallback_pos = self
                .patchguardop
                .as_ref()
                .map(|p| p.rd_resume_position)
                .filter(|&p| snapshot_contains(&self.snapshot_boxes, p));
            if let Some(fb_pos) = fallback_pos {
                op.rd_resume_position = fb_pos;
                // resume.py:570 _add_optimizer_sections: forward knowledge
                // to the patchguardop snapshot so heap/class/loopinvariant
                // sections are serialized into rd_numb. RPython's finish()
                // always serializes current optimizer knowledge regardless
                // of which snapshot provides the frame boxes.
                self.finalize_guard_resume_data(op, knowledge, pending_setfields);
                return;
            }
            // resume.py:396-397 parity:
            //   `resume_position = self.guard_op.rd_resume_position`
            //   `assert resume_position >= 0`
            // RPython asserts unconditionally here.  Tests that construct
            // guards directly must seed snapshot_boxes explicitly instead of
            // inventing a fail_args-derived fallback in this path.
            panic!(
                "store_final_boxes_in_guard: guard {:?} (pos={:?}, \
                 resume_pos={}) has no snapshot and no patchguardop \
                 ancestor — RPython resume.py:397 \
                 `assert resume_position >= 0` parity",
                op.opcode, op.pos, op.rd_resume_position
            );
        }

        // RPython parity: snapshot path handles ALL guards with snapshots,
        // including guards with rd_virtuals. The snapshot uses original boxes
        // and PtrInfo to correctly assign TAGVIRTUAL via _number_boxes.
        // _number_virtuals then builds rd_virtuals from PtrInfo.
        let snapshot_boxes = snapshot_get(&self.snapshot_boxes, op.rd_resume_position)
            .cloned()
            .unwrap_or_default();
        let vable_oprefs = snapshot_get(&self.snapshot_vable_boxes, op.rd_resume_position)
            .cloned()
            .unwrap_or_default();
        let vref_oprefs = snapshot_get(&self.snapshot_vref_boxes, op.rd_resume_position)
            .cloned()
            .unwrap_or_default();
        let frame_pcs = snapshot_get(&self.snapshot_frame_pcs, op.rd_resume_position)
            .cloned()
            .unwrap_or_default();

        // resume.py:201-202 get_box_replacement parity:
        // Pass ORIGINAL (unresolved) snapshot boxes. _number_boxes calls
        // env.get_box_replacement per-box, which resolves through the
        // replacement chain while preserving virtual identity.
        let frame_sizes = snapshot_get(&self.snapshot_frame_sizes, op.rd_resume_position);
        let mut snapshot = if let Some(sizes) = frame_sizes.filter(|s| s.len() > 1) {
            // Multi-frame: split snapshot_boxes into per-frame chunks.
            let mut frames = Vec::new();
            let mut offset = 0;
            for (i, &size) in sizes.iter().enumerate() {
                let end = (offset + size).min(snapshot_boxes.len());
                let frame_boxes: Vec<SnapshotBox> = snapshot_boxes[offset..end].to_vec();
                let (jitcode_index, pc) = frame_pcs.get(i).copied().unwrap_or((0, 0));
                frames.push((jitcode_index, pc, frame_boxes));
                offset = end;
            }
            Snapshot::multi_frame_boxes(frames)
        } else {
            let (jitcode_index, pc) = frame_pcs.first().copied().unwrap_or((0, 0));
            Snapshot::single_frame_boxes(jitcode_index, pc, snapshot_boxes.clone())
        };
        // pyjitpl.py:2588: vable_array stores virtualizable_boxes.
        // ni/vsd are constants (TAGINT/TAGCONST) so they don't affect
        // TAGBOX numbering. The same OpRefs also appear in fail_args —
        // _number_boxes deduplicates via liveboxes HashMap.
        snapshot.vable_array = vable_oprefs;
        // resume.py:243-247 _number_boxes also reads vref_array as a
        // separate section after vable_array. opencoder.py:767
        // create_top_snapshot writes both arrays into the snapshot.
        snapshot.vref_array = vref_oprefs;

        if majit_log_enabled() && op.opcode == OpCode::GuardNotForced2 {
            let env = OptBoxEnv { ctx: self };
            let snapshot_debug: Vec<(OpRef, OpRef, bool, Type)> = snapshot_boxes
                .iter()
                .copied()
                .map(|boxref| {
                    let boxref = boxref.opref;
                    let resolved = self.get_box_replacement(boxref);
                    let is_virtual = self.is_virtual_via_box(resolved);
                    let tp = majit_ir::BoxEnv::get_type(&env, boxref);
                    (boxref, resolved, is_virtual, tp)
                })
                .collect();
            let vable_debug: Vec<(OpRef, OpRef, bool, Type)> = snapshot
                .vable_array
                .iter()
                .copied()
                .map(|boxref| {
                    let boxref = boxref.opref;
                    let resolved = self.get_box_replacement(boxref);
                    let is_virtual = self.is_virtual_via_box(resolved);
                    let tp = majit_ir::BoxEnv::get_type(&env, boxref);
                    (boxref, resolved, is_virtual, tp)
                })
                .collect();
            eprintln!(
                "[jit][guard-resume] pos={:?} snapshot={:?} vable={:?}",
                op.pos, snapshot_debug, vable_debug
            );
        }

        // resume.py:389-452: delegate to ResumeDataVirtualAdder.finish()
        let env = OptBoxEnv { ctx: self };
        let mut memo = ResumeDataLoopMemo::new();
        let Ok(numb_state) = memo.number(&snapshot, &env, -1) else {
            return;
        };

        // resume.py:428-445, 520-558: pending_setfields are passed to finish()
        // which handles register_box, visitor_walk_recursive, and tagging.
        let (rd_numb, rd_consts, rd_virtuals, liveboxes, livebox_types) =
            memo.finish(numb_state, &env, &mut pending_setfields, knowledge.as_ref());

        if majit_log_enabled() && op.opcode == OpCode::GuardNotForced2 {
            eprintln!(
                "[jit][guard-resume] pos={:?} liveboxes={:?} rd_virtuals={} livebox_types={:?}",
                op.pos,
                liveboxes,
                rd_virtuals.len(),
                livebox_types
            );
        }

        // RPython Box.type parity: types captured at numbering time via
        // env.get_type(), equivalent to RPython's intrinsic Box.type.
        // Replaces the fragile 7-level type resolution cascade.
        let new_types: Vec<majit_ir::Type> = liveboxes
            .iter()
            .map(|opref| {
                if opref.is_none() {
                    return majit_ir::Type::Ref;
                }
                livebox_types
                    .get(opref)
                    .copied()
                    .unwrap_or(majit_ir::Type::Ref)
            })
            .collect();

        op.store_final_boxes(liveboxes);
        op.fail_arg_types = Some(new_types.clone());
        // optimizer.py:722-730 `store_final_boxes_in_guard` parity:
        //   if op.getdescr() is not None:
        //       descr = op.getdescr()
        //       assert isinstance(descr, compile.ResumeGuardDescr)
        //   else:
        //       descr = compile.invent_fail_descr_for_op(op.getopnum(), self)
        //       op.setdescr(descr)
        // RPython preserves the existing descr object (and its
        // `fail_index`, subtype, vector_info) and only mutates its
        // `fail_arg_types`. Pyre's MetaFailDescr / ResumeGuardDescr /
        // ResumeAtPositionDescr / CompileLoopVersionDescr keep `types`
        // in `UnsafeCell<Vec<Type>>`, exposed via
        // `FailDescr::set_fail_arg_types`, so we mutate in place — the
        // load-bearing contract that subtype markers
        // (`is_resume_at_position()`, `loop_version()`) survive
        // `store_final_boxes_in_guard` (compile.py:1035-1043, mirrored
        // at pyjitpl/mod.rs:6799 `is_resume_at_position()`).
        match op.descr.as_ref() {
            Some(existing) => {
                if let Some(fd) = existing.as_fail_descr() {
                    fd.set_fail_arg_types(new_types);
                }
            }
            None => {
                // RPython compile.py:919-937 `invent_fail_descr_for_op`
                // dispatches on opcode:
                //   GUARD_NOT_FORCED / GUARD_NOT_FORCED_2 → ResumeGuardForcedDescr
                //   GUARD_EXCEPTION  / GUARD_NO_EXCEPTION → ResumeGuardExcDescr
                //   else                                  → ResumeGuardDescr
                // The exception-flow / async-forcing special cases at
                // `pyjitpl/mod.rs` opcode-check sites (e.g. the
                // GUARD_EXCEPTION → `is_exception_guard` and
                // GUARD_NOT_FORCED chains) can migrate to descr-keyed
                // dispatch via `is_guard_exc()` / `is_guard_forced()`
                // without reshaping this match arm.
                use majit_ir::OpCode;
                op.descr = Some(match op.opcode {
                    OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                        crate::compile::make_resume_guard_forced_descr_typed(new_types)
                    }
                    OpCode::GuardException | OpCode::GuardNoException => {
                        crate::compile::make_resume_guard_exc_descr_typed(new_types)
                    }
                    _ => crate::compile::make_resume_guard_descr_typed(new_types),
                });
            }
        }
        // compile.py:855 ResumeGuardDescr `_attrs_` parity: write the
        // post-numbering resume payload onto the descr that
        // store_final_boxes_in_guard just minted (or onto the existing
        // ResumeGuardDescr / ResumeAtPositionDescr / ResumeGuardForcedDescr
        // / ResumeGuardExcDescr that capture_resumedata stamped earlier).
        // The descr is the single source of truth — readers go through
        // FailDescr::rd_*().
        let descr_rd_virtuals = if rd_virtuals.is_empty() {
            None
        } else {
            Some(rd_virtuals)
        };
        let descr_pending = if pending_setfields.is_empty() {
            None
        } else {
            Some(pending_setfields)
        };
        if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_fail_descr()) {
            fd.set_rd_numb(Some(rd_numb));
            fd.set_rd_consts(Some(rd_consts));
            fd.set_rd_virtuals(descr_rd_virtuals);
            fd.set_rd_pendingfields(descr_pending);
        }
        // resume.py: RPython does NOT carry frame sizes out-of-band.
        // The decoder reads jitcode liveness (jitcode.position_info) at
        // each frame's resume pc. majit routes this through the global
        // `frame_value_count_at` callback registered by pyre-jit-trace.
        let _ = frame_sizes;
    }

    /// Get the IntBound for an OpRef, if known from forwarded info or constants.
    /// Returns `None` for boxes that have no IntBound in `box._forwarded`.
    /// Equivalent to `peek_intbound_via_box`; preserved for legacy callers in
    /// rewrite.rs that gate optimizations on "is a bound known?".
    pub fn get_int_bound(&self, opref: OpRef) -> Option<crate::optimizeopt::intutils::IntBound> {
        self.peek_intbound_via_box(opref)
    }

    /// Allocate a fresh constant OpRef and store the value.
    ///
    /// RPython equivalent: `ConstInt(value)` — constants in RPython are
    /// first-class Const objects, not boxes. majit's constant pool model
    /// reserves an OpRef in the constant namespace and stores the value
    /// via `seed_constant`.
    ///
    /// NOTE: do NOT route through `make_constant`. That helper is the
    /// `optimizer.py:make_constant(box, constbox)` analogue and is meant
    /// to forward an existing **box** OpRef to a constant value. It bails
    /// out early when the input is already a constant OpRef
    /// (`is_constant()` true), which would silently drop the new entry.
    pub fn make_constant_int(&mut self, value: i64) -> OpRef {
        let pos = self.reserve_const_ref(Type::Int);
        self.seed_constant(pos, Value::Int(value));
        pos
    }

    pub fn make_constant_ref(&mut self, value: GcRef) -> OpRef {
        let pos = self.reserve_const_ref(Type::Ref);
        self.seed_constant(pos, Value::Ref(value));
        pos
    }

    pub fn make_constant_float(&mut self, value: f64) -> OpRef {
        let pos = self.reserve_const_ref(Type::Float);
        self.seed_constant(pos, Value::Float(value));
        pos
    }

    /// Record a constant-folded value and return its OpRef.
    ///
    /// If `opref` is not already a known constant, records the value.
    /// Returns `opref` (which is now known to be this constant).
    pub fn find_or_record_constant_int(&mut self, opref: OpRef, value: i64) -> OpRef {
        self.make_constant(opref, Value::Int(value));
        opref
    }

    /// Get constant float value, if known.
    pub fn get_constant_float(&self, opref: OpRef) -> Option<f64> {
        self.get_constant(opref).and_then(|v| match v {
            Value::Float(f) => Some(f),
            _ => None,
        })
    }

    /// optimizer.py: make_equal_to(op, value)
    /// Replace an op's result with a known value (forwarding + constant).
    pub fn make_equal_to(&mut self, opref: OpRef, target: OpRef) {
        self.replace_op(opref, target);
    }

    /// Look up the operation that produces a given OpRef.
    /// Searches emitted operations and input ops.
    /// Used for pattern matching nested operations (e.g., int_add(int_add(x, C1), C2)).
    /// Returns a clone to avoid borrow conflicts with mutable ctx methods.
    pub fn get_producing_op(&self, opref: OpRef) -> Option<Op> {
        let opref = self.get_box_replacement(opref);
        self.new_operations
            .iter()
            .find(|op| op.pos == opref)
            .cloned()
    }

    /// Number of emitted operations so far.
    pub fn num_emitted(&self) -> usize {
        self.new_operations.len()
    }

    /// Get the last emitted operation, if any.
    pub fn last_emitted_operation(&self) -> Option<&Op> {
        self.new_operations.last()
    }

    /// optimizer.py: get_constant_box(opref)
    /// Get a constant Value for an OpRef, or None if not constant.
    pub fn get_constant_box(&self, opref: OpRef) -> Option<Value> {
        self.get_constant(opref)
    }

    /// optimizer.py:783-790: constant_fold(op).
    /// Calls protect_speculative_operation, then execute_nonspec_const.
    /// Returns None on SpeculativeError (fold skipped).
    pub fn constant_fold(&self, op: &Op) -> Option<Value> {
        self.protect_speculative_operation(op)?;
        self.execute_nonspec_const(op)
    }

    /// optimizer.py:791-840: protect_speculative_operation(op).
    /// Validates that constant GcRef args are safe to dereference.
    /// Returns None (SpeculativeError) if validation fails.
    fn protect_speculative_operation(&self, op: &Op) -> Option<()> {
        // llmodel.py:555-567: protect_speculative_field.
        // When supports_guard_gc_type is false (majit has no GC type registry),
        // only null check is performed (llmodel.py:556-557).
        if op.opcode.is_getfield() {
            let gcref = match self.get_constant_box(op.arg(0))? {
                Value::Ref(r) => r,
                _ => return None,
            };
            if gcref.is_null() {
                return None; // SpeculativeError
            }
        }
        Some(())
    }

    /// executor.py:555 execute_nonspec_const → _execute_arglist →
    /// do_getfield_gc_i → cpu.bh_getfield_gc_i → llmodel.py:467
    /// read_int_at_mem → llop.raw_load(TYPE, gcref, ofs).
    ///
    /// RPython's constant_fold is ultimately a direct memory read.
    /// Safety is ensured by protect_speculative_operation (null + type
    /// check) BEFORE this function is called.
    ///
    /// GC safety: Ref constants are rooted on the shadow stack via
    /// ConstantPool (gcreftracer.py parity). GC updates shadow stack
    /// entries in place; refresh_from_gc propagates to the HashMap
    /// before optimization reads them. Constants are live during
    /// optimization (no Python allocations trigger GC).
    fn execute_nonspec_const(&self, op: &Op) -> Option<Value> {
        if !op.opcode.is_getfield() {
            return None;
        }
        let arg0 = op.arg(0);
        let resolved = self.get_box_replacement(arg0);
        // Resolve the receiver GcRef via the BoxRef chain only — never via
        // `self.constants[]` (which `seed_constant` pre-populates with
        // trace-time Ref values that may not match runtime; same
        // parity rationale as `pure.rs:835` `constant_ptr_value`). Three
        // valid sources, all PyPy `getinfo()`-equivalent:
        //   (a) walker advanced to a Const-namespace OpRef → const_pool
        //   (b) BoxRef terminal carries `Forwarded::Box(constbox)` →
        //       `BoxKind::Const.value` (orthodox `optimizer.py:432`)
        //   (c) BoxRef terminal carries legacy
        //       `Forwarded::Info(OpInfo::Constant(_))`
        let gcref = if resolved.is_constant() {
            match self.get_constant_box(resolved)? {
                Value::Ref(r) => r,
                _ => return None,
            }
        } else {
            use crate::optimizeopt::info::OpInfo;
            let b = self.get_box_replacement_box(resolved)?;
            if let Some(Value::Ref(r)) = b.const_value() {
                r
            } else if let crate::r#box::Forwarded::Info(OpInfo::Constant(Value::Ref(r))) =
                &*b.get_forwarded()
            {
                *r
            } else {
                return None;
            }
        };
        if gcref.is_null() {
            return None;
        }
        let descr = op.descr.as_ref()?;
        let fd = descr.as_field_descr()?;
        let addr = gcref.0 + fd.offset();
        // llmodel.py:467-478 read_int_at_mem / read_ref_at_mem dispatch.
        match (fd.field_type(), fd.field_size()) {
            (majit_ir::Type::Int, 8) => Some(Value::Int(unsafe { *(addr as *const i64) })),
            (majit_ir::Type::Int, 4) => {
                if fd.is_field_signed() {
                    Some(Value::Int(unsafe { *(addr as *const i32) as i64 }))
                } else {
                    Some(Value::Int(unsafe { *(addr as *const u32) as i64 }))
                }
            }
            (majit_ir::Type::Int, 2) => {
                if fd.is_field_signed() {
                    Some(Value::Int(unsafe { *(addr as *const i16) as i64 }))
                } else {
                    Some(Value::Int(unsafe { *(addr as *const u16) as i64 }))
                }
            }
            (majit_ir::Type::Int, 1) => Some(Value::Int(unsafe { *(addr as *const u8) as i64 })),
            (majit_ir::Type::Float, 8) => {
                let bits = unsafe { *(addr as *const u64) };
                Some(Value::Float(f64::from_bits(bits)))
            }
            (majit_ir::Type::Ref, _) => {
                let ptr = unsafe { *(addr as *const usize) };
                Some(Value::Ref(majit_ir::GcRef(ptr)))
            }
            _ => None,
        }
    }

    /// Look up the producing `Op` for an OpRef in `new_operations`.
    /// Returns `None` for inputargs, constants, and OpRefs not yet emitted.
    ///
    /// RPython equivalent: holding a reference to the producing `Box`
    /// itself (every Box is a Python object, so identity lookup is the
    /// `is` operator — O(1)). pyre's flat `OpRef(u32)` cannot mirror
    /// that in O(1) without an auxiliary index; mutation patterns on
    /// `new_operations` (in-place replace at `optimizer.rs:3391`,
    /// `rewrite.rs:1579/1674`, plus `remove(jump_idx)` at
    /// `optimizer.rs:2605`) make a maintained `pos_to_index` brittle.
    /// Slice 0.2 unifies on this single API; converting it to O(1)
    /// (via a maintained index or layout invariant) is deferred to a
    /// later slice once those mutation patterns are stabilised.
    pub fn op_at(&self, opref: OpRef) -> Option<&Op> {
        if let Some(op) = self.new_operations.iter().rev().find(|op| op.pos == opref) {
            return Some(op);
        }
        // Phase 1 emit-op fallback (history.py:220 box.type parity for
        // cross-phase OpRefs). Reverse scan mirrors `new_operations` so a
        // later replacement of the same `pos` wins. Returned `&Op` is
        // safe to read for `.type_` and other intrinsic attributes; arg
        // / descr fields refer to Phase 1's namespace and should not be
        // dereferenced through this path (Phase 2 callers only consume
        // `op.type_` via `get_op_result_type` / `opref_type`).
        self.phase1_emit_ops.iter().rev().find(|op| op.pos == opref)
    }

    /// RPython box.type parity: find the result type of the operation
    /// that produces this OpRef. Returns None if the OpRef is an
    /// inputarg or was not produced by any emitted operation.
    pub fn get_op_result_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        let op = self.op_at(opref)?;
        // history.py:220 ConstInt.type, resoperation.py:567 IntOp.type:
        // Box.type is intrinsic to the producing op. `op.type_` was
        // populated at construction (Slice 0.1) from `opcode.result_type()`.
        if op.type_ == majit_ir::Type::Void {
            None
        } else {
            Some(op.type_)
        }
    }

    /// optimizer.py: clear_newoperations()
    /// Clear the output operation list (used when restarting optimization).
    pub fn clear_newoperations(&mut self) {
        self.new_operations.clear();
        // Reset next_pos to the iteration's first fresh OpRef position
        // (right after the inputarg slice in the OpRef namespace), but
        // never below an already-registered value_types position. The
        // context survives across iterations (e.g. Phase 2 final_ctx
        // reused as `jump_ctx` for short-preamble inlining), so every
        // previously emitted op still sits in `value_types`. `reserve_pos`
        // only skips constants; without accounting for the typed slots
        // here, a subsequent `alloc_op_position` would hand out a pos
        // already typed in the prior iteration and trip the Box.type
        // retype invariant at `register_value_type`.
        let base = self.inputarg_base + self.num_inputs;
        self.next_pos = if let Some(&max_typed_pos) = self.value_types.keys().max() {
            base.max(max_typed_pos.saturating_add(1))
        } else {
            base
        };
        self.const_infos.clear();
    }

    /// Get a mutable reference to the last emitted operation.
    pub fn last_emitted_operation_mut(&mut self) -> Option<&mut Op> {
        self.new_operations.last_mut()
    }

    /// resoperation.py: `op.type` parity. RPython Boxes carry their
    /// type intrinsically (`AbstractValue.type` ∈ {`'i'`, `'r'`, `'f'`,
    /// `'v'`}). majit's flat OpRef model has no such intrinsic field,
    /// so the type is reconstructed from the available metadata sources
    /// in priority order:
    ///
    /// 1. The seeded constant value's intrinsic Rust type. A
    ///    `Value::Int` is `'i'`, `Value::Float` is `'f'`, `Value::Ref`
    ///    is `'r'`. The `constant_types_for_numbering` override is a
    ///    raw-pointer marker on `'i'`-typed Boxes (RPython's
    ///    `getrawptrinfo` ConstInt path) — it does NOT change `op.type`
    ///    from `'i'` to `'r'`, matching the upstream invariant that a
    ///    raw-pointer `ConstInt` Box stays `op.type == 'i'` while still
    ///    becoming `ConstPtrInfo` through `getrawptrinfo`.
    /// 2. `value_types`, populated when an op is emitted via
    ///    `OptContext::emit` (mirrors RPython `op.type` lookup on
    ///    operations with a known result type).
    /// 3. The producing op's static `result_type()` (last resort for
    ///    OpRefs that have not been emitted yet but exist in
    ///    `new_operations`).
    ///
    /// Returns `None` only when none of the above sources have type
    /// information for the OpRef. Callers must treat `None` like
    /// RPython's "unknown type" path and avoid making structural
    /// assumptions about it.
    pub fn opref_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        let resolved = self.get_box_replacement(opref);
        // 0. RPython `AbstractValue.type` (resoperation.py:29) parity. The
        //    Phase 1-5 OpRef enum encodes `box.type` directly in the variant
        //    tag (`ConstInt`/`InputArgInt`/`IntOp` → Int, etc.), so reading
        //    the tag is the line-by-line equivalent of upstream `box.type`.
        //    `Untyped(_)` and `None` return `None` here and fall through to
        //    the legacy chain (Slice 0.5 in-progress: producers that still
        //    allocate `Untyped` positions are being migrated to typed
        //    factories so this becomes the only path).
        if let Some(tp) = resolved.ty() {
            // Slice 0.5 dual-source check: when `value_types` carries an
            // entry for a typed-variant OpRef it MUST agree with the
            // variant tag. Surface any divergence so we can fix the
            // producer, not paper over with a side-table read.
            #[cfg(debug_assertions)]
            if let Some(&table_tp) = self.value_types.get(&resolved.raw()) {
                debug_assert_eq!(
                    tp, table_tp,
                    "opref_type: OpRef variant ({:?}) disagrees with \
                     value_types entry ({:?}) for {:?}",
                    tp, table_tp, resolved,
                );
            }
            return Some(tp);
        }
        // 1. Seeded constant — read the intrinsic Rust shape (history.py:220
        //    `ConstInt.type = INT` parity).
        //    The constant_types_for_numbering override is intentionally
        //    NOT consulted here — it is a raw-pointer marker on
        //    'i'-typed Boxes, not a type-changing annotation.
        if let Some(val) = self.get_constant(resolved) {
            return Some(val.get_type());
        }
        // 2. Producing op's intrinsic `type_` (resoperation.py:1693
        //    `opclasses[opnum].type` parity). Slice 0.1 populates this
        //    at construction; this is now the primary fast path.
        if let Some(tp) = self.get_op_result_type(resolved) {
            // Slice 0.3 dual-source check: when `value_types` carries an
            // entry it must agree with the intrinsic. Disagreement = bug
            // the side-table has been masking; surface during dev.
            #[cfg(debug_assertions)]
            if let Some(&table_tp) = self.value_types.get(&resolved.raw()) {
                debug_assert_eq!(
                    tp, table_tp,
                    "opref_type: Op.type_ ({:?}) disagrees with \
                     value_types entry ({:?}) for {:?}",
                    tp, table_tp, resolved,
                );
            }
            return Some(tp);
        }
        // 3. `value_types` side-table — covers OpRefs that have no
        //    producing Op yet (e.g. registered via `register_value_type`
        //    by passes that synthesize an aliased OpRef). Removed in
        //    Slice 0.5 once all writers route through Op intrinsics or
        //    the dedicated `inputarg_types` source below.
        if let Some(&tp) = self.value_types.get(&resolved.raw()) {
            return Some(tp);
        }
        // 4. Inputarg slot (recorder-side `InputArg{Int,Ref,Float}.tp`,
        //    history.py:220 parity). Slice 0.5 prep — placed last so
        //    legacy `value_types` writers keep priority during the
        //    transition; once Slice 0.5 drops the inputarg fan-in this
        //    becomes the authoritative inputarg-type source.
        if let Some(tp) = self.inputarg_type(resolved) {
            return Some(tp);
        }
        // 5. PtrInfo-derived type (history.py:220 box.type parity for
        //    virtual heads across phase boundaries). Phase 1 virtualizes
        //    NewWithVtable / New / NewArray etc. by returning
        //    `OptimizationResult::Remove` from the relevant `optimize_*`
        //    method (virtualize.py:208-225) — the op never lands in
        //    `new_operations` and is therefore absent from
        //    `phase1_emit_ops`. Phase 2 imports the virtual head's
        //    `PtrInfo` via `setinfo_from_preamble` (unroll.py:55-64) but
        //    starts with a fresh `value_types` map, so the prior four
        //    sources all miss. RPython preserves the type intrinsically
        //    on the Box object; pyre recovers it from the PtrInfo variant
        //    because every variant maps to a unique RPython box type.
        //    Ref-typed: Virtual / VirtualArray / VirtualStruct /
        //    VirtualArrayStruct / Array / Struct / Instance / Constant /
        //    NonNull / Virtualizable / Str (RPython instances of
        //    AbstractStructPtrInfo / ArrayPtrInfo / StrPtrInfo carry 'r').
        //    Int-typed: VirtualRawBuffer / VirtualRawSlice
        //    (info.py:865 RawBufferPtrInfo + getrawptrinfo() — these
        //    describe raw pointers stored in 'i' Boxes).
        if let Some(info) = self.peek_ptr_info_via_box(resolved) {
            return Some(match info {
                crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(_)
                | crate::optimizeopt::info::PtrInfo::VirtualRawSlice(_) => majit_ir::Type::Int,
                _ => majit_ir::Type::Ref,
            });
        }
        None
    }

    /// Inputarg slot lookup. Returns `Some(tp)` when `opref` falls in either
    /// the current context's own inputarg range
    /// `[inputarg_base, inputarg_base + num_inputs)` (RPython invariant) or
    /// the shared low range `[0, num_inputs)` (Phase 1's inputarg slot OpRefs
    /// referenced from Phase 2 via `imported_label_args`). Returns `None` for
    /// constants, sentinels, out-of-range OpRefs, Void-typed slots, or empty
    /// `inputarg_types` (test contexts that bypass `setup_optimizations`).
    ///
    /// `[0, num_inputs)` fallback: in RPython each `InputArgRef`/`InputArgInt`
    /// Box carries its `.type` intrinsically, so Phase 2 reads the same
    /// `box.type` regardless of which iteration's TraceIterator produced
    /// the box. Pyre's flat `OpRef(u32)` namespace separates Phase 1
    /// inputargs (at `[0, num_inputs)`) from Phase 2 inputargs (at
    /// `[phase2_inputarg_base, phase2_inputarg_base + num_inputs)`), but
    /// `Optimizer.trace_inputarg_types` is identical between phases (single
    /// recorder source — see `unroll.rs:313` and `unroll.rs:510`). Indexing
    /// the same `inputarg_types` Vec by raw position recovers Phase 1
    /// slot types from Phase 2 without a separate side-table
    /// (history.py:220 parity).
    ///
    /// Note: `inputarg_types` may be over-allocated (auto-seeded
    /// `vec![Ref; 1024]` test stubs from optimizer.rs:1751); the
    /// authoritative inputarg-slot count is `num_inputs` and the cap
    /// reflects opencoder's `inputargs occupy [inputarg_base,
    /// inputarg_base + num_inputs)` invariant.
    pub fn inputarg_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        if opref.is_none() || opref.is_constant() {
            return None;
        }
        let raw = opref.raw();
        let ni = self.num_inputs as usize;
        let idx = if raw >= self.inputarg_base && (raw - self.inputarg_base) < self.num_inputs {
            (raw - self.inputarg_base) as usize
        } else if self.inputarg_base > 0 && raw < self.num_inputs {
            // Phase 1 inputarg slot, accessed from a non-Phase-1 context
            // (Phase 2 / bridge). RPython resolves these through Box
            // identity; flat-OpRef pyre uses the shared inputarg_types Vec.
            raw as usize
        } else {
            return None;
        };
        if idx >= ni {
            return None;
        }
        let tp = *self.inputarg_types.get(idx)?;
        if tp == majit_ir::Type::Void {
            None
        } else {
            Some(tp)
        }
    }

    /// Look up the declared type of inputarg slot `idx` (zero-based) from
    /// the `inputarg_types` Vec seeded by `setup_optimizations`. Returns
    /// `None` if the slot is out of range, the type is `Void`, or the Vec
    /// has not been populated. Mirrors the per-Box `box.type` lookup that
    /// `init_virtualizable` and other consumers use to mint typed
    /// `OpRef::input_arg_*` matching the producer side.
    pub fn inputarg_type_at(&self, idx: usize) -> Option<majit_ir::Type> {
        let tp = *self.inputarg_types.get(idx)?;
        if tp == majit_ir::Type::Void {
            None
        } else {
            Some(tp)
        }
    }

    /// `Box.type` strict accessor. Panics when no source carries a type for
    /// `opref`, matching `history.py:802 record_same_as(box.type)`'s
    /// no-guess-on-miss policy: RPython Boxes always have an intrinsic type,
    /// so a missing type is a structural bug.
    ///
    /// Use this whenever the call site previously read a HashMap with
    /// `unwrap_or(Type::Int|Ref)` — those defaults silently absorbed bugs
    /// that would have been audible under the upstream invariant.  Sites
    /// that legitimately need a fallback (e.g. inputarg-stub harnesses,
    /// out-of-process compile.rs paths without an `OptContext`) should
    /// stay on `opref_type` and document the deviation.
    #[track_caller]
    pub fn op_type_strict(&self, opref: OpRef) -> majit_ir::Type {
        self.opref_type(opref).unwrap_or_else(|| {
            panic!(
                "op_type_strict: no Box.type for {:?} (resolved={:?}); \
                 every OpRef must have a type via constant / value_types / \
                 producer-op result_type. history.py:802 parity.",
                opref,
                self.get_box_replacement(opref),
            )
        })
    }

    /// Transitional: classify an [`OpRef`] into the [`AbstractValue`]
    /// variant that matches its RPython class.
    ///
    /// RPython's `AbstractValue` carries `type` and class identity
    /// (Const/InputArg/ResOp + i/r/f) intrinsically on the Python
    /// object. In pyre, [`OpRef`] is a numeric handle and the type
    /// information lives in side tables (constant pool, `value_types`,
    /// `get_op_result_type`); this method composes those into the
    /// canonical [`AbstractValue`] shape so callers can match on
    /// RPython class identity directly.
    ///
    /// Phase 1 of the OpRef → AbstractValue migration ships this as a
    /// non-breaking addition. Subsequent phases migrate construction
    /// sites to thread typed values through the call graph; the final
    /// phase replaces the [`OpRef`] tuple struct with an enum that IS
    /// [`AbstractValue`], at which point this method becomes
    /// unnecessary.
    ///
    /// Returns [`AbstractValue::None`] for [`OpRef::NONE`] and for
    /// OpRefs whose type cannot be resolved (a bookkeeping bug, but
    /// here treated as "unknown" rather than panicking — strict
    /// `expect()` callers should consult `opref_type` directly first).
    pub fn abstract_value(&self, opref: OpRef) -> majit_ir::AbstractValue {
        use majit_ir::{AbstractValue, Type};
        if opref.is_none() {
            return AbstractValue::None;
        }
        let resolved = self.get_box_replacement(opref);
        // Const family: constant-namespace OpRefs.
        if resolved.is_constant() {
            let idx = resolved.const_index();
            return match self.get_constant(resolved).map(|v| v.get_type()) {
                Some(Type::Int) => AbstractValue::ConstInt(idx),
                Some(Type::Float) => AbstractValue::ConstFloat(idx),
                Some(Type::Ref) => AbstractValue::ConstPtr(idx),
                _ => AbstractValue::None,
            };
        }
        let raw = resolved.raw();
        let ty = self.opref_type(resolved);
        // resoperation.py:719 AbstractInputArg parity: positions in the
        // [inputarg_base, inputarg_base + num_inputs) range are
        // InputArg{Int,Float,Ref}; everything else is an emitted
        // AbstractResOp result.
        let is_input_arg =
            raw >= self.inputarg_base && raw < self.inputarg_base.saturating_add(self.num_inputs);
        match (ty, is_input_arg) {
            (Some(Type::Int), true) => AbstractValue::InputArgInt(raw),
            (Some(Type::Float), true) => AbstractValue::InputArgFloat(raw),
            (Some(Type::Ref), true) => AbstractValue::InputArgRef(raw),
            (Some(Type::Int), false) => AbstractValue::IntOp(raw),
            (Some(Type::Float), false) => AbstractValue::FloatOp(raw),
            (Some(Type::Ref), false) => AbstractValue::RefOp(raw),
            _ => AbstractValue::None,
        }
    }

    /// info.py:865-878 `getrawptrinfo(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getrawptrinfo(op):
    ///     from rpython.jit.metainterp.optimizeopt.intutils import IntBound
    ///     assert op.type == 'i'
    ///     op = op.get_box_replacement()
    ///     assert op.type == 'i'
    ///     if isinstance(op, ConstInt):
    ///         return ConstPtrInfo(op)
    ///     fw = op.get_forwarded()
    ///     if isinstance(fw, IntBound):
    ///         return None
    ///     if fw is not None:
    ///         assert isinstance(fw, AbstractRawPtrInfo)
    ///         return fw
    ///     return None
    /// ```
    ///
    /// majit's only structural difference is the `ConstInt` arm: RPython
    /// trusts the upstream caller to have selected `getrawptrinfo` only
    /// for raw-pointer Boxes (i.e. the upstream caller is statically
    /// certain that this `ConstInt` carries a raw pointer rather than a
    /// plain integer). majit's flat constant pool cannot recover that
    /// caller-side intent from the `ConstInt` alone, so the raw-pointer
    /// signal is encoded as a `Type::Ref` entry in
    /// `constant_types_for_numbering`. A plain `Value::Int` constant
    /// without that annotation is treated as a plain integer (returns
    /// `None`), preventing counters/indices from masquerading as null
    /// pointers and triggering spurious `is_null` optimizations.
    pub fn getrawptrinfo(&self, opref: OpRef) -> Option<std::borrow::Cow<'_, PtrInfo>> {
        // assert op.type == 'i'
        debug_assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "getrawptrinfo: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        // op = op.get_box_replacement()
        let resolved = self.get_box_replacement(opref);
        // assert op.type == 'i'
        debug_assert!(matches!(
            self.opref_type(resolved),
            Some(majit_ir::Type::Int) | None
        ));
        // if isinstance(op, ConstInt): return ConstPtrInfo(op)
        // The `Type::Ref` override is majit's raw-pointer marker on the
        // 'i'-typed constant pool entry — see opref_type docs.
        if let Some(Value::Int(bits)) = self.get_constant(resolved) {
            if let Some(majit_ir::Type::Ref) = self
                .constant_types_for_numbering
                .get(&resolved.raw())
                .copied()
            {
                return Some(std::borrow::Cow::Owned(PtrInfo::Constant(majit_ir::GcRef(
                    bits as usize,
                ))));
            }
            // Plain integer ConstInt (no raw-pointer marker) → upstream
            // would never reach this branch from a properly-typed call
            // site; majit returns None to mirror that "no info".
            return None;
        }
        // fw = op.get_forwarded()
        // if isinstance(fw, IntBound): return None  →  peek_ptr_info_via_box
        //   only returns Some for Forwarded::Info(PtrInfo). An int-typed
        //   box that holds Forwarded::IntBound returns None here, matching
        //   the upstream early-return on IntBound forwarding.
        // if fw is not None: assert isinstance(fw, AbstractRawPtrInfo); return fw
        //   AbstractRawPtrInfo ↔ PtrInfo::VirtualRawBuffer / VirtualRawSlice
        //   in majit (see is_raw_ptr).
        // BoxRef-routing reader (H-3.2c slice 56). Always returns Owned —
        // test callers binding `Cow::Borrowed` keep working through deref.
        self.peek_ptr_info_via_box(resolved)
            .map(std::borrow::Cow::Owned)
    }

    /// info.py:880-894 `getptrinfo(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getptrinfo(op):
    ///     if op.type == 'i':
    ///         return getrawptrinfo(op)
    ///     elif op.type == 'f':
    ///         return None
    ///     assert op.type == 'r'
    ///     op = get_box_replacement(op)
    ///     assert op.type == 'r'
    ///     if isinstance(op, ConstPtr):
    ///         return ConstPtrInfo(op)
    ///     fw = op.get_forwarded()
    ///     if fw is not None:
    ///         assert isinstance(fw, PtrInfo)
    ///         return fw
    ///     return None
    /// ```
    ///
    /// The `op.type == 'r'` ConstPtr arm corresponds to a `Value::Ref`
    /// constant in majit's pool. The `op.type == 'i'` arm delegates to
    /// `getrawptrinfo`, which is the only path by which an integer
    /// constant becomes `ConstPtrInfo` — matching upstream's
    /// raw-pointer-only routing.
    pub fn getptrinfo(&self, opref: OpRef) -> Option<std::borrow::Cow<'_, PtrInfo>> {
        // if op.type == 'i': return getrawptrinfo(op)
        // elif op.type == 'f': return None
        // assert op.type == 'r'
        match self.opref_type(opref) {
            Some(majit_ir::Type::Int) => return self.getrawptrinfo(opref),
            Some(majit_ir::Type::Float) => return None,
            Some(majit_ir::Type::Ref) => {}
            // Type::Void or unknown — RPython would have asserted; majit
            // returns the forwarded slot (or None) to mirror "no info
            // available" rather than panicking on traces that haven't
            // populated value_types for every transitively reachable
            // OpRef.
            _ => {}
        }
        // op = get_box_replacement(op)
        let resolved = self.get_box_replacement(opref);
        // if isinstance(op, ConstPtr): return ConstPtrInfo(op)
        if let Some(Value::Ref(gcref)) = self.get_constant(resolved) {
            return Some(std::borrow::Cow::Owned(PtrInfo::Constant(gcref)));
        }
        // fw = op.get_forwarded()
        // if fw is not None: assert isinstance(fw, PtrInfo); return fw
        // BoxRef-routing reader (H-3.2c slice 56). Always returns Owned —
        // test callers binding `Cow::Borrowed` keep working through deref.
        self.peek_ptr_info_via_box(resolved)
            .map(std::borrow::Cow::Owned)
    }

    /// `info.py:880 getptrinfo(op)` BoxRef-routing reader. Returns owned
    /// `PtrInfo` (matching `peek_ptr_info_via_box` semantics) — synthesizes
    /// `ConstPtrInfo` for ConstPtr inputs (RPython parity at info.py:888-889).
    ///
    /// Mirror of [`Self::getptrinfo`] that routes through
    /// `box_pool[idx].forwarded`.
    pub fn getptrinfo_via_box(&self, opref: OpRef) -> Option<PtrInfo> {
        // if op.type == 'i': return getrawptrinfo(op)
        // elif op.type == 'f': return None
        match self.opref_type(opref) {
            Some(majit_ir::Type::Int) => return self.getrawptrinfo_via_box(opref),
            Some(majit_ir::Type::Float) => return None,
            Some(majit_ir::Type::Ref) => {}
            _ => {}
        }
        // op = get_box_replacement(op)
        let resolved = self.get_box_replacement(opref);
        // info.py:888-889: if isinstance(op, ConstPtr): return ConstPtrInfo(op)
        if let Some(Value::Ref(gcref)) = self.get_constant(resolved) {
            return Some(PtrInfo::Constant(gcref));
        }
        // fw = op.get_forwarded() — routed through BoxRef when pool plumbed.
        self.peek_ptr_info_via_box(resolved)
    }

    /// `info.py:840 getrawptrinfo(op)` BoxRef-routing reader. Returns owned
    /// `PtrInfo` synthesizing `ConstPtrInfo` for `Type::Ref`-overridden
    /// `ConstInt` (raw-pointer marker) entries.
    ///
    /// Mirror of [`Self::getrawptrinfo`]. See that method for the
    /// raw-pointer routing semantics.
    pub fn getrawptrinfo_via_box(&self, opref: OpRef) -> Option<PtrInfo> {
        debug_assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "getrawptrinfo_via_box: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let resolved = self.get_box_replacement(opref);
        if let Some(Value::Int(bits)) = self.get_constant(resolved) {
            if let Some(majit_ir::Type::Ref) = self
                .constant_types_for_numbering
                .get(&resolved.raw())
                .copied()
            {
                return Some(PtrInfo::Constant(majit_ir::GcRef(bits as usize)));
            }
            return None;
        }
        self.peek_ptr_info_via_box(resolved)
    }

    /// info.py:880 `getptrinfo(op).get_known_class(cpu)` parity.
    ///
    /// Delegates to `getptrinfo` (which synthesizes `ConstPtrInfo` for
    /// constant Refs) and then `PtrInfo::get_known_class`, so constant
    /// pointers are handled via `cls_of_box` the same way
    /// `Instance` / `Virtual` read their stored `known_class`.
    pub fn get_known_class(&self, opref: OpRef) -> Option<majit_ir::GcRef> {
        self.getptrinfo_via_box(opref)?.get_known_class()
    }

    /// optimizer.py:127-135 `getnullness(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getnullness(self, op):
    ///     if op.type == 'r' or self.is_raw_ptr(op):
    ///         ptrinfo = getptrinfo(op)
    ///         if ptrinfo is None:
    ///             return info.INFO_UNKNOWN
    ///         return ptrinfo.getnullness()
    ///     elif op.type == 'i':
    ///         return self.getintbound(op).getnullness()
    ///     assert False
    /// ```
    ///
    /// Returns one of `INFO_NULL` / `INFO_NONNULL` / `INFO_UNKNOWN`
    /// (info.py:13-15) so callers can compare directly against the
    /// upstream constants.
    ///
    /// Takes `&mut self` because the upstream `getintbound` lazily
    /// installs an unbounded `IntBound` on first access (optimizer.py:
    /// 102-112), and majit mirrors that side effect via
    /// `OptContext::getintbound`.
    pub fn getnullness(&mut self, opref: OpRef) -> i8 {
        let tp = self.opref_type(opref);
        // optimizer.py:128: op.type == 'r' or self.is_raw_ptr(op)
        if matches!(tp, Some(majit_ir::Type::Ref)) || self.is_raw_ptr(opref) {
            // ptrinfo = getptrinfo(op)
            // if ptrinfo is None: return INFO_UNKNOWN
            // return ptrinfo.getnullness()
            return match self.getptrinfo_via_box(opref) {
                None => INFO_UNKNOWN,
                Some(info) => info.getnullness(),
            };
        }
        // optimizer.py:133-134: elif op.type == 'i': return getintbound(op).getnullness()
        if matches!(tp, Some(majit_ir::Type::Int) | None) {
            return self.getintbound(opref).getnullness();
        }
        // optimizer.py:135: assert False  →  Float never reaches here.
        INFO_UNKNOWN
    }

    /// optimizer.py:154-158 `is_raw_ptr(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def is_raw_ptr(self, op):
    ///     fw = get_box_replacement(op).get_forwarded()
    ///     if isinstance(fw, info.AbstractRawPtrInfo):
    ///         return True
    ///     return False
    /// ```
    ///
    /// `AbstractRawPtrInfo` is the upstream base for `RawBufferPtrInfo`,
    /// `RawStructPtrInfo`, `RawSlicePtrInfo` (info.py:374-485). Of these:
    ///
    /// - `RawBufferPtrInfo` ↔ majit `PtrInfo::VirtualRawBuffer` (created
    ///   by `OptVirtualize` from `RAW_MALLOC_VARSIZE_CHAR`).
    /// - `RawSlicePtrInfo` ↔ majit `PtrInfo::VirtualRawSlice` (created
    ///   by `OptVirtualize::optimize_int_add` slice creator,
    ///   virtualize.py:60 make_virtual_raw_slice).
    /// - `RawStructPtrInfo` is defined at info.py:452 but never
    ///   instantiated anywhere in upstream (`grep -rn "RawStructPtrInfo("
    ///   rpython/jit/` returns only the class definition). It is dead
    ///   reservation code, so the absence of a majit variant is not a
    ///   parity gap.
    ///
    /// `ConstPtrInfo` is NOT a subclass of `AbstractRawPtrInfo` in
    /// upstream, so a constant raw-pointer `ConstInt` is `False` here
    /// (matches `isinstance(fw, AbstractRawPtrInfo)` returning `False`
    /// for `ConstPtrInfo`).
    pub fn is_raw_ptr(&self, opref: OpRef) -> bool {
        let resolved = self.get_box_replacement(opref);
        matches!(
            self.peek_ptr_info_via_box(resolved),
            Some(PtrInfo::VirtualRawBuffer(_)) | Some(PtrInfo::VirtualRawSlice(_))
        )
    }

    /// info.py: op.set_forwarded(info) — set PtrInfo for an OpRef.
    /// Ensure a PtrInfo exists for the given OpRef. Creates an empty
    /// Instance if none exists, so that set_field can store values.
    pub fn ensure_ptr_info(&mut self, opref: OpRef) {
        if opref.is_constant() {
            return;
        }
        use crate::optimizeopt::info::OpInfo;
        let idx = opref.raw() as usize;
        // BoxRef-authoritative probe. `ensure_box_at` lazily allocates a
        // placeholder when the recorder did not pre-seed this position.
        let b = self.ensure_box_at(idx);
        let already_set = !matches!(*b.get_forwarded(), crate::r#box::Forwarded::None);
        if !already_set {
            b.set_forwarded_info(OpInfo::Ptr(PtrInfo::instance(None, None)));
        }
    }

    /// Set PtrInfo without clearing forwarding.
    /// RPython parity: set PtrInfo at the terminal of opref's forwarding chain.
    /// In RPython, `box.set_forwarded(info)` sets info on the Box directly.
    /// `get_box_replacement(box)` then returns the terminal Box which has the info.
    /// In majit, we follow the Op chain to the terminal OpRef and set Info there.
    fn ensure_ptr_info_preserve_forwarding(&mut self, opref: OpRef, info: PtrInfo) {
        use crate::optimizeopt::info::OpInfo;
        let terminal = self.get_box_replacement(opref);
        if terminal.is_constant() {
            return;
        }
        let idx = terminal.raw() as usize;
        let b = self.ensure_box_at(idx);
        let already_set = !matches!(*b.get_forwarded(), crate::r#box::Forwarded::None);
        if !already_set {
            b.set_forwarded_info(OpInfo::Ptr(info));
        }
    }

    /// info.py:718-726 `ConstPtrInfo._get_info(descr, optheap)` parity.
    ///
    /// ```python
    /// def _get_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    ///     info = optheap.const_infos.get(ref, None)
    ///     if info is None:
    ///         info = StructPtrInfo(descr)
    ///         optheap.const_infos[ref] = info
    ///     return info
    /// ```
    ///
    /// majit's port: route through `getptrinfo` (which encapsulates the
    /// RPython `op.type` dispatch + `ConstPtrInfo` synthesis), then read
    /// `_const.getref_base()` from the resulting `PtrInfo::Constant`.
    /// Both `Value::Ref` constants and `Value::Int` constants tagged
    /// with a `Type::Ref` override hash to the same `const_infos` slot
    /// — the upstream invariant that any `ConstPtrInfo._get_info()`
    /// call on the same address returns the same shared
    /// `StructPtrInfo`.
    ///
    /// Returns `None` only when `opref` is not a constant pointer at all
    /// (matching PyPy's `getrawptrinfo` returning `None` for non-pointer
    /// boxes — there's no `_get_info` to call). For a constant pointer
    /// that resolves to a null `gcref`, this raises `InvalidLoop` via
    /// `panic_any`, exactly as PyPy `info.py:720-721` does:
    ///
    /// ```python
    /// def _get_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    /// ```
    ///
    /// The trace was constant-folding through a null base pointer, which
    /// is an impossible execution path; the optimizer aborts so the JIT
    /// can retry with a different shape.
    /// Like `get_const_info_mut` but does NOT create an entry on miss.
    /// Returns `None` when:
    /// - `opref` is not a constant pointer
    /// - The constant is null
    /// - No `const_infos` entry has been created yet
    ///
    /// Used by array invalidation paths that only need to clear existing
    /// items, not install new PtrInfo variants.
    pub fn get_const_info_mut_if_exists(
        &mut self,
        opref: OpRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        use crate::optimizeopt::info::PtrInfo;
        let gcref = match self.getptrinfo_via_box(opref) {
            Some(PtrInfo::Constant(g)) => g,
            _ => return None,
        };
        if gcref.is_null() {
            return None;
        }
        self.const_infos.get_mut(&gcref.0)
    }

    /// info.py:715-726 `ConstPtrInfo._get_info(descr, optheap)` parity.
    ///
    /// `parent_descr` is the parent SizeDescr, passed so that the
    /// vacant-slot case creates `StructPtrInfo(descr)` (info.py:724)
    /// rather than a bare `PtrInfo::instance(None, None)`. Callers
    /// that don't have the parent descr (e.g. the field read path)
    /// extract it from the field descr via
    /// `descr.as_field_descr().get_parent_descr()`.
    pub fn get_const_info_mut(
        &mut self,
        opref: OpRef,
        parent_descr: Option<DescrRef>,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        use crate::optimizeopt::info::PtrInfo;
        // info.py:719: ref = self._const.getref_base()
        let gcref = match self.getptrinfo_via_box(opref) {
            Some(PtrInfo::Constant(g)) => g,
            _ => return None,
        };
        // info.py:720-721: if not ref: raise InvalidLoop
        if gcref.is_null() {
            std::panic::panic_any(crate::optimize::InvalidLoop(
                "ConstPtrInfo._get_info: null constant base pointer",
            ));
        }
        let addr = gcref.0;
        use std::collections::hash_map::Entry;
        match self.const_infos.entry(addr) {
            // info.py:722-725: info = optheap.const_infos.get(ref, None)
            //                  if info is None: info = StructPtrInfo(descr)
            //                  optheap.const_infos[ref] = info
            Entry::Occupied(e) => Some(e.into_mut()),
            Entry::Vacant(e) => {
                // info.py:724: StructPtrInfo(descr)
                let info = match parent_descr {
                    Some(d) => PtrInfo::struct_ptr(d),
                    None => PtrInfo::instance(None, None),
                };
                Some(e.insert(info))
            }
        }
    }

    /// info.py:728-735 `ConstPtrInfo._get_array_info(descr, optheap)`
    /// parity:
    ///
    /// ```python
    /// def _get_array_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    ///     info = optheap.const_infos.get(ref, None)
    ///     if info is None:
    ///         info = ArrayPtrInfo(descr)
    ///         optheap.const_infos[ref] = info
    ///     return info
    /// ```
    ///
    /// Companion to `get_const_info_mut` for the array path. Both share
    /// the same `const_infos` slot keyed by `gcref` — PyPy's invariant
    /// is that a given constant ref is used as either a struct base or
    /// an array base, never both. The Vacant entry inserts an
    /// `ArrayPtrInfo` (descr + `nonnegative` lenbound) so subsequent
    /// `setitem`/`getitem` calls land on the right variant.
    pub fn get_const_info_array_mut(
        &mut self,
        opref: OpRef,
        descr: DescrRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        use crate::optimizeopt::info::PtrInfo;
        // info.py:729: ref = self._const.getref_base() — same dispatch as
        // _get_info; route through getptrinfo for the op.type contract.
        let gcref = match self.getptrinfo_via_box(opref) {
            Some(PtrInfo::Constant(g)) => g,
            _ => return None,
        };
        // info.py:730-731: if not ref: raise InvalidLoop
        if gcref.is_null() {
            std::panic::panic_any(crate::optimize::InvalidLoop(
                "ConstPtrInfo._get_array_info: null constant base pointer",
            ));
        }
        let addr = gcref.0;
        use std::collections::hash_map::Entry;
        match self.const_infos.entry(addr) {
            Entry::Occupied(e) => Some(e.into_mut()),
            Entry::Vacant(e) => Some(e.insert(crate::optimizeopt::info::PtrInfo::array(
                descr,
                crate::optimizeopt::intutils::IntBound::nonnegative(),
            ))),
        }
    }

    /// info.py:750-752 `ConstPtrInfo.setfield` + info.py:203-211
    /// `AbstractStructPtrInfo.setfield` parity (line-by-line PyPy
    /// `structinfo.setfield(...)` routing).
    ///
    /// ```python
    /// # ConstPtrInfo
    /// def setfield(self, fielddescr, struct, op, optheap=None, cf=None):
    ///     info = self._get_info(fielddescr.get_parent_descr(), optheap)
    ///     info.setfield(fielddescr, struct, op, optheap=optheap, cf=cf)
    ///
    /// # AbstractStructPtrInfo
    /// def setfield(self, fielddescr, struct, op, optheap=None, cf=None):
    ///     self.init_fields(fielddescr.get_parent_descr(),
    ///                      fielddescr.get_index())
    ///     self._fields[fielddescr.get_index()] = op
    /// ```
    ///
    /// The Rust port routes both branches through one helper so heap.rs
    /// callers don't need to special-case the constant arg0 path. The
    /// constant case lands on `const_infos[gcref]`; the regular case
    /// runs `ensure_ptr_info_arg0(op).as_mut().setfield(...)`.
    pub fn structinfo_setfield(&mut self, op: &Op, field_idx: u32, value: OpRef) {
        let arg0 = self.get_box_replacement(op.arg(0));
        if arg0.is_constant() || self.get_constant(arg0).is_some() {
            // info.py:750-752 ConstPtrInfo.setfield → _get_info(parent_descr, optheap)
            let parent_descr = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .and_then(|fd| fd.get_parent_descr());
            if let Some(info) = self.get_const_info_mut(arg0, parent_descr) {
                info.setfield(field_idx, value);
            }
            return;
        }
        // info.py:203-211 AbstractStructPtrInfo.setfield: mutate `_fields`
        // in the PtrInfo object stored in the BoxRef's `_forwarded` slot.
        // PyPy has the same single-object behavior via `box._forwarded`.
        self.with_ensured_ptr_info_arg0(op, |mut handle| {
            if let Some(mut pi) = handle.as_mut() {
                pi.setfield(field_idx, value);
            }
        });
    }

    /// info.py:746-748 `ConstPtrInfo.setitem` + info.py: ArrayPtrInfo
    /// `setitem` parity. Same shape as `structinfo_setfield` but routes
    /// through `_get_array_info` (`get_const_info_array_mut`) for the
    /// constant arg0 path so the const_infos slot is created as
    /// `PtrInfo::Array` rather than `PtrInfo::Instance`.
    pub fn arrayinfo_setitem(&mut self, op: &Op, index: usize, value: OpRef) {
        let arg0 = self.get_box_replacement(op.arg(0));
        if arg0.is_constant() || self.get_constant(arg0).is_some() {
            // info.py:746-748 ConstPtrInfo.setitem → _get_array_info.
            if let Some(descr) = op.descr.clone() {
                if let Some(info) = self.get_const_info_array_mut(arg0, descr) {
                    info.setitem(index, value);
                }
            }
            return;
        }
        // info.py: ArrayPtrInfo.setitem: mutate `_items` in the PtrInfo object
        // stored in the BoxRef's `_forwarded` slot.
        self.with_ensured_ptr_info_arg0(op, |mut handle| {
            if let Some(mut pi) = handle.as_mut() {
                pi.setitem(index, value);
            }
        });
    }

    /// RPython set_forwarded parity: setting Info on a box REPLACES
    /// any forwarding. In pyre, ptr_info and forwarding must be
    /// mutually exclusive — the last writer wins.
    /// optimizer.py:437-448: make_nonnull — record that a Ref box is nonnull.
    /// Only sets NonNull if no existing PtrInfo is present.
    pub fn make_nonnull(&mut self, opref: OpRef) {
        let resolved = self.get_box_replacement(opref);
        // PyPy `op.is_constant()` returns True for Const boxes after
        // `get_box_replacement`. The BoxRef-level probe also protects legacy
        // `Info(Constant)` fixtures where the walker terminates at the source
        // InputArg/ResOp and `resolved.is_constant()` would be false.
        if self.resolved_is_constant_via_box(resolved) {
            return;
        }
        // optimizer.py:446: if opinfo is not None: assert opinfo.is_nonnull(); return
        if self.has_ptr_info_via_box(resolved) {
            return;
        }
        // optimizer.py:448: op.set_forwarded(info.NonNullPtrInfo())
        self.set_ptr_info(resolved, PtrInfo::NonNull { last_guard_pos: -1 });
    }

    /// optimizer.py:461-499 `ensure_ptr_info_arg0(op)` — direct line-by-line
    /// port that returns the same kind of value as PyPy.
    ///
    /// ```python
    /// def ensure_ptr_info_arg0(self, op):
    ///     from rpython.jit.metainterp.optimizeopt import vstring
    ///     arg0 = self.get_box_replacement(op.getarg(0))
    ///     if arg0.is_constant():
    ///         return info.ConstPtrInfo(arg0)
    ///     opinfo = arg0.get_forwarded()
    ///     if isinstance(opinfo, info.AbstractVirtualPtrInfo):
    ///         return opinfo
    ///     elif opinfo is not None:
    ///         last_guard_pos = opinfo.get_last_guard_pos()
    ///     else:
    ///         last_guard_pos = -1
    ///     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
    ///     opnum = op.opnum
    ///     if (rop.is_getfield(opnum) or opnum == rop.SETFIELD_GC or
    ///         opnum == rop.QUASIIMMUT_FIELD):
    ///         descr = op.getdescr()
    ///         parent_descr = descr.get_parent_descr()
    ///         if parent_descr.is_object():
    ///             opinfo = info.InstancePtrInfo(parent_descr)
    ///         else:
    ///             opinfo = info.StructPtrInfo(parent_descr)
    ///         opinfo.init_fields(parent_descr, descr.get_index())
    ///     elif (rop.is_getarrayitem(opnum) or opnum == rop.SETARRAYITEM_GC or
    ///           opnum == rop.ARRAYLEN_GC):
    ///         opinfo = info.ArrayPtrInfo(op.getdescr())
    ///     elif opnum in (rop.GUARD_CLASS, rop.GUARD_NONNULL_CLASS):
    ///         opinfo = info.InstancePtrInfo()
    ///     elif opnum in (rop.STRLEN,):
    ///         opinfo = vstring.StrPtrInfo(vstring.mode_string)
    ///     elif opnum in (rop.UNICODELEN,):
    ///         opinfo = vstring.StrPtrInfo(vstring.mode_unicode)
    ///     else:
    ///         assert False, "operations %s unsupported" % op
    ///     assert isinstance(opinfo, info.NonNullPtrInfo)
    ///     opinfo.last_guard_pos = last_guard_pos
    ///     arg0.set_forwarded(opinfo)
    ///     return opinfo
    /// ```
    ///
    /// Returns an [`EnsuredPtrInfo`] discriminating the constant arg0 path
    /// (`Constant(GcRef)` ↔ `info.ConstPtrInfo(arg0)`) from the regular
    /// path (`Forwarded(&mut PtrInfo)` ↔ `arg0.set_forwarded(opinfo); return
    /// opinfo`). Callers invoke methods on the return value directly,
    /// matching PyPy's `structinfo.setfield(...)` /
    /// `arrayinfo.getlenbound(...)` patterns.
    pub fn ensure_ptr_info_arg0(&mut self, op: &Op) -> EnsuredPtrInfo {
        // optimizer.py:464: arg0 = self.get_box_replacement(op.getarg(0))
        let arg0 = self.get_box_replacement(op.arg(0));
        // optimizer.py:465-466: if arg0.is_constant(): return info.ConstPtrInfo(arg0)
        //
        // PyPy's `info.ConstPtrInfo(arg0)` wraps the constant box itself,
        // which can be either a `ConstPtr` (Ref) or a `ConstInt` (raw
        // pointer). PyPy doesn't reject either at this point — downstream
        // code calls `_const.getref_base()` and raises `InvalidLoop` only
        // when the ref is null. The Rust port matches that permissive
        // contract: extract whatever GcRef we can (Ref → the gcref, raw
        // pointer Int → cast, anything else → null sentinel) and let the
        // downstream user decide whether to act on it.
        if arg0.is_constant() || self.get_constant(arg0).is_some() {
            let gcref = match self.get_constant(arg0) {
                Some(Value::Ref(g)) => g,
                Some(Value::Int(bits)) => majit_ir::GcRef(bits as usize),
                // Float / Void / no-constant fall back to a null sentinel —
                // PyPy's getref_base would return null and InvalidLoop guard
                // the dereference at the actual use site.
                _ => majit_ir::GcRef(0),
            };
            // info.py:810-822 `ConstPtrInfo.getstrlen1(mode)`: clone the
            // resolver Arc into the EnsuredPtrInfo so subsequent
            // `getlenbound(Some(mode))` calls can ask the runtime for an
            // exact constant string length without re-borrowing self.
            let resolver = self.string_length_resolver.clone();
            return EnsuredPtrInfo::Constant {
                gcref,
                string_length_resolver: resolver,
            };
        }
        // optimizer.py:467-474:
        //     opinfo = arg0.get_forwarded()
        //     if isinstance(opinfo, info.AbstractVirtualPtrInfo):
        //         return opinfo
        //     elif opinfo is not None:
        //         last_guard_pos = opinfo.get_last_guard_pos()
        //     else:
        //         last_guard_pos = -1
        //     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
        //
        // The PyPy class hierarchy that drives the AbstractVirtualPtrInfo
        // early-return:
        //
        //     PtrInfo
        //       NonNullPtrInfo                       ← only this falls through
        //         AbstractVirtualPtrInfo
        //           AbstractStructPtrInfo
        //             InstancePtrInfo                ← Instance / Virtual
        //             StructPtrInfo                  ← Struct / VirtualStruct
        //           AbstractRawPtrInfo
        //             RawBufferPtrInfo               ← VirtualRawBuffer
        //             RawSlicePtrInfo                ← VirtualRawSlice
        //           ArrayPtrInfo                     ← Array / VirtualArray
        //             ArrayStructInfo                ← VirtualArrayStruct
        //         vstring.StrPtrInfo                 ← Str
        //       ConstPtrInfo                         ← Constant (handled before)
        //
        // The early-return path uses a `&'s mut PtrInfo` whose lifetime
        // matches the function return. Once that mutable borrow is taken,
        // the borrow checker conservatively prevents any further write to
        // the same `_forwarded` slot even on the construction branch (which
        // never executes when we early-returned). To stay close to PyPy's
        // single-`opinfo` shape we read the slot immutably with
        // `get_ptr_info` to compute `last_guard_pos`, drop that read, and
        // then either re-borrow mutably for the early return or fall
        // through to the upgrade.
        // BoxRef-routing read. Owned PtrInfo from `peek_ptr_info_via_box` is
        // consumed by `matches!` so no borrow is held when the mutable
        // re-borrow of the BoxRef slot runs below for the early return.
        if matches!(
            self.peek_ptr_info_via_box(arg0),
            Some(
                PtrInfo::Instance(_)
                    | PtrInfo::Virtual(_)
                    | PtrInfo::Struct(_)
                    | PtrInfo::VirtualStruct(_)
                    | PtrInfo::Array(_)
                    | PtrInfo::VirtualArray(_)
                    | PtrInfo::VirtualArrayStruct(_)
                    | PtrInfo::VirtualRawBuffer(_)
                    | PtrInfo::VirtualRawSlice(_)
                    | PtrInfo::Virtualizable(_)
                    | PtrInfo::Str(_)
            )
        ) {
            // optimizer.py:469: return opinfo. `ensure_box_at` guarantees a
            // BoxRef even for positions whose pool slot was skipped by the
            // recorder.
            let idx = arg0.raw() as usize;
            let bx = self.ensure_box_at(idx);
            return EnsuredPtrInfo::ForwardedBox(bx);
        }
        let last_guard_pos = if let Some(opinfo) = self.peek_ptr_info_via_box(arg0) {
            // optimizer.py:474:
            //     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
            debug_assert!(
                matches!(opinfo, PtrInfo::NonNull { .. }),
                "ensure_ptr_info_arg0: existing non-virtual PtrInfo must be NonNullPtrInfo before upgrade, got {:?}",
                opinfo
            );
            // optimizer.py:471: last_guard_pos = opinfo.get_last_guard_pos()
            opinfo.last_guard_pos().unwrap_or(-1)
        } else {
            // optimizer.py:472-473: else: last_guard_pos = -1
            -1
        };
        // optimizer.py:475-495: dispatch on opcode to construct the right
        // PtrInfo class. The Rust port reuses PtrInfo factory constructors
        // (`PtrInfo::array`, `PtrInfo::instance`, `PtrInfo::struct_ptr`,
        // and the StrPtrInfo struct literal).
        let mut new_info = if op.opcode.is_getfield()
            || op.opcode == OpCode::SetfieldGc
            || op.opcode == OpCode::QuasiimmutField
        {
            // optimizer.py:476-484:
            //     descr = op.getdescr()
            //     parent_descr = descr.get_parent_descr()
            //     if parent_descr.is_object():
            //         opinfo = info.InstancePtrInfo(parent_descr)
            //     else:
            //         opinfo = info.StructPtrInfo(parent_descr)
            //     opinfo.init_fields(parent_descr, descr.get_index())
            let field_descr = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .expect("ensure_ptr_info_arg0: field op without FieldDescr");
            // optimizer.py:479-484: parent_descr.is_object() decides Instance vs Struct.
            let parent_descr = field_descr.get_parent_descr().unwrap_or_else(|| {
                panic!(
                    "ensure_ptr_info_arg0: FieldDescr.get_parent_descr() returned None \
                     for opcode={:?} descr={:?} field_name={:?} index_in_parent={} \
                     offset={} field_type={:?}; the FieldDescr implementation must \
                     override get_parent_descr() for parity with optimizer.py:478",
                    op.opcode,
                    op.descr,
                    field_descr.field_name(),
                    field_descr.index_in_parent(),
                    field_descr.offset(),
                    field_descr.field_type(),
                )
            });
            let is_object = parent_descr
                .as_size_descr()
                .expect(
                    "ensure_ptr_info_arg0: FieldDescr.get_parent_descr() must point at a SizeDescr",
                )
                .is_object();
            let mut new_info = if is_object {
                PtrInfo::instance(Some(parent_descr.clone()), None)
            } else {
                PtrInfo::struct_ptr(parent_descr.clone())
            };
            // optimizer.py:484: opinfo.init_fields(parent_descr, descr.get_index())
            // info.py:180-188 init_fields(parent_descr, index) sets self.descr
            // and pre-allocates _fields by parent slot count.
            new_info.init_fields(parent_descr, field_descr.index_in_parent());
            new_info
        } else if op.opcode.is_getarrayitem()
            || op.opcode == OpCode::SetarrayitemGc
            || op.opcode == OpCode::ArraylenGc
        {
            // optimizer.py:485-487: getarrayitem / setarrayitem_gc / arraylen_gc
            // → ArrayPtrInfo(op.getdescr())
            let descr = op
                .descr
                .clone()
                .expect("ensure_ptr_info_arg0: array op without descr");
            PtrInfo::array(descr, crate::optimizeopt::intutils::IntBound::nonnegative())
        } else if op.opcode == OpCode::GuardClass || op.opcode == OpCode::GuardNonnullClass {
            // optimizer.py:488-489: guard_class / guard_nonnull_class
            // → InstancePtrInfo()
            PtrInfo::instance(None, None)
        } else if op.opcode == OpCode::Strlen {
            // optimizer.py:490-491: strlen → StrPtrInfo(mode_string)
            PtrInfo::Str(crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            })
        } else if op.opcode == OpCode::Unicodelen {
            // optimizer.py:492-493: unicodelen → StrPtrInfo(mode_unicode)
            PtrInfo::Str(crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 1,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            })
        } else {
            // optimizer.py:494-495: assert False, "operations %s unsupported"
            panic!("ensure_ptr_info_arg0: opcode {:?} unsupported", op.opcode);
        };
        // optimizer.py:496: assert isinstance(opinfo, info.NonNullPtrInfo)
        // — every constructed PtrInfo above is a NonNullPtrInfo subclass.
        // optimizer.py:497: opinfo.last_guard_pos = last_guard_pos
        new_info.set_last_guard_pos(last_guard_pos);
        // optimizer.py:498: arg0.set_forwarded(opinfo)
        let idx = arg0.raw() as usize;
        let bx = self.ensure_box_at(idx);
        use crate::optimizeopt::info::OpInfo;
        bx.set_forwarded_info(OpInfo::Ptr(new_info));
        // optimizer.py:499: return opinfo — hand back the BoxRef so subsequent
        // mutations land on the authoritative slot.
        EnsuredPtrInfo::ForwardedBox(bx)
    }

    /// optimizer.py:453-459: make_nonnull_str — record StrPtrInfo on a string box.
    /// Only sets if no existing PtrInfo is present or existing is not StrPtrInfo.
    pub fn make_nonnull_str(&mut self, opref: OpRef, mode: u8) {
        let resolved = self.get_box_replacement(opref);
        // Same Const-detection caveat as `make_nonnull`: PyPy's
        // Forwarded::Box(constbox) shape and legacy `Info(Constant)` fixtures
        // both need the BoxRef-level probe before `set_ptr_info`.
        if self.resolved_is_constant_via_box(resolved) {
            return;
        }
        if let Some(PtrInfo::Str(_)) = self.peek_ptr_info_via_box(resolved) {
            return;
        }
        self.set_ptr_info(
            resolved,
            PtrInfo::Str(crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
    }

    /// rewrite.py:434-435: isinstance(old_guard_op.getdescr(),
    /// compile.ResumeAtPositionDescr).
    /// guard_pos is a _newoperations index (info.py:100-103).
    pub fn is_resume_at_position_guard(&self, guard_pos: i32) -> bool {
        if guard_pos < 0 {
            return false;
        }
        self.new_operations
            .get(guard_pos as usize)
            .and_then(|op| op.descr.as_ref())
            .map_or(false, |descr| descr.is_resume_at_position())
    }

    /// Take ownership of PtrInfo, replacing with None.
    /// Used by force_box to mutate info in-place (RPython parity).
    pub fn take_ptr_info(&mut self, opref: OpRef) -> Option<PtrInfo> {
        use crate::optimizeopt::info::OpInfo;
        let r = self.get_box_replacement(opref);
        let idx = r.raw() as usize;
        // BoxRef-authoritative read. PtrInfo writers populate the BoxRef via
        // `ensure_box_at`, so any PtrInfo present at `idx` lives on the
        // BoxRef.
        let b = self.box_pool.get(idx).cloned()?;
        let info = match &*b.get_forwarded() {
            crate::r#box::Forwarded::Info(OpInfo::Ptr(p)) => Some(p.clone()),
            _ => None,
        };
        if info.is_some() {
            b.clear_forwarded();
        }
        info
    }

    pub fn set_ptr_info(&mut self, opref: OpRef, info: PtrInfo) {
        if opref.is_constant() {
            return;
        }
        use crate::optimizeopt::info::OpInfo;
        let idx = opref.raw() as usize;
        // `ensure_box_at` lazily allocates a Type::Void placeholder when no
        // recorder-allocated box exists at this position (test fixtures).
        let b = self.ensure_box_at(idx);
        b.set_forwarded_info(OpInfo::Ptr(info));
    }

    /// Lazy-allocate a `BoxRef::new_resop(Type::Void, idx)` placeholder at
    /// `box_pool[idx]` (and any preceding holes) when absent, returning a
    /// clone of the BoxRef at that position.
    ///
    /// PRE-EXISTING-ADAPTATION audit point (H): PyPy has no notion of a
    /// synthetic placeholder box — every box in the trace was allocated by
    /// `Trace.record()` / `inputarg()`. The lazy-alloc path here exists
    /// solely for test fixtures that build `OptContext::with_num_inputs(...)`
    /// directly and drive writers without going through the recorder or
    /// `emit()` pipeline. Production paths (recorder plumbed pool +
    /// `reserve_pos` / `reserve_pos_typed` eagerly extending on every
    /// `emit()`, mod.rs:1647-1662, 1689-1702) never reach `idx >=
    /// self.box_pool.len()`. Verified by an opt-in
    /// `MAJIT_PROBE_ENSURE_BOX_AT=1` probe on the release `pyre-dynasm`
    /// binary across fib_recursive, nbody, fannkuch, list_setslice,
    /// nested_loop benches: 0 lazy-alloc fires against a non-empty pool.
    /// Tests with partially-populated pools (e.g. `with_num_inputs(8, 2)`
    /// + direct writer calls) legitimately exercise the branch — retiring
    /// it under `#[cfg(test)]` requires migrating those fixtures and is
    /// tracked as Slice 5 of the strict-parity `_forwarded` epic.
    ///
    /// Phantom intermediate placeholders carry `Type::Void`, which the
    /// chain walker treats as "type-erased" — it preserves the source
    /// `OpRef` variant via `with_raw` instead of promoting to
    /// `void_op`/`input_arg_typed(_, Void)`, so callers that thread
    /// `OpRef::from_raw(_)` through the optimizer continue to round-trip.
    fn ensure_box_at(&mut self, idx: usize) -> crate::r#box::BoxRef {
        if idx >= self.box_pool.len() {
            let start = self.box_pool.len();
            for i in start..=idx {
                self.box_pool.push(crate::r#box::BoxRef::new_resop(
                    majit_ir::Type::Void,
                    i as u32,
                ));
            }
        }
        self.box_pool.get(idx).cloned().expect("just ensured")
    }

    /// optimizer.py: replace_op_with(old, new_op, ctx)
    /// Replace old opref AND emit the new op.
    pub fn replace_op_with(&mut self, old: OpRef, new_op: Op) -> OpRef {
        let new_ref = self.emit(new_op);
        self.replace_op(old, new_ref);
        new_ref
    }
}

/// An optimization pass.
///
/// optimizer.py: Optimization base class.
pub trait Optimization {
    /// Process an operation. Called for each operation in the trace.
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult;

    /// optimizer.py:71 propagate_postprocess — called AFTER the op has been
    /// emitted through all passes and added to new_operations. Runs in
    /// REVERSE pass order. RPython uses this for bounds propagation
    /// (intbounds.py postprocess_GUARD_TRUE) and heap cache updates
    /// (heap.py postprocess_GETFIELD_GC_I).
    fn propagate_postprocess(&mut self, _op: &Op, _ctx: &mut OptContext) {}

    /// optimizer.py:74-75 have_postprocess
    fn have_postprocess(&self) -> bool {
        false
    }

    /// optimizer.py:77-79 have_postprocess_op(opnum)
    fn have_postprocess_op(&self, _opcode: OpCode) -> bool {
        self.have_postprocess()
    }

    /// Called once before optimization starts.
    fn setup(&mut self) {}

    /// Called after all operations have been processed.
    fn flush(&mut self, _ctx: &mut OptContext) {}

    /// Mark this pass as Phase 2 (loop body). Phase 2 should not fully
    /// virtualize New() ops because guard recovery_layout is not yet
    /// populated. Default: no-op.
    fn set_phase2(&mut self, _phase2: bool) {}

    /// warmstate.py: pureop_historylength.
    /// Only OptPure consumes this; other passes ignore it.
    fn set_pureop_historylength(&mut self, _limit: usize) {}

    /// optimizer.py:517 propagate_all_forward(trace, call_pure_results, flush).
    /// Only OptPure consumes this; other passes ignore it.
    fn set_call_pure_results(
        &mut self,
        _results: &std::collections::HashMap<Vec<majit_ir::Value>, majit_ir::Value>,
    ) {
    }

    /// Name of this pass (for debugging).
    fn name(&self) -> &'static str;

    /// optimizer.py:557 parity hook — drain this pass's accumulated
    /// `Counters.*` bumps into `staticdata.profiler` and reset the
    /// internal accumulators.
    ///
    /// Each pass that records its own `Counters.*` bumps
    /// (vector.py:139/146 OPT_VECTORIZE_TRY/OPT_VECTORIZED, heap.py
    /// HEAPCACHED_OPS, ...) overrides this; the default impl does
    /// nothing for passes that have no counters of their own.
    /// `Optimizer::update_counters` calls this on every pass after
    /// each `propagate_all_forward` exit.
    fn drain_profiler_counters(&mut self, _profiler: &crate::jitprof::JitProfiler) {}

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Contribute operations to the short preamble builder.
    /// Called after preamble optimization to collect ops that bridges need to replay.
    /// RPython passes `optimizer` for PtrInfo access. We pass `ctx`.
    fn produce_potential_short_preamble_ops(
        &self,
        _sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        _ctx: &mut OptContext,
    ) {
        // Default: no contribution
    }

    /// heap.py:825-846 serialize_optheap(available_boxes) — export struct field triples.
    /// `available_boxes`: None = no filter (accept all), Some = RPython filter.
    fn export_cached_fields(
        &self,
        _ctx: &mut OptContext,
        _available_boxes: Option<&std::collections::HashMap<OpRef, ()>>,
    ) -> Vec<(OpRef, majit_ir::DescrRef, OpRef)> {
        Vec::new()
    }

    /// heap.py:870-883 deserialize_optheap — import struct fields.
    fn import_cached_fields(
        &mut self,
        _entries: &[(OpRef, majit_ir::DescrRef, OpRef)],
        _ctx: &mut OptContext,
    ) {
    }

    /// heap.py:847-868 serialize_optheap(available_boxes) — export array item triples.
    /// `available_boxes`: None = no filter (accept all), Some = RPython filter.
    fn export_cached_arrayitems(
        &self,
        _ctx: &mut OptContext,
        _available_boxes: Option<&std::collections::HashMap<OpRef, ()>>,
    ) -> Vec<(OpRef, i64, majit_ir::DescrRef, OpRef)> {
        Vec::new()
    }

    /// heap.py:885-894 deserialize_optheap — import array item triples.
    fn import_cached_arrayitems(
        &mut self,
        _entries: &[(OpRef, i64, majit_ir::DescrRef, OpRef)],
        _ctx: &mut OptContext,
    ) {
    }

    /// rewrite.py:828-834 serialize_optrewrite
    fn serialize_optrewrite(&self) -> Vec<(i64, OpRef)> {
        Vec::new()
    }

    /// rewrite.py:836-838 deserialize_optrewrite
    fn deserialize_optrewrite(&mut self, _entries: &[(i64, OpRef)]) {}

    /// shortpreamble.py:112-126: PureOp.produce_op / LoopInvariantOp.produce_op
    /// Transfer imported PreambleOp entries from OptContext to this pass.
    /// RPython calls `opt.optimizer.optpure` directly during produce_op.
    /// In majit, the Optimization trait mediates this transfer.
    fn install_preamble_pure_ops(&mut self, _ctx: &OptContext) {}

    /// RPython unroll.py: exported_infos also carries widened IntBound knowledge.
    fn export_arg_int_bounds(
        &self,
        _args: &[OpRef],
        _ctx: &OptContext,
    ) -> HashMap<OpRef, IntBound> {
        HashMap::new()
    }

    /// optimizer.py: is_virtual(opref)
    /// Whether an opref refers to a virtual object (for this pass).
    fn is_virtual(&self, _opref: OpRef) -> bool {
        false
    }

    /// RPython optimizer.py: emitting_operation(op)
    /// Called before any operation is emitted to the output, regardless of
    /// which pass emits it. This enables passes like OptHeap to force lazy
    /// sets before guards, even when the guard is emitted by an earlier pass.
    ///
    /// `self_pass_idx` is this pass's own index in the optimizer pipeline.
    /// RPython uses `self.next_optimization` to route lazy-set emissions
    /// starting AFTER the current pass. In majit, pass this index to
    /// `emit_extra` to achieve the same behavior.
    fn emitting_operation(&mut self, _op: &Op, _ctx: &mut OptContext, _self_pass_idx: usize) {}
}

#[cfg(test)]
pub(crate) fn seed_guard_snapshots_with<F>(
    ops: &[Op],
    mut snapshot_for_guard: F,
) -> (Vec<Op>, SnapshotBoxes)
where
    F: FnMut(&Op) -> Vec<OpRef>,
{
    let mut seeded = ops.to_vec();
    let mut snapshots: SnapshotBoxes = Vec::new();
    let mut next_resume_pos = 0i32;
    for op in seeded.iter_mut().filter(|op| op.opcode.is_guard()) {
        let snapshot_boxes = snapshot_for_guard(op);
        let resume_pos = if op.rd_resume_position >= 0
            && !snapshot_contains(&snapshots, op.rd_resume_position)
        {
            op.rd_resume_position
        } else {
            while snapshot_contains(&snapshots, next_resume_pos) {
                next_resume_pos += 1;
            }
            let resume_pos = next_resume_pos;
            next_resume_pos += 1;
            resume_pos
        };
        op.rd_resume_position = resume_pos;
        snapshot_insert(
            &mut snapshots,
            resume_pos,
            snapshot_boxes.into_iter().map(SnapshotBox::from).collect(),
        );
    }
    (seeded, snapshots)
}

/// Test fixture helper for optimizer tests whose guard resume state is
/// intentionally irrelevant.  RPython would still have a
/// `capture_resumedata()` entry for such a guard; the explicit empty list
/// models an empty active frame snapshot rather than deriving anything from
/// `guard.fail_args`.
#[cfg(test)]
pub(crate) fn seed_empty_guard_snapshots(ops: &[Op]) -> (Vec<Op>, SnapshotBoxes) {
    seed_guard_snapshots_with(ops, |_| Vec::new())
}

#[cfg(test)]
mod boxref_forwarding_tests {
    //! BoxRef `_forwarded` invariants: the four writers (`set_ptr_info`,
    //! `setintbound`, `make_constant`, `replace_op`) install PyPy-style
    //! forwarding state on the authoritative BoxRef slot.
    use super::*;
    use crate::r#box::{BoxRef, Forwarded as BoxForwarded};
    use crate::optimizeopt::info::{OpInfo, PtrInfo};
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::{OpRef, Type, Value};

    fn ctx_with_two_int_boxes() -> (OptContext, BoxRef, BoxRef) {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let b0 = BoxRef::new_inputarg(Type::Int, Some(0));
        let b1 = BoxRef::new_inputarg(Type::Int, Some(1));
        ctx.box_pool = vec![b0.clone(), b1.clone()].into();
        (ctx, b0, b1)
    }

    /// `replace_op(old, new)` mirrors `old_box.set_forwarded_box(new_box)`.
    #[test]
    fn h3_1_replace_op_mirrors_box_forward() {
        let (mut ctx, b0, b1) = ctx_with_two_int_boxes();
        ctx.replace_op(OpRef::int_op(0), OpRef::int_op(1));
        match &*b0.get_forwarded() {
            BoxForwarded::Box(target) => assert_eq!(target, &b1),
            other => panic!("expected Forwarded::Box, got {:?}", other),
        }
    }

    /// `replace_op(old, NONE)` mirrors `old_box.clear_forwarded()`.
    #[test]
    fn h3_1_replace_op_to_none_clears_box_forward() {
        let (mut ctx, b0, _b1) = ctx_with_two_int_boxes();
        ctx.replace_op(OpRef::int_op(0), OpRef::int_op(1));
        ctx.replace_op(OpRef::int_op(0), OpRef::NONE);
        assert!(matches!(*b0.get_forwarded(), BoxForwarded::None));
    }

    /// `optimizer.py:387-400 make_equal_to` Info transfer parity: when
    /// `old` carries `Forwarded::IntBound(_)` and is forwarded to a
    /// non-constant `new`, the IntBound moves to `new`'s slot.
    #[test]
    fn h3_1_replace_op_transfers_int_bound_to_new() {
        let (mut ctx, b0, b1) = ctx_with_two_int_boxes();
        let bound = IntBound::from_constant(7);
        ctx.setintbound(OpRef::int_op(0), &bound);
        ctx.replace_op(OpRef::int_op(0), OpRef::int_op(1));
        // After: old's IntBound transferred to new (PyPy:
        // `newop.set_forwarded(opinfo)`). old now forwards to new.
        match &*b1.get_forwarded() {
            BoxForwarded::Info(OpInfo::IntBound(b)) => assert_eq!(b.lower, 7),
            other => panic!("BoxRef[1] should carry IntBound, got {:?}", other),
        }
        // old's slot now points to new (forwarding chain head).
        match &*b0.get_forwarded() {
            BoxForwarded::Box(target) => assert_eq!(target, &b1),
            other => panic!("expected b0 to forward to b1, got {:?}", other),
        }
    }

    /// `optimizer.py:400` guard: transfer is **skipped** when `new` is
    /// constant. PyPy short-circuits via `not newop.is_constant()`.
    #[test]
    fn h3_1_replace_op_skips_info_transfer_when_new_is_constant() {
        let (mut ctx, b0, _b1) = ctx_with_two_int_boxes();
        // Seed an IntBound on old.
        let bound = IntBound::from_constant(42);
        ctx.setintbound(OpRef::int_op(0), &bound);
        // Forward to a Const target.
        let const_opref = OpRef::const_int(0);
        ctx.const_pool
            .insert(const_opref.const_index(), Value::Int(42));
        ctx.replace_op(OpRef::int_op(0), const_opref);
        // The IntBound on old is gone (overwritten by Box(const_box)).
        // Const targets do not carry transferred info — PyPy skips this case.
        match &*b0.get_forwarded() {
            BoxForwarded::Box(target) => assert!(target.is_constant()),
            other => panic!("expected b0 to forward to const_box, got {:?}", other),
        }
    }

    /// `set_ptr_info(opref, info)` mirrors `box.set_forwarded(PtrInfo)`.
    #[test]
    fn h3_1_set_ptr_info_mirrors_box_info() {
        // PtrInfo applies to ref-typed boxes.
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let b = BoxRef::new_inputarg(Type::Ref, Some(0));
        ctx.box_pool = vec![b.clone()].into();
        let info = PtrInfo::NonNull { last_guard_pos: -1 };
        ctx.set_ptr_info(OpRef::int_op(0), info);
        match &*b.get_forwarded() {
            BoxForwarded::Info(OpInfo::Ptr(PtrInfo::NonNull { .. })) => {}
            other => panic!("expected Info(Ptr(NonNull)), got {:?}", other),
        }
    }

    /// PyPy optimizer.py:432 parity: after
    /// `make_constant(opref, Value::Ref(_))` writes `Forwarded::Box(constbox)`
    /// on the InputArg's BoxRef, a subsequent `make_nonnull(opref)` MUST NOT
    /// overwrite the Const slot with `OpInfo::Ptr(NonNull)`.
    #[test]
    fn audit_a_make_nonnull_preserves_box_constant_slot() {
        use majit_ir::GcRef;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let b = BoxRef::new_inputarg(Type::Ref, Some(0));
        ctx.box_pool = vec![b.clone()].into();
        let opref = OpRef::input_arg_typed(0, Type::Ref);
        ctx.make_constant(opref, Value::Ref(GcRef(0xdead_beef)));
        match &*b.get_forwarded() {
            BoxForwarded::Box(target) => {
                assert_eq!(target.const_value(), Some(Value::Ref(GcRef(0xdead_beef))));
            }
            other => panic!("expected Box(ConstRef) post make_constant, got {:?}", other),
        }
        ctx.make_nonnull(opref);
        match &*b.get_forwarded() {
            BoxForwarded::Box(target) => {
                assert_eq!(
                    target.const_value(),
                    Some(Value::Ref(GcRef(0xdead_beef))),
                    "make_nonnull must not overwrite the Const Box slot"
                );
            }
            other => panic!("make_nonnull clobbered Const Box slot — got {:?}", other),
        }
    }

    /// `resolved_is_constant_via_box` accepts all three Const shapes:
    /// (a) Const-namespace OpRef, (b) `Forwarded::Box(constbox)` chain,
    /// (c) legacy `Forwarded::Info(OpInfo::Constant)`. Used by
    /// `make_nonnull` / `make_nonnull_str` write-side gates.
    #[test]
    fn audit_a_resolved_is_constant_via_box_recognizes_three_shapes() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let b0 = BoxRef::new_inputarg(Type::Int, Some(0));
        let b1 = BoxRef::new_inputarg(Type::Int, Some(1));
        ctx.box_pool = vec![b0, b1.clone()].into();
        // (a) Const-namespace OpRef.
        let const_opref = OpRef::const_int(0);
        ctx.const_pool
            .insert(const_opref.const_index(), Value::Int(7));
        assert!(ctx.resolved_is_constant_via_box(const_opref));
        // (b) `Forwarded::Box(constbox)` chain on a non-Const-namespace OpRef.
        ctx.replace_op(OpRef::from_raw(0), const_opref);
        assert!(ctx.resolved_is_constant_via_box(OpRef::from_raw(0)));
        // (c) legacy `Forwarded::Info(OpInfo::Constant)` shape. Readers keep
        // accepting it for existing tests and bridge-state fixtures, while
        // make_constant itself now writes PyPy's Forwarded::Box(constbox).
        b1.set_forwarded_info(OpInfo::Constant(Value::Int(42)));
        assert!(ctx.resolved_is_constant_via_box(OpRef::from_raw(1)));
        // Negative case: OpRef with no constant forwarding.
        let mut ctx2 = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let nb = BoxRef::new_inputarg(Type::Int, Some(0));
        ctx2.box_pool = vec![nb].into();
        assert!(!ctx2.resolved_is_constant_via_box(OpRef::from_raw(0)));
    }

    /// `make_constant` mirrors PyPy optimizer.py:432
    /// `box.set_forwarded(constbox)` as `Forwarded::Box(Const)`.
    #[test]
    fn h3_1_make_constant_mirrors_box_info_constant() {
        let (mut ctx, b0, _b1) = ctx_with_two_int_boxes();
        ctx.make_constant(OpRef::int_op(0), Value::Int(42));
        match &*b0.get_forwarded() {
            BoxForwarded::Box(target) => {
                assert_eq!(target.const_value(), Some(Value::Int(42)));
            }
            other => panic!("expected Forwarded::Box(ConstInt 42), got {:?}", other),
        }
    }

    /// `setintbound(opref, bound)` mirrors `box.set_forwarded(IntBound)`.
    #[test]
    fn h3_1_setintbound_mirrors_box_info() {
        let (mut ctx, b0, _b1) = ctx_with_two_int_boxes();
        let bound = IntBound::from_constant(7);
        ctx.setintbound(OpRef::int_op(0), &bound);
        match &*b0.get_forwarded() {
            BoxForwarded::Info(OpInfo::IntBound(b)) => {
                assert_eq!(b.lower, 7);
                assert_eq!(b.upper, 7);
            }
            other => panic!("expected Info(IntBound), got {:?}", other),
        }
    }

    /// `replace_op(old, ConstX)` mirrors onto `old_box.set_forwarded_box(
    /// fresh_const_box)`. Per RPython parity (`optimizer.py:393`,
    /// `history.py:220` ConstInt construction), the const target is built
    /// fresh from `const_pool[const_index]` per call site — no dedup, value
    /// equality via `same_constant`. The mirror must produce a Const-kind
    /// BoxRef carrying the same Value as the seeded constant.
    #[test]
    fn h3_4_replace_op_const_target_mirrors_value_box() {
        let (mut ctx, b0, _b1) = ctx_with_two_int_boxes();
        let const_opref = OpRef::const_int(0);
        ctx.const_pool
            .insert(const_opref.const_index(), Value::Int(42));
        ctx.replace_op(OpRef::int_op(0), const_opref);
        match &*b0.get_forwarded() {
            BoxForwarded::Box(target) => {
                assert!(target.is_constant());
                assert_eq!(target.const_value(), Some(Value::Int(42)));
            }
            other => panic!("expected Forwarded::Box(Const), got {:?}", other),
        }
    }

    /// resoperation.py:58 get_box_replacement(not_const=True) stops before
    /// stepping into a Const target. This is required for guard fail args:
    /// resume numbering encodes constants as TAGCONST, while backend liveboxes
    /// keep the runtime Box identity.
    #[test]
    fn get_box_replacement_not_const_stops_before_const_target() {
        let (mut ctx, _b0, _b1) = ctx_with_two_int_boxes();
        let const_opref = OpRef::const_int(0);
        ctx.const_pool
            .insert(const_opref.const_index(), Value::Int(42));
        ctx.replace_op(OpRef::int_op(0), const_opref);

        assert_eq!(ctx.get_box_replacement(OpRef::int_op(0)), const_opref);
        assert_eq!(
            ctx.get_box_replacement_not_const(OpRef::int_op(0)),
            OpRef::int_op(0)
        );

        ctx.replace_op(OpRef::int_op(1), OpRef::int_op(0));
        assert_eq!(ctx.get_box_replacement(OpRef::int_op(1)), const_opref);
        assert_eq!(
            ctx.get_box_replacement_not_const(OpRef::int_op(1)),
            OpRef::input_arg_typed(0, Type::Int)
        );
    }

    /// When `const_pool` lacks a Value for the const target, BoxRef
    /// `Forwarded::Box(BoxRef::new_const_with_index(value, idx))` cannot be
    /// constructed — the value would have to be invented. PyPy parity
    /// (`history.py:220` ConstInt(value)) has no notion of a "Const
    /// without a Value", so `replace_op` panics instead of silently
    /// skipping; the caller is broken if it produces a Const OpRef without
    /// seeding the pool.
    #[test]
    #[should_panic(expected = "missing from const_pool")]
    fn h3_4_replace_op_const_target_without_const_pool_panics() {
        let (mut ctx, _b0, _b1) = ctx_with_two_int_boxes();
        // const_pool is empty — bug in caller.
        let const_opref = OpRef::const_int(0);
        ctx.replace_op(OpRef::int_op(0), const_opref);
    }

    /// H-3.4 slice 77b follow-up: Phase 2's `box_pool` carries placeholder
    /// `BoxRef::new_resop(Type::Void)` at indices `[0..phase2_inputarg_base)`
    /// (the Phase 1 emit-position region; Phase 1 emit ops do NOT appear in
    /// Phase 2's trace iteration, so Phase 2's iter has no `cls()` allocation
    /// for them). Replicates the import_state pattern at unroll.rs:3105:
    ///
    ///   1. `replace_op(source_p2, target_p1)` writes
    ///      `source._forwarded = Box(placeholder_at_target_p1.raw)`.
    ///   2. Phase 2 imports info via `set_ptr_info(target_p1, info)` writes
    ///      `placeholder._forwarded = Info(info)`.
    ///   3. Reading source via `peek_ptr_info_via_box` walks
    ///      `source → placeholder` and sees the placeholder's info.
    ///
    /// PyPy parity is preserved structurally even though Phase 1's actual
    /// Box is not shared across phases (per the H-3.4 first-77b aliasing
    /// fix): the placeholder absorbs Phase 2's import writes the same way
    /// Phase 1's Box would in PyPy.
    #[test]
    fn h3_4_phase2_placeholder_forwarding_yields_consistent_reads() {
        // Layout: indices 0..2 are Phase 1 emit-position placeholders,
        // indices 2..4 are Phase 2 inputarg BoxRefs. PyPy `box.type`
        // invariant prevents `replace_op(Ref, Void)` (cross-type forward),
        // so place Ref-typed boxes on both sides — the test models a
        // Phase 1 RefOp result acting as the import target.
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 4, 0, 4);
        let placeholder_target = crate::r#box::BoxRef::new_resop(majit_ir::Type::Ref, 0);
        let placeholder_other = crate::r#box::BoxRef::new_resop(majit_ir::Type::Ref, 1);
        let source_box = crate::r#box::BoxRef::new_inputarg(majit_ir::Type::Ref, Some(2));
        let other_box = crate::r#box::BoxRef::new_inputarg(majit_ir::Type::Ref, Some(3));
        ctx.box_pool = vec![
            placeholder_target.clone(),
            placeholder_other,
            source_box.clone(),
            other_box,
        ]
        .into();

        // BoxRef-first chain walker reconstructs the variant tag from
        // `box.type_()`; placeholders and source are both Ref, so use the
        // typed factories that match.
        let target_p1 = OpRef::ref_op(0);
        let source_p2 = OpRef::input_arg_ref(2);

        // Step 1: import_state's `source.set_forwarded(target)` equivalent.
        ctx.replace_op(source_p2, target_p1);

        // Step 2: setinfo_from_preamble's terminal write.
        // `setinfo_from_preamble(source, info)` first walks the chain via
        // `get_box_replacement` (mod.rs:2538) which returns `target_p1`,
        // then calls `set_ptr_info(target_p1, info)`. Replicate the
        // post-walk write directly.
        let info = PtrInfo::NonNull { last_guard_pos: -1 };
        ctx.set_ptr_info(target_p1, info.clone());

        // Read via BoxRef-routing path: walk source's chain to placeholder.
        let via_box = ctx
            .peek_ptr_info_via_box(source_p2)
            .expect("BoxRef path must see info");
        assert!(matches!(via_box, PtrInfo::NonNull { .. }));

        // Chain walk lands on target_p1.
        let resolved = ctx.get_box_replacement(source_p2);
        assert_eq!(resolved, target_p1);

        // Placeholder Box absorbed the mirror write, so its _forwarded now
        // carries the info — equivalent to PyPy's Phase 1 Box receiving
        // setinfo_from_preamble.
        match &*placeholder_target.get_forwarded() {
            BoxForwarded::Info(OpInfo::Ptr(PtrInfo::NonNull { .. })) => {}
            other => panic!(
                "placeholder must carry Info(NonNull) after set_ptr_info, got {:?}",
                other
            ),
        }
    }

    /// H-3.4 slice 77b follow-up: complementary to
    /// `h3_4_phase2_placeholder_forwarding_yields_consistent_reads`. Pre-import
    /// (no `setinfo_from_preamble` call), reading `target_p1` info via either
    /// path returns None — consistent within pyre. PyPy parity here depends on
    /// `ExportedState.exported_infos` (`unroll.py:529` canonical field)
    /// carrying every Phase 1 op info Phase 2 needs; the placeholder cannot
    /// fabricate Phase 1 info that wasn't exported. PyPy itself uses the same
    /// serialization map for the import (PyPy's Phase 2 reads exported_infos
    /// → setinfo_from_preamble too), so structural narrowness here matches
    /// PyPy's own dispatch.
    #[test]
    fn h3_4_phase2_placeholder_without_import_returns_none_consistently() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 4, 0, 4);
        // Same Ref-typed alignment as the sibling test: forwarding a Ref
        // source to the placeholder requires placeholder type to match
        // (PyPy `box.type` invariant; `replace_op` cross-type assertion).
        let placeholder_target = crate::r#box::BoxRef::new_resop(majit_ir::Type::Ref, 0);
        let placeholder_other = crate::r#box::BoxRef::new_resop(majit_ir::Type::Ref, 1);
        let source_box = crate::r#box::BoxRef::new_inputarg(majit_ir::Type::Ref, Some(2));
        let other_box = crate::r#box::BoxRef::new_inputarg(majit_ir::Type::Ref, Some(3));
        ctx.box_pool = vec![
            placeholder_target.clone(),
            placeholder_other,
            source_box,
            other_box,
        ]
        .into();

        let target_p1 = OpRef::ref_op(0);
        let source_p2 = OpRef::input_arg_ref(2);

        // import_state's replace_op fires, but Phase 2 chose NOT to import
        // info (e.g. exported_infos didn't carry an entry for target_p1).
        ctx.replace_op(source_p2, target_p1);

        // BoxRef-routing reader: chain walks source → placeholder → None.
        assert!(ctx.peek_ptr_info_via_box(source_p2).is_none());

        // Legacy Vec reader: chain walks source → target_p1 → None
        // (Phase 2's fresh Vec has no entry for target_p1).
        let resolved = ctx.get_box_replacement(source_p2);
        assert_eq!(resolved, target_p1);

        // Placeholder Box was not mutated (no info import fired) — still None.
        assert!(matches!(
            *placeholder_target.get_forwarded(),
            BoxForwarded::None
        ));
    }

    /// `seed_constant` populates `const_pool` (Value map) for
    /// `is_constant()` OpRefs. Const BoxRefs are NOT cached —
    /// `get_box_replacement_box` allocates a fresh `BoxRef::new_const(value)`
    /// per call (RPython `history.py:220` per-call-site `ConstInt(value)`
    /// parity).
    #[test]
    fn h3_4_seed_constant_populates_const_pool_only() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 0, 0, 0);
        let const_opref = OpRef::const_int(7);
        ctx.seed_constant(const_opref, Value::Int(123));

        assert_eq!(
            ctx.const_pool.get(&const_opref.const_index()),
            Some(&Value::Int(123))
        );

        // Subsequent reader (e.g. `get_box_replacement_box`) materializes
        // a Const BoxRef on demand — value-equivalent across calls but no
        // identity dedup.
        let b1 = ctx.get_box_replacement_box(const_opref).unwrap();
        let b2 = ctx.get_box_replacement_box(const_opref).unwrap();
        assert!(b1.is_constant());
        assert_eq!(b1.const_value(), Some(Value::Int(123)));
        assert_eq!(b2.const_value(), b1.const_value());
    }

    /// H-3.2b: with a populated `box_pool` and no forwarding, the
    /// BoxRef-returning reader returns the pool entry unchanged.
    /// `resoperation.py:57-68` walker terminates on `None` immediately.
    #[test]
    fn h3_2b_get_box_replacement_box_returns_pool_entry_when_no_forward() {
        let (ctx, b0, _b1) = ctx_with_two_int_boxes();
        let got = ctx
            .get_box_replacement_box(OpRef::int_op(0))
            .expect("pool entry exists");
        // Pointer identity: same `Rc` allocation as `b0`.
        assert_eq!(got, b0);
    }

    /// H-3.2b: with a forwarding chain installed via `replace_op`, the
    /// BoxRef walker reaches the terminal Box (`b1`). RPython parity:
    /// `optimizer.py:393 box.set_forwarded(newop)` → reader walks until
    /// `Forwarded::None` and returns the last Box.
    #[test]
    fn h3_2b_get_box_replacement_box_walks_forwarded_chain() {
        let (mut ctx, b0, b1) = ctx_with_two_int_boxes();
        ctx.replace_op(OpRef::int_op(0), OpRef::int_op(1));
        let got = ctx
            .get_box_replacement_box(OpRef::int_op(0))
            .expect("pool entry exists");
        // b0 → b1 through the BoxRef `_forwarded` slot; terminal is b1.
        assert_eq!(got, b1);
        // b0 itself is not the terminal.
        assert_ne!(got, b0);
    }

    /// H-3.2b: empty `box_pool` (test/retrace baseline) makes the
    /// BoxRef-returning reader return `None`; the OpRef-returning walker
    /// cannot resolve a Box identity without a pool entry either.
    #[test]
    fn h3_2b_get_box_replacement_box_returns_none_when_pool_empty() {
        let ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        // box_pool empty + const_pool has no entry for the const OpRef.
        assert!(ctx.get_box_replacement_box(OpRef::int_op(0)).is_none());
        assert!(ctx.get_box_replacement_box(OpRef::const_int(0)).is_none());
    }

    /// H-3.2b: `OpRef::NONE` sentinel returns `None` — the BoxRef reader
    /// has no Box to root the walk on. The OpRef-returning reader handles
    /// the sentinel independently by returning it unchanged.
    #[test]
    fn h3_2b_get_box_replacement_box_handles_none_sentinel() {
        let (ctx, _b0, _b1) = ctx_with_two_int_boxes();
        assert!(ctx.get_box_replacement_box(OpRef::NONE).is_none());
    }

    /// `get_box_replacement_box` for a const OpRef returns a fresh
    /// `BoxRef::new_const(value)` materialized from `const_pool` —
    /// per-call-site allocation matches RPython `history.py:220` ConstInt
    /// construction. Identity is irrelevant; readers compare via Value.
    #[test]
    fn h3_4_get_box_replacement_box_materializes_const_from_const_pool() {
        let (mut ctx, _b0, _b1) = ctx_with_two_int_boxes();
        let const_opref = OpRef::const_int(0);
        ctx.const_pool
            .insert(const_opref.const_index(), Value::Int(42));
        let got = ctx
            .get_box_replacement_box(const_opref)
            .expect("const_pool entry exists");
        assert!(got.is_constant());
        assert_eq!(got.const_value(), Some(Value::Int(42)));
    }

    /// H-3.2b: when the chain terminates at `Forwarded::Info(_)`, the
    /// walker returns the Box that holds the Info — `box.rs::BoxRef::
    /// get_box_replacement` stops before descending into Info, matching
    /// PyPy `resoperation.py:60 isinstance(next, AbstractInfo)`.
    #[test]
    fn h3_2b_get_box_replacement_box_stops_at_info_terminal() {
        let (mut ctx, b0, _b1) = ctx_with_two_int_boxes();
        ctx.setintbound(OpRef::int_op(0), &IntBound::from_constant(7));
        let got = ctx
            .get_box_replacement_box(OpRef::int_op(0))
            .expect("pool entry exists");
        // Walker terminates at b0 (its `_forwarded` is Info, not Box).
        assert_eq!(got, b0);
    }

    // BoxRef-routing helpers `is_virtual_via_box` / `is_nonnull_via_box`
    // read the same `_forwarded` slot that PyPy's getptrinfo() inspects.

    fn ctx_with_one_ref_box() -> (OptContext, BoxRef) {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let b = BoxRef::new_inputarg(Type::Ref, Some(0));
        ctx.box_pool = vec![b.clone()].into();
        (ctx, b)
    }

    #[derive(Debug)]
    struct DummySizeDescr;
    impl majit_ir::Descr for DummySizeDescr {}

    #[test]
    fn h3_2c_is_virtual_via_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        let info = PtrInfo::Virtual(crate::optimizeopt::info::VirtualInfo {
            descr: std::sync::Arc::new(DummySizeDescr),
            known_class: None,
            ob_type_descr: None,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        });
        ctx.set_ptr_info(OpRef::from_raw(0), info);
        // Legacy and BoxRef-routing path agree.
        assert!(
            ctx.peek_ptr_info_via_box(OpRef::from_raw(0))
                .is_some_and(|i| i.is_virtual())
        );
        assert!(ctx.is_virtual_via_box(OpRef::from_raw(0)));
    }

    #[test]
    fn h3_2c_is_virtual_via_box_returns_false_for_nonnull_only() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(!ctx.is_virtual_via_box(OpRef::from_raw(0)));
    }

    #[test]
    fn h3_2c_is_virtual_via_box_returns_false_for_unset() {
        let (ctx, _b) = ctx_with_one_ref_box();
        assert!(!ctx.is_virtual_via_box(OpRef::from_raw(0)));
    }

    #[test]
    fn h3_2c_is_nonnull_via_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(
            ctx.peek_ptr_info_via_box(OpRef::from_raw(0))
                .is_some_and(|i| i.is_nonnull())
        );
        assert!(ctx.is_nonnull_via_box(OpRef::from_raw(0)));
    }

    #[test]
    fn h3_2c_is_nonnull_via_box_returns_false_for_unset() {
        let (ctx, _b) = ctx_with_one_ref_box();
        assert!(!ctx.is_nonnull_via_box(OpRef::from_raw(0)));
    }

    #[test]
    fn h3_2c_peek_intbound_via_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, _b0, _b1) = ctx_with_two_int_boxes();
        ctx.setintbound(OpRef::from_raw(0), &IntBound::from_constant(42));
        let legacy = ctx.peek_intbound(OpRef::from_raw(0)).expect("legacy bound");
        let via_box = ctx
            .peek_intbound_via_box(OpRef::from_raw(0))
            .expect("box bound");
        assert!(legacy.is_constant());
        assert_eq!(legacy.get_constant_int(), 42);
        assert!(via_box.is_constant());
        assert_eq!(via_box.get_constant_int(), 42);
    }

    #[test]
    fn h3_2c_peek_intbound_via_box_returns_none_for_unset() {
        let (ctx, _b0, _b1) = ctx_with_two_int_boxes();
        assert!(ctx.peek_intbound_via_box(OpRef::from_raw(0)).is_none());
    }

    #[test]
    fn h3_2c_last_guard_pos_via_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: 5 });
        assert_eq!(ctx.last_guard_pos_via_box(OpRef::from_raw(0)), Some(5));
        // legacy path agrees.
        assert_eq!(
            ctx.peek_ptr_info_via_box(OpRef::from_raw(0))
                .and_then(|i| i.get_last_guard_pos()),
            Some(5)
        );
    }

    #[test]
    fn h3_2c_last_guard_pos_via_box_returns_none_for_unset() {
        let (ctx, _b) = ctx_with_one_ref_box();
        assert!(ctx.last_guard_pos_via_box(OpRef::from_raw(0)).is_none());
    }

    #[test]
    fn h3_2c_last_guard_pos_via_box_returns_none_when_no_recorded_guard() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        // info.py:91 last_guard_pos == -1 → get_last_guard_pos returns None.
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(ctx.last_guard_pos_via_box(OpRef::from_raw(0)).is_none());
    }

    #[test]
    fn h3_2c_is_virtualizable_via_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            OpRef::from_raw(0),
            PtrInfo::Virtualizable(crate::optimizeopt::info::VirtualizableFieldState {
                fields: Vec::new(),
                field_descrs: Vec::new(),
                arrays: Vec::new(),
                last_guard_pos: -1,
            }),
        );
        assert!(ctx.is_virtualizable_via_box(OpRef::from_raw(0)));
        assert!(matches!(
            ctx.peek_ptr_info_via_box(OpRef::from_raw(0)),
            Some(PtrInfo::Virtualizable(_))
        ));
    }

    #[test]
    fn h3_2c_is_virtualizable_via_box_returns_false_for_nonnull_only() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(!ctx.is_virtualizable_via_box(OpRef::from_raw(0)));
    }

    #[test]
    fn h3_2c_is_virtualizable_via_box_returns_false_for_unset() {
        let (ctx, _b) = ctx_with_one_ref_box();
        assert!(!ctx.is_virtualizable_via_box(OpRef::from_raw(0)));
    }

    #[test]
    fn h3_2c_has_ptr_info_via_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(ctx.has_ptr_info_via_box(OpRef::from_raw(0)));
        assert!(ctx.peek_ptr_info_via_box(OpRef::from_raw(0)).is_some());
    }

    #[test]
    fn h3_2c_has_ptr_info_via_box_returns_false_for_unset() {
        let (ctx, _b) = ctx_with_one_ref_box();
        assert!(!ctx.has_ptr_info_via_box(OpRef::from_raw(0)));
    }

    #[test]
    fn h3_2c_peek_ptr_info_via_box_returns_set_info_when_pool_plumbed() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: 5 });
        let via_box = ctx
            .peek_ptr_info_via_box(OpRef::from_raw(0))
            .expect("box clone");
        assert!(matches!(via_box, PtrInfo::NonNull { last_guard_pos: 5 }));
    }

    #[test]
    fn h3_2c_peek_ptr_info_via_box_returns_none_for_unset() {
        let (ctx, _b) = ctx_with_one_ref_box();
        assert!(ctx.peek_ptr_info_via_box(OpRef::from_raw(0)).is_none());
    }

    // H-3.2c slice 29: `with_ptr_info_mut(opref, |info| ...)` runs a
    // closure against the legacy `&mut PtrInfo` and auto-mirrors the
    // post-closure slot onto `box_pool[idx].forwarded` so subsequent
    // BoxRef-routing readers (`peek_ptr_info_via_box`,
    // `last_guard_pos_via_box`) see the mutation. This closes the
    // mirror gap codex round-5 audit identified for inline
    // `get_ptr_info_mut` consumers.

    #[test]
    fn h3_2c_with_ptr_info_mut_mirrors_after_mutation_when_pool_plumbed() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: 0 });
        // Pre-condition: BoxRef snapshot matches legacy at pos 0.
        assert_eq!(ctx.last_guard_pos_via_box(OpRef::from_raw(0)), Some(0));
        // Mutate inner state via closure.
        let returned = ctx
            .with_ptr_info_mut(OpRef::from_raw(0), |info| {
                info.set_last_guard_pos(42);
                "ok"
            })
            .expect("closure runs");
        assert_eq!(returned, "ok");
        // Post-condition: BoxRef snapshot reflects mutation (mirror ran).
        assert_eq!(ctx.last_guard_pos_via_box(OpRef::from_raw(0)), Some(42));
        // Legacy slot also reflects mutation (closure ran on &mut PtrInfo).
        assert_eq!(
            ctx.peek_ptr_info_via_box(OpRef::from_raw(0))
                .and_then(|i| i.get_last_guard_pos()),
            Some(42)
        );
    }

    #[test]
    fn h3_2c_with_ptr_info_mut_returns_none_for_unset() {
        let (mut ctx, _b) = ctx_with_one_ref_box();
        // No PtrInfo installed at OpRef(0).
        let invoked = std::cell::Cell::new(false);
        let result = ctx.with_ptr_info_mut(OpRef::from_raw(0), |_info| {
            invoked.set(true);
        });
        assert!(result.is_none());
        assert!(!invoked.get(), "closure must not run when info is absent");
    }
}

#[cfg(test)]
mod constant_ptr_info_tests {
    //! info.py:706-758 + 865-894 ConstPtrInfo / getptrinfo / getrawptrinfo
    //! parity tests for the typed-Int constant override path. RPython
    //! treats `ConstInt` (raw pointer) and `ConstPtr` uniformly via
    //! `_const.getref_base()`; majit must do the same regardless of how
    //! the constant pool stored the bits (`Value::Ref` vs `Value::Int`
    //! with a `Type::Ref` override).
    use super::*;
    use crate::optimizeopt::info::{
        PtrInfo, VStringVariant, VirtualRawBufferInfo, VirtualRawSliceInfo,
    };
    use majit_ir::{GcRef, OpRef, Type, Value};
    use std::borrow::Cow;

    /// info.py:880-894 getptrinfo(ConstPtr) → ConstPtrInfo(op).
    /// A `Value::Ref` constant must be wrapped in `PtrInfo::Constant`.
    #[test]
    fn getptrinfo_returns_constant_for_value_ref() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(10_000);
        ctx.seed_constant(opref, Value::Ref(GcRef(0xdead_beef)));
        match ctx.getptrinfo(opref) {
            Some(Cow::Owned(PtrInfo::Constant(g))) => assert_eq!(g.0, 0xdead_beef),
            other => panic!("expected ConstPtrInfo(0xdeadbeef), got {other:?}"),
        }
    }

    /// info.py:865-878 getrawptrinfo(ConstInt) → ConstPtrInfo(op).
    /// A `Value::Int` whose `constant_types_for_numbering` entry is
    /// `Type::Ref` (the typed-override path used for static class
    /// pointers) must also produce `PtrInfo::Constant`.
    #[test]
    fn getptrinfo_returns_constant_for_int_with_ref_override() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(10_001);
        ctx.seed_constant(opref, Value::Int(0x1234_5678));
        ctx.constant_types_for_numbering
            .insert(opref.raw(), Type::Ref);
        match ctx.getptrinfo(opref) {
            Some(Cow::Owned(PtrInfo::Constant(g))) => assert_eq!(g.0, 0x1234_5678),
            other => panic!("expected ConstPtrInfo(0x12345678), got {other:?}"),
        }
    }

    /// info.py:881-882 getptrinfo(ConstInt) without a Ref-typed box →
    /// `getrawptrinfo` returns None for non-raw-pointer constants. majit
    /// honors that by leaving plain `Value::Int` constants without an
    /// override unwrapped.
    #[test]
    fn getptrinfo_skips_int_without_ref_override() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(10_002);
        ctx.seed_constant(opref, Value::Int(42));
        // No constant_types_for_numbering entry → no Ref override.
        assert!(ctx.getptrinfo(opref).is_none());
    }

    /// info.py:865-878 getrawptrinfo(ConstInt) parity: a raw-pointer
    /// `ConstInt` (here represented as a `Value::Int` constant tagged
    /// with `Type::Ref`) becomes `ConstPtrInfo(op)` even when the
    /// pointer bits are NULL. Downstream callers null-check via
    /// `is_null`/`getref_base`.
    ///
    /// The plain `Value::Int(0)` shape (no Type::Ref override) is a
    /// regular integer constant and is covered by
    /// `getptrinfo_skips_int_without_ref_override` — it must NOT be
    /// promoted to `ConstPtrInfo`, matching RPython's `op.type == 'i'`
    /// raw-pointer-only routing.
    #[test]
    fn getptrinfo_null_int_constant_with_ref_override_is_constant_null() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(10_003);
        ctx.seed_constant(opref, Value::Int(0));
        ctx.constant_types_for_numbering
            .insert(opref.raw(), Type::Ref);
        match ctx.getptrinfo(opref) {
            Some(Cow::Owned(PtrInfo::Constant(g))) => assert_eq!(g.0, 0),
            other => panic!("expected ConstPtrInfo(NULL), got {other:?}"),
        }
    }

    /// info.py:881-882 getptrinfo(ConstInt(0)) without a Ref-typed box →
    /// `getrawptrinfo` is the only path into `ConstPtrInfo` for integer
    /// constants, and it is only called for raw-pointer Boxes (`op.type
    /// == 'i'` annotated as a pointer). A plain integer `ConstInt(0)`
    /// must therefore NOT become `ConstPtrInfo(NULL)` — otherwise an
    /// integer counter at zero would erroneously trigger `is_null`
    /// optimizations on integer slots.
    #[test]
    fn getptrinfo_int_zero_without_ref_override_is_none() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(10_009);
        ctx.seed_constant(opref, Value::Int(0));
        // No constant_types_for_numbering entry → no Ref override.
        assert!(ctx.getptrinfo(opref).is_none());
    }

    /// info.py:718-726 ConstPtrInfo._get_info(descr, optheap) parity:
    /// the same constant must always resolve to the same shared
    /// `const_infos[ref]` slot. Calling `get_const_info_mut` twice on a
    /// `Value::Ref` constant returns identical info — and a mutation
    /// observed via the second call confirms the slot identity.
    #[test]
    fn const_info_mut_returns_same_slot_for_value_ref() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(10_004);
        ctx.seed_constant(opref, Value::Ref(GcRef(0xa5a5_a5a5)));
        // First lookup: install Instance via the Vacant entry path,
        // then mark a known class so the second lookup observes it.
        {
            let info = ctx
                .get_const_info_mut(opref, None)
                .expect("Ref constant should have const_infos slot");
            *info = PtrInfo::known_class(GcRef(0x1111_2222), true);
        }
        // Second lookup: the slot must contain the previously written
        // PtrInfo, not a freshly minted Instance.
        let info = ctx
            .get_const_info_mut(opref, None)
            .expect("Ref constant should still have const_infos slot");
        match info {
            PtrInfo::Instance(iinfo) => {
                assert_eq!(iinfo.known_class.map(|c| c.0), Some(0x1111_2222));
            }
            other => panic!("expected Instance(known_class=Some) after re-lookup, got {other:?}"),
        }
    }

    /// info.py:718-726 ConstPtrInfo._get_info parity for the typed-Int
    /// override path: a `Value::Int` constant tagged as `Type::Ref` must
    /// share its `const_infos` slot with any other reference to the same
    /// address (whether that other reference came in as `Value::Ref` or
    /// as another tagged Int). Without this, two `getfield` paths on the
    /// same vtable address would maintain disjoint heap caches.
    #[test]
    fn const_info_mut_shares_slot_between_ref_and_tagged_int() {
        let mut ctx = OptContext::new(0);
        let ref_op = OpRef::from_raw(10_005);
        let int_op = OpRef::from_raw(10_006);
        let addr: usize = 0xfeed_face;
        ctx.seed_constant(ref_op, Value::Ref(GcRef(addr)));
        ctx.seed_constant(int_op, Value::Int(addr as i64));
        ctx.constant_types_for_numbering
            .insert(int_op.raw(), Type::Ref);

        // Mutate via the Ref-typed constant.
        {
            let info = ctx
                .get_const_info_mut(ref_op, None)
                .expect("Ref constant should resolve");
            *info = PtrInfo::known_class(GcRef(0xc0de_cafe), true);
        }
        // Read back via the typed-Int alias — must observe the same
        // PtrInfo because both keys hash to the same const_infos entry.
        let info = ctx
            .get_const_info_mut(int_op, None)
            .expect("typed-Int alias should resolve to the same slot");
        match info {
            PtrInfo::Instance(iinfo) => {
                assert_eq!(iinfo.known_class.map(|c| c.0), Some(0xc0de_cafe));
            }
            other => panic!("expected shared Instance(known_class=Some), got {other:?}"),
        }
    }

    /// info.py:719-720 `if not ref: raise InvalidLoop` — null protection.
    /// `get_const_info_mut` raises `InvalidLoop` (via `panic_any`) when
    /// the constant pointer resolves to a null `gcref`. Callers in PyPy
    /// rely on the exception to abort the impossible trace shape so the
    /// JIT can retry; the Rust port mirrors that contract.
    ///
    /// `panic_any(InvalidLoop)` is not a string panic so we use
    /// `catch_unwind` + downcast to assert the typed payload, matching
    /// how other optimizer passes catch the same exception.
    #[test]
    fn const_info_mut_raises_on_null_value_ref_constant() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ctx = OptContext::new(0);
            let ref_null = OpRef::from_raw(10_007);
            ctx.seed_constant(ref_null, Value::Ref(GcRef(0)));
            let _ = ctx.get_const_info_mut(ref_null, None);
        }));
        let err = result.expect_err("expected InvalidLoop panic");
        let invalid = err
            .downcast_ref::<crate::optimize::InvalidLoop>()
            .expect("expected InvalidLoop payload");
        assert!(invalid.0.contains("null constant base pointer"));
    }

    /// Same `if not ref: raise InvalidLoop` path for the typed-Int alias
    /// — `Value::Int(0)` tagged as `Type::Ref` is the raw-pointer
    /// representation of NULL and must trip the same protection.
    #[test]
    fn const_info_mut_raises_on_null_typed_int_constant() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ctx = OptContext::new(0);
            let int_null = OpRef::from_raw(10_008);
            ctx.seed_constant(int_null, Value::Int(0));
            ctx.constant_types_for_numbering
                .insert(int_null.raw(), Type::Ref);
            let _ = ctx.get_const_info_mut(int_null, None);
        }));
        let err = result.expect_err("expected InvalidLoop panic");
        let invalid = err
            .downcast_ref::<crate::optimize::InvalidLoop>()
            .expect("expected InvalidLoop payload");
        assert!(invalid.0.contains("null constant base pointer"));
    }

    /// Plain `Value::Int(0)` (no `Type::Ref` override) is not a constant
    /// pointer at all — `getrawptrinfo` returns `None` long before the
    /// null protection runs, matching the integer-counter case where
    /// the value just happens to be zero.
    #[test]
    fn const_info_mut_returns_none_for_plain_int_zero() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(10_010);
        ctx.seed_constant(opref, Value::Int(0));
        // No constant_types_for_numbering entry → no Ref override.
        assert!(ctx.get_const_info_mut(opref, None).is_none());
    }

    /// optimizer.py:154-158 `is_raw_ptr(op)` parity for
    /// `info.RawSlicePtrInfo`: once a raw slice PtrInfo is present, it
    /// must be classified as an `AbstractRawPtrInfo` exactly like its
    /// parent raw buffer.
    #[test]
    fn is_raw_ptr_accepts_virtual_raw_slice() {
        let mut ctx = OptContext::new(0);
        let parent = OpRef::from_raw(10_010);
        let slice = OpRef::from_raw(10_011);

        ctx.set_ptr_info(
            parent,
            PtrInfo::VirtualRawBuffer(VirtualRawBufferInfo {
                func: 0,
                size: 32,
                offsets: Vec::new(),
                lengths: Vec::new(),
                descrs: Vec::new(),
                values: Vec::new(),
                last_guard_pos: -1,
                calldescr: None,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
        ctx.set_ptr_info(
            slice,
            PtrInfo::VirtualRawSlice(VirtualRawSliceInfo {
                offset: 8,
                parent,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        assert!(ctx.is_raw_ptr(parent));
        assert!(ctx.is_raw_ptr(slice));
    }

    /// vstring.py:50 `StrPtrInfo.__init__(mode, is_virtual=False, length=-1)`
    /// parity for non-virtual strings: `make_nonnull_str()` must install
    /// a base `StrPtrInfo`, not one of the virtual subclasses.
    #[test]
    fn make_nonnull_str_initializes_ptr_variant() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(10_012);

        ctx.make_nonnull_str(opref, 0);

        match ctx.peek_ptr_info_via_box(opref) {
            Some(PtrInfo::Str(sinfo)) => {
                assert_eq!(sinfo.mode, 0);
                assert_eq!(sinfo.length, -1);
                assert!(sinfo.lenbound.is_none());
                assert!(matches!(sinfo.variant, VStringVariant::Ptr));
            }
            other => panic!("expected base StrPtrInfo, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod ensure_ptr_info_arg0_tests {
    //! optimizer.py:461-499 `ensure_ptr_info_arg0` parity tests.
    //!
    //! Each test mirrors a single PyPy branch in `ensure_ptr_info_arg0`:
    //! the constant arg0 path, the AbstractVirtualPtrInfo early-return path,
    //! the NonNullPtrInfo upgrade path, and the assertion that fires on
    //! unexpected forwarded info shapes.
    use super::*;
    use crate::optimizeopt::info::{ArrayPtrInfo, EnsuredPtrInfo, PtrInfo};
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::{Descr, DescrRef, GcRef, Op, OpCode, OpRef, SizeDescr, Type, Value};
    use std::sync::Arc;

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

    fn struct_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0000,
            is_object: false,
        })
    }

    fn instance_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0001,
            is_object: true,
        })
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        index: u32,
        parent: DescrRef,
    }

    impl Descr for TestFieldDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
            Some(self)
        }
    }

    impl majit_ir::FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            0
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
        fn index_in_parent(&self) -> usize {
            0
        }
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(self.parent.clone())
        }
    }

    fn field_op_with_parent(parent: DescrRef) -> Op {
        let descr: DescrRef = Arc::new(TestFieldDescr { index: 0, parent });
        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[OpRef::from_raw(0)], descr);
        op.pos = OpRef::int_op(1);
        op
    }

    fn array_op() -> Op {
        let descr: DescrRef = Arc::new(TestSizeDescr {
            index: 7,
            is_object: false,
        });
        let mut op = Op::with_descr(OpCode::ArraylenGc, &[OpRef::from_raw(0)], descr);
        op.pos = OpRef::int_op(1);
        op
    }

    /// optimizer.py:465-466: `if arg0.is_constant(): return info.ConstPtrInfo(arg0)`
    /// Constant `Value::Ref` arg0 → `EnsuredPtrInfo::Constant(gcref)`.
    #[test]
    fn ensure_ptr_info_arg0_returns_constant_for_value_ref() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        ctx.seed_constant(OpRef::from_raw(0), Value::Ref(GcRef(0xdead_beef)));
        let op = field_op_with_parent(struct_parent_descr());
        let info = ctx.ensure_ptr_info_arg0(&op);
        match info {
            EnsuredPtrInfo::Constant { gcref, .. } => assert_eq!(gcref.0, 0xdead_beef),
            _ => panic!("expected EnsuredPtrInfo::Constant"),
        }
    }

    /// optimizer.py:465-466 parity for plain `Value::Int` constants — PyPy
    /// returns `info.ConstPtrInfo(arg0)` regardless of the box's exact type.
    /// majit's port mirrors that by returning `Constant(GcRef(bits))`; null
    /// or unsafe pointers are filtered downstream by `_get_info`'s null
    /// protection (info.py:719-720).
    #[test]
    fn ensure_ptr_info_arg0_returns_constant_for_value_int() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        ctx.seed_constant(OpRef::from_raw(0), Value::Int(1));
        let op = field_op_with_parent(struct_parent_descr());
        let info = ctx.ensure_ptr_info_arg0(&op);
        assert!(matches!(info, EnsuredPtrInfo::Constant { .. }));
    }

    /// info.py:796-822 `ConstPtrInfo.getlenbound(mode_string)` returns
    /// `IntBound.from_constant(length)` when `getstrlen1(mode)` knows the
    /// exact length. The Rust port consults the `string_length_resolver`
    /// hook the host runtime registered on `OptContext`.
    #[test]
    fn ensure_ptr_info_arg0_constant_string_returns_exact_length_via_resolver() {
        use std::sync::Arc;
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        ctx.seed_constant(OpRef::from_raw(0), Value::Ref(GcRef(0xC0FE)));
        // Resolver pretends every constant has byte-string length 5 in
        // mode_string and unicode length 7 in mode_unicode.
        ctx.string_length_resolver = Some(Arc::new(|gcref: GcRef, mode: u8| {
            assert_eq!(gcref.0, 0xC0FE);
            match mode {
                0 => Some(5),
                1 => Some(7),
                _ => None,
            }
        }));
        let op = {
            let descr: DescrRef = Arc::new(TestSizeDescr {
                index: 1,
                is_object: false,
            });
            let mut op = Op::with_descr(OpCode::Strlen, &[OpRef::from_raw(0)], descr);
            op.pos = OpRef::int_op(1);
            op
        };
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(Some(0))
            .expect("constant string length should resolve");
        assert_eq!(bound.lower, 5);
        assert_eq!(bound.upper, 5);
        let bound = info
            .getlenbound(Some(1))
            .expect("constant unicode length should resolve");
        assert_eq!(bound.lower, 7);
        assert_eq!(bound.upper, 7);
    }

    /// info.py:799-801 `if length < 0: return IntBound.nonnegative()` —
    /// no resolver registered → conservative nonnegative fallback.
    #[test]
    fn ensure_ptr_info_arg0_constant_string_falls_back_to_nonnegative_without_resolver() {
        use std::sync::Arc;
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        ctx.seed_constant(OpRef::from_raw(0), Value::Ref(GcRef(0x1234)));
        let op = {
            let descr: DescrRef = Arc::new(TestSizeDescr {
                index: 1,
                is_object: false,
            });
            let mut op = Op::with_descr(OpCode::Strlen, &[OpRef::from_raw(0)], descr);
            op.pos = OpRef::int_op(1);
            op
        };
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(Some(0))
            .expect("nonnegative fallback should be Some");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
        assert!(!bound.is_constant());
    }

    /// optimizer.py:475-484 GETFIELD branch with `parent_descr.is_object() == false`
    /// → `info.StructPtrInfo(parent_descr)`. The Rust port returns the
    /// freshly-installed `PtrInfo::Struct` via `Forwarded(&mut PtrInfo)`.
    #[test]
    fn ensure_ptr_info_arg0_constructs_struct_for_non_object_field() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        // Keep a strong reference to the parent alive for the duration
        // of the test: `SimpleFieldDescr::parent_descr` is a
        // `Weak<DescrRef>` (breaks the cycle between SizeDescr.all_fielddescrs
        // and FieldDescr.parent_descr), so the test must hold the parent
        // Arc until `get_parent_descr()` has been called.
        let _parent = struct_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let pi = info.as_mut().expect("Forwarded variant expected");
        assert!(matches!(&*pi, PtrInfo::Struct(_)));
    }

    /// optimizer.py:480-484 GETFIELD branch with `parent_descr.is_object() == true`
    /// → `info.InstancePtrInfo(parent_descr)`.
    #[test]
    fn ensure_ptr_info_arg0_constructs_instance_for_object_field() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let _parent = instance_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let pi = info.as_mut().expect("Forwarded variant expected");
        assert!(matches!(&*pi, PtrInfo::Instance(_)));
    }

    /// optimizer.py:485-487 ARRAYLEN_GC branch → `info.ArrayPtrInfo(descr)`.
    /// The PyPy primitive returns the same arrayinfo across calls so
    /// callers can read `arrayinfo.getlenbound(None)` directly. The Rust
    /// port mirrors that and the `getlenbound` call resolves to the
    /// pre-installed `nonnegative` lenbound on the freshly-built ArrayPtrInfo.
    #[test]
    fn ensure_ptr_info_arg0_arraylen_returns_array_with_nonnegative_lenbound() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = array_op();
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(None)
            .expect("ArrayPtrInfo.getlenbound(None) should be Some");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
    }

    /// info.py:796-802 `ConstPtrInfo.getlenbound(mode)` returns
    /// `IntBound.nonnegative()` whenever `getstrlen1(mode)` produces a
    /// negative length. info.py:823-824 makes `mode is None` (no
    /// vstring mode) one of those cases via the `else: return -1`
    /// branch. The Rust port must therefore answer `Some(nonnegative())`
    /// — not `None` — for `Constant.getlenbound(None)` so the
    /// ARRAYLEN_GC postprocess on a constant array still propagates a
    /// non-negative bound.
    #[test]
    fn ensure_ptr_info_arg0_constant_arraylen_returns_nonnegative() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        ctx.seed_constant(OpRef::from_raw(0), Value::Ref(GcRef(0xfeed)));
        let op = array_op();
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(None)
            .expect("ConstPtrInfo.getlenbound(None) must mirror PyPy nonnegative fallback");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
        assert_eq!(bound.upper, IntBound::nonnegative().upper);
    }

    /// optimizer.py:467-469 `if isinstance(opinfo, AbstractVirtualPtrInfo):
    /// return opinfo` parity. A second call must return the SAME PtrInfo
    /// (verified by mutating via the first call and observing the mutation
    /// via the second). PyPy's structinfo identity is the test of record;
    /// the Rust port checks via state preserved across calls.
    #[test]
    fn ensure_ptr_info_arg0_returns_existing_array_unchanged() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = array_op();
        // First call constructs the ArrayPtrInfo and tightens the lenbound
        // through the helper.
        {
            let mut info = ctx.ensure_ptr_info_arg0(&op);
            let mut handle = info.as_mut().expect("expected fresh ArrayPtrInfo");
            if let PtrInfo::Array(arr) = &mut *handle {
                let _ = arr.lenbound.make_gt_const(7);
            } else {
                panic!("expected fresh ArrayPtrInfo");
            }
        }
        // Second call returns the same ArrayPtrInfo (lenbound preserved).
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info.as_mut().expect("second call must still return Array");
        match &mut *handle {
            PtrInfo::Array(ArrayPtrInfo { lenbound, .. }) => {
                assert!(
                    lenbound.lower >= 8,
                    "second call must return the previously-mutated ArrayPtrInfo (lower={})",
                    lenbound.lower
                );
            }
            _ => panic!("second call must still return Array"),
        }
    }

    /// optimizer.py:470-474 `elif opinfo is not None: ...; assert opinfo is
    /// None or opinfo.__class__ is info.NonNullPtrInfo`. A pre-existing
    /// NonNullPtrInfo flows through the upgrade path; its `last_guard_pos`
    /// is preserved on the freshly-installed PtrInfo.
    #[test]
    fn ensure_ptr_info_arg0_upgrades_nonnull_to_struct() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        // Pre-install a NonNullPtrInfo with a specific last_guard_pos.
        ctx.set_ptr_info(OpRef::from_raw(0), PtrInfo::NonNull { last_guard_pos: 7 });
        let _parent = struct_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info.as_mut().expect("expected upgraded Struct, got None");
        match &mut *handle {
            pi @ PtrInfo::Struct(_) => {
                assert_eq!(pi.last_guard_pos(), Some(7));
            }
            other => panic!("expected upgraded Struct, got {other:?}"),
        }
    }

    /// optimizer.py:474 assertion: an unexpected forwarded info shape (e.g.
    /// a `Forwarded::Box` redirect that resolved to a non-PtrInfo state)
    /// must NOT silently overwrite. We seed an `Instance` PtrInfo, then
    /// hand it a field op with a different parent — the early-return path
    /// hits, and the existing Instance is returned without overwrite.
    #[test]
    fn ensure_ptr_info_arg0_does_not_overwrite_existing_instance() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        ctx.set_ptr_info(
            OpRef::from_raw(0),
            PtrInfo::instance(Some(instance_parent_descr()), Some(GcRef(0xc0de))),
        );
        let op = field_op_with_parent(struct_parent_descr());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info
            .as_mut()
            .expect("expected Instance preserved, got None");
        match &mut *handle {
            PtrInfo::Instance(_) => {} // unchanged
            other => panic!("expected Instance preserved, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod rd_virtual_info_builder_tests {
    use super::*;
    use crate::walkvirtual::VirtualVisitor;
    use majit_ir::{Descr, DescrRef, FieldDescr, SizeDescr, Type};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
        type_id: u32,
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
            16
        }

        fn type_id(&self) -> u32 {
            self.type_id
        }

        fn is_immutable(&self) -> bool {
            false
        }

        fn is_object(&self) -> bool {
            self.is_object
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
    }

    impl Descr for TestFieldDescr {
        fn index(&self) -> u32 {
            self.index
        }

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

    #[test]
    fn visit_virtual_preserves_field_descr_indices() {
        let mut builder = RdVirtualInfoBuilder;
        let size_descr: DescrRef = Arc::new(TestSizeDescr {
            index: 0x3000_0001,
            type_id: 7,
            is_object: true,
        });
        let field0: DescrRef = Arc::new(TestFieldDescr {
            index: 0x1000_0123,
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
        });
        let field1: DescrRef = Arc::new(TestFieldDescr {
            index: 0x1000_0456,
            offset: 24,
            field_size: 8,
            field_type: Type::Ref,
        });

        let Some(majit_ir::RdVirtualInfo::VirtualInfo { fielddescrs, .. }) =
            builder.visit_virtual(&size_descr, &[], &[field0.clone(), field1.clone()])
        else {
            panic!("expected VirtualInfo");
        };

        assert_eq!(fielddescrs[0].index, field0.index());
        assert_eq!(fielddescrs[1].index, field1.index());
    }
}

#[cfg(test)]
mod intbound_invariant_tests {
    use super::*;
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::{GcRef, OpRef, Value};

    #[test]
    #[should_panic]
    fn getintbound_rejects_non_int_boxes() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(20_000);
        ctx.seed_constant(opref, Value::Ref(GcRef(0xdead_beef)));
        let _ = ctx.getintbound(opref);
    }

    #[test]
    #[should_panic]
    fn setintbound_rejects_non_int_boxes() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::from_raw(20_001);
        ctx.seed_constant(opref, Value::Ref(GcRef(0xcafe_babe)));
        ctx.setintbound(opref, &IntBound::nonnegative());
    }
}

#[cfg(test)]
mod imported_short_preamble_fallback_tests {
    use super::*;
    use majit_ir::{Op, OpCode, OpRef};

    #[test]
    fn force_op_from_preamble_replays_pop_without_builder_lookup() {
        // 2 Ref inputargs for the body label — typical loop-body shape.
        let mut ctx =
            OptContext::with_inputarg_types(16, &[majit_ir::Type::Ref, majit_ir::Type::Ref]);
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef::from_raw(0), OpRef::from_raw(1)],
            &[OpRef::from_raw(7), OpRef::from_raw(8)],
            &[],
        );

        let mut replay_op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(7), OpRef::from_raw(8)]);
        replay_op.pos = OpRef::int_op(14);
        // shortpreamble.py:120 non-invented PureOp.produce_op: `op = self.res`.
        // pop.op carries the body-visible OpRef directly (no forwarding chain
        // installed for non-invented Pure).
        let pop = crate::optimizeopt::info::PreambleOp {
            op: OpRef::from_raw(41),
            invented_name: false,
            preamble_op: replay_op,
        };

        let forced = ctx.force_op_from_preamble_op(&pop);
        assert_eq!(forced, OpRef::from_raw(41));

        let sp = ctx
            .build_imported_short_preamble()
            .expect("imported short preamble builder should exist");
        assert_eq!(sp.ops.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(
            sp.ops[0].op.args.as_slice(),
            &[OpRef::from_raw(7), OpRef::from_raw(8)]
        );
        assert_eq!(sp.ops[0].op.pos, OpRef::int_op(14));
    }
}

#[cfg(test)]
mod opt_box_env_tests {
    use super::*;
    use crate::optimizeopt::info::VirtualInfo;
    use majit_ir::{BoxEnv, DescrRef, GcRef, OpRef};
    use std::sync::Arc;

    #[derive(Debug)]
    struct DummySizeDescr;

    impl majit_ir::Descr for DummySizeDescr {
        fn index(&self) -> u32 {
            0
        }

        fn clone_descr(&self) -> Option<DescrRef> {
            Some(Arc::new(DummySizeDescr))
        }

        fn as_size_descr(&self) -> Option<&dyn majit_ir::SizeDescr> {
            Some(self)
        }
    }

    impl majit_ir::SizeDescr for DummySizeDescr {
        fn size(&self) -> usize {
            24
        }

        fn vtable(&self) -> usize {
            0x1234
        }

        fn type_id(&self) -> u32 {
            7
        }

        fn is_immutable(&self) -> bool {
            false
        }
    }

    #[test]
    fn opt_box_env_is_virtual_ref_follows_box_replacement() {
        let mut ctx = OptContext::with_num_inputs(16, 0);
        let source = OpRef::from_raw(12);
        let target = OpRef::from_raw(21);
        ctx.replace_op(source, target);
        ctx.set_ptr_info(
            target,
            PtrInfo::Virtual(VirtualInfo {
                descr: Arc::new(DummySizeDescr),
                known_class: Some(GcRef(0x1234)),
                ob_type_descr: None,
                fields: Vec::new(),
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        let env = OptBoxEnv { ctx: &ctx };
        assert!(
            env.is_virtual_ref(source),
            "forwarded snapshot boxes must classify as virtual via replacement"
        );
    }
}
