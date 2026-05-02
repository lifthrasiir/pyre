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
        replay.pos = source;
        replay.descr = descr.clone();
        ImportedShortPureOp {
            opcode,
            descr,
            args,
            result,
            pop: crate::optimizeopt::info::PreambleOp {
                op: source,
                resolved: result,
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

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedShortSource {
    pub result: OpRef,
    pub source: OpRef,
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
    /// resoperation.py: _forwarded — unified forwarding + PtrInfo.
    /// Forwarded::Op(target) = forwarding, Forwarded::Info(info) = terminal.
    pub forwarded: Vec<crate::optimizeopt::info::Forwarded>,
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
    /// RPython shortpreamble.py: invented SameAs names preserved from exported
    /// short boxes. Phase 2 can later re-materialize these aliases when
    /// building the short preamble for bridges.
    pub imported_short_aliases: Vec<ImportedShortAlias>,
    /// (base_len, short_args): virtual field values start at base_len
    /// within short_args. Used by install_imported_virtuals.
    pub imported_virtual_args: Option<(usize, Vec<OpRef>)>,
    /// Original preamble result box for each imported short-box result.
    /// This preserves RPython PreambleOp.op identity so phase 2 assembly
    /// can remap any surviving synthetic imported boxes back to the
    /// corresponding preamble value.
    pub imported_short_sources: Vec<ImportedShortSource>,
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
    /// RPython import_state: maps original inputarg index → fresh virtual head OpRef.
    /// Used by ensure_linked_list_head to return the imported virtual.
    pub imported_virtual_heads: Vec<(usize, OpRef)>,
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

    fn is_const(&self, opref: OpRef) -> bool {
        // RPython resume.py:204: isinstance(box, Const)
        // True Const = constant-namespace OpRef or PtrInfo::Constant.
        // NOT optimizer-known values from make_constant() on operation results.
        if opref.is_constant() {
            return true;
        }
        // optimizer.py:432: make_constant → Forwarded::Const terminal.
        // resume.py:204: isinstance(box, Const)
        if matches!(
            self.ctx.forwarded.get(opref.0 as usize),
            Some(crate::optimizeopt::info::Forwarded::Const(_))
        ) {
            return true;
        }
        // info.py: ConstPtrInfo.is_constant() → True
        matches!(
            self.ctx.get_ptr_info(opref),
            Some(crate::optimizeopt::info::PtrInfo::Constant(_))
        )
    }

    fn get_const(&self, opref: OpRef) -> (i64, majit_ir::Type) {
        // optimizer.py:432: make_constant → Forwarded::Const — check first.
        if let Some(crate::optimizeopt::info::Forwarded::Const(val)) =
            self.ctx.forwarded.get(opref.0 as usize)
        {
            let (raw, tp) = match val {
                Value::Int(v) => (*v, majit_ir::Type::Int),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                Value::Void => (0, majit_ir::Type::Int),
            };
            let tp = self
                .ctx
                .constant_types_for_numbering
                .get(&opref.0)
                .copied()
                .unwrap_or(tp);
            return (raw, tp);
        }
        // RPython ConstPtr parity: check numbering type overrides first.
        let type_override = self.ctx.constant_types_for_numbering.get(&opref.0).copied();
        match self.ctx.get_constant(opref) {
            Some(Value::Int(v)) => (*v, type_override.unwrap_or(majit_ir::Type::Int)),
            Some(Value::Float(f)) => (f.to_bits() as i64, majit_ir::Type::Float),
            Some(Value::Ref(r)) => (r.0 as i64, majit_ir::Type::Ref),
            _ => {
                // info.py: ConstPtrInfo — GcRef constant stored in PtrInfo
                if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) =
                    self.ctx.get_ptr_info(opref)
                {
                    (gcref.0 as i64, majit_ir::Type::Ref)
                } else {
                    (0, majit_ir::Type::Int)
                }
            }
        }
    }

    fn get_type(&self, opref: OpRef) -> majit_ir::Type {
        // Check constant type first (history.py:220 ConstInt.type parity).
        // The `constant_types_for_numbering` override is a raw-pointer
        // marker on `'i'`-typed Boxes (RPython `getrawptrinfo` ConstInt
        // path); it does NOT change `box.type` from `'i'` to `'r'`, so
        // we never read it here as a type source.  This matches
        // `OptContext::opref_type`'s documented invariant.
        if let Some(val) = self.ctx.get_constant(opref) {
            return val.get_type();
        }
        // Producing op intrinsic type (resoperation.py:1693
        // `opclasses[opnum].type` parity).
        let resolved = self.ctx.get_box_replacement(opref);
        if let Some(op) = self.ctx.op_at(resolved) {
            if op.type_ != majit_ir::Type::Void {
                return op.type_;
            }
        }
        // value_types aggregates the four upstream sources (original trace
        // ops, prev-phase carry, inputarg types, transformed trace result
        // types) seeded in `Optimizer::make_ctx`. Removed in Slice 0.5
        // once writers feed Op.type_ exclusively.
        if let Some(&tp) = self.ctx.value_types.get(&opref.0) {
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
        if self.ctx.get_ptr_info(opref).is_some() {
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
            .get_ptr_info(resolved)
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
            .get_ptr_info(resolved)
            .and_then(|info| info.get_known_class())
            .is_some()
    }

    fn get_virtual_fields(&self, opref: OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        let resolved = self.ctx.get_box_replacement(opref);
        let info = self.ctx.get_ptr_info(resolved)?;
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
        let info = self.ctx.get_ptr_info(resolved)?;
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
        }
        Some(shared)
    }

    fn virtual_info_would_be_reused(&self, opref: OpRef, fieldnums: &[i16]) -> bool {
        let resolved = self.ctx.get_box_replacement(opref);
        let Some(info) = self.ctx.get_ptr_info(resolved) else {
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
        if let Some(&existing) = self.value_types.get(&opref.0) {
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
        self.value_types.insert(opref.0, tp);
    }

    pub fn new(estimated_ops: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            const_pool: HashMap::new(),
            forwarded: Vec::new(),
            num_inputs: 0,
            inputarg_base: 0,
            next_pos: 0,
            next_const_idx: 0,
            extra_operations_after: VecDeque::new(),
            pending_guard_class_postprocess: None,
            pending_mark_last_guard: None,
            pending_finish_guard_postprocess: None,
            imported_short_pure_ops: Vec::new(),
            imported_short_aliases: Vec::new(),
            imported_virtual_args: None,
            imported_short_sources: Vec::new(),
            imported_loop_invariant_results: HashMap::new(),
            imported_short_preamble_builder: None,
            const_infos: HashMap::new(),
            imported_short_preamble_used: HashSet::new(),

            potential_extra_ops: HashMap::new(),
            active_short_preamble_producer: None,
            exported_short_boxes: Vec::new(),
            imported_virtual_heads: Vec::new(),
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
            snapshot_frame_pcs: Vec::new(),

            constant_types_for_numbering: HashMap::new(),
            value_types: HashMap::new(),
            inputarg_types: Vec::new(),
            phase1_emit_ops: Vec::new(),
            last_guard_idx: None,
            last_seen_snapshot_pos: None,
        }
    }

    pub fn with_num_inputs(estimated_ops: usize, num_inputs: usize) -> Self {
        Self::with_num_inputs_and_start_pos(estimated_ops, num_inputs, 0, num_inputs as u32)
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
            forwarded: Vec::new(),
            num_inputs: num_inputs as u32,
            inputarg_base,
            next_pos: start_next_pos,
            next_const_idx: 0,
            extra_operations_after: VecDeque::new(),
            pending_guard_class_postprocess: None,
            pending_mark_last_guard: None,
            pending_finish_guard_postprocess: None,
            imported_short_pure_ops: Vec::new(),
            imported_short_aliases: Vec::new(),
            imported_virtual_args: None,
            imported_short_sources: Vec::new(),
            imported_loop_invariant_results: HashMap::new(),
            imported_short_preamble_builder: None,
            const_infos: HashMap::new(),
            imported_short_preamble_used: HashSet::new(),

            potential_extra_ops: HashMap::new(),
            active_short_preamble_producer: None,
            exported_short_boxes: Vec::new(),
            imported_virtual_heads: Vec::new(),
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
            snapshot_frame_pcs: Vec::new(),

            constant_types_for_numbering: HashMap::new(),
            value_types: HashMap::new(),
            inputarg_types: Vec::new(),
            phase1_emit_ops: Vec::new(),
            last_guard_idx: None,
            last_seen_snapshot_pos: None,
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    /// Allocate a fresh OpRef position (for imported virtual heads).
    pub fn alloc_op_position(&mut self) -> OpRef {
        self.reserve_pos()
    }

    /// Allocate a fresh OpRef in the constant namespace.
    pub(crate) fn reserve_const_ref(&mut self) -> OpRef {
        let opref = OpRef::from_const(self.next_const_idx);
        self.next_const_idx += 1;
        opref
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
        let pos_ref = self.reserve_pos();
        let mut op = Op::new(OpCode::SameAsI, &[pos_ref]);
        op.pos = pos_ref;
        let opref = self.emit_extra(self.current_pass_idx, op);
        self.make_constant(opref, Value::Int(value));
        opref
    }

    /// Emit a boxed reference constant through the optimizer pipeline and
    /// return the resulting OpRef.
    pub fn emit_constant_ref(&mut self, value: GcRef) -> OpRef {
        let pos_ref = self.reserve_pos();
        let mut op = Op::new(OpCode::SameAsR, &[pos_ref]);
        op.pos = pos_ref;
        let opref = self.emit_extra(self.current_pass_idx, op);
        self.make_constant(opref, Value::Ref(value));
        opref
    }

    /// Emit a boxed float constant through the optimizer pipeline and return
    /// the resulting OpRef.
    pub fn emit_constant_float(&mut self, value: f64) -> OpRef {
        let pos_ref = self.reserve_pos();
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
        if let Some(info) = self.getptrinfo(resolved) {
            if let Some(lgtop) = info.as_ref().get_cached_lgtop() {
                return lgtop;
            }
        }
        // vstring.py:174/253: constant or structurally-known length
        let known_len = self
            .getptrinfo(resolved)
            .and_then(|info| info.as_ref().get_known_str_length(self, mode));
        if let Some(len) = known_len {
            let len_opref = self.make_constant_int(len);
            self.set_str_lgtop(resolved, len_opref);
            return len_opref;
        }
        // vstring.py:281-295: VStringConcatInfo.getstrlen — recursive
        // dispatch: getstrlen on each child, then _int_add.
        // Borrow-checker adaptation: extract vleft/vright before &mut self calls.
        let concat_children = self.getptrinfo(resolved).and_then(|info| {
            use crate::optimizeopt::info::VStringVariant;
            if let PtrInfo::Str(sinfo) = info.as_ref() {
                if let VStringVariant::Concat(c) = &sinfo.variant {
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
    /// Borrow-checker adaptation: direct forwarded[] mutation because
    /// RPython mutates self.lgtop directly (vstring.py:117, 174, 293).
    pub(crate) fn set_str_lgtop(&mut self, opref: OpRef, lgtop: OpRef) {
        use crate::optimizeopt::info::Forwarded;
        let resolved = self.get_box_replacement(opref);
        if resolved.is_constant() {
            return;
        }
        let idx = resolved.0 as usize;
        if let Some(Forwarded::Info(PtrInfo::Str(si))) = self.forwarded.get_mut(idx) {
            si.lgtop = Some(lgtop);
        }
    }

    /// vstring.py:62-70 StrPtrInfo.getlenbound(mode) — get lenbound from
    /// StrPtrInfo, lazily initializing it from self.length.
    /// Borrow-checker adaptation: direct forwarded[] access.
    fn get_str_lenbound(&mut self, opref: OpRef) -> Option<crate::optimizeopt::intutils::IntBound> {
        use crate::optimizeopt::info::Forwarded;
        let resolved = self.get_box_replacement(opref);
        if resolved.is_constant() {
            return None;
        }
        let idx = resolved.0 as usize;
        if let Some(Forwarded::Info(PtrInfo::Str(si))) = self.forwarded.get_mut(idx) {
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
    }

    pub(crate) fn reserve_pos(&mut self) -> OpRef {
        // opencoder.py:271 _index parity: floor at the iteration's inputarg
        // base + num_inputs + emitted-op count, so reserve_pos never returns
        // an OpRef inside the inputarg slice or below the parent trace's
        // high water mark when called from a Phase 2 / bridge OptContext.
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
            !OpRef(self.next_pos).is_constant(),
            "reserve_pos overflowed into constant namespace: {}",
            self.next_pos
        );
        let pos_ref = OpRef(self.next_pos);
        self.next_pos += 1;
        pos_ref
    }

    /// Emit an operation to the output.
    ///
    /// If the op has no pos assigned (NONE), sets it to `num_inputs + idx`
    /// so the backend's variable numbering stays consistent.
    pub fn emit(&mut self, mut op: Op) -> OpRef {
        if op.pos.is_none() || op.pos.is_constant() {
            op.pos = self.reserve_pos();
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
            // (c) `import_state` only creates `Forwarded::Op` chains on
            //     inputarg slots (in `[inputarg_base..inputarg_base +
            //     num_inputs)`) — never on op-result positions that a
            //     later `emit` would try to use.
            //
            // Together these guarantee that:
            //   - `new_operations` never contains two ops at the same pos
            //   - an op being emitted whose pos is a non-void result does
            //     not already have Forwarded::Op set
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
                "emit: Forwarded::Op set on non-void result position {:?} — \
                 import_state should only forward inputarg slots in \
                 [inputarg_base..inputarg_base + num_inputs), and Phase 2 op results \
                 live in a disjoint range [p2_high_water..) (Commit D1).",
                op.pos,
            );
            self.next_pos = self.next_pos.max(op.pos.0.saturating_add(1));
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
            op.pos = self.reserve_pos();
        } else {
            self.next_pos = self.next_pos.max(op.pos.0.saturating_add(1));
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
        for (&const_idx, _) in &self.const_pool {
            builder.note_known_constant(OpRef::from_const(const_idx));
        }
        self.imported_short_preamble_builder = Some(builder);
        self.imported_short_preamble_used.clear();
    }

    pub fn initialize_imported_short_preamble_builder_from_exported_ops(
        &mut self,
        short_args: &[OpRef],
        short_inputargs: &[OpRef],
        exported_short_ops: &[crate::optimizeopt::unroll::ExportedShortOp],
    ) -> bool {
        use crate::optimizeopt::shortpreamble::{
            PreambleOpKind, ProducedShortOp, ShortPreambleBuilder,
        };
        use crate::optimizeopt::unroll::{ExportedShortArg, ExportedShortOp, ExportedShortResult};

        let mut produced: Vec<(OpRef, ProducedShortOp)> =
            Vec::with_capacity(exported_short_ops.len());
        let mut produced_results: Vec<OpRef> = Vec::with_capacity(exported_short_ops.len());
        let mut temporary_results: Vec<Option<OpRef>> = Vec::new();
        let mut imported_constants: HashMap<OpRef, OpRef> = HashMap::new();
        let imported_result_for_source = |source: OpRef, this: &Self| {
            this.imported_short_sources
                .iter()
                .find_map(|entry| (entry.source == source).then_some(entry.result))
        };

        let mut resolve_result = |result: &ExportedShortResult, this: &mut Self| match result {
            ExportedShortResult::Slot(slot) => short_args.get(*slot).copied(),
            ExportedShortResult::Temporary(index) => {
                if *index >= temporary_results.len() {
                    temporary_results.resize(index + 1, None);
                }
                let slot = &mut temporary_results[*index];
                Some(*slot.get_or_insert_with(|| this.alloc_op_position()))
            }
        };

        // shortpreamble.py:PreambleOp.add_op_to_short — Pure ops whose
        // opcode is a Call get rewritten to the CallPure* equivalent so the
        // short preamble can replay the cached call without re-executing
        // arbitrary side effects.
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
        for entry in exported_short_ops {
            match entry {
                ExportedShortOp::Pure {
                    source,
                    opcode,
                    descr,
                    args,
                    result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(result_opref) = imported_result_for_source(*source, self)
                        .or_else(|| resolve_result(result, self))
                    else {
                        return false;
                    };
                    let mut resolved_args = Vec::with_capacity(args.len());
                    for arg in args {
                        let resolved = match arg {
                            ExportedShortArg::Slot(slot) => short_args.get(*slot).copied(),
                            ExportedShortArg::Produced(index) => {
                                produced_results.get(*index).copied()
                            }
                            ExportedShortArg::Const { source, value } => {
                                let opref =
                                    imported_constants.entry(*source).or_insert_with(|| {
                                        let opref = self.alloc_op_position();
                                        self.make_constant(opref, value.clone());
                                        opref
                                    });
                                Some(*opref)
                            }
                        };
                        let Some(resolved) = resolved else {
                            return false;
                        };
                        resolved_args.push(resolved);
                    }
                    let mut op = Op::new(pure_call_opcode(*opcode), &resolved_args);
                    op.pos = result_opref;
                    op.descr = descr.clone();
                    let produced_op = ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        preamble_op: op,
                        invented_name: *invented_name,
                        same_as_source: *same_as_source,
                    };
                    produced.push((*source, produced_op.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, produced_op));
                    }
                    produced_results.push(result_opref);
                }
                ExportedShortOp::HeapField {
                    source,
                    object,
                    descr,
                    result_type,
                    result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(result_opref) = imported_result_for_source(*source, self)
                        .or_else(|| resolve_result(result, self))
                    else {
                        return false;
                    };
                    let obj = match object {
                        ExportedShortArg::Slot(slot) => short_args.get(*slot).copied(),
                        ExportedShortArg::Const { source, value } => {
                            let opref = imported_constants.entry(*source).or_insert_with(|| {
                                let opref = self.alloc_op_position();
                                self.make_constant(opref, value.clone());
                                opref
                            });
                            Some(*opref)
                        }
                        ExportedShortArg::Produced(_) => None,
                    };
                    let Some(obj) = obj else {
                        return false;
                    };
                    let opcode = match result_type {
                        majit_ir::Type::Int => OpCode::GetfieldGcI,
                        majit_ir::Type::Ref => OpCode::GetfieldGcR,
                        majit_ir::Type::Float => OpCode::GetfieldGcF,
                        majit_ir::Type::Void => return false,
                    };
                    let mut op = Op::new(opcode, &[obj]);
                    op.pos = result_opref;
                    op.descr = Some(descr.clone());
                    let produced_op = ProducedShortOp {
                        kind: PreambleOpKind::Heap,
                        preamble_op: op,
                        invented_name: *invented_name,
                        same_as_source: *same_as_source,
                    };
                    produced.push((*source, produced_op.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, produced_op));
                    }
                    produced_results.push(result_opref);
                }
                ExportedShortOp::HeapArrayItem {
                    source,
                    object,
                    descr,
                    index,
                    result_type,
                    result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(result_opref) = imported_result_for_source(*source, self)
                        .or_else(|| resolve_result(result, self))
                    else {
                        return false;
                    };
                    // shortpreamble.py:128 sb.produce_arg() parity: all args
                    // (including const base objects) go through the same
                    // import-local constant path so the builder tracks them.
                    let obj = match object {
                        ExportedShortArg::Slot(slot) => short_args.get(*slot).copied(),
                        ExportedShortArg::Const { source, value } => {
                            let opref = imported_constants.entry(*source).or_insert_with(|| {
                                let opref = self.alloc_op_position();
                                self.make_constant(opref, value.clone());
                                opref
                            });
                            Some(*opref)
                        }
                        ExportedShortArg::Produced(_) => None,
                    };
                    let Some(obj) = obj else {
                        return false;
                    };
                    let opcode = match result_type {
                        majit_ir::Type::Int => OpCode::GetarrayitemGcI,
                        majit_ir::Type::Ref => OpCode::GetarrayitemGcR,
                        majit_ir::Type::Float => OpCode::GetarrayitemGcF,
                        majit_ir::Type::Void => return false,
                    };
                    let index_opref = {
                        let opref = self.alloc_op_position();
                        self.make_constant(opref, majit_ir::Value::Int(*index));
                        opref
                    };
                    let mut op = Op::new(opcode, &[obj, index_opref]);
                    op.pos = result_opref;
                    op.descr = Some(descr.clone());
                    let produced_op = ProducedShortOp {
                        kind: PreambleOpKind::Heap,
                        preamble_op: op,
                        invented_name: *invented_name,
                        same_as_source: *same_as_source,
                    };
                    produced.push((*source, produced_op.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, produced_op));
                    }
                    produced_results.push(result_opref);
                }
                ExportedShortOp::LoopInvariant {
                    source,
                    func_ptr,
                    result_type,
                    result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(result_opref) = imported_result_for_source(*source, self)
                        .or_else(|| resolve_result(result, self))
                    else {
                        return false;
                    };
                    // shortpreamble.py LoopInvariantOp replays the call with
                    // its original func_ptr argument. The serialization only
                    // carries the func ptr — the call had no other args
                    // (collect_exported_short_ops only emits LoopInvariant
                    // for zero-arg calls today). Synthesize the func_ptr
                    // constant from the serialized value.
                    let func_opref = {
                        let opref = self.alloc_op_position();
                        self.make_constant(opref, majit_ir::Value::Int(*func_ptr));
                        opref
                    };
                    let opcode = loop_invariant_opcode(*result_type);
                    let mut op = Op::new(opcode, &[func_opref]);
                    op.pos = result_opref;
                    let produced_op = ProducedShortOp {
                        kind: PreambleOpKind::LoopInvariant,
                        preamble_op: op,
                        invented_name: *invented_name,
                        same_as_source: *same_as_source,
                    };
                    produced.push((*source, produced_op.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, produced_op));
                    }
                    produced_results.push(result_opref);
                }
            }
        }

        let mut builder = ShortPreambleBuilder::new(short_args, &produced, short_inputargs);
        for (&const_idx, _) in &self.const_pool {
            builder.note_known_constant(OpRef::from_const(const_idx));
        }
        // Imported constants are bridge-local stand-ins for RPython Const
        // objects; make them available to produce_arg() just like const_pool
        // entries instead of leaving trace-local source OpRefs in short ops.
        for &opref in imported_constants.values() {
            builder.note_known_constant(opref);
        }
        self.imported_short_preamble_builder = Some(builder);
        self.imported_short_preamble_used.clear();
        true
    }

    pub fn note_imported_short_use(&mut self, result: OpRef) {
        let _ = self.force_op_from_preamble(result);
    }

    pub(crate) fn imported_short_source(&self, result: OpRef) -> OpRef {
        self.imported_short_sources
            .iter()
            .find_map(|entry| (entry.result == result).then_some(entry.source))
            .unwrap_or(result)
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
        let result = preamble_op.resolved;
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
            // shortpreamble.py:404: optimizer.setinfo_from_preamble(box, info, None)
            if let Some(info) = self.get_ptr_info(preamble_source).cloned() {
                self.setinfo_from_preamble(result, &info, None);
            }
            // RPython PreambleOp carries Box.type intrinsically. majit's
            // imported replay uses a distinct `resolved` OpRef, so preserve
            // the replay result type on first force when the import path did
            // not already seed `value_types` (e.g. focused unit fixtures).
            self.value_types.entry(result.0).or_insert(result_type);
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
        // unroll.py:38: return preamble_op.op
        // RPython uses a single Box identity for Phase 1 and Phase 2.
        // In majit, preamble_source (Phase 1) and result (Phase 2) are
        // distinct OpRefs. Phase 2 operations reference `result`, so
        // returning it is structurally equivalent to RPython's `return op`.
        result
    }

    /// Backward compat: force_op_from_preamble by OpRef lookup.
    /// unroll.py:26-39: force_op_from_preamble(preamble_op)
    ///
    /// Backward compat wrapper: builds PreambleOp from OpRef lookup.
    /// RPython receives a PreambleOp directly; this path looks up
    /// the produced_short_op to reconstruct it.
    pub fn force_op_from_preamble(&mut self, result: OpRef) -> OpRef {
        let preamble_source = self.imported_short_source(result);
        let produced = self
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(preamble_source))
            .or_else(|| {
                self.active_short_preamble_producer
                    .as_ref()
                    .and_then(|b| b.produced_short_op(preamble_source))
            });
        // unroll.py:27: isinstance(preamble_op, PreambleOp) — RPython
        // always receives PreambleOp directly. All callers with a
        // PreambleOp should use force_op_from_preamble_op() instead.
        let produced = produced.expect(
            "force_op_from_preamble: produced_short_op not found — \
             caller should use force_op_from_preamble_op(&PreambleOp)",
        );
        let pop = crate::optimizeopt::info::PreambleOp {
            op: preamble_source,
            resolved: result,
            invented_name: produced.invented_name,
            preamble_op: produced.preamble_op,
        };
        self.force_op_from_preamble_op(&pop)
    }

    /// shortpreamble.py:383-396,401-406: collect guards from the forwarded
    /// info of preamble_op's args and result. RPython's `info = arg.get_forwarded()`
    /// returns whatever is stored — PtrInfo *or* IntBound — and calls
    /// `info.make_guards(...)` uniformly.
    fn collect_use_box_guards(&mut self, preamble_op: &Op) -> (Vec<Op>, Vec<Op>) {
        use crate::optimizeopt::info::Forwarded;
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

        // Snapshot forwarded state per arg to avoid borrow conflicts during
        // guard emission. `Forwarded::Info` carries PtrInfo; `Forwarded::IntBound`
        // carries integer range info — both produce guards.
        enum ForwardedInfo {
            Ptr(PtrInfo),
            Int(crate::optimizeopt::intutils::IntBound),
        }
        let snapshot_forwarded = |ctx: &Self, arg: OpRef| -> Option<ForwardedInfo> {
            let replaced = ctx.get_box_replacement(arg);
            let idx = replaced.0 as usize;
            if idx >= ctx.forwarded.len() {
                return None;
            }
            match &ctx.forwarded[idx] {
                Forwarded::Info(info) => Some(ForwardedInfo::Ptr(info.clone())),
                Forwarded::IntBound(b) => Some(ForwardedInfo::Int(b.clone())),
                _ => None,
            }
        };
        let arg_infos: Vec<(OpRef, ForwardedInfo)> = preamble_op
            .args
            .iter()
            .filter(|arg| short_inputargs.contains(arg))
            .filter_map(|&arg| snapshot_forwarded(self, arg).map(|info| (arg, info)))
            .collect();
        let result_info: Option<(OpRef, ForwardedInfo)> =
            snapshot_forwarded(self, preamble_op.pos).map(|info| (preamble_op.pos, info));

        // Now generate guards — alloc_const directly allocates constant
        // OpRefs, matching RPython where ConstInt/ConstPtr are created inline.
        let mut arg_guards = Vec::new();
        let mut alloc_const = |value: Value| -> OpRef {
            let pos = self.reserve_const_ref();
            self.seed_constant(pos, value);
            pos
        };
        for (arg, info) in &arg_infos {
            match info {
                ForwardedInfo::Ptr(p) => p.make_guards(*arg, &mut arg_guards, &mut alloc_const),
                ForwardedInfo::Int(b) => b.make_guards(*arg, &mut arg_guards, &mut alloc_const),
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
            }
        }
        (arg_guards, result_guards)
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
        if self.get_ptr_info(op).is_some() || self.is_replaced(op) {
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

    /// unroll.py:49: item.set_forwarded(None)
    fn clear_forwarded(&mut self, opref: OpRef) {
        use crate::optimizeopt::info::Forwarded;
        let idx = opref.0 as usize;
        if idx < self.forwarded.len() {
            self.forwarded[idx] = Forwarded::None;
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
                    .map(|(&i, val)| (OpRef::from_const(i).0, value_to_raw(val)))
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
                        .entry(OpRef::from_const(i).0)
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
        use crate::optimizeopt::info::Forwarded;
        let idx = old.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        if new.is_none() {
            self.forwarded[idx] = Forwarded::None;
        } else {
            self.forwarded[idx] = Forwarded::Op(new);
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
        if let Some(info) = self.get_ptr_info_mut(opref) {
            info.set_last_guard_pos(pos);
        }
    }

    /// info.py:100-103: get_last_guard.
    /// _newoperations[last_guard_pos].
    pub fn get_last_guard(&self, opref: OpRef) -> Option<&Op> {
        let pos = self.get_ptr_info(opref)?.get_last_guard_pos()?;
        self.new_operations.get(pos)
    }

    /// resoperation.py:57-68 get_box_replacement: follow the forwarding
    /// chain (op._forwarded) until we reach a terminal. RPython: walks
    /// op → op._forwarded → ... until None or Info instance.
    ///
    /// RPython invariant: get_box_replacement NEVER returns None.
    /// `_forwarded = None` means "no forwarding" (terminal), NOT
    /// "forwarded to None". Forwarded::Op(OpRef::NONE) is treated
    /// as terminal — the chain stops at the current opref.
    ///
    /// NEVER consults mapping dicts — RPython's get_box_replacement only
    /// follows the _forwarded chain on the box itself.
    pub fn get_box_replacement(&self, mut opref: OpRef) -> OpRef {
        use crate::optimizeopt::info::Forwarded;
        loop {
            let idx = opref.0 as usize;
            if idx >= self.forwarded.len() {
                return opref;
            }
            match &self.forwarded[idx] {
                Forwarded::Op(next) if !next.is_none() => opref = *next,
                _ => return opref,
            }
        }
    }

    /// resoperation.py: op.get_forwarded() is not None — check if OpRef
    /// has any forwarding entry (Op, Info, IntBound, Const).
    pub fn has_forwarding(&self, opref: OpRef) -> bool {
        let idx = opref.0 as usize;
        if idx >= self.forwarded.len() {
            return false;
        }
        !matches!(
            &self.forwarded[idx],
            crate::optimizeopt::info::Forwarded::None
        )
    }

    /// True only when opref has a positional redirect (Forwarded::Op).
    /// Info/Const/IntBound are terminal metadata, not import_state redirects.
    pub fn has_op_forwarding(&self, opref: OpRef) -> bool {
        let idx = opref.0 as usize;
        if idx >= self.forwarded.len() {
            return false;
        }
        matches!(
            &self.forwarded[idx],
            crate::optimizeopt::info::Forwarded::Op(_)
        )
    }

    /// Store a constant value WITHOUT setting Forwarded::Const.
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
            self.const_pool.insert(opref.const_index(), value);
        } else {
            let idx = opref.0 as usize;
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
    pub fn peek_intbound(&self, opref: OpRef) -> Option<crate::optimizeopt::intutils::IntBound> {
        use crate::optimizeopt::info::Forwarded;
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
                *v as i64,
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
        let idx = replaced.0 as usize;
        if idx < self.forwarded.len() {
            if let Forwarded::IntBound(b) = &self.forwarded[idx] {
                return Some(b.clone());
            }
        }
        None
    }

    /// optimizer.py:99-113: getintbound(op) — get or create IntBound for
    /// an int-typed box. Lazy: creates unbounded on first access and stores
    /// it in forwarded[].
    pub fn getintbound(&mut self, opref: OpRef) -> crate::optimizeopt::intutils::IntBound {
        use crate::optimizeopt::info::Forwarded;
        // optimizer.py:100: assert op.type == 'i'
        assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "getintbound: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let replaced = self.get_box_replacement(opref);
        // optimizer.py:102-103: if isinstance(op, ConstInt): return from_constant
        if let Some(Value::Int(v)) = self.get_constant(replaced) {
            return crate::optimizeopt::intutils::IntBound::from_constant(*v as i64);
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
        let idx = replaced.0 as usize;
        if idx < self.forwarded.len() {
            match &self.forwarded[idx] {
                // optimizer.py:106-107: isinstance(fw, IntBound) → return fw
                Forwarded::IntBound(b) => return b.clone(),
                // optimizer.py:108-109: fw is not None but not IntBound
                // (rare: RawBufferPtrInfo etc.) → return unbounded, do NOT
                // overwrite existing forwarding.
                Forwarded::None => {}
                _ => return crate::optimizeopt::intutils::IntBound::unbounded(),
            }
        }
        // optimizer.py:110-112: fw is None → create unbounded and store
        let intbound = crate::optimizeopt::intutils::IntBound::unbounded();
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        self.forwarded[idx] = Forwarded::IntBound(intbound.clone());
        intbound
    }

    /// optimizer.py:115-125: setintbound(op, bound) — intersect existing
    /// bound with new bound, or set if none exists.
    pub fn setintbound(&mut self, opref: OpRef, bound: &crate::optimizeopt::intutils::IntBound) {
        use crate::optimizeopt::info::Forwarded;
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
        let idx = replaced.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        match &mut self.forwarded[idx] {
            Forwarded::IntBound(cur) => {
                let _ = cur.intersect(bound);
            }
            fwd @ Forwarded::None => *fwd = Forwarded::IntBound(bound.clone()),
            _ => {}
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
        use crate::optimizeopt::info::Forwarded;
        // optimizer.py:99: assert op.type == 'i'
        assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "with_intbound_mut: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let replaced = self.get_box_replacement(opref);
        if let Some(Value::Int(v)) = self.get_constant(replaced) {
            let mut tmp = crate::optimizeopt::intutils::IntBound::from_constant(*v as i64);
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
        let idx = replaced.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        match &mut self.forwarded[idx] {
            Forwarded::IntBound(b) => f(b),
            fwd @ Forwarded::None => {
                // optimizer.py:110-112 first-access: materialize unbounded.
                let mut new_bound = crate::optimizeopt::intutils::IntBound::unbounded();
                let result = f(&mut new_bound);
                *fwd = Forwarded::IntBound(new_bound);
                result
            }
            _ => {
                // Forwarded::Const/Op/Info — RPython's "rare case" arm:
                // return IntBound.unbounded() without overwriting forwarding.
                let mut tmp = crate::optimizeopt::intutils::IntBound::unbounded();
                f(&mut tmp)
            }
        }
    }

    /// optimizer.py:410-432 make_constant(box, constbox).
    /// Forwarded::Const(value) is the terminal — get_box_replacement stops
    /// here and returns the opref. is_const detects Forwarded::Const.
    /// optimizer.py:410-432: make_constant(box, constbox)
    pub fn make_constant(&mut self, opref: OpRef, value: Value) {
        use crate::optimizeopt::info::Forwarded;
        // optimizer.py:412: box = get_box_replacement(box)
        let replaced = self.get_box_replacement(opref);
        // optimizer.py:415-426: safety check — if box.get_forwarded() is
        // IntBound and the constant is Int, validate contains() + make_eq_const().
        // RPython checks ONE authoritative source: box._forwarded.
        if let Value::Int(intval) = value {
            let ridx = replaced.0 as usize;
            if ridx < self.forwarded.len() {
                if let Forwarded::IntBound(bound) = &mut self.forwarded[ridx] {
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
        if replaced.is_constant()
            || matches!(
                self.forwarded.get(replaced.0 as usize),
                Some(Forwarded::Const(_))
            )
        {
            return;
        }
        // optimizer.py:429-431 — when promoting a virtual to a constant,
        // call `copy_fields_to_const(constinfo, optheap)` so the cached
        // field/item state survives via the const_infos pool.
        if let Value::Ref(gcref) = value {
            self.copy_fields_to_const(replaced, gcref);
        }
        // Store in constants array for get_constant() callers.
        let idx = replaced.0 as usize;
        if idx >= self.constants.len() {
            self.constants.resize(idx + 1, None);
        }
        self.constants[idx] = Some(value.clone());
        // optimizer.py:432: box.set_forwarded(constbox)
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        self.forwarded[idx] = Forwarded::Const(value);
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
        use crate::optimizeopt::info::{
            ArrayPtrInfo, FieldEntry, Forwarded, PtrInfo, StructPtrInfo,
        };
        let Some(Forwarded::Info(info)) = self.forwarded.get(source.0 as usize) else {
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
        let type_override = self.constant_types_for_numbering.get(&opref.0).copied();
        // Check constant pool (through replacement chain).
        if let Some(val) = self.get_constant(opref) {
            let (raw, tp) = match val {
                Value::Int(v) => (*v, type_override.unwrap_or(majit_ir::Type::Int)),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                _ => return None,
            };
            return Some((raw, tp));
        }
        // Check raw constants (before replacement).
        if let Some(val) = self
            .constants
            .get(opref.0 as usize)
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
        if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) = self.get_ptr_info(opref) {
            return Some((gcref.0 as i64, majit_ir::Type::Ref));
        }
        None
    }

    /// Get the constant value for an operation, if known.
    pub fn get_constant(&self, opref: OpRef) -> Option<&Value> {
        let opref = self.get_box_replacement(opref);
        if opref.is_constant() {
            return self.const_pool.get(&opref.const_index());
        }
        // Check Forwarded::Const first (constant-folded operations)
        if let Some(crate::optimizeopt::info::Forwarded::Const(val)) =
            self.forwarded.get(opref.0 as usize)
        {
            return Some(val);
        }
        let idx = opref.0 as usize;
        self.constants.get(idx).and_then(|v| v.as_ref())
    }

    /// Whether `opref` has a known constant value.
    pub fn is_constant(&self, opref: OpRef) -> bool {
        self.get_constant(opref).is_some()
    }

    /// Get constant integer value, if known.
    pub fn get_constant_int(&self, opref: OpRef) -> Option<i64> {
        self.get_constant(opref).and_then(|v| match v {
            Value::Int(i) => Some(*i),
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
    /// Mirrors Optimizer.force_box contract: resolve imported short source,
    /// handle tracked preamble ops, then force virtuals to concrete.
    fn force_box_inline(&mut self, opref: OpRef) -> OpRef {
        let preamble_source = self.imported_short_source(opref);
        let resolved = self.get_box_replacement(opref);
        let tracked = self
            .take_potential_extra_op(resolved)
            .or_else(|| self.take_potential_extra_op(opref))
            .or_else(|| {
                (preamble_source != resolved && preamble_source != opref)
                    .then(|| self.take_potential_extra_op(preamble_source))
                    .flatten()
            });
        if let Some(preamble_op) = tracked {
            let resolved_for_pop = self.get_box_replacement(preamble_op.op);
            if let Some(builder) = self.active_short_preamble_producer_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
            } else if let Some(builder) = self.imported_short_preamble_builder.as_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
            }
        }
        if let Some(mut info) = self.get_ptr_info(resolved).cloned() {
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

        if majit_log_enabled() && op.opcode == OpCode::GuardNotForced2 {
            let env = OptBoxEnv { ctx: self };
            let snapshot_debug: Vec<(OpRef, OpRef, bool, Type)> = snapshot_boxes
                .iter()
                .copied()
                .map(|boxref| {
                    let boxref = boxref.opref;
                    let resolved = self.get_box_replacement(boxref);
                    let is_virtual = self
                        .get_ptr_info(resolved)
                        .is_some_and(|info| info.is_virtual());
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
                    let is_virtual = self
                        .get_ptr_info(resolved)
                        .is_some_and(|info| info.is_virtual());
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
                    .get(&opref.0)
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
    /// Equivalent to `peek_intbound`; preserved for legacy callers in
    /// rewrite.rs that gate optimizations on "is a bound known?".
    pub fn get_int_bound(&self, opref: OpRef) -> Option<crate::optimizeopt::intutils::IntBound> {
        self.peek_intbound(opref)
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
        let pos = self.reserve_const_ref();
        self.seed_constant(pos, Value::Int(value));
        pos
    }

    pub fn make_constant_ref(&mut self, value: GcRef) -> OpRef {
        let pos = self.reserve_const_ref();
        self.seed_constant(pos, Value::Ref(value));
        pos
    }

    pub fn make_constant_float(&mut self, value: f64) -> OpRef {
        let pos = self.reserve_const_ref();
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
            Value::Float(f) => Some(*f),
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
        self.get_constant(opref).cloned()
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
        if !resolved.is_constant() {
            return None;
        }
        let gcref = match self.get_constant_box(arg0)? {
            Value::Ref(r) => r,
            _ => return None,
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

    // ── info.py: per-OpRef pointer info ──

    /// Read the forwarded `PtrInfo` slot for an OpRef without synthesizing
    /// `ConstPtrInfo` for constant pointers. Callers that need the
    /// RPython-faithful lookup (which also sees constants as
    /// `ConstPtrInfo`) should use `getptrinfo` instead.
    pub fn get_ptr_info(&self, opref: OpRef) -> Option<&PtrInfo> {
        use crate::optimizeopt::info::Forwarded;
        let r = self.get_box_replacement(opref);
        match self.forwarded.get(r.0 as usize)? {
            Forwarded::Info(info) => Some(info),
            _ => None,
        }
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
            if let Some(&table_tp) = self.value_types.get(&resolved.0) {
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
        if let Some(&tp) = self.value_types.get(&resolved.0) {
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
        let raw = opref.0;
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
            if let Some(majit_ir::Type::Ref) =
                self.constant_types_for_numbering.get(&resolved.0).copied()
            {
                return Some(std::borrow::Cow::Owned(PtrInfo::Constant(majit_ir::GcRef(
                    *bits as usize,
                ))));
            }
            // Plain integer ConstInt (no raw-pointer marker) → upstream
            // would never reach this branch from a properly-typed call
            // site; majit returns None to mirror that "no info".
            return None;
        }
        // fw = op.get_forwarded()
        // if isinstance(fw, IntBound): return None  →  get_ptr_info
        //   only returns Some for Forwarded::Info(PtrInfo). An int-typed
        //   box that holds Forwarded::IntBound returns None here, matching
        //   the upstream early-return on IntBound forwarding.
        // if fw is not None: assert isinstance(fw, AbstractRawPtrInfo); return fw
        //   AbstractRawPtrInfo ↔ PtrInfo::VirtualRawBuffer / VirtualRawSlice
        //   in majit (see is_raw_ptr).
        self.get_ptr_info(resolved).map(std::borrow::Cow::Borrowed)
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
            return Some(std::borrow::Cow::Owned(PtrInfo::Constant(*gcref)));
        }
        // fw = op.get_forwarded()
        // if fw is not None: assert isinstance(fw, PtrInfo); return fw
        self.get_ptr_info(resolved).map(std::borrow::Cow::Borrowed)
    }

    /// info.py:880 `getptrinfo(op).get_known_class(cpu)` parity.
    ///
    /// Delegates to `getptrinfo` (which synthesizes `ConstPtrInfo` for
    /// constant Refs) and then `PtrInfo::get_known_class`, so constant
    /// pointers are handled via `cls_of_box` the same way
    /// `Instance` / `Virtual` read their stored `known_class`.
    pub fn get_known_class(&self, opref: OpRef) -> Option<majit_ir::GcRef> {
        self.getptrinfo(opref)?.get_known_class()
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
            return match self.getptrinfo(opref) {
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
            self.get_ptr_info(resolved),
            Some(PtrInfo::VirtualRawBuffer(_)) | Some(PtrInfo::VirtualRawSlice(_))
        )
    }

    /// info.py: getptrinfo(op) — mutable variant.
    pub fn get_ptr_info_mut(&mut self, opref: OpRef) -> Option<&mut PtrInfo> {
        use crate::optimizeopt::info::Forwarded;
        let r = self.get_box_replacement(opref);
        match self.forwarded.get_mut(r.0 as usize)? {
            Forwarded::Info(info) => Some(info),
            _ => None,
        }
    }

    /// info.py: op.set_forwarded(info) — set PtrInfo for an OpRef.
    /// Ensure a PtrInfo exists for the given OpRef. Creates an empty
    /// Instance if none exists, so that set_field can store values.
    pub fn ensure_ptr_info(&mut self, opref: OpRef) {
        if opref.is_constant() {
            return;
        }
        use crate::optimizeopt::info::Forwarded;
        let idx = opref.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        if matches!(self.forwarded[idx], Forwarded::None) {
            self.forwarded[idx] = Forwarded::Info(PtrInfo::instance(None, None));
        }
    }

    /// Set PtrInfo without clearing forwarding.
    /// RPython parity: set PtrInfo at the terminal of opref's forwarding chain.
    /// In RPython, `box.set_forwarded(info)` sets info on the Box directly.
    /// `get_box_replacement(box)` then returns the terminal Box which has the info.
    /// In majit, we follow the Op chain to the terminal OpRef and set Info there.
    fn ensure_ptr_info_preserve_forwarding(&mut self, opref: OpRef, info: PtrInfo) {
        use crate::optimizeopt::info::Forwarded;
        let terminal = self.get_box_replacement(opref);
        if terminal.is_constant() {
            return;
        }
        let idx = terminal.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        if matches!(self.forwarded[idx], Forwarded::None) {
            self.forwarded[idx] = Forwarded::Info(info);
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
        let gcref = match self.getptrinfo(opref).as_deref() {
            Some(PtrInfo::Constant(g)) => *g,
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
        let gcref = match self.getptrinfo(opref).as_deref() {
            Some(PtrInfo::Constant(g)) => *g,
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
        let gcref = match self.getptrinfo(opref).as_deref() {
            Some(PtrInfo::Constant(g)) => *g,
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
        // info.py:203-211 AbstractStructPtrInfo.setfield: mutate `_fields`.
        let mut handle = self.ensure_ptr_info_arg0(op);
        if let Some(pi) = handle.as_mut() {
            pi.setfield(field_idx, value);
        }
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
        // info.py: ArrayPtrInfo.setitem: mutate `_items`.
        let mut handle = self.ensure_ptr_info_arg0(op);
        if let Some(pi) = handle.as_mut() {
            pi.setitem(index, value);
        }
    }

    /// RPython set_forwarded parity: setting Info on a box REPLACES
    /// any forwarding. In pyre, ptr_info and forwarding must be
    /// mutually exclusive — the last writer wins.
    /// optimizer.py:437-448: make_nonnull — record that a Ref box is nonnull.
    /// Only sets NonNull if no existing PtrInfo is present.
    pub fn make_nonnull(&mut self, opref: OpRef) {
        let resolved = self.get_box_replacement(opref);
        if resolved.is_constant() {
            return;
        }
        // optimizer.py:446: if opinfo is not None: assert opinfo.is_nonnull(); return
        if self.get_ptr_info(resolved).is_some() {
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
    pub fn ensure_ptr_info_arg0<'s>(&'s mut self, op: &Op) -> EnsuredPtrInfo<'s> {
        use crate::optimizeopt::info::Forwarded;
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
                Some(Value::Ref(g)) => *g,
                Some(Value::Int(bits)) => majit_ir::GcRef(*bits as usize),
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
        // `self.forwarded[idx]` even on the construction branch (which
        // never executes when we early-returned). To stay close to PyPy's
        // single-`opinfo` shape we read the slot immutably with
        // `get_ptr_info` to compute `last_guard_pos`, drop that read, and
        // then either re-borrow mutably for the early return or fall
        // through to the upgrade.
        if let Some(
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
            | PtrInfo::Str(_),
        ) = self.get_ptr_info(arg0)
        {
            // optimizer.py:469: return opinfo. The immutable borrow above
            // ends at the `if let` brace, freeing `self.forwarded[idx]`
            // for the mutable re-borrow below.
            let idx = arg0.0 as usize;
            let info = match &mut self.forwarded[idx] {
                Forwarded::Info(info) => info,
                other => unreachable!(
                    "ensure_ptr_info_arg0: forwarded[{}] changed under us: {:?}",
                    idx, other
                ),
            };
            return EnsuredPtrInfo::Forwarded(info);
        }
        let last_guard_pos = if let Some(opinfo) = self.get_ptr_info(arg0) {
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
        let idx = arg0.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        self.forwarded[idx] = Forwarded::Info(new_info);
        // optimizer.py:499: return opinfo — re-borrow the freshly-installed
        // PtrInfo so the caller can mutate it via Forwarded variant methods.
        let info = match &mut self.forwarded[idx] {
            Forwarded::Info(info) => info,
            _ => unreachable!(),
        };
        EnsuredPtrInfo::Forwarded(info)
    }

    /// optimizer.py:453-459: make_nonnull_str — record StrPtrInfo on a string box.
    /// Only sets if no existing PtrInfo is present or existing is not StrPtrInfo.
    pub fn make_nonnull_str(&mut self, opref: OpRef, mode: u8) {
        let resolved = self.get_box_replacement(opref);
        if resolved.is_constant() {
            return;
        }
        if let Some(PtrInfo::Str(_)) = self.get_ptr_info(resolved) {
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
        use crate::optimizeopt::info::Forwarded;
        let r = self.get_box_replacement(opref);
        let slot = self.forwarded.get_mut(r.0 as usize)?;
        match slot {
            Forwarded::Info(_) => {
                let old = std::mem::replace(slot, Forwarded::None);
                match old {
                    Forwarded::Info(info) => Some(info),
                    _ => unreachable!(),
                }
            }
            _ => None,
        }
    }

    pub fn set_ptr_info(&mut self, opref: OpRef, info: PtrInfo) {
        if opref.is_constant() {
            return;
        }
        use crate::optimizeopt::info::Forwarded;
        let idx = opref.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        self.forwarded[idx] = Forwarded::Info(info);
    }

    /// optimizer.py: replace_op_with(old, new_op, ctx)
    /// Replace old opref AND emit the new op.
    pub fn replace_op_with(&mut self, old: OpRef, new_op: Op) -> OpRef {
        let new_ref = self.emit(new_op);
        self.replace_op(old, new_ref);
        new_ref
    }

    /// Check if an opref has been replaced (forwarded).
    pub fn is_replaced(&self, opref: OpRef) -> bool {
        use crate::optimizeopt::info::Forwarded;
        let idx = opref.0 as usize;
        if idx < self.forwarded.len() {
            matches!(self.forwarded[idx], Forwarded::Op(_))
        } else {
            false
        }
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
        let opref = OpRef(10_000);
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
        let opref = OpRef(10_001);
        ctx.seed_constant(opref, Value::Int(0x1234_5678));
        ctx.constant_types_for_numbering.insert(opref.0, Type::Ref);
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
        let opref = OpRef(10_002);
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
        let opref = OpRef(10_003);
        ctx.seed_constant(opref, Value::Int(0));
        ctx.constant_types_for_numbering.insert(opref.0, Type::Ref);
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
        let opref = OpRef(10_009);
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
        let opref = OpRef(10_004);
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
        let ref_op = OpRef(10_005);
        let int_op = OpRef(10_006);
        let addr: usize = 0xfeed_face;
        ctx.seed_constant(ref_op, Value::Ref(GcRef(addr)));
        ctx.seed_constant(int_op, Value::Int(addr as i64));
        ctx.constant_types_for_numbering.insert(int_op.0, Type::Ref);

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
            let ref_null = OpRef(10_007);
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
            let int_null = OpRef(10_008);
            ctx.seed_constant(int_null, Value::Int(0));
            ctx.constant_types_for_numbering
                .insert(int_null.0, Type::Ref);
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
        let opref = OpRef(10_010);
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
        let parent = OpRef(10_010);
        let slice = OpRef(10_011);

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
        let opref = OpRef(10_012);

        ctx.make_nonnull_str(opref, 0);

        match ctx.get_ptr_info(opref) {
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
    use majit_ir::{Descr, DescrRef, GcRef, Op, OpCode, OpRef, SizeDescr, Value};
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
        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], descr);
        op.pos = OpRef(1);
        op
    }

    fn array_op() -> Op {
        let descr: DescrRef = Arc::new(TestSizeDescr {
            index: 7,
            is_object: false,
        });
        let mut op = Op::with_descr(OpCode::ArraylenGc, &[OpRef(0)], descr);
        op.pos = OpRef(1);
        op
    }

    /// optimizer.py:465-466: `if arg0.is_constant(): return info.ConstPtrInfo(arg0)`
    /// Constant `Value::Ref` arg0 → `EnsuredPtrInfo::Constant(gcref)`.
    #[test]
    fn ensure_ptr_info_arg0_returns_constant_for_value_ref() {
        let mut ctx = OptContext::with_num_inputs(4, 1);
        ctx.seed_constant(OpRef(0), Value::Ref(GcRef(0xdead_beef)));
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
        let mut ctx = OptContext::with_num_inputs(4, 1);
        ctx.seed_constant(OpRef(0), Value::Int(1));
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
        let mut ctx = OptContext::with_num_inputs(4, 1);
        ctx.seed_constant(OpRef(0), Value::Ref(GcRef(0xC0FE)));
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
            let mut op = Op::with_descr(OpCode::Strlen, &[OpRef(0)], descr);
            op.pos = OpRef(1);
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
        let mut ctx = OptContext::with_num_inputs(4, 1);
        ctx.seed_constant(OpRef(0), Value::Ref(GcRef(0x1234)));
        let op = {
            let descr: DescrRef = Arc::new(TestSizeDescr {
                index: 1,
                is_object: false,
            });
            let mut op = Op::with_descr(OpCode::Strlen, &[OpRef(0)], descr);
            op.pos = OpRef(1);
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
        let mut ctx = OptContext::with_num_inputs(4, 1);
        // Keep a strong reference to the parent alive for the duration
        // of the test: `SimpleFieldDescr::parent_descr` is a
        // `Weak<DescrRef>` (breaks the cycle between SizeDescr.all_fielddescrs
        // and FieldDescr.parent_descr), so the test must hold the parent
        // Arc until `get_parent_descr()` has been called.
        let _parent = struct_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let pi = info.as_mut().expect("Forwarded variant expected");
        assert!(matches!(pi, PtrInfo::Struct(_)));
    }

    /// optimizer.py:480-484 GETFIELD branch with `parent_descr.is_object() == true`
    /// → `info.InstancePtrInfo(parent_descr)`.
    #[test]
    fn ensure_ptr_info_arg0_constructs_instance_for_object_field() {
        let mut ctx = OptContext::with_num_inputs(4, 1);
        let _parent = instance_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let pi = info.as_mut().expect("Forwarded variant expected");
        assert!(matches!(pi, PtrInfo::Instance(_)));
    }

    /// optimizer.py:485-487 ARRAYLEN_GC branch → `info.ArrayPtrInfo(descr)`.
    /// The PyPy primitive returns the same arrayinfo across calls so
    /// callers can read `arrayinfo.getlenbound(None)` directly. The Rust
    /// port mirrors that and the `getlenbound` call resolves to the
    /// pre-installed `nonnegative` lenbound on the freshly-built ArrayPtrInfo.
    #[test]
    fn ensure_ptr_info_arg0_arraylen_returns_array_with_nonnegative_lenbound() {
        let mut ctx = OptContext::with_num_inputs(4, 1);
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
        let mut ctx = OptContext::with_num_inputs(4, 1);
        ctx.seed_constant(OpRef(0), Value::Ref(GcRef(0xfeed)));
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
        let mut ctx = OptContext::with_num_inputs(4, 1);
        let op = array_op();
        // First call constructs the ArrayPtrInfo and tightens the lenbound
        // through the helper.
        {
            let mut info = ctx.ensure_ptr_info_arg0(&op);
            if let Some(PtrInfo::Array(arr)) = info.as_mut() {
                let _ = arr.lenbound.make_gt_const(7);
            } else {
                panic!("expected fresh ArrayPtrInfo");
            }
        }
        // Second call returns the same ArrayPtrInfo (lenbound preserved).
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        match info.as_mut() {
            Some(PtrInfo::Array(ArrayPtrInfo { lenbound, .. })) => {
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
        let mut ctx = OptContext::with_num_inputs(4, 1);
        // Pre-install a NonNullPtrInfo with a specific last_guard_pos.
        ctx.set_ptr_info(OpRef(0), PtrInfo::NonNull { last_guard_pos: 7 });
        let _parent = struct_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        match info.as_mut() {
            Some(pi @ PtrInfo::Struct(_)) => {
                assert_eq!(pi.last_guard_pos(), Some(7));
            }
            other => panic!("expected upgraded Struct, got {other:?}"),
        }
    }

    /// optimizer.py:474 assertion: an unexpected forwarded info shape (e.g.
    /// a `Forwarded::Op` redirect that resolved to a non-PtrInfo state)
    /// must NOT silently overwrite. We seed an `Instance` PtrInfo, then
    /// hand it a field op with a different parent — the early-return path
    /// hits, and the existing Instance is returned without overwrite.
    #[test]
    fn ensure_ptr_info_arg0_does_not_overwrite_existing_instance() {
        let mut ctx = OptContext::with_num_inputs(4, 1);
        ctx.set_ptr_info(
            OpRef(0),
            PtrInfo::instance(Some(instance_parent_descr()), Some(GcRef(0xc0de))),
        );
        let op = field_op_with_parent(struct_parent_descr());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        match info.as_mut() {
            Some(PtrInfo::Instance(_)) => {} // unchanged
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
        let opref = OpRef(20_000);
        ctx.seed_constant(opref, Value::Ref(GcRef(0xdead_beef)));
        let _ = ctx.getintbound(opref);
    }

    #[test]
    #[should_panic]
    fn setintbound_rejects_non_int_boxes() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef(20_001);
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
        let mut ctx = OptContext::with_num_inputs(16, 2);
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef(0), OpRef(1)],
            &[OpRef(7), OpRef(8)],
            &[],
        );

        let mut replay_op = Op::new(OpCode::IntAdd, &[OpRef(7), OpRef(8)]);
        replay_op.pos = OpRef(14);
        let pop = crate::optimizeopt::info::PreambleOp {
            op: OpRef(14),
            resolved: OpRef(41),
            invented_name: false,
            preamble_op: replay_op,
        };

        let forced = ctx.force_op_from_preamble_op(&pop);
        assert_eq!(forced, OpRef(41));

        let sp = ctx
            .build_imported_short_preamble()
            .expect("imported short preamble builder should exist");
        assert_eq!(sp.ops.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(sp.ops[0].op.args.as_slice(), &[OpRef(7), OpRef(8)]);
        assert_eq!(sp.ops[0].op.pos, OpRef(14));
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
        let source = OpRef(12);
        let target = OpRef(21);
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
