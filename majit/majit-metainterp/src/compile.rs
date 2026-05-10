//! Trace compilation helpers.
//!
//! Mirrors RPython's `compile.py`: guard metadata building, exit layout
//! management, backend layout merging, trace post-processing (unboxing),
//! and the ResumeGuard-descriptor class hierarchy
//! (`compile.py:730-940 AbstractResumeGuardDescr` →
//! `ResumeGuardDescr` → `{ResumeAtPositionDescr, ResumeGuardForcedDescr,
//! ResumeGuardExcDescr, CompileLoopVersionDescr}`,
//! `ResumeGuardCopiedDescr` / `ResumeGuardCopiedExcDescr`,
//! `invent_fail_descr_for_op`).  Pyre wraps upstream's direct attribute
//! assignments in `UnsafeCell` so the optimizer can mutate the descr
//! through `Arc<dyn Descr>` shared ownership; the only structural
//! difference vs upstream is the use of cells, not a separate module.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use smallvec::smallvec;

use majit_backend::{
    Backend, BackendError, CompiledLoopToken, CompiledTraceInfo, ExitFrameLayout,
    ExitRecoveryLayout, FailDescrLayout, JitCellToken, TerminalExitLayout,
};
use majit_ir::{
    AccumInfo, Const, Descr, DescrRef, FailDescr, GcRef, GuardPendingFieldEntry, InputArg, Op,
    OpCode, OpRef, RdVirtualInfo, Type, Value,
};

use crate::blackhole::ExceptionState;
use crate::pyjitpl::{CompiledTrace, StoredExitLayout};
use crate::resume::{
    ResumeData, ResumeDataLoopMemo, ResumeDataVirtualAdder, ResumeFrameLayoutSummary,
    ResumeLayoutSummary, ResumeValueSource,
};

/// `compile.py:166-169` `make_jitcell_token(jitdriver_sd)`.
///
/// ```python
/// def make_jitcell_token(jitdriver_sd):
///     jitcell_token = JitCellToken()
///     jitcell_token.outermost_jitdriver_sd = jitdriver_sd
///     return jitcell_token
/// ```
///
/// Pyre takes `number` separately because `JitCellToken::new(number)`
/// requires a unique number allocated up-front (RPython relies on
/// `__init__` increment of `_GLOBAL_NUMBER_OF_TOKENS`; pyre uses
/// `WarmEnterState::alloc_token_number` for the same purpose).  The
/// `outermost_jitdriver_index: Option<usize>` is the pyre analog of
/// RPython's `outermost_jitdriver_sd` attribute (pyre stores a slot
/// index into `metainterp_sd.jitdrivers_sd` instead of a direct
/// reference, since the descriptor lives in another crate).
///
/// Returns the canonical `Arc<JitCellToken>` immediately.  Callers can
/// mutate the token before it is cloned with `Arc::get_mut`, then pass
/// the same Arc through `compile_loop` (`compile.py:266`) →
/// `attach_procedure_to_interp` (`compile.py:1019`) →
/// `MemoryManager.keep_loop_alive` (`compile.py:567`/`:1149`).
pub fn make_jitcell_token(number: u64, jd_index: Option<usize>) -> Arc<JitCellToken> {
    let mut token = JitCellToken::new(number);
    token.outermost_jitdriver_index = jd_index;
    Arc::new(token)
}

/// `compile.py:180-181` `wref = weakref.ref(original_jitcell_token);
/// clt.loop_token_wref = wref` parity. Must be called *after* every
/// `Arc::get_mut(&mut token)` mutation has settled, because creating
/// the `Weak` increments the weak count and `Arc::get_mut` requires
/// `weak_count == 0`. Practically this means: configure the token
/// fields first (`configure_loop_token_for_driver`, `inputarg_types`
/// etc.), then call this helper before the token is published into
/// `compiled_loops` / `attach_procedure_with_redirect`.
pub fn wire_clt_loop_token_wref(token: &Arc<JitCellToken>) {
    if let Some(clt) = token.compiled_loop_token.as_ref() {
        clt.set_loop_token_wref(Arc::downgrade(token));
    }
}

/// Resolve the type of an OpRef in guard fail_args.
/// OpRef::NONE is a virtual slot placeholder (null GC ref).
/// RPython resume.py:389-452 parity: guard metadata sees only the
/// Box.type values available at the guard's numbering point, not later
/// operations in the same trace. `value_types` is the incremental
/// guard-point map maintained by `build_guard_metadata`.
fn fail_arg_type(opref: &OpRef, value_types: &HashMap<u32, Type>) -> Type {
    if *opref == OpRef::NONE {
        return Type::Ref;
    }
    if opref.is_constant() {
        value_types.get(&opref.raw()).copied().unwrap_or(Type::Int)
    } else {
        value_types.get(&opref.raw()).copied().unwrap_or(Type::Ref)
    }
}

/// Derive slot_types from ExitValueSourceLayout + exit_types.
/// RPython resume.py:1410 load_next_value_of_type parity:
/// slot type is the DECLARED type of the variable.
/// ExitValue(idx) → exit_types[idx]; Constant → Int; others → Ref.
fn derive_slot_types(
    slots: &[majit_backend::ExitValueSourceLayout],
    exit_types: &[Type],
) -> Vec<Type> {
    slots
        .iter()
        .map(|slot| match slot {
            majit_backend::ExitValueSourceLayout::ExitValue(idx) => {
                exit_types.get(*idx).copied().unwrap_or(Type::Ref)
            }
            majit_backend::ExitValueSourceLayout::Constant(_) => Type::Int,
            _ => Type::Ref,
        })
        .collect()
}

// ── Compilation result types (compile.py) ───────────────────────────────

/// Static exit metadata for a compiled guard or finish point.
#[derive(Debug, Clone)]
pub struct CompiledExitLayout {
    /// compile.py:186 rd_loop_token: the green_key of the compiled loop
    /// that owns this guard. Used by handle_fail to find the owning
    /// compiled entry without scanning all entries.
    pub rd_loop_token: u64,
    pub trace_id: u64,
    pub fail_index: u32,
    pub source_op_index: Option<usize>,
    pub exit_types: Vec<Type>,
    pub is_finish: bool,
    pub gc_ref_slots: Vec<usize>,
    pub force_token_slots: Vec<usize>,
    pub recovery_layout: Option<ExitRecoveryLayout>,
    pub resume_layout: Option<ResumeLayoutSummary>,
    /// compile.py:853 `ResumeGuardDescr` storage handle — shared
    /// pool with rd_numb / rd_consts / rd_virtuals / rd_pendingfields.
    pub storage: Option<std::sync::Arc<crate::resume::ResumeStorage>>,
}

/// Typed result from running compiled code.
pub struct CompileResult<'a, M> {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub meta: &'a M,
    pub fail_index: u32,
    pub trace_id: u64,
    /// `cpu.get_latest_descr(deadframe)` (`history.py:125`) — the
    /// backend-side guard descr recovered from the failing frame.  Pyre
    /// currently has split metainterp/backend descr objects, so this is
    /// the runtime descr carrying the same `rd_loop_token_clt` /
    /// `fail_index_per_trace` identity used for bridge routing.
    pub descr_arc: std::sync::Arc<dyn majit_ir::FailDescr>,
    pub is_finish: bool,
    /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity:
    /// true when the FINISH descriptor was
    /// `sd.exit_frame_with_exception_descr_ref` (emitted via
    /// `pyjitpl.py:3238-3245 compile_exit_frame_with_exception`).
    /// jitdriver routes this to `jitexc.ExitFrameWithExceptionRef`.
    pub is_exit_frame_with_exception: bool,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
    /// compile.py:741-745: ResumeGuardDescr.status read at guard failure.
    pub status: u64,
}

/// Raw (lightweight) result from running compiled code.
pub struct RawCompileResult<'a, M> {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub meta: &'a M,
    pub fail_index: u32,
    pub trace_id: u64,
    /// `cpu.get_latest_descr(deadframe)` (`history.py:125`,
    /// `compile.py:701`) — runtime descr Arc owning this exit.  Always
    /// set: routed through `Backend::get_latest_descr_arc` via
    /// `RawExecResult::descr_arc`, so FINISH / `DoneWithThisFrame*` /
    /// `ExitFrameWithExceptionDescrRef` singletons return their global
    /// Arc identity instead of `None`.  Bridge consumers
    /// (`start_bridge_tracing`, `_trace_and_compile_from_bridge`) read
    /// `rd_loop_token_clt` / `fail_index_per_trace` directly from this.
    pub descr_arc: std::sync::Arc<dyn majit_ir::FailDescr>,
    pub is_finish: bool,
    /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity —
    /// mirrors `CompileResult::is_exit_frame_with_exception`.
    pub is_exit_frame_with_exception: bool,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
    /// compile.py:741-745: ResumeGuardDescr.status read at guard failure.
    pub status: u64,
}

/// Terminal exit layout for a FINISH or JUMP op.
#[derive(Debug, Clone)]
pub struct CompiledTerminalExitLayout {
    pub op_index: usize,
    pub exit_layout: CompiledExitLayout,
}

/// Full trace compilation layout with all exits.
#[derive(Debug, Clone)]
pub struct CompiledTraceLayout {
    pub trace_id: u64,
    pub exit_layouts: Vec<CompiledExitLayout>,
    pub terminal_exit_layouts: Vec<CompiledTerminalExitLayout>,
}

/// Artifacts extracted from a backend DeadFrame.
#[derive(Debug, Clone)]
pub struct DeadFrameArtifacts {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
}

// ── Compilation helper functions ────────────────────────────────────────

/// Build guard metadata for a compiled trace.
///
/// The backend numbers every guard and finish in a single exit table, so this
/// helper mirrors that numbering and records only the guard entries that need
/// resume data plus the corresponding op index for blackhole fallback.
/// resume.py ResumeDataLoopMemo parity: `constant_types` maps OpRef.0 → Type.
/// Used by `getconst` to encode virtual field constants as TAGINT/TAGCONST
/// instead of TAGBOX.
pub(crate) fn build_guard_metadata(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
    pc: u64,
    constant_types: &std::collections::HashMap<u32, Type>,
    callinfocollection: Option<&majit_ir::CallInfoCollection>,
) -> (
    HashMap<u32, crate::resume::ResumeLayoutSummary>,
    HashMap<u32, StoredExitLayout>,
) {
    let mut result = HashMap::new();
    let mut exit_layouts = HashMap::new();
    let mut fail_index = 0u32;
    let mut resume_memo = ResumeDataLoopMemo::new();
    // RPython box.type parity: type lookup is `inputarg.tp` /
    // `constant.type` / `op.type_` direct attribute access — pyre
    // gathers these into OpTypeIndex since OpRef carries no type.
    //
    // RPython resume.py:389-520 captures each guard's livebox types at
    // numbering time, so a guard at trace position K only sees ops
    // that have already been emitted (positions <= K). pyre's flat
    // `OpRef(u32)` namespace can re-use the same numeric position
    // across phases (Phase 2 `shift_back` remaps fresh OpRefs onto
    // Phase 1's inputarg slots), so a global `type_index.opref_type`
    // would let later op result types leak back into earlier guards.
    // Mirror main's incremental walk: maintain `value_types` seeded
    // from inputargs + constants, then insert each op's result type
    // before consulting it.
    let mut value_types: HashMap<u32, Type> =
        inputargs.iter().map(|arg| (arg.index, arg.tp)).collect();
    for (&idx, &tp) in constant_types {
        value_types.entry(idx).or_insert(tp);
    }

    for (op_idx, op) in ops.iter().enumerate() {
        if !op.pos.is_none() && op.result_type() != Type::Void {
            value_types.insert(op.pos.raw(), op.result_type());
        }

        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;
        if !is_guard && !is_finish {
            continue;
        }

        if is_guard {
            // F.5-orthodox.1: drop the `guard_op_indices` HashMap.
            // Every reader is now routed through descr-side identity
            // (`op.descr.as_fail_descr().fail_index_per_trace()`
            // forward; op-position lookup
            // `trace.ops[guard_index].descr.as_fail_descr()
            // .fail_index_per_trace()` reverse).  Mirrors RPython's
            // `compile.py:184 op.getdescr()` predicate where the descr
            // identity replaces the side table entirely.
            //
            // The readers compare against `fail_index_per_trace()`
            // (this slot, set by `set_fail_index_per_trace` below),
            // not `fail_index()` (which is the global
            // `alloc_fail_index()` id at `descr.rs:1065` — a separate
            // structural slot the readers do not consult).
            //
            // Pyre-only: stamp the per-trace `fail_index` onto the
            // metainterp ResumeGuardDescr so `(trace_id, fail_index)`
            // lookups can resolve through the descr Arc directly.
            // Skip non-resume FailDescrs (whose
            // `set_fail_index_per_trace` panics by default).
            if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_fail_descr()) {
                if op
                    .descr
                    .as_ref()
                    .map_or(false, |d| d.is_resume_guard() || d.is_resume_guard_copied())
                {
                    fd.set_fail_index_per_trace(fail_index);
                }
            }
        }

        // RPython Box.type parity: each fail-arg's type is `livebox.type`,
        // captured at numbering time inside `store_final_boxes_in_guard`
        // (resume.py:520, optimizer.py:728). majit stores that snapshot on
        // the descr's `fail_arg_types()` (post-numbering, post-virtual-
        // materialization) and mirrors it to `op.fail_arg_types` for
        // sharing-path guards (mod.rs:3068-3088). After the codex #3 fix
        // (tracer-stage descr=None, dbd452a640c), every guard's descr is
        // minted by `store_final_boxes_in_guard` carrying the
        // post-numbering type vector, so descr-first priority no longer
        // exposes stale tracer types. Fall back to `op.fail_arg_types`
        // and finally the incremental guard-point `value_types` map.
        let descr_types = op
            .descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map(|fd| fd.fail_arg_types());
        let exit_types: Vec<Type> = if is_finish {
            // FINISH ops are always emitted with one of the
            // `_DoneWithThisFrameDescr` family (compile.py:623-672) or
            // `ExitFrameWithExceptionDescrRef`, all of which carry a
            // fixed `fail_arg_types` (Void → empty, Int → [Int],
            // Ref → [Ref], Float → [Float]). Prefer the descr's
            // typing — it matches RPython where `handle_fail` reads
            // `cpu.get_*_value(deadframe, 0)` keyed by the descr
            // class, not by per-arg inference.
            if let Some(types) = descr_types {
                if types.len() == op.args.len() {
                    types.to_vec()
                } else {
                    // Arity mismatch (synthetic test ops without a
                    // type-shaped descr): reconstruct per-arg from the
                    // incremental `value_types`. Production FINISH always
                    // matches the descr arity.
                    op.args
                        .iter()
                        .map(|opref| value_types.get(&opref.raw()).copied().unwrap_or(Type::Int))
                        .collect()
                }
            } else {
                // No descr — synthetic test FINISH only.
                op.args
                    .iter()
                    .map(|opref| value_types.get(&opref.raw()).copied().unwrap_or(Type::Int))
                    .collect()
            }
        } else if let Some(ref fail_args) = op.fail_args {
            // `store_final_boxes_in_guard` (resume.py:397) writes the
            // reduced liveboxes' types authoritatively. Prefer the descr's
            // fail_arg_types (single source of truth, matches RPython
            // `ResumeGuardDescr.fail_arg_types`); fall back to op-level
            // `fail_arg_types` on sharing-path (no descr); fall back to
            // per-arg reconstruction via guard-point `value_types` when arity
            // mismatches.
            let fa_types = op.fail_arg_types.as_ref();
            if let Some(types) = descr_types {
                if types.len() == fail_args.len() {
                    types.to_vec()
                } else {
                    fail_args
                        .iter()
                        .enumerate()
                        .map(|(i, opref)| {
                            if let Some(&tp) = types.get(i) {
                                return tp;
                            }
                            if let Some(fa) = fa_types {
                                if let Some(&tp) = fa.get(i) {
                                    return tp;
                                }
                            }
                            if let Some(&tp) = value_types.get(&opref.raw()) {
                                return tp;
                            }
                            fail_arg_type(opref, &value_types)
                        })
                        .collect()
                }
            } else if let Some(types) = fa_types {
                if types.len() == fail_args.len() {
                    types.clone()
                } else {
                    fail_args
                        .iter()
                        .enumerate()
                        .map(|(i, opref)| {
                            if let Some(&tp) = types.get(i) {
                                return tp;
                            }
                            if let Some(&tp) = value_types.get(&opref.raw()) {
                                return tp;
                            }
                            fail_arg_type(opref, &value_types)
                        })
                        .collect()
                }
            } else {
                fail_args
                    .iter()
                    .map(|opref| {
                        if let Some(&tp) = value_types.get(&opref.raw()) {
                            return tp;
                        }
                        fail_arg_type(opref, &value_types)
                    })
                    .collect()
            }
        } else if let Some(dt) = descr_types {
            dt.to_vec()
        } else if let Some(ref types) = op.fail_arg_types {
            types.clone()
        } else {
            inputargs.iter().map(|arg| arg.tp).collect()
        };
        let resume_layout;
        let storage = if is_guard {
            let mut builder = ResumeDataVirtualAdder::new();

            // store_final_boxes parity: when rd_numb is present, fail_args
            // are normalized to liveboxes only (no constants/virtuals).
            // Build resume_layout from rd_numb so that TAGCONST/TAGINT
            // slots produce Constant entries in the reconstructed state.
            // Multi-frame: push_frame per frame with correct pc.
            //
            // `resolved_rd_*` chases `descr.prev` (compile.py:849
            // ResumeGuardCopiedDescr.get_resumestorage) so a shared guard
            // reads the donor's resume data without an owned copy.
            if let (Some(rd_numb_bytes), Some(rd_consts_data)) =
                (op.resolved_rd_numb(), op.resolved_rd_consts())
            {
                use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};
                let fvc = majit_ir::resumedata::get_frame_value_count_fn();
                let fvc_ref: Option<&dyn Fn(i32, i32) -> usize> =
                    fvc.as_ref().map(|f| f as &dyn Fn(i32, i32) -> usize);
                let (_num_failargs, vable_values, _vref_values, frames) =
                    rebuild_from_numbering(rd_numb_bytes, rd_consts_data, &exit_types, fvc_ref);
                let vable_array = vable_values
                    .iter()
                    .map(|val| match val {
                        RebuiltValue::Box(idx, _) => ResumeValueSource::FailArg(*idx),
                        RebuiltValue::Const(c) => ResumeValueSource::Constant(*c),
                        RebuiltValue::Virtual(vidx) => ResumeValueSource::Virtual(*vidx),
                        RebuiltValue::Unassigned => ResumeValueSource::Unavailable,
                    })
                    .collect::<Vec<_>>();
                builder.set_vable_array(vable_array);
                let add_slot =
                    |builder: &mut ResumeDataVirtualAdder, slot_idx: usize, val: &RebuiltValue| {
                        match val {
                            RebuiltValue::Box(idx, _) => {
                                builder.map_slot(slot_idx, *idx);
                            }
                            RebuiltValue::Const(c) => {
                                builder.set_slot_constant(slot_idx, *c);
                            }
                            RebuiltValue::Virtual(vidx) => {
                                builder.set_slot_virtual(slot_idx, *vidx);
                            }
                            RebuiltValue::Unassigned => {
                                builder.set_slot_uninitialized(slot_idx);
                            }
                        }
                    };
                // After `opencoder.py:217` `framestack.reverse()` parity,
                // both rd_numb and `ResumeData.frames` agree on outermost-
                // first ordering, so push frames in stream order.
                //
                // RPython resume.py keeps vable_array/vref_array/framestack
                // as separate sections. Do not merge vable_array entries into
                // the innermost frame slots here.
                for frame in frames.iter() {
                    builder.push_frame(frame.jitcode_index, frame.pc as u64);
                    let mut slot_idx = 0usize;
                    for val in &frame.values {
                        add_slot(&mut builder, slot_idx, val);
                        slot_idx += 1;
                    }
                }
            } else {
                // No rd_numb: single frame, 1:1 mapping (fail_args[i] → state[i]).
                builder.push_frame(0, pc);
                let num_slots = op
                    .fail_args
                    .as_ref()
                    .map(|fa| fa.len())
                    .unwrap_or(exit_types.len());
                for slot_idx in 0..num_slots {
                    builder.map_slot(slot_idx, slot_idx);
                }
            }

            let layout = resume_memo.encode_shared(&builder.build()).layout_summary();
            resume_layout = Some(layout.clone());
            // compile.py:853 `ResumeGuardDescr` storage — build the shared
            // Arc once from the guard op's `rd_*` fields so every reader
            // (StoredExitLayout, bridge retrace, blackhole resume, GC
            // root walker) observes the same pool.  Resolve through
            // descr.prev (`resolved_rd_*` chases the copied-descr chain)
            // so a sharing-path guard's ResumeStorage points at the same
            // byte stream the donor was built from (RPython compile.py:832
            // ResumeGuardCopiedDescr).
            let storage_for_guard = if let Some(numb) = op.resolved_rd_numb() {
                Some(crate::resume::ResumeStorage::new(
                    numb.to_vec(),
                    op.resolved_rd_consts()
                        .map(<[Const]>::to_vec)
                        .unwrap_or_default(),
                    op.resolved_rd_virtuals()
                        .map(<[std::rc::Rc<majit_ir::RdVirtualInfo>]>::to_vec)
                        .unwrap_or_default(),
                    op.resolved_rd_pendingfields()
                        .map(<[majit_ir::GuardPendingFieldEntry]>::to_vec)
                        .unwrap_or_default(),
                ))
            } else {
                None
            };
            result.insert(fail_index, layout);
            storage_for_guard
        } else {
            resume_layout = None;
            None
        };

        // rd_* values are now carried inside `storage` (an
        // `Arc<ResumeStorage>` installed above). They still feed into
        // `recovery_layout` below via the guard op's rd_numb / rd_consts.
        // Sharing-path guards (mod.rs::sharing-guard) own a
        // ResumeGuardCopiedDescr whose `prev` points at the donor;
        // the `resolved_rd_*` helpers chase that descr-side pointer
        // (compile.py:849 get_resumestorage).
        let recovery_layout = if op.resolved_rd_numb().is_some() {
            // Consumer switchover path: rd_numb contains the full frame encoding.
            // Build recovery_layout from rd_numb + rd_virtuals.
            use majit_backend::{ExitRecoveryLayout, ExitValueSourceLayout};
            let (num_failargs, vable_layout, vref_layout, frames_layout) =
                if let (Some(rd_numb_bytes), Some(rd_consts_data)) =
                    (op.resolved_rd_numb(), op.resolved_rd_consts())
                {
                    use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};
                    let fvc = majit_ir::resumedata::get_frame_value_count_fn();
                    let fvc_ref: Option<&dyn Fn(i32, i32) -> usize> =
                        fvc.as_ref().map(|f| f as &dyn Fn(i32, i32) -> usize);
                    let (num_failargs, vable_values, vref_values, frames) =
                        rebuild_from_numbering(rd_numb_bytes, rd_consts_data, &exit_types, fvc_ref);
                    debug_assert!(
                        vref_values.len() & 1 == 0,
                        "vref_values length must be even, got {}",
                        vref_values.len(),
                    );
                    let to_exit_source = |val: &RebuiltValue| match val {
                        RebuiltValue::Box(idx, _) => ExitValueSourceLayout::ExitValue(*idx),
                        RebuiltValue::Virtual(vidx) => ExitValueSourceLayout::Virtual(*vidx),
                        RebuiltValue::Const(c) => ExitValueSourceLayout::Constant(c.as_raw_i64()),
                        RebuiltValue::Unassigned => ExitValueSourceLayout::Uninitialized,
                    };
                    (
                        num_failargs,
                        vable_values.iter().map(to_exit_source).collect::<Vec<_>>(),
                        vref_values.iter().map(to_exit_source).collect::<Vec<_>>(),
                        frames
                            .iter()
                            .map(|frame| {
                                let mut slots = Vec::new();
                                slots.extend(frame.values.iter().map(to_exit_source));
                                let slot_types = derive_slot_types(&slots, &exit_types);
                                majit_backend::ExitFrameLayout {
                                    trace_id: None,
                                    header_pc: Some(frame.pc as u64),
                                    source_guard: None,
                                    pc: frame.pc as u64,
                                    jitcode_index: frame.jitcode_index,
                                    slots,
                                    slot_types: Some(slot_types),
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    (exit_types.len() as i32, vec![], vec![], vec![])
                };
            // Collect slots from ALL frames for virtual target_slot lookup.
            // RPython resolves virtuals across the entire frame stack, not
            // just the innermost frame (resume.py:1410).
            let frame_slots: Vec<ExitValueSourceLayout> = frames_layout
                .iter()
                .flat_map(|frame| frame.slots.iter().cloned())
                .collect();
            // resume.py:576-860 parity: resolve fieldnums tags for recovery.
            // Follow `descr.prev` so sharing-path guards see the donor's
            // const pool (compile.py:849 get_resumestorage).
            let rd_consts_ref = op.resolved_rd_consts().unwrap_or(&[]);
            let resolve_tagged_source = |tagged: i16| -> ExitValueSourceLayout {
                let (val, tagbits) = majit_ir::resumedata::untag(tagged);
                match tagbits {
                    majit_ir::resumedata::TAGBOX => {
                        let idx = if val >= 0 {
                            val as usize
                        } else {
                            (num_failargs + val) as usize
                        };
                        ExitValueSourceLayout::ExitValue(idx)
                    }
                    majit_ir::resumedata::TAGVIRTUAL => {
                        ExitValueSourceLayout::Virtual(val as usize)
                    }
                    majit_ir::resumedata::TAGINT => ExitValueSourceLayout::Constant(val as i64),
                    majit_ir::resumedata::TAGCONST => {
                        let idx = (val - majit_ir::resumedata::TAG_CONST_OFFSET) as usize;
                        let c = rd_consts_ref.get(idx).copied().unwrap_or(Const::Int(0));
                        ExitValueSourceLayout::Constant(c.as_raw_i64())
                    }
                    _ => ExitValueSourceLayout::Constant(0),
                }
            };
            let resolve_fieldnums = |fieldnums: &[i16],
                                     fielddescr_indices: &[u32]|
             -> Vec<(u32, ExitValueSourceLayout)> {
                fieldnums
                    .iter()
                    .enumerate()
                    .map(|(fi, &fnum)| {
                        let fdi = fielddescr_indices.get(fi).copied().unwrap_or(fi as u32);
                        (fdi, resolve_tagged_source(fnum))
                    })
                    .collect()
            };
            // Sharing-path follows `descr.prev` to read the donor's
            // virtual table (compile.py:849 get_resumestorage parity).
            let virtual_layouts: Vec<majit_backend::ExitVirtualLayout> = op
                .resolved_rd_virtuals()
                .map(|entries| {
                    entries
                        .iter()
                        .enumerate()
                        .map(|(vidx, entry_rc)| {
                            let entry: &majit_ir::RdVirtualInfo = entry_rc.as_ref();
                            let target_slot = frame_slots.iter().position(
                                |s| matches!(s, ExitValueSourceLayout::Virtual(v) if *v == vidx),
                            );
                            match entry {
                                majit_ir::RdVirtualInfo::VirtualInfo {
                                    descr,
                                    type_id,
                                    descr_index,
                                    known_class,
                                    fielddescrs,
                                    fieldnums,
                                    descr_size,
                                } => {
                                    let idx: Vec<u32> =
                                        fielddescrs.iter().map(|fd| fd.index).collect();
                                    majit_backend::ExitVirtualLayout::Object {
                                        descr: descr.clone(),
                                        type_id: *type_id,
                                        descr_index: *descr_index,
                                        known_class: *known_class,
                                        fields: resolve_fieldnums(fieldnums, &idx),
                                        target_slot,
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
                                    let idx: Vec<u32> =
                                        fielddescrs.iter().map(|fd| fd.index).collect();
                                    majit_backend::ExitVirtualLayout::Struct {
                                        typedescr: typedescr.clone(),
                                        type_id: *type_id,
                                        descr_index: *descr_index,
                                        fields: resolve_fieldnums(fieldnums, &idx),
                                        target_slot,
                                        fielddescrs: fielddescrs.clone(),
                                        descr_size: *descr_size,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VArrayInfoClear {
                                    arraydescr: _,
                                    descr_index,
                                    kind,
                                    fieldnums,
                                }
                                | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                                    arraydescr: _,
                                    descr_index,
                                    kind,
                                    fieldnums,
                                } => {
                                    let clear = matches!(
                                        entry,
                                        majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
                                    );
                                    let items = fieldnums
                                        .iter()
                                        .map(|&fnum| resolve_tagged_source(fnum))
                                        .collect();
                                    majit_backend::ExitVirtualLayout::Array {
                                        descr_index: *descr_index,
                                        clear,
                                        kind: *kind,
                                        items,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VArrayStructInfo {
                                    arraydescr,
                                    descr_index,
                                    fielddescrs,
                                    size,
                                    fielddescr_indices,
                                    fieldnums,
                                    ..
                                } => {
                                    let fpe = if *size > 0 {
                                        fieldnums.len() / *size
                                    } else {
                                        0
                                    };
                                    let element_fields = (0..*size)
                                        .map(|ei| {
                                            let s = ei * fpe;
                                            let e = (s + fpe).min(fieldnums.len());
                                            resolve_fieldnums(&fieldnums[s..e], fielddescr_indices)
                                        })
                                        .collect();
                                    majit_backend::ExitVirtualLayout::ArrayStruct {
                                        descr_index: *descr_index,
                                        arraydescr: arraydescr.clone(),
                                        fielddescrs: fielddescrs.clone(),
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
                                    let values = fieldnums
                                        .iter()
                                        .map(|&fnum| resolve_tagged_source(fnum))
                                        .collect();
                                    majit_backend::ExitVirtualLayout::RawBuffer {
                                        func: *func,
                                        size: *size,
                                        offsets: offsets.clone(),
                                        descrs: descrs.clone(),
                                        values,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
                                    // resume.py:717: VRawSliceInfo — base_buffer + offset.
                                    let base = fieldnums
                                        .first()
                                        .map(|&fnum| resolve_tagged_source(fnum))
                                        .unwrap_or(ExitValueSourceLayout::Constant(0));
                                    majit_backend::ExitVirtualLayout::RawSlice {
                                        offset: *offset,
                                        base,
                                    }
                                }
                                // resume.py:763 VStrPlainInfo /
                                // resume.py:817 VUniPlainInfo —
                                // length = len(fieldnums).
                                majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums } => {
                                    let chars = fieldnums
                                        .iter()
                                        .map(|&fnum| resolve_tagged_source(fnum))
                                        .collect();
                                    majit_backend::ExitVirtualLayout::StrPlain {
                                        is_unicode: false,
                                        chars,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
                                    let chars = fieldnums
                                        .iter()
                                        .map(|&fnum| resolve_tagged_source(fnum))
                                        .collect();
                                    majit_backend::ExitVirtualLayout::StrPlain {
                                        is_unicode: true,
                                        chars,
                                    }
                                }
                                // resume.py:781 VStrConcatInfo /
                                // resume.py:836 VUniConcatInfo —
                                // decoder.concat_strings(left, right).
                                majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums }
                                | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
                                    let is_unicode = matches!(
                                        entry,
                                        majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
                                    );
                                    let oopspec = if is_unicode {
                                        majit_ir::effectinfo::OopSpecIndex::UniConcat
                                    } else {
                                        majit_ir::effectinfo::OopSpecIndex::StrConcat
                                    };
                                    let cic = callinfocollection.expect(
                                        "build_guard_metadata: callinfocollection \
                                         required for VStr/VUni Concat recovery",
                                    );
                                    let (calldescr, func) =
                                        cic.callinfo_for_oopspec(oopspec).expect(
                                            "callinfo_for_oopspec missing OS_STR_CONCAT/OS_UNI_CONCAT",
                                        );
                                    let left = resolve_tagged_source(fieldnums[0]);
                                    let right = resolve_tagged_source(fieldnums[1]);
                                    majit_backend::ExitVirtualLayout::StrConcat {
                                        is_unicode,
                                        func: *func as i64,
                                        calldescr: calldescr.clone(),
                                        left,
                                        right,
                                    }
                                }
                                // resume.py:801 VStrSliceInfo /
                                // resume.py:856 VUniSliceInfo —
                                // decoder.slice_string(largerstr, start, length).
                                majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums }
                                | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
                                    let is_unicode = matches!(
                                        entry,
                                        majit_ir::RdVirtualInfo::VUniSliceInfo { .. }
                                    );
                                    let oopspec = if is_unicode {
                                        majit_ir::effectinfo::OopSpecIndex::UniSlice
                                    } else {
                                        majit_ir::effectinfo::OopSpecIndex::StrSlice
                                    };
                                    let cic = callinfocollection.expect(
                                        "build_guard_metadata: callinfocollection \
                                         required for VStr/VUni Slice recovery",
                                    );
                                    let (calldescr, func) =
                                        cic.callinfo_for_oopspec(oopspec).expect(
                                            "callinfo_for_oopspec missing OS_STR_SLICE/OS_UNI_SLICE",
                                        );
                                    let str_src = resolve_tagged_source(fieldnums[0]);
                                    let start = resolve_tagged_source(fieldnums[1]);
                                    let length = resolve_tagged_source(fieldnums[2]);
                                    majit_backend::ExitVirtualLayout::StrSlice {
                                        is_unicode,
                                        func: *func as i64,
                                        calldescr: calldescr.clone(),
                                        str_src,
                                        start,
                                        length,
                                    }
                                }
                                majit_ir::RdVirtualInfo::Empty => {
                                    panic!("[jit] rd_virtuals[{vidx}] is Empty");
                                }
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            // resume.py:926,993: rd_pendingfields → pending_field_layouts.
            // PENDINGFIELDSTRUCT.lldescr parity: derive field_offset /
            // field_size / field_type from `pf.descr` (FieldDescr or
            // ArrayDescr) at consume time. The cache used to live on
            // GuardPendingFieldEntry; dropping it matches the RPython
            // struct shape (lldescr / num / fieldnum / itemindex).
            //
            // resume.py:993 _prepare_pendingfields parity:
            //   if itemindex < 0: setfield(struct, fieldnum, descr)
            //   else:             setarrayitem(struct, itemindex, fieldnum, descr)
            // The layout carries the RPython shape verbatim — `field_offset`
            // is the descr's base offset (struct field offset for FieldDescr,
            // array base offset for ArrayDescr), `is_array_item` mirrors
            // RPython's `itemindex >= 0` test, and `item_index` is the array
            // index when present.  Consumers (eval.rs:5009-5016 dynasm,
            // compiler.rs:1314 cranelift) reconstruct the address per
            // resume.py:1509-1530 (`base + offset + item_index * item_size`
            // for arrays, `base + offset` for fields).
            let pending_field_layouts: Vec<majit_backend::ExitPendingFieldLayout> = op
                .resolved_rd_pendingfields()
                .map(|entries| {
                    entries
                        .iter()
                        .map(|pf| {
                            // resume.py:1000 PENDINGFIELDSTRUCT.lldescr is
                            // always present in RPython — the descr is
                            // captured directly off the Setfield_gc /
                            // Setarrayitem_gc op that produced the pending
                            // field (heap.py force_lazy_sets_for_guard).
                            // Pyre's producer at optimizer.rs:3389 mirrors
                            // this: `pf.descr = pf_op.descr.clone()` where
                            // pf_op is always a descr-bearing setfield op.
                            let descr = pf
                                .descr
                                .as_ref()
                                .expect("resume.py:1000 PENDINGFIELDSTRUCT.lldescr must be set");
                            let (field_offset, field_size, field_type, item_index) =
                                if let Some(fd) = descr.as_field_descr() {
                                    // setfield: itemindex < 0 in RPython.
                                    (fd.offset(), fd.field_size(), fd.field_type(), None)
                                } else if let Some(ad) = descr.as_array_descr() {
                                    // setarrayitem: itemindex >= 0.  Carry
                                    // base_size as the offset and item_index
                                    // separately; consumers add
                                    // `item_index * item_size`.
                                    let idx = pf.item_index.max(0) as usize;
                                    (ad.base_size(), ad.item_size(), ad.item_type(), Some(idx))
                                } else {
                                    panic!(
                                        "pending field descr must be FieldDescr or ArrayDescr (descr_index={})",
                                        pf.descr_index,
                                    );
                                };
                            majit_backend::ExitPendingFieldLayout {
                                descr_index: pf.descr_index,
                                is_array_item: item_index.is_some(),
                                item_index,
                                target: resolve_tagged_source(pf.target_tagged),
                                value: resolve_tagged_source(pf.value_tagged),
                                field_offset,
                                field_size,
                                field_type,
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            Some(ExitRecoveryLayout {
                vable_array: vable_layout,
                vref_array: vref_layout,
                frames: frames_layout,
                virtual_layouts,
                pending_field_layouts,
            })
        } else {
            // No rd_numb: identity recovery layout.
            // Every guard has at minimum an identity mapping from
            // fail_args → frame slots, with exit_types as slot_types.
            // `jitcode_index: 0` is a placeholder for the no-rd_numb
            // path — `patch_backend_guard_recovery_layouts_for_trace`
            // (compile.rs:1596) overwrites this with the resume_layout
            // derived from `Snapshot::single_frame(jitcode_index, pc, ...)`.
            // The outermost-frame rule at eval.rs:3938-3951 means a
            // stale `jitcode_index: 0` is never consulted for code lookup
            // on the sole frame of a single-frame identity layout — code
            // comes from the vable instead.
            let slots: Vec<majit_backend::ExitValueSourceLayout> = (0..exit_types.len())
                .map(majit_backend::ExitValueSourceLayout::ExitValue)
                .collect();
            Some(ExitRecoveryLayout {
                vable_array: vec![],
                vref_array: vec![],
                frames: vec![majit_backend::ExitFrameLayout {
                    trace_id: None,
                    header_pc: Some(pc),
                    source_guard: None,
                    pc,
                    jitcode_index: 0,
                    slots,
                    slot_types: Some(exit_types.clone()),
                }],
                virtual_layouts: vec![],
                pending_field_layouts: vec![],
            })
        };

        exit_layouts.insert(
            fail_index,
            StoredExitLayout {
                source_op_index: Some(op_idx),
                gc_ref_slots: exit_types
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, tp)| (*tp == Type::Ref).then_some(slot))
                    .collect(),
                force_token_slots: Vec::new(),
                recovery_layout,
                resume_layout,
                storage,
                descr: op.descr.clone(),
                op_arg_types_for_jump: None,
            },
        );
        fail_index += 1;
    }

    (result, exit_layouts)
}

pub(crate) fn merge_backend_exit_layouts(
    exit_layouts: &mut HashMap<u32, StoredExitLayout>,
    backend_layouts: &[FailDescrLayout],
    ops: &[majit_ir::Op],
) {
    for layout in backend_layouts {
        // compile.py:861 copy_all_attributes_from parity: when the backend
        // exposes resume data (rd_numb / rd_consts / rd_virtuals /
        // rd_pendingfields) for an exit the frontend never saw, assemble
        // them into a `ResumeStorage` so downstream consumers
        // (rebuild_guard_fail_state, build_blackhole_frames_from_deadframe)
        // see the same shared pool they get on the frontend-primed path.
        let storage_from_backend = layout.rd_numb.clone().map(|numb| {
            crate::resume::ResumeStorage::new(
                numb,
                layout.rd_consts.clone().unwrap_or_default(),
                layout.rd_virtuals.clone().unwrap_or_default(),
                layout.rd_pendingfields.clone().unwrap_or_default(),
            )
        });
        // Pre-resolve the source-op `descr` for backend-only entries so
        // `entry.descr` matches what `build_guard_metadata` would have
        // primed had the frontend seen this exit.  When the source op
        // is no longer reachable (frontend trace evicted but backend
        // exit layout persists across previous-token fallback) the
        // backend's `fail_arg_types` is the canonical typed-data
        // carrier; mint a fresh `MetaFailDescr` carrying that vector
        // so descr-side readers stay populated.  Branch on
        // `layout.is_finish` so a backend-only FINISH entry synthesizes
        // a `_DoneWithThisFrameDescr`-flavored handle (is_finish=true)
        // — matching the terminal path at compile.rs:1305-1311.
        // Without the branch, `pyjitpl/mod.rs:262 descr.is_finish()`
        // returns false on the synthesized guard descr and breaks
        // PyPy's `DoneWithThisFrameDescr` /
        // `ExitFrameWithExceptionDescrRef` identity check
        // (compile.py:658-662 / compile.py:701-784).
        let descr_from_op = layout
            .source_op_index
            .and_then(|idx| ops.get(idx))
            .and_then(|op| op.descr.clone())
            .or_else(|| {
                Some(if layout.is_finish {
                    make_finish_fail_descr_typed(layout.fail_arg_types.clone())
                } else {
                    make_fail_descr_typed(layout.fail_arg_types.clone())
                })
            });
        let entry: &mut StoredExitLayout =
            exit_layouts
                .entry(layout.fail_index)
                .or_insert_with(|| StoredExitLayout {
                    source_op_index: layout.source_op_index,
                    gc_ref_slots: layout.gc_ref_slots.clone(),
                    force_token_slots: layout.force_token_slots.clone(),
                    recovery_layout: layout.recovery_layout.clone(),
                    resume_layout: None,
                    storage: storage_from_backend.clone(),
                    descr: descr_from_op.clone(),
                    op_arg_types_for_jump: None,
                });
        entry.source_op_index = layout.source_op_index;
        // Backfill descr if the entry was inserted before `op.descr` was
        // available (or the op walk produced a fresh handle). Keeps
        // descr-side identity in sync with `build_guard_metadata`'s
        // priority-1 source.
        if entry.descr.is_none() {
            entry.descr = descr_from_op;
        }
        entry.gc_ref_slots = layout.gc_ref_slots.clone();
        entry.force_token_slots = layout.force_token_slots.clone();
        entry.recovery_layout = layout.recovery_layout.clone();
        // compile.py:861 copy_all_attributes_from parity: fill missing
        // resume storage from the backend-side rd_* propagation. Entries
        // already primed by the frontend keep their Arc (same bytes
        // either way, but preserving the handle avoids dropping shared
        // downstream references).
        if entry.storage.is_none() {
            entry.storage = storage_from_backend.clone();
        }

        // Merge backend frame_stack metadata into the stored resume layout.
        if let Some(frame_stack) = &layout.frame_stack {
            merge_frame_stack_into_resume_layout(entry, frame_stack);
        }
    }
    validate_exit_layouts(exit_layouts);
}

/// Validate that guard exit layouts with recovery_layout have complete
/// metadata. Called after merge_backend_exit_layouts to enforce:
/// if recovery_layout is present, header_pc and slot_types must be set.
///
/// Guards that HAVE recovery_layout (all production guards after backend
/// merge) must satisfy the full invariant. Guards without (only possible
/// in unit tests with mock backends) are warned but not fatal.
pub(crate) fn validate_exit_layouts(exit_layouts: &HashMap<u32, StoredExitLayout>) {
    for (&fail_index, layout) in exit_layouts {
        if layout.resolve_is_finish() {
            continue;
        }
        let Some(ref recovery) = layout.recovery_layout else {
            // No recovery_layout — backend didn't provide one (test mock).
            // In production, identity_recovery_layout always creates it.
            continue;
        };
        for (fi, frame) in recovery.frames.iter().enumerate() {
            // header_pc and slot_types are filled by both
            // build_guard_metadata (metainterp) and identity_recovery_layout
            // (backend). After backend merge, both must be present.
            // Backend-provided layouts always have them (compiler.rs:4482,4488).
            // Metainterp-provided layouts fill them since Step 1.
            if frame.header_pc.is_none() || frame.slot_types.is_none() {
                // Backend test mocks may omit these — not fatal in tests.
                #[cfg(not(test))]
                {
                    debug_assert!(
                        frame.header_pc.is_some(),
                        "guard fail_index={fail_index} frame[{fi}] has no header_pc"
                    );
                    debug_assert!(
                        frame.slot_types.is_some(),
                        "guard fail_index={fail_index} frame[{fi}] has no slot_types"
                    );
                }
                continue;
            }
            let st = frame.slot_types.as_ref().unwrap();
            debug_assert_eq!(
                st.len(),
                frame.slots.len(),
                "guard fail_index={fail_index} frame[{fi}]: slot_types.len()={} != slots.len()={}",
                st.len(),
                frame.slots.len(),
            );
        }
    }
}

/// Merge backend-origin `frame_stack` metadata into a `StoredExitLayout`'s
/// resume layout, enriching or creating `frame_layouts` entries with slot
/// types from the backend's `ExitFrameLayout`.
pub(crate) fn merge_frame_stack_into_resume_layout(
    entry: &mut StoredExitLayout,
    frame_stack: &[ExitFrameLayout],
) {
    if frame_stack.is_empty() {
        return;
    }

    let frame_layouts: Vec<ResumeFrameLayoutSummary> = frame_stack
        .iter()
        .map(ResumeFrameLayoutSummary::from_exit_frame_layout)
        .collect();

    if let Some(ref mut resume_layout) = entry.resume_layout {
        // Merge slot types from frame_stack into existing frame_layouts.
        let shared = resume_layout.frame_layouts.len().min(frame_layouts.len());
        for offset in 0..shared {
            let resume_index = resume_layout.frame_layouts.len() - 1 - offset;
            let fs_index = frame_layouts.len() - 1 - offset;
            let target = &mut resume_layout.frame_layouts[resume_index];
            let source = &frame_layouts[fs_index];

            if target.trace_id.is_none() {
                target.trace_id = source.trace_id;
            }
            if target.header_pc.is_none() {
                target.header_pc = source.header_pc;
            }
            if target.source_guard.is_none() {
                target.source_guard = source.source_guard;
            }

            let needs_slot_types = target
                .slot_types
                .as_ref()
                .map_or(true, |types| types.len() != target.slot_layouts.len());
            if needs_slot_types
                && source
                    .slot_types
                    .as_ref()
                    .is_some_and(|types| types.len() == target.slot_layouts.len())
            {
                target.slot_types = source.slot_types.clone();
            }
        }

        // If the frame_stack has more frames than the existing resume layout,
        // prepend the extra outer frames.
        if frame_layouts.len() > resume_layout.frame_layouts.len() {
            let extra_count = frame_layouts.len() - resume_layout.frame_layouts.len();
            let mut new_frames = frame_layouts[..extra_count].to_vec();
            new_frames.append(&mut resume_layout.frame_layouts);
            resume_layout.frame_layouts = new_frames;
            resume_layout.num_frames = resume_layout.frame_layouts.len();
            resume_layout.frame_pcs = resume_layout.frame_layouts.iter().map(|f| f.pc).collect();
            resume_layout.frame_slot_counts = resume_layout
                .frame_layouts
                .iter()
                .map(|f| f.slot_layouts.len())
                .collect();
        }
    } else {
        // No existing resume layout; create one from the frame_stack.
        entry.resume_layout = Some(ResumeLayoutSummary {
            num_frames: frame_layouts.len(),
            frame_pcs: frame_layouts.iter().map(|f| f.pc).collect(),
            frame_slot_counts: frame_layouts.iter().map(|f| f.slot_layouts.len()).collect(),
            frame_layouts,
            num_virtuals: 0,
            virtual_kinds: Vec::new(),
            virtual_layouts: Vec::new(),
            pending_field_count: 0,
            pending_field_layouts: Vec::new(),
            const_pool_size: 0,
        });
    }
}

/// Enrich an `Option<ResumeLayoutSummary>` with backend-origin `frame_stack`
/// metadata at runtime, merging slot types and outer frames.
pub(crate) fn enrich_resume_layout_with_frame_stack(
    resume_layout: &mut Option<ResumeLayoutSummary>,
    frame_stack: Option<&[ExitFrameLayout]>,
) {
    let Some(frame_stack) = frame_stack else {
        return;
    };
    if frame_stack.is_empty() {
        return;
    }

    let frame_layouts: Vec<ResumeFrameLayoutSummary> = frame_stack
        .iter()
        .map(ResumeFrameLayoutSummary::from_exit_frame_layout)
        .collect();

    if let Some(layout) = resume_layout {
        let shared = layout.frame_layouts.len().min(frame_layouts.len());
        for offset in 0..shared {
            let resume_index = layout.frame_layouts.len() - 1 - offset;
            let fs_index = frame_layouts.len() - 1 - offset;
            let target = &mut layout.frame_layouts[resume_index];
            let source = &frame_layouts[fs_index];

            if target.trace_id.is_none() {
                target.trace_id = source.trace_id;
            }
            if target.header_pc.is_none() {
                target.header_pc = source.header_pc;
            }
            if target.source_guard.is_none() {
                target.source_guard = source.source_guard;
            }

            let needs_slot_types = target
                .slot_types
                .as_ref()
                .map_or(true, |types| types.len() != target.slot_layouts.len());
            if needs_slot_types
                && source
                    .slot_types
                    .as_ref()
                    .is_some_and(|types| types.len() == target.slot_layouts.len())
            {
                target.slot_types = source.slot_types.clone();
            }
        }

        if frame_layouts.len() > layout.frame_layouts.len() {
            let extra_count = frame_layouts.len() - layout.frame_layouts.len();
            let mut new_frames = frame_layouts[..extra_count].to_vec();
            new_frames.append(&mut layout.frame_layouts);
            layout.frame_layouts = new_frames;
            layout.num_frames = layout.frame_layouts.len();
            layout.frame_pcs = layout.frame_layouts.iter().map(|f| f.pc).collect();
            layout.frame_slot_counts = layout
                .frame_layouts
                .iter()
                .map(|f| f.slot_layouts.len())
                .collect();
        }
    } else {
        *resume_layout = Some(ResumeLayoutSummary {
            num_frames: frame_layouts.len(),
            frame_pcs: frame_layouts.iter().map(|f| f.pc).collect(),
            frame_slot_counts: frame_layouts.iter().map(|f| f.slot_layouts.len()).collect(),
            frame_layouts,
            num_virtuals: 0,
            virtual_kinds: Vec::new(),
            virtual_layouts: Vec::new(),
            pending_field_count: 0,
            pending_field_layouts: Vec::new(),
            const_pool_size: 0,
        });
    }
}

pub(crate) fn merge_backend_terminal_exit_layouts(
    terminal_exit_layouts: &mut HashMap<usize, StoredExitLayout>,
    backend_layouts: &[TerminalExitLayout],
    ops: &[majit_ir::Op],
) {
    for layout in backend_layouts {
        // Pre-resolve the source-op `descr` for backend-only entries so
        // `entry.descr` matches what `build_terminal_exit_layouts` would
        // have primed had the frontend seen this exit.  Source op
        // priority: a frontend `op.descr` (LoopTargetDescr for JUMP,
        // _DoneWithThisFrameDescr* / ExitFrameWithExceptionDescrRef for
        // FINISH).  Fallback when source op evicted or its descr handle
        // was dropped: synthesize per the backend layout's own
        // `is_finish` discriminator — a `_DoneWithThisFrameDescr`-flavored
        // `make_finish_fail_descr_typed` (is_finish=true) for FINISH; a
        // guard-flavored `make_fail_descr_typed` (is_finish=false) for
        // JUMP, since `LoopTargetDescr` (history.py:470) carries no
        // `fail_arg_types` and a synthetic FINISH descr would silently
        // mark the exit as terminating.  Both retain `descr=Some` so
        // downstream readers that probe `entry.descr.is_some_and(...)`
        // (e.g. `assign_guard_hashes` via the backend layout, exit-type
        // resolution paths) keep working uniformly.
        let source_op = ops.get(layout.op_index);
        let is_jump = match source_op {
            Some(op) => op.opcode == OpCode::Jump,
            None => !layout.is_finish,
        };
        let descr_from_op = source_op.and_then(|op| op.descr.clone()).or_else(|| {
            Some(if layout.is_finish {
                make_finish_fail_descr_typed(layout.exit_types.clone())
            } else {
                make_fail_descr_typed(layout.exit_types.clone())
            })
        });
        let op_arg_types_for_jump = is_jump.then(|| layout.exit_types.clone());
        let entry = terminal_exit_layouts
            .entry(layout.op_index)
            .or_insert_with(|| StoredExitLayout {
                source_op_index: Some(layout.op_index),
                gc_ref_slots: layout.gc_ref_slots.clone(),
                force_token_slots: layout.force_token_slots.clone(),
                recovery_layout: layout.recovery_layout.clone(),
                resume_layout: None,
                storage: None,
                descr: descr_from_op.clone(),
                op_arg_types_for_jump: op_arg_types_for_jump.clone(),
            });
        entry.source_op_index = Some(layout.op_index);
        entry.gc_ref_slots = layout.gc_ref_slots.clone();
        entry.force_token_slots = layout.force_token_slots.clone();
        entry.recovery_layout = layout.recovery_layout.clone();
        if entry.descr.is_none() {
            entry.descr = descr_from_op;
        }
        if entry.op_arg_types_for_jump.is_none() && is_jump {
            entry.op_arg_types_for_jump = op_arg_types_for_jump;
        }
    }
}

pub(crate) fn enrich_resume_layout_with_trace_metadata(
    layout: &mut ResumeLayoutSummary,
    trace_id: u64,
    inputargs: &[InputArg],
    trace_info: Option<&CompiledTraceInfo>,
    recovery_layout: Option<&ExitRecoveryLayout>,
) {
    if layout.frame_layouts.is_empty() {
        return;
    }

    if let Some(recovery_layout) = recovery_layout {
        let shared_frames = layout.frame_layouts.len().min(recovery_layout.frames.len());
        for offset in 0..shared_frames {
            let layout_index = layout.frame_layouts.len() - 1 - offset;
            let recovery_index = recovery_layout.frames.len() - 1 - offset;
            let recovery_frame = &recovery_layout.frames[recovery_index];
            let frame = &mut layout.frame_layouts[layout_index];
            if frame.trace_id.is_none() {
                frame.trace_id = recovery_frame.trace_id;
            }
            if frame.header_pc.is_none() {
                frame.header_pc = recovery_frame.header_pc;
            }
            if frame.source_guard.is_none() {
                frame.source_guard = recovery_frame.source_guard;
            }
            let needs_slot_types = match frame.slot_types.as_ref() {
                Some(slot_types) => slot_types.len() != frame.slot_layouts.len(),
                None => true,
            };
            if needs_slot_types
                && recovery_frame
                    .slot_types
                    .as_ref()
                    .is_some_and(|slot_types| slot_types.len() == frame.slot_layouts.len())
            {
                frame.slot_types = recovery_frame.slot_types.clone();
            }
        }
    }

    let last_index = layout.frame_layouts.len() - 1;
    let innermost = &mut layout.frame_layouts[last_index];
    if innermost.trace_id.is_none() {
        innermost.trace_id = Some(trace_id);
    }
    if innermost.header_pc.is_none() {
        innermost.header_pc = trace_info.map(|info| info.header_pc);
    }
    if innermost.source_guard.is_none() {
        innermost.source_guard = trace_info.and_then(|info| info.source_guard);
    }
    let needs_slot_types = match innermost.slot_types.as_ref() {
        Some(slot_types) => slot_types.len() != innermost.slot_layouts.len(),
        None => true,
    };
    if needs_slot_types && inputargs.len() == innermost.slot_layouts.len() {
        innermost.slot_types = Some(inputargs.iter().map(|arg| arg.tp).collect());
    }
}

pub(crate) fn find_fail_index_for_exit_op(ops: &[majit_ir::Op], op_index: usize) -> Option<u32> {
    let mut fail_index = 0u32;
    for (idx, op) in ops.iter().enumerate() {
        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            if idx == op_index {
                return Some(fail_index);
            }
            fail_index += 1;
        }
    }
    None
}

pub(crate) fn infer_terminal_exit_layout(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
    constant_types: &HashMap<u32, Type>,
    owning_key: u64,
    trace_id: u64,
    op_index: usize,
) -> Option<CompiledExitLayout> {
    let op = ops.get(op_index)?;
    let is_finish = op.opcode == OpCode::Finish;
    if !is_finish && op.opcode != OpCode::Jump {
        return None;
    }
    let fail_index = find_fail_index_for_exit_op(ops, op_index).unwrap_or(u32::MAX);
    let type_index = majit_ir::OpTypeIndex::new(inputargs, ops, constant_types);
    let exit_types: Vec<Type> = op
        .args
        .iter()
        .map(|opref| {
            type_index
                .opref_type_at(*opref, op_index)
                .unwrap_or(Type::Int)
        })
        .collect();
    let force_token_slots: Vec<usize> = op
        .args
        .iter()
        .enumerate()
        .filter_map(|(slot, opref)| {
            type_index
                .op_at(*opref)
                .map(|op| op.opcode)
                .filter(|opcode| *opcode == OpCode::ForceToken)
                .map(|_| slot)
        })
        .collect();
    let gc_ref_slots: Vec<usize> = exit_types
        .iter()
        .enumerate()
        .filter_map(|(slot, tp)| {
            (*tp == Type::Ref && !force_token_slots.contains(&slot)).then_some(slot)
        })
        .collect();
    Some(CompiledExitLayout {
        rd_loop_token: owning_key, // compile.py:186
        trace_id,
        fail_index,
        source_op_index: Some(op_index),
        exit_types,
        is_finish,
        gc_ref_slots,
        force_token_slots,
        recovery_layout: None,
        resume_layout: None,
        storage: None,
    })
}

pub(crate) fn build_terminal_exit_layouts(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
    constant_types: &HashMap<u32, Type>,
) -> HashMap<usize, StoredExitLayout> {
    let mut layouts = HashMap::new();
    for (op_index, op) in ops.iter().enumerate() {
        if op.opcode != OpCode::Finish && op.opcode != OpCode::Jump {
            continue;
        }
        if let Some(layout) =
            infer_terminal_exit_layout(inputargs, ops, constant_types, 0, 0, op_index)
        {
            // For JUMP exits the descr is `LoopTargetDescr`
            // (`history.py:470`), which has no `fail_arg_types`.  Cache
            // the per-arg types so `StoredExitLayout::resolve_exit_types()`
            // can fall back to them — see the field's docstring.
            // FINISH carries `_DoneWithThisFrameDescr*` /
            // `ExitFrameWithExceptionDescrRef`, both of which expose
            // `fail_arg_types()` directly, so the cache stays `None`.
            let op_arg_types_for_jump =
                (op.opcode == OpCode::Jump).then(|| layout.exit_types.clone());
            layouts.insert(
                op_index,
                StoredExitLayout {
                    source_op_index: Some(op_index),
                    gc_ref_slots: layout.gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: None,
                    resume_layout: None,
                    storage: None,
                    descr: op.descr.clone(),
                    op_arg_types_for_jump,
                },
            );
        }
    }
    layouts
}

pub(crate) fn terminal_exit_layout_for_trace(
    trace: &CompiledTrace,
    owning_key: u64,
    trace_id: u64,
    op_index: usize,
) -> Option<CompiledExitLayout> {
    if let Some(layout) = trace.terminal_exit_layouts.get(&op_index) {
        return Some(layout.public(
            owning_key,
            trace_id,
            find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
        ));
    }
    if let Some(fail_index) = find_fail_index_for_exit_op(&trace.ops, op_index) {
        if let Some(layout) = trace.exit_layouts.get(&fail_index) {
            return Some(layout.public(owning_key, trace_id, fail_index));
        }
    }
    infer_terminal_exit_layout(
        &trace.inputargs,
        &trace.ops,
        &trace.constant_types,
        owning_key,
        trace_id,
        op_index,
    )
}

pub(crate) fn decode_values_with_layout(
    raw_values: &[i64],
    layout: &CompiledExitLayout,
) -> Vec<Value> {
    layout
        .exit_types
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            let raw = raw_values.get(index).copied().unwrap_or(0);
            match tp {
                Type::Int => Value::Int(raw),
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                Type::Void => Value::Void,
            }
        })
        .collect()
}

pub(crate) fn normalize_closing_jump_args(
    mut ops: Vec<Op>,
    constants: &std::collections::HashMap<u32, i64>,
    num_inputs: usize,
) -> Vec<Op> {
    let Some(label_args) = ops
        .iter()
        .rev()
        .find(|op| op.opcode == OpCode::Label)
        .map(|op| op.args.clone())
    else {
        return ops;
    };

    let defined: std::collections::HashSet<OpRef> = ops
        .iter()
        .filter(|op| op.result_type() != majit_ir::Type::Void && !op.pos.is_none())
        .map(|op| op.pos)
        .collect();

    let Some(jump) = ops.iter_mut().rfind(|op| op.opcode == OpCode::Jump) else {
        return ops;
    };

    for (idx, arg) in jump.args.iter_mut().enumerate() {
        if idx >= label_args.len() {
            break;
        }
        if constants.contains_key(&arg.raw()) {
            continue;
        }
        if (arg.raw() as usize) < num_inputs {
            continue;
        }
        if defined.contains(arg) {
            continue;
        }
        *arg = label_args[idx];
    }

    ops
}

/// `rpython/jit/metainterp/compile.py:425-461`
/// `patch_new_loop_to_load_virtualizable_fields`.
///
/// ```python
/// def patch_new_loop_to_load_virtualizable_fields(loop, jitdriver_sd, vable):
///     vinfo = jitdriver_sd.virtualizable_info
///     extra_ops = []
///     inputargs = loop.inputargs
///     vable_box = inputargs[jitdriver_sd.index_of_virtualizable]
///     i = jitdriver_sd.num_red_args
///     loop.inputargs = inputargs[:i]
///     for descr in vinfo.static_field_descrs:
///         assert i < len(inputargs)
///         box = inputargs[i]
///         opnum = OpHelpers.getfield_for_descr(descr)
///         emit_op(extra_ops,
///                 ResOperation(opnum, [vable_box], descr=descr))
///         box.set_forwarded(extra_ops[-1])
///         i += 1
///     arrayindex = 0
///     for descr in vinfo.array_field_descrs:
///         arraylen = vinfo.get_array_length(vable, arrayindex)
///         arrayop = ResOperation(rop.GETFIELD_GC_R, [vable_box], descr=descr)
///         emit_op(extra_ops, arrayop)
///         arraydescr = vinfo.array_descrs[arrayindex]
///         assert i + arraylen <= len(inputargs)
///         for index in range(arraylen):
///             opnum = OpHelpers.getarrayitem_for_descr(arraydescr)
///             box = inputargs[i]
///             emit_op(extra_ops,
///                 ResOperation(opnum,
///                              [arrayop, ConstInt(index)],
///                              descr=arraydescr))
///             i += 1
///             box.set_forwarded(extra_ops[-1])
///         arrayindex += 1
///     assert i == len(inputargs)
///     for op in loop.operations:
///         emit_op(extra_ops, op)
///     loop.operations = extra_ops
/// ```
///
/// Called from `send_loop_to_backend` (compile.py:504-511) after the loop
/// has been optimized but before it is handed to the CPU backend. The
/// virtualizable's static and array fields ride through the optimizer as
/// expanded trace inputargs; this function strips them and reconstructs
/// each field at loop entry with a `GETFIELD_GC` / `GETARRAYITEM_GC` op
/// so the compiled loop's `len(inputargs) == num_red_args` matches
/// `execute_token`'s `clt._debug_nbargs` and CA's `op.args.len()`.
///
/// `vable_array_lengths` mirrors RPython's `vinfo.get_array_length(vable, i)`
/// reads: one length per array field (in `vinfo.array_fields` order), taken
/// from the concrete virtualizable at trace-start time.
pub(crate) fn patch_new_loop_to_load_virtualizable_fields(
    ops: &mut Vec<Op>,
    inputargs: &mut Vec<InputArg>,
    vinfo: &crate::virtualizable::VirtualizableInfo,
    vable_array_lengths: &[usize],
    num_red_args: usize,
    index_of_virtualizable: usize,
    constants: &mut HashMap<u32, i64>,
    constant_types: &mut HashMap<u32, Type>,
) {
    // PRE-EXISTING-ADAPTATION (Rust language constraint, not a logic
    // divergence): RPython `compile.py:425-461` calls
    // `box.set_forwarded(extra_ops[-1])` to set Python-Box-attached
    // forwarding pointers, which `emit_op`'s default `get_box_replacement`
    // walks transitively when later body ops reference the original box.
    // Pyre uses a flat-`OpRef` IR (no per-Box mutable forwarding cell),
    // so the equivalent rewrite uses a function-local
    // `forwarding: Vec<OpRef>` indexed by source `OpRef.0`. The Vec is
    // discarded when the function returns; its lifetime mirrors the
    // single in-place loop rewrite that RPython's `_forwarded` model
    // accomplishes via Box mutation. No semantic divergence.
    use majit_ir::{Op, OpCode, OpRef, descr::ArrayFlag};

    // PRE-EXISTING-ADAPTATION (Rust language constraint, not a logic
    // divergence): RPython `compile.py:425-461` calls
    // `box.set_forwarded(extra_ops[-1])` to set Python-Box-attached
    // forwarding pointers, which `emit_op`'s default `get_box_replacement`
    // walks transitively when later body ops reference the original box.
    // Pyre uses a flat-`OpRef` IR (no per-Box mutable forwarding cell),
    // so the equivalent rewrite uses a function-local
    // `forwarding: Vec<OpRef>` indexed by source `OpRef.0`. The Vec is
    // discarded when the function returns; its lifetime mirrors the
    // single in-place loop rewrite that RPython's `_forwarded` model
    // accomplishes via Box mutation. No semantic divergence.
    fn set_local_forwarded(forwarding: &mut Vec<OpRef>, source: OpRef, target: OpRef) {
        if source.is_none() || source.is_constant() {
            return;
        }
        let idx = source.raw() as usize;
        if idx >= forwarding.len() {
            forwarding.resize(idx + 1, OpRef::NONE);
        }
        forwarding[idx] = target;
    }

    fn get_local_box_replacement(forwarding: &[OpRef], mut opref: OpRef) -> OpRef {
        if opref.is_none() || opref.is_constant() {
            return opref;
        }
        loop {
            let idx = opref.raw() as usize;
            if idx >= forwarding.len() {
                return opref;
            }
            let next = forwarding[idx];
            if next.is_none() {
                return opref;
            }
            opref = next;
        }
    }

    fn emit_forwarded_patch_op(
        extra_ops: &mut Vec<Op>,
        op: &Op,
        forwarding: &mut Vec<OpRef>,
        next_opref: &mut u32,
    ) {
        let mut emitted = op.clone();
        let mut replaced = false;

        for i in 0..op.args.len() {
            let orig_arg = op.args[i];
            let arg = get_local_box_replacement(forwarding, orig_arg);
            if orig_arg != arg {
                if !replaced {
                    emitted = op.copy_and_change(op.opcode, None, None);
                    if op.result_type() != Type::Void && !op.pos.is_none() {
                        let new_pos = OpRef::op_typed(*next_opref, op.result_type());
                        *next_opref += 1;
                        emitted.pos = new_pos;
                        // compile.py:414-418 `orig_op.set_forwarded(op)`:
                        // in the flat OpRef model this temporary vector is
                        // the local equivalent of Box._forwarded.
                        set_local_forwarded(forwarding, op.pos, new_pos);
                    }
                    replaced = true;
                }
                emitted.args[i] = arg;
            }
        }

        if op.opcode.is_guard() {
            if !replaced {
                emitted = op.copy_and_change(op.opcode, None, None);
            }
            if let Some(fail_args) = emitted.fail_args.as_mut() {
                for arg in fail_args.iter_mut() {
                    *arg = get_local_box_replacement(forwarding, *arg);
                }
            }
        }

        extra_ops.push(emitted);
    }

    assert!(
        index_of_virtualizable < num_red_args,
        "virtualizable must live inside the red args (pyjitpl.py:3589 index_of_virtualizable < num_red_args)"
    );
    if inputargs.len() <= num_red_args {
        // Already reduced or no virtualizable expansion in the trace.
        return;
    }

    let expanded_inputargs = inputargs.clone();

    // compile.py:429-430 — vable_box = inputargs[index_of_virtualizable].
    let vable_box = OpRef::input_arg_typed(
        expanded_inputargs[index_of_virtualizable].index,
        expanded_inputargs[index_of_virtualizable].tp,
    );

    // compile.py keeps Box identities disjoint automatically; in the flat
    // OpRef model we must allocate above every runtime ref already reachable
    // from the trace so copied ops can stand in for `orig_op.set_forwarded(op)`.
    let max_runtime_ref = ops
        .iter()
        .flat_map(|op| {
            std::iter::once(op.pos)
                .chain(op.args.iter().copied())
                .chain(op.fail_args.iter().flatten().copied())
        })
        .chain(expanded_inputargs.iter().map(|ia| ia.opref()))
        .filter(|opref| !opref.is_none() && !opref.is_constant())
        .map(|opref| opref.raw())
        .max()
        .unwrap_or(0);
    let mut next_opref = max_runtime_ref + 1;

    // Allocate fresh const indices above the existing max.
    // Index-keyed pool namespace probe (Slice P3 category E):
    // raw u32 keys carry the constant-namespace bit directly, so use
    // the bit-helpers instead of round-tripping through
    // `OpRef::from_raw(k)` which lands on the to-be-retired
    // `OpRef::Untyped` variant.
    let mut next_const_idx = constants
        .keys()
        .filter_map(|&k| OpRef::raw_is_constant(k).then(|| OpRef::raw_const_index(k)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut forwarding = vec![OpRef::NONE; (max_runtime_ref as usize).saturating_add(1)];
    let mut extra_ops: Vec<Op> = Vec::new();
    let mut i = num_red_args;

    // compile.py:432 — loop.inputargs = inputargs[:i].
    inputargs.truncate(num_red_args);

    // compile.py:433-440 — GETFIELD_GC per static field.
    let static_descrs = vinfo.static_field_descrs();
    for (fi, field) in vinfo.static_fields.iter().enumerate() {
        assert!(
            i < expanded_inputargs.len(),
            "static field {fi} exceeds inputargs ({} <= {})",
            i,
            expanded_inputargs.len()
        );
        let descr = static_descrs
            .get(fi)
            .cloned()
            .expect("static_field_descrs must be populated by set_parent_descr");
        let opcode = match field.field_type {
            Type::Int => OpCode::GetfieldGcI,
            Type::Ref => OpCode::GetfieldGcR,
            Type::Float => OpCode::GetfieldGcF,
            Type::Void => panic!("virtualizable static field {fi} has Void type"),
        };
        let old_opref =
            OpRef::input_arg_typed(expanded_inputargs[i].index, expanded_inputargs[i].tp);
        let new_opref = OpRef::op_typed(next_opref, field.field_type);
        next_opref += 1;
        let mut op = Op::new(opcode, &[vable_box]);
        op.pos = new_opref;
        op.descr = Some(descr);
        extra_ops.push(op);
        set_local_forwarded(&mut forwarding, old_opref, new_opref);
        i += 1;
    }

    // compile.py:441-457 — GETFIELD_GC_R (array ptr) + GETARRAYITEM_GC per element.
    let array_descrs_list = vinfo.array_field_descrs();
    for (ai, array_field_descr) in array_descrs_list.iter().enumerate() {
        let array_len = vable_array_lengths.get(ai).copied().unwrap_or(0);
        assert!(
            i + array_len <= expanded_inputargs.len(),
            "array {ai} length {array_len} would overrun inputargs (i={i}, len={})",
            expanded_inputargs.len()
        );
        // GETFIELD_GC_R(vable_box, array_field_descr) → array pointer (Ref-typed).
        let array_opref = OpRef::ref_op(next_opref);
        next_opref += 1;
        let mut arr_load = Op::new(OpCode::GetfieldGcR, &[vable_box]);
        arr_load.pos = array_opref;
        arr_load.descr = Some(array_field_descr.clone());
        extra_ops.push(arr_load);

        let array_descr = vinfo
            .array_descrs
            .get(ai)
            .cloned()
            .expect("VirtualizableInfo.array_descrs must cover every array_field");
        let array_info = &vinfo.array_fields[ai];
        let (item_opcode, item_descr, item_base) = match array_info.item_type {
            Type::Int => (OpCode::GetarrayitemGcI, array_descr.clone(), array_opref),
            Type::Ref => (OpCode::GetarrayitemGcR, array_descr.clone(), array_opref),
            Type::Float => (OpCode::GetarrayitemGcF, array_descr.clone(), array_opref),
            Type::Void => panic!("virtualizable array {ai} has Void item_type"),
        };
        let (item_opcode, item_descr, item_base) = match array_info.storage {
            crate::virtualizable::VableArrayStorage::DirectPointer => {
                (item_opcode, item_descr, item_base)
            }
            crate::virtualizable::VableArrayStorage::EmbeddedArray { ptr_offset } => {
                // PRE-EXISTING-ADAPTATION (heap layout divergence):
                // RPython's `vinfo.array_field_descrs` points at a real
                // GC array field; `compile.py:445 ResOperation(GETFIELD_GC_R)`
                // returns the array Box and `:451 ResOperation(GETARRAYITEM_GC_*)`
                // reads the items directly. Pyre's `FixedObjectArray` is
                // a `{ ptr: *T, len: usize }` container struct, so we
                // first emit `GetfieldGcI` to project the backing-storage
                // pointer (via `ptr_offset`), then use `GetarrayitemRaw*`
                // (raw because the items live behind a non-GC pointer).
                // The `make_array_descr(0, item_size, item_type)` mirrors
                // RPython's `array_descrs[arrayindex]` shape; only the
                // base-pointer indirection step is added. Convergence
                // would require switching pyre's `FixedObjectArray` to
                // RPython's flat GC-array layout — out of scope.
                let ptr_opref = OpRef::int_op(next_opref);
                next_opref += 1;
                let mut ptr_load = Op::new(OpCode::GetfieldGcI, &[array_opref]);
                ptr_load.pos = ptr_opref;
                ptr_load.descr = Some(majit_ir::descr::make_field_descr(
                    ptr_offset,
                    std::mem::size_of::<usize>(),
                    Type::Int,
                    ArrayFlag::Unsigned,
                ));
                extra_ops.push(ptr_load);

                let raw_opcode = match array_info.item_type {
                    Type::Int => OpCode::GetarrayitemRawI,
                    Type::Ref => OpCode::GetarrayitemRawR,
                    Type::Float => OpCode::GetarrayitemRawF,
                    Type::Void => unreachable!(),
                };
                let raw_descr = majit_ir::descr::make_array_descr(
                    0,
                    crate::virtualizable::item_size_for_type(array_info.item_type),
                    array_info.item_type,
                );
                (raw_opcode, raw_descr, ptr_opref)
            }
        };
        for index in 0..array_len {
            // compile.py:453 — ConstInt(index) for the array subscript.
            let const_opref = OpRef::const_int(next_const_idx);
            next_const_idx += 1;
            constants.insert(const_opref.raw(), index as i64);
            constant_types.insert(const_opref.raw(), Type::Int);

            let old_opref =
                OpRef::input_arg_typed(expanded_inputargs[i].index, expanded_inputargs[i].tp);
            let new_opref = OpRef::op_typed(next_opref, vinfo.array_fields[ai].item_type);
            next_opref += 1;
            let mut elem_op = Op::new(item_opcode, &[item_base, const_opref]);
            elem_op.pos = new_opref;
            elem_op.descr = Some(item_descr.clone());
            extra_ops.push(elem_op);
            set_local_forwarded(&mut forwarding, old_opref, new_opref);
            i += 1;
        }
    }

    assert!(
        i == expanded_inputargs.len(),
        "compile.py:458 assert i == len(inputargs) failed ({i} != {})",
        expanded_inputargs.len()
    );

    // compile.py:459-461 — emit_op walks the existing ops re-emitting
    // each one with `get_box_replacement` applied to args + fail_args.
    let original_ops = std::mem::take(ops);
    for op in original_ops.iter() {
        emit_forwarded_patch_op(&mut extra_ops, op, &mut forwarding, &mut next_opref);
    }
    *ops = extra_ops;
}

/// RPython dependency.py requires GUARD_(NO_)OVERFLOW to be scheduled only
/// when there is a live preceding INT_*_OVF operation to consume.
/// intbounds.py:231-242: optimizer raises InvalidLoop for stray overflow
/// guards. This function is a post-optimization safety net: if any stray
/// guard survived, strip it to prevent backend panic.
pub(crate) fn strip_stray_overflow_guards(ops: Vec<Op>) -> Vec<Op> {
    use majit_ir::OpCode;

    let mut pending_ovf = false;
    let mut result = Vec::with_capacity(ops.len());
    for op in ops {
        match op.opcode {
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf => {
                pending_ovf = true;
                result.push(op);
            }
            OpCode::GuardNoOverflow | OpCode::GuardOverflow => {
                if pending_ovf {
                    result.push(op);
                }
                // else: stray guard — strip it (intbounds.py:231 InvalidLoop
                // should have caught it; this is a safety net).
                pending_ovf = false;
            }
            OpCode::Label | OpCode::Jump | OpCode::Finish => {
                pending_ovf = false;
                result.push(op);
            }
            _ => {
                result.push(op);
            }
        }
    }
    result
}

pub(crate) fn enrich_guard_resume_layouts_for_trace(
    resume_layouts: &mut HashMap<u32, crate::resume::ResumeLayoutSummary>,
    exit_layouts: &mut HashMap<u32, StoredExitLayout>,
    trace_id: u64,
    inputargs: &[InputArg],
    trace_info: Option<&CompiledTraceInfo>,
) {
    for (fail_index, layout) in resume_layouts.iter_mut() {
        let recovery_layout = exit_layouts
            .get(fail_index)
            .and_then(|exit_layout| exit_layout.recovery_layout.clone());
        enrich_resume_layout_with_trace_metadata(
            layout,
            trace_id,
            inputargs,
            trace_info,
            recovery_layout.as_ref(),
        );
        if let Some(exit_layout) = exit_layouts.get_mut(fail_index) {
            exit_layout.resume_layout = Some(layout.clone());
        }
    }
}

pub(crate) fn patch_backend_guard_recovery_layouts_for_trace(
    backend: &mut dyn majit_backend::Backend,
    token: &majit_backend::JitCellToken,
    trace_id: u64,
    exit_layouts: &mut HashMap<u32, StoredExitLayout>,
) {
    for (&fail_index, exit_layout) in exit_layouts.iter_mut() {
        let Some(resume_layout) = exit_layout.resume_layout.as_ref() else {
            continue;
        };
        let recovery_layout = resume_layout
            .to_exit_recovery_layout_with_caller_prefix(exit_layout.recovery_layout.as_ref());
        if backend.update_fail_descr_recovery_layout(
            token,
            trace_id,
            fail_index,
            recovery_layout.clone(),
        ) {
            exit_layout.recovery_layout = Some(recovery_layout);
        }
    }
}

pub(crate) fn patch_backend_terminal_recovery_layouts_for_trace(
    backend: &mut dyn majit_backend::Backend,
    token: &majit_backend::JitCellToken,
    trace_id: u64,
    terminal_exit_layouts: &mut HashMap<usize, StoredExitLayout>,
) {
    for (&op_index, exit_layout) in terminal_exit_layouts.iter_mut() {
        let Some(resume_layout) = exit_layout.resume_layout.as_ref() else {
            continue;
        };
        let recovery_layout = resume_layout
            .to_exit_recovery_layout_with_caller_prefix(exit_layout.recovery_layout.as_ref());
        if backend.update_terminal_exit_recovery_layout(
            token,
            trace_id,
            op_index,
            recovery_layout.clone(),
        ) {
            exit_layout.recovery_layout = Some(recovery_layout);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// `rpython/jit/metainterp/compile.py:623-674` — finish/propagate descrs.
//
// These are ported as backend-agnostic `FailDescr` impls on the
// `majit-metainterp` side so `compile_tmp_callback` and
// `finish_setup` can reference the same singletons RPython does.
//
// `pyjitpl.py:2222` `compile.make_and_attach_done_descrs([self, cpu])` —
// RPython attaches a *single* `DoneWithThisFrameDescr*` object to both
// `MetaInterpStaticData` and the CPU so the FINISH descr pointer the
// backend observes is the same Arc the metainterp reads back in
// `handle_fail`. pyre mirrors the same shape through the
// `DescrContainer` trait implemented on both `MetaInterpStaticData`
// (pyjitpl/mod.rs) and `Backend` (majit-backend/lib.rs via the blanket
// impl below), so `MetaInterp::new` installs a single `Arc` on both
// halves; `attach_descrs_to_cpu` forwards the clones to the backend.
// ──────────────────────────────────────────────────────────────────────

/// `compile.py:623-624` `class _DoneWithThisFrameDescr(AbstractFailDescr):
/// final_descr = True`.
///
/// Shared base fields for the four `DoneWithThisFrame*` subclasses —
/// a stable `fail_arg_types` vector plus the `final_descr = True`
/// marker exposed through `FailDescr::is_finish()`.
#[derive(Debug)]
struct DoneWithThisFrameDescrBase {
    /// `history.py:122` `index = -1`.  For this descriptor family
    /// `set_descr_index` is never called (no `setup_descrs` pass); we
    /// keep the AbstractDescr default of -1.
    descr_index: std::sync::atomic::AtomicI32,
    /// `handle_fail` (`compile.py:632`, 641, 650, 659) reads the result
    /// out of `deadframe[0]`.  pyre carries the same one-slot shape via
    /// `fail_arg_types`.
    fail_arg_types: Vec<Type>,
}

impl DoneWithThisFrameDescrBase {
    fn new(fail_arg_types: Vec<Type>) -> Self {
        Self {
            descr_index: std::sync::atomic::AtomicI32::new(-1),
            fail_arg_types,
        }
    }
}

/// `compile.py:626-629` `class DoneWithThisFrameDescrVoid(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrVoid(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrVoid {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(Vec::new()))
    }
}

impl Default for DoneWithThisFrameDescrVoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for DoneWithThisFrameDescrVoid {
    fn get_descr_index(&self) -> i32 {
        self.0
            .descr_index
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0
            .descr_index
            .store(index, std::sync::atomic::Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrVoid {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        // `compile.py:624` `final_descr = True`.
        true
    }
}

/// `compile.py:631-638` `class DoneWithThisFrameDescrInt(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrInt(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrInt {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(vec![Type::Int]))
    }
}

impl Default for DoneWithThisFrameDescrInt {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for DoneWithThisFrameDescrInt {
    fn get_descr_index(&self) -> i32 {
        self.0
            .descr_index
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0
            .descr_index
            .store(index, std::sync::atomic::Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrInt {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        true
    }
}

/// `compile.py:640-647` `class DoneWithThisFrameDescrRef(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrRef(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrRef {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(vec![Type::Ref]))
    }
}

impl Default for DoneWithThisFrameDescrRef {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for DoneWithThisFrameDescrRef {
    fn get_descr_index(&self) -> i32 {
        self.0
            .descr_index
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0
            .descr_index
            .store(index, std::sync::atomic::Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrRef {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        true
    }
}

/// `compile.py:649-656` `class DoneWithThisFrameDescrFloat(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrFloat(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrFloat {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(vec![Type::Float]))
    }
}

impl Default for DoneWithThisFrameDescrFloat {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for DoneWithThisFrameDescrFloat {
    fn get_descr_index(&self) -> i32 {
        self.0
            .descr_index
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0
            .descr_index
            .store(index, std::sync::atomic::Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrFloat {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        true
    }
}

/// `compile.py:658-662` `class ExitFrameWithExceptionDescrRef(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct ExitFrameWithExceptionDescrRef(DoneWithThisFrameDescrBase);

impl ExitFrameWithExceptionDescrRef {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(vec![Type::Ref]))
    }
}

impl Default for ExitFrameWithExceptionDescrRef {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for ExitFrameWithExceptionDescrRef {
    fn get_descr_index(&self) -> i32 {
        self.0
            .descr_index
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0
            .descr_index
            .store(index, std::sync::atomic::Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for ExitFrameWithExceptionDescrRef {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        // `compile.py:658` inherits `final_descr = True` from `_DoneWithThisFrameDescr`.
        true
    }
    fn is_exit_frame_with_exception(&self) -> bool {
        // `compile.py:658` subclass identity: ExitFrameWithExceptionDescrRef
        // dispatches to `jitexc.ExitFrameWithExceptionRef` via `handle_fail`.
        true
    }
}

/// `compile.py:1092-1099` `class PropagateExceptionDescr(AbstractFailDescr)`.
///
/// `handle_fail` reads the exception out of the `deadframe` and raises
/// `jitexc.ExitFrameWithExceptionRef`.  Stored on
/// `JitDriverStaticData.propagate_exc_descr` and on
/// `MetaInterpStaticData.propagate_exception_descr` so
/// `compile_tmp_callback` can reference it when emitting the
/// `GUARD_NO_EXCEPTION` descriptor.
#[derive(Debug, Default)]
pub struct PropagateExceptionDescr {
    /// `history.py:122` `index = -1` default.
    descr_index: std::sync::atomic::AtomicI32,
    /// Unified-Descr Port Epic Session 2 scaffold: empty placeholder
    /// for the dynasm-backend payload (mirror of the `ResumeGuardDescr`
    /// scaffold).  RPython parity: `compile.py:1092
    /// PropagateExceptionDescr(AbstractFailDescr)` carries `rd_locs`
    /// and `adr_jump_offset` from `history.py:121` `_attrs_` directly.
    dynasm: DynasmDescrPayload,
    /// Mirror of `dynasm` for the cranelift backend.
    cranelift: CraneliftDescrPayload,
}

impl PropagateExceptionDescr {
    pub fn new() -> Self {
        Self {
            descr_index: std::sync::atomic::AtomicI32::new(-1),
            dynasm: DynasmDescrPayload::default(),
            cranelift: CraneliftDescrPayload::default(),
        }
    }
}

impl Descr for PropagateExceptionDescr {
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(std::sync::atomic::Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index
            .store(index, std::sync::atomic::Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for PropagateExceptionDescr {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        // `compile.py:1141` `ResOperation(rop.GUARD_NO_EXCEPTION, [], descr=faildescr)`
        // `operations[1].setfailargs([])` — no fail args.
        &[]
    }
    fn is_finish(&self) -> bool {
        // `compile.py:1092` `class PropagateExceptionDescr(AbstractFailDescr)` —
        // inherits `final_descr = False`.  This is a guard descr, not a finish.
        false
    }
}

/// `compile.py:665-674` `def make_and_attach_done_descrs(targets)`.
///
/// Creates one instance of each `DoneWithThisFrameDescr*` +
/// `ExitFrameWithExceptionDescrRef` and attaches them to each target
/// under the attributes `done_with_this_frame_descr_{void,int,ref,float}`
/// and `exit_frame_with_exception_descr_ref`.
///
/// pyre's `DescrContainer` trait (implemented by `MetaInterpStaticData`
/// and the CPU stand-in) exposes the five `set_*` hooks that mirror
/// the RPython `setattr(target, name, descr)` loop.
pub fn make_and_attach_done_descrs(targets: &mut [&mut dyn DescrContainer]) {
    let void: DescrRef = Arc::new(DoneWithThisFrameDescrVoid::new());
    let int: DescrRef = Arc::new(DoneWithThisFrameDescrInt::new());
    let ref_: DescrRef = Arc::new(DoneWithThisFrameDescrRef::new());
    let float: DescrRef = Arc::new(DoneWithThisFrameDescrFloat::new());
    let exc_ref: DescrRef = Arc::new(ExitFrameWithExceptionDescrRef::new());
    for target in targets.iter_mut() {
        target.set_done_with_this_frame_descr_void(void.clone());
        target.set_done_with_this_frame_descr_int(int.clone());
        target.set_done_with_this_frame_descr_ref(ref_.clone());
        target.set_done_with_this_frame_descr_float(float.clone());
        target.set_exit_frame_with_exception_descr_ref(exc_ref.clone());
    }
}

/// Trait hooked by `make_and_attach_done_descrs`.
///
/// `compile.py:673-674` `setattr(target, name, descr)` — in RPython each
/// target is a Python object with settable attributes; in Rust the
/// five setters make the contract explicit.  `MetaInterpStaticData`
/// and `dyn Backend` both implement this trait so a single call to
/// `make_and_attach_done_descrs(&mut [&mut sd, &mut *backend])`
/// mirrors RPython's `make_and_attach_done_descrs([self, cpu])`
/// exactly.
pub trait DescrContainer {
    fn set_done_with_this_frame_descr_void(&mut self, descr: DescrRef);
    fn set_done_with_this_frame_descr_int(&mut self, descr: DescrRef);
    fn set_done_with_this_frame_descr_ref(&mut self, descr: DescrRef);
    fn set_done_with_this_frame_descr_float(&mut self, descr: DescrRef);
    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: DescrRef);
}

/// `DescrContainer` blanket impl for `dyn Backend`.  Each setter
/// forwards to the corresponding `Backend::set_*` trait method so
/// backends that care about FINISH descr identity (dynasm / cranelift)
/// can override and store the `Arc`.  Backends that don't override
/// fall through to the no-op defaults — identity parity is still
/// maintained on the `MetaInterpStaticData` side.
impl DescrContainer for dyn Backend + '_ {
    fn set_done_with_this_frame_descr_void(&mut self, descr: DescrRef) {
        Backend::set_done_with_this_frame_descr_void(self, descr);
    }
    fn set_done_with_this_frame_descr_int(&mut self, descr: DescrRef) {
        Backend::set_done_with_this_frame_descr_int(self, descr);
    }
    fn set_done_with_this_frame_descr_ref(&mut self, descr: DescrRef) {
        Backend::set_done_with_this_frame_descr_ref(self, descr);
    }
    fn set_done_with_this_frame_descr_float(&mut self, descr: DescrRef) {
        Backend::set_done_with_this_frame_descr_float(self, descr);
    }
    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: DescrRef) {
        Backend::set_exit_frame_with_exception_descr_ref(self, descr);
    }
}

/// `rpython/jit/metainterp/compile.py:1101-1150` `compile_tmp_callback`.
///
/// Make a `JitCellToken` that corresponds to assembler code that just
/// calls back the interpreter.  Used temporarily: a fully compiled
/// version of the code may end up replacing it via
/// `redirect_call_assembler`.
///
/// The RPython-orthodox approach has **no separate "pending target
/// registry"**: every `JitCellToken` points at a real compiled body.
/// For an unfinished callee the body is a 3-op stub —
/// `CALL portal_runner_adr(funcbox, *greenboxes, *inputargs)` →
/// `GUARD_NO_EXCEPTION` → `FINISH` — which bounces control back into
/// the interpreter. Once the real trace compiles,
/// `redirect_call_assembler` (`x86/assembler.py:1138`) in-place patches
/// `_ll_function_addr` so callers reach the real loop transparently.
///
/// # Wiring status
///
/// The recursive `CALL_ASSEMBLER` path (`pyjitpl/mod.rs::handle_call_assembler`)
/// already routes pending callees through this helper. Step 2 — wiring
/// `warmstate::get_assembler_token` (`warmstate.py:714-723`) to mark the
/// resulting cell with `tmp=true` — and Step 3 — dropping
/// `register_pending_target` plus the cranelift/dynasm pending placeholder
/// registries entirely — remain pending (Tasks #195/#211).
///
/// # Parameters
///
/// `jitdriver_sd` must have `portal_runner_adr`, `portal_calldescr`,
/// `portal_finishtoken`, and `propagate_exc_descr` populated.  The
/// first three are set by `warmspot.py:1010-1017`; the last two by
/// `pyjitpl.py:2279-2281` (see
/// `MetaInterpStaticData::finish_setup_descrs_for_jitdrivers`).
pub fn compile_tmp_callback(
    backend: &mut dyn Backend,
    jitdriver_sd: &crate::jitdriver::JitDriverStaticData,
    token_number: u64,
    green_key: u64,
    greenboxes: &[Value],
    red_arg_types: &[Type],
) -> Result<Arc<JitCellToken>, BackendError> {
    // S2.1 invariant (wiggly-barto plan): every `JitDriverStaticData` reaching
    // `compile_tmp_callback` must have `portal_runner_adr` AND `portal_calldescr`
    // populated. `portal_runner_adr == 0` is the "attribute absent" sentinel
    // upstream (`warmspot.py:1010-1012` sets the address before any tmp_callback
    // can fire); `portal_calldescr.is_none()` means
    // `MetaInterpStaticData::finish_setup_descrs_for_jitdrivers` (pyjitpl/mod.rs:
    // 12336-12338, mirroring `pyjitpl.py:2274-2281` + `warmspot.py:1013-1017`)
    // never ran for this driver, so `funcbox` would dereference a null portal
    // address and the resulting tmp callback would jump to 0x0. `debug_assert!`
    // catches the misuse in dev/test builds (the bench harness runs in dev
    // profile so violations surface in `pyre/check.sh`); release builds opt
    // out for the same hot-path reason upstream avoids per-call asserts.
    debug_assert!(
        jitdriver_sd.portal_runner_adr != 0,
        "compile_tmp_callback: jitdriver_sd.portal_runner_adr is 0 — \
         warmspot.py:1010-1012 must populate portal_runner_adr before tmp_callback \
         can build a real funcbox"
    );
    debug_assert!(
        jitdriver_sd.portal_calldescr.is_some(),
        "compile_tmp_callback: jitdriver_sd.portal_calldescr is None — \
         MetaInterpStaticData::register_jitdriver_sd must have run \
         finish_setup_descrs_for_jitdrivers (pyjitpl.py:2274-2281 + \
         warmspot.py:1013-1017) so the CALL_* descr is available"
    );
    // The caller-supplied `red_arg_types` ↔ `jd.red_args_types` consistency
    // check lives below the upstream length assertion (`compile.py:1113`)
    // adjacent to the InputArg loop so the two signal-pairs (length /
    // typed-shape) sit together. See the `debug_assert_eq!` block at
    // `compile.py:1113` parity below.
    // `compile.py:1107` `jitcell_token = make_jitcell_token(jitdriver_sd)`.
    // Pyre adaptation: the token carries `green_key` (enabling cell lookup
    // on later CALL_ASSEMBLER) and `virtualizable_arg_index` (cached from
    // `jitdriver_sd.index_of_virtualizable`, matching the fields populated
    // on real-loop tokens at `compile_loop`).
    //
    // `compile.py:168` `jitcell_token.outermost_jitdriver_sd = jitdriver_sd`
    // is set inside `make_jitcell_token`.
    let mut jitcell_token = make_jitcell_token(token_number, jitdriver_sd.index);
    {
        let token = Arc::get_mut(&mut jitcell_token)
            .expect("fresh tmp callback JitCellToken must be uniquely owned");
        token.green_key = green_key;
        token.virtualizable_arg_index = jitdriver_sd.virtualizable_arg_index();
    }
    //
    // `compile.py:1110` `jl.tmp_callback(jitcell_token)` — JIT logger
    // marker.  PRE-EXISTING-ADAPTATION: `rpython/rlib/jit.py`'s `jl`
    // module is not ported; skip.
    //
    // `compile.py:1112` `nb_red_args = jitdriver_sd.num_red_args`.
    let nb_red_args = jitdriver_sd.num_red_args();
    // `compile.py:1113` `assert len(redargtypes) == nb_red_args`.
    assert_eq!(
        red_arg_types.len(),
        nb_red_args,
        "compile_tmp_callback: red_arg_types length mismatch",
    );
    // S2.4 contract (wiggly-barto plan): caller-passed `red_arg_types`
    // must match `jd.red_args_types` — upstream's `compile.py:1107-1124`
    // reads `redargtypes` from `jitdriver_sd.red_args_types` directly so
    // the tmp-callback signature is owned by the jd, not by the call
    // site. Pyre still threads `red_arg_types` through the parameter
    // list while runtime callers derive kinds from CALL_ASSEMBLER args
    // (see pyjitpl/mod.rs:10444-10451); this assertion locks the
    // invariant so the S2.4 cutover can drop the parameter without
    // a silent semantic shift.
    debug_assert_eq!(
        red_arg_types,
        jitdriver_sd.red_arg_types_as_ir_types().as_slice(),
        "compile_tmp_callback: caller-provided red_arg_types must match \
         jd.red_args_types — warmspot.py:664 makes the jd the source of truth"
    );
    // `compile.py:1114-1124` build `inputargs`:
    //     for kind in redargtypes:
    //         if kind == history.INT:   box = InputArgInt()
    //         elif kind == history.REF: box = InputArgRef()
    //         elif kind == history.FLOAT: box = InputArgFloat()
    //         ...
    //         inputargs.append(box)
    let inputargs: Vec<InputArg> = red_arg_types
        .iter()
        .enumerate()
        .map(|(i, kind)| match kind {
            Type::Int => InputArg::new_int(i as u32),
            Type::Ref => InputArg::new_ref(i as u32),
            Type::Float => InputArg::new_float(i as u32),
            Type::Void => panic!("compile_tmp_callback: void red arg is invalid"),
        })
        .collect();
    let num_inputs = inputargs.len() as u32;
    //
    // `compile.py:1125-1126`
    //     k = jitdriver_sd.portal_runner_adr
    //     funcbox = history.ConstInt(adr2int(k))
    //
    // `compile.py:1127` `callargs = [funcbox] + greenboxes + inputargs`.
    //
    // pyre layout: the CALL op's `args` slots reference `OpRef` numbers.
    // InputArgs occupy `0..num_inputs` and are minted via
    // `InputArg::opref()` so they carry RPython-parity InputArg{Int,
    // Float,Ref} variants (resoperation.py:719/727/739). Constants
    // (funcbox + greens) live in the dedicated CONST_BIT namespace and
    // are minted via `OpRef::const_typed` so each one is a Const{Int,
    // Float,Ptr} (history.py:220/261/307).
    const CONST_BASE: u32 = 10_000;
    let mut constants: HashMap<u32, i64> = HashMap::new();
    let mut constant_types: HashMap<u32, Type> = HashMap::new();
    // `compile.py:1126` funcbox = ConstInt(adr2int(k)).
    let funcbox_ref = OpRef::const_int(CONST_BASE);
    constants.insert(funcbox_ref.raw(), jitdriver_sd.portal_runner_adr);
    constant_types.insert(funcbox_ref.raw(), Type::Int);
    // Green boxes follow in declaration order.
    let mut callargs: Vec<OpRef> = Vec::with_capacity(1 + greenboxes.len() + inputargs.len());
    callargs.push(funcbox_ref);
    for (i, gb) in greenboxes.iter().enumerate() {
        let (raw, tp) = match *gb {
            Value::Int(v) => (v, Type::Int),
            Value::Ref(r) => (r.0 as i64, Type::Ref),
            Value::Float(f) => (f.to_bits() as i64, Type::Float),
            Value::Void => panic!("compile_tmp_callback: void greenbox"),
        };
        let g_ref = OpRef::const_typed(CONST_BASE + 1 + i as u32, tp);
        constants.insert(g_ref.raw(), raw);
        constant_types.insert(g_ref.raw(), tp);
        callargs.push(g_ref);
    }
    // Red args — inputargs occupy contiguous low OpRefs. Use
    // `InputArg::opref()` so each typed inputarg keeps its
    // resoperation.py:719/727/739 variant.
    for ia in inputargs.iter() {
        callargs.push(ia.opref());
    }
    //
    let portal_calldescr = jitdriver_sd
        .portal_calldescr
        .as_ref()
        .expect("compile_tmp_callback: jd.portal_calldescr not set")
        .clone();
    let portal_finishtoken = jitdriver_sd
        .portal_finishtoken
        .as_ref()
        .expect("compile_tmp_callback: jd.portal_finishtoken not set")
        .clone();
    let propagate_exc_descr = jitdriver_sd
        .propagate_exc_descr
        .as_ref()
        .expect("compile_tmp_callback: jd.propagate_exc_descr not set")
        .clone();
    //
    // `compile.py:1130` `jd = jitdriver_sd`.
    // `compile.py:1131` `opnum = OpHelpers.call_for_descr(jd.portal_calldescr)`.
    let call_opcode = OpCode::call_for_type(jitdriver_sd.result_type);
    // `compile.py:1132` `call_op = ResOperation(opnum, callargs,
    // descr=jd.portal_calldescr)`.
    let mut call_op = Op::with_descr(call_opcode, &callargs, portal_calldescr);
    //
    // `compile.py:1133-1136` `if call_op.type != 'v': finishargs = [call_op]
    // else: finishargs = []`.
    //
    // A void CALL leaves no result OpRef — match `Op::default_pos()` /
    // `OpRef::NONE` so dynasm/cranelift backends that only emit a store
    // when `op.pos != NONE` (e.g. `x86/assembler.rs` CALL handler) don't
    // produce a bogus result slot.
    let finishargs: Vec<OpRef> = if jitdriver_sd.result_type == Type::Void {
        Vec::new()
    } else {
        // The CALL writes to the first free OpRef after inputargs.
        // resoperation.py:564-638 IntOp/FloatOp/RefOp mixin: the result
        // box of a typed CALL is a typed ResOp variant.
        let call_result_ref = OpRef::op_typed(num_inputs, jitdriver_sd.result_type);
        call_op.pos = call_result_ref;
        vec![call_result_ref]
    };
    //
    // `compile.py:1138-1144` operations = [call_op,
    //   GUARD_NO_EXCEPTION(descr=faildescr),
    //   FINISH(finishargs, descr=jd.portal_finishtoken)].
    let mut guard_op = Op::with_descr(OpCode::GuardNoException, &[], propagate_exc_descr);
    // `compile.py:1144` `operations[1].setfailargs([])` — no fail args.
    guard_op.fail_args = Some(smallvec![]);
    let finish_op = Op::with_descr(OpCode::Finish, &finishargs, portal_finishtoken);
    let operations = vec![call_op, guard_op, finish_op];
    //
    // `compile.py:1145` `operations = get_deep_immutable_oplist(operations)` —
    // pyre has no immutable-list transformation.
    //
    // `compile.py:1146` `cpu.compile_loop(inputargs, operations, jitcell_token,
    // log=False)`.
    backend.set_constants(constants);
    backend.set_constant_types(constant_types);
    backend.compile_loop(
        &inputargs,
        &operations,
        Arc::get_mut(&mut jitcell_token)
            .expect("tmp callback JitCellToken must stay uniquely owned until backend compile"),
    )?;
    // `compile.py:180-181` wire wref now that all `Arc::get_mut` writes
    // have settled.  `compile_tmp_callback` doesn't go through
    // `record_loop_or_bridge` (the tmp callback is a synthetic
    // 3-instruction loop, not a real trace), so wref wiring happens
    // directly here.
    wire_clt_loop_token_wref(&jitcell_token);
    //
    // `compile.py:1148-1149` `if memory_manager is not None:
    //   memory_manager.keep_loop_alive(jitcell_token)` — pyre's
    // `BaseJitCell` holds the `Arc<JitCellToken>` once
    // `set_procedure_token(token, tmp=true)` runs in `warmstate.rs`.
    //
    // `compile.py:1150` `return jitcell_token`.
    // `compile.py:179-180` record_loop_or_bridge: the tmp-callback loop is a
    // real compiled loop even though MetaInterp never inserts it into
    // compiled_loops. Register it with the backend so `find_descr_by_ptr`
    // can still walk its fail_descrs on cross-token guard resolution.
    backend.track_compiled_token(Arc::clone(&jitcell_token));
    Ok(jitcell_token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::make_fail_descr_with_index;
    use crate::resume::{ResumeDataLoopMemo, SimpleBoxEnv, Snapshot, SnapshotFrame};
    use majit_ir::{ArrayFlag, Op, OpCode, OpRef};

    #[test]
    fn test_build_guard_metadata_keeps_vable_array_out_of_frame_slots() {
        use majit_backend::ExitValueSourceLayout;

        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.types.insert(0, Type::Ref);
        env.types.insert(1, Type::Int);
        env.constants
            .insert(OpRef::const_int(1).raw(), (8, Type::Int));
        env.constants
            .insert(OpRef::const_int(2).raw(), (777, Type::Int)); // code object payload
        env.constants
            .insert(OpRef::const_int(3).raw(), (2, Type::Int));
        env.constants
            .insert(OpRef::const_int(4).raw(), (999, Type::Int)); // namespace payload

        let snapshot = Snapshot {
            vable_array: vec![
                OpRef::input_arg_ref(0).into(),
                OpRef::const_int(1).into(),
                OpRef::const_int(2).into(),
                OpRef::const_int(3).into(),
                OpRef::const_int(4).into(),
            ],
            vref_array: vec![],
            framestack: vec![SnapshotFrame {
                jitcode_index: 0,
                pc: 8,
                boxes: vec![OpRef::input_arg_int(1).into()],
            }],
        };
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();
        let rd_consts = memo.consts().to_vec();

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef::input_arg_int(1)]);
        let descr = crate::compile::make_resume_guard_descr_typed(vec![Type::Ref, Type::Int]);
        if let Some(fd) = descr.as_fail_descr() {
            fd.set_rd_numb(Some(rd_numb));
            fd.set_rd_consts(Some(rd_consts));
        }
        guard.descr = Some(descr);
        guard.fail_args = Some(smallvec::smallvec![
            OpRef::input_arg_ref(0),
            OpRef::input_arg_int(1)
        ]);
        guard.fail_arg_types = Some(vec![Type::Ref, Type::Int]);

        let (_resume_data, exit_layouts) =
            build_guard_metadata(&inputargs, &[guard], 8, &HashMap::new(), None);
        let exit = exit_layouts.get(&0).expect("guard exit layout");

        let resume_layout = exit.resume_layout.as_ref().expect("resume_layout");
        assert_eq!(resume_layout.frame_layouts.len(), 1);
        assert_eq!(
            resume_layout.frame_layouts[0]
                .slot_layouts
                .iter()
                .map(|slot| slot.fail_arg_index)
                .collect::<Vec<_>>(),
            vec![1]
        );

        let recovery = exit.recovery_layout.as_ref().expect("recovery_layout");
        assert_eq!(recovery.frames.len(), 1);
        assert_eq!(
            recovery.frames[0].slots,
            vec![ExitValueSourceLayout::ExitValue(1),]
        );
        assert_eq!(
            recovery.frames[0].slot_types.as_ref().unwrap(),
            &vec![Type::Int]
        );
    }

    #[test]
    fn test_build_guard_metadata_prefers_explicit_fail_arg_types_over_stale_inputarg_types() {
        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_ref(1),
            InputArg::new_ref(2),
            InputArg::new_ref(3),
        ];
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef::input_arg_ref(0)]);
        let fail_arg_types = vec![Type::Ref, Type::Ref, Type::Int, Type::Int];
        let descr = make_fail_descr_with_index(0, fail_arg_types.len());
        descr
            .as_fail_descr()
            .unwrap()
            .set_fail_arg_types(fail_arg_types.clone());
        guard.descr = Some(descr);
        guard.fail_args = Some(smallvec::smallvec![
            OpRef::input_arg_ref(0),
            OpRef::input_arg_ref(1),
            OpRef::input_arg_ref(2),
            OpRef::input_arg_ref(3)
        ]);
        guard.fail_arg_types = Some(fail_arg_types);

        let (_resume_data, exit_layouts) =
            build_guard_metadata(&inputargs, &[guard], 0, &HashMap::new(), None);
        let exit = exit_layouts.get(&0).expect("guard exit layout");

        assert_eq!(
            exit.resolve_exit_types(),
            &[Type::Ref, Type::Ref, Type::Int, Type::Int][..]
        );
    }

    #[test]
    fn test_patch_new_loop_reemits_ops_through_forwarded_results() {
        let mut vinfo = crate::virtualizable::VirtualizableInfo::new(0);
        vinfo.add_field("obj", Type::Ref, 8);
        vinfo.set_parent_descr(majit_ir::descr::make_size_descr(16));

        let mut ops = vec![
            {
                let mut op = Op::new(OpCode::SameAsR, &[OpRef::input_arg_ref(1)]);
                op.pos = OpRef::ref_op(10);
                op
            },
            Op::new(OpCode::Label, &[OpRef::input_arg_ref(0), OpRef::ref_op(10)]),
            {
                let mut op = Op::new(OpCode::GetfieldGcPureI, &[OpRef::ref_op(10)]);
                op.pos = OpRef::int_op(11);
                op.descr = Some(majit_ir::descr::make_field_descr(
                    16,
                    8,
                    Type::Int,
                    ArrayFlag::Signed,
                ));
                op
            },
        ];
        let mut inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let mut constants = HashMap::new();
        let mut constant_types = HashMap::new();

        patch_new_loop_to_load_virtualizable_fields(
            &mut ops,
            &mut inputargs,
            &vinfo,
            &[],
            1,
            0,
            &mut constants,
            &mut constant_types,
        );

        assert_eq!(inputargs, vec![InputArg::new_ref(0)]);
        assert_eq!(ops.len(), 4);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        let vable_field = ops[0].pos;

        assert_eq!(ops[1].opcode, OpCode::SameAsR);
        assert_eq!(ops[1].args.as_slice(), &[vable_field]);
        let forwarded_same_as = ops[1].pos;
        assert_ne!(forwarded_same_as, OpRef::ref_op(10));

        assert_eq!(ops[2].opcode, OpCode::Label);
        assert_eq!(
            ops[2].args.as_slice(),
            &[OpRef::input_arg_ref(0), forwarded_same_as]
        );

        assert_eq!(ops[3].opcode, OpCode::GetfieldGcPureI);
        assert_eq!(ops[3].args.as_slice(), &[forwarded_same_as]);
    }

    #[test]
    fn test_patch_new_loop_reads_embedded_array_items_from_backing_storage() {
        let mut vinfo = crate::virtualizable::VirtualizableInfo::new(0);
        vinfo.add_embedded_array_field(
            "locals_cells_stack_w",
            Type::Ref,
            8,
            0,
            8,
            0,
            majit_ir::descr::make_array_descr(0, 8, Type::Ref),
        );
        vinfo.set_parent_descr(majit_ir::descr::make_size_descr(16));

        let mut ops = vec![Op::new(
            OpCode::Label,
            &[
                OpRef::input_arg_ref(0),
                OpRef::input_arg_ref(1),
                OpRef::input_arg_ref(2),
            ],
        )];
        let mut inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_ref(1),
            InputArg::new_ref(2),
        ];
        let mut constants = HashMap::new();
        let mut constant_types = HashMap::new();

        patch_new_loop_to_load_virtualizable_fields(
            &mut ops,
            &mut inputargs,
            &vinfo,
            &[2],
            1,
            0,
            &mut constants,
            &mut constant_types,
        );

        assert_eq!(inputargs, vec![InputArg::new_ref(0)]);
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[1].args.as_slice(), &[ops[0].pos]);
        assert_eq!(ops[2].opcode, OpCode::GetarrayitemRawR);
        assert_eq!(ops[2].args[0], ops[1].pos);
        assert_eq!(ops[3].opcode, OpCode::GetarrayitemRawR);
        assert_eq!(ops[3].args[0], ops[1].pos);
        assert_eq!(ops[4].opcode, OpCode::Label);
        assert_eq!(
            ops[4].args.as_slice(),
            &[OpRef::input_arg_ref(0), ops[2].pos, ops[3].pos]
        );
    }
}
/// `compile.py:855` ResumeGuardDescr `_attrs_ = ('rd_numb', 'rd_consts',
/// 'rd_virtuals', 'rd_pendingfields', 'status')` — the per-guard
/// resume payload shared by every concrete `AbstractResumeGuardDescr`
/// subclass.  Pyre stores them in `UnsafeCell` so the optimizer can
/// mutate the descr in place via `FailDescr::set_rd_*` without
/// breaking the `Arc<dyn FailDescr>` identity stamped on the op.
///
/// Each slot wraps `Arc<[T]>` so `copy_all_attributes_from`
/// (compile.py:861-867) — `self.rd_consts = other.rd_consts` etc. —
/// can mirror RPython's reference-share semantics with a single
/// `Arc::clone()` rather than a `Vec::clone()` that would deep-copy
/// the bytes.  External setters still accept `Option<Vec<T>>`; the
/// conversion to `Arc<[T]>` is one move per (rare) write.
#[derive(Debug)]
struct RdPayload {
    rd_numb: UnsafeCell<Option<Arc<[u8]>>>,
    rd_consts: UnsafeCell<Option<Arc<[Const]>>>,
    rd_virtuals: UnsafeCell<Option<Arc<[Rc<RdVirtualInfo>]>>>,
    rd_pendingfields: UnsafeCell<Option<Arc<[GuardPendingFieldEntry]>>>,
}

impl RdPayload {
    fn empty() -> Self {
        Self {
            rd_numb: UnsafeCell::new(None),
            rd_consts: UnsafeCell::new(None),
            rd_virtuals: UnsafeCell::new(None),
            rd_pendingfields: UnsafeCell::new(None),
        }
    }

    /// `clone()` shares every field — used by `clone_descr()`, which
    /// mirrors RPython's `ResumeGuardDescr.clone()` (compile.py:844-846).
    ///
    /// RPython `copy_all_attributes_from` (compile.py:861-867) does
    /// `self.rd_consts = other.rd_consts` etc. — list reference share.
    /// `Arc<[T]>` provides the equivalent: `Arc::clone()` only bumps a
    /// refcount.  In-place mutation never happens (the only writer is
    /// `set_rd_*` which swap-replaces the whole slot), so sharing is
    /// safe and observably identical to RPython.
    fn deep_clone(&self) -> Self {
        Self {
            rd_numb: UnsafeCell::new(unsafe { (*self.rd_numb.get()).clone() }),
            rd_consts: UnsafeCell::new(unsafe { (*self.rd_consts.get()).clone() }),
            rd_virtuals: UnsafeCell::new(unsafe { (*self.rd_virtuals.get()).clone() }),
            rd_pendingfields: UnsafeCell::new(unsafe { (*self.rd_pendingfields.get()).clone() }),
        }
    }

    fn rd_numb(&self) -> Option<&[u8]> {
        unsafe { (*self.rd_numb.get()).as_deref() }
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        unsafe { (*self.rd_numb.get()).clone() }
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        unsafe { *self.rd_numb.get() = value.map(Arc::from) }
    }
    fn set_rd_numb_arc(&self, value: Option<Arc<[u8]>>) {
        unsafe { *self.rd_numb.get() = value }
    }

    fn rd_consts(&self) -> Option<&[Const]> {
        unsafe { (*self.rd_consts.get()).as_deref() }
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        unsafe { (*self.rd_consts.get()).clone() }
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        unsafe { *self.rd_consts.get() = value.map(Arc::from) }
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<[Const]>>) {
        unsafe { *self.rd_consts.get() = value }
    }

    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        unsafe { (*self.rd_virtuals.get()).as_deref() }
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        unsafe { (*self.rd_virtuals.get()).clone() }
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        unsafe { *self.rd_virtuals.get() = value.map(Arc::from) }
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        unsafe { *self.rd_virtuals.get() = value }
    }

    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        unsafe { (*self.rd_pendingfields.get()).as_deref() }
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        unsafe { (*self.rd_pendingfields.get()).clone() }
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        unsafe { *self.rd_pendingfields.get() = value.map(Arc::from) }
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        unsafe { *self.rd_pendingfields.get() = value }
    }
}

fn push_vector_info(head: &mut Option<Box<AccumInfo>>, mut info: AccumInfo) {
    info.prev = head.take();
    *head = Some(Box::new(info));
}

fn flatten_vector_info(head: Option<&AccumInfo>) -> Vec<AccumInfo> {
    let mut result = Vec::new();
    let mut current = head;
    while let Some(info) = current {
        result.push(info.clone());
        current = info.prev.as_deref();
    }
    result
}

/// `compile.py:869 self.rd_vector_info = other.rd_vector_info.clone()`
/// rebuild helper: takes the donor's flattened chain (head at index 0)
/// and assembles the equivalent linked-list head suitable for writing
/// through `vector_info: UnsafeCell<Option<Box<AccumInfo>>>`.
fn build_vector_info_chain(chain: Vec<AccumInfo>) -> Option<Box<AccumInfo>> {
    let mut current: Option<Box<AccumInfo>> = None;
    for mut info in chain.into_iter().rev() {
        info.prev = current;
        current = Some(Box::new(info));
    }
    current
}

/// Global counter for unique fail_index allocation.
///
/// Mirrors RPython's ResumeGuardDescr numbering — each guard in every
/// compiled trace receives a unique fail_index so the backend can
/// report exactly which guard failed.
static NEXT_FAIL_INDEX: AtomicU32 = AtomicU32::new(1);

/// Reset the global fail_index counter (for testing).
#[cfg(test)]
pub fn reset_fail_index_counter() {
    NEXT_FAIL_INDEX.store(1, Ordering::SeqCst);
}

/// Allocate the next unique fail_index.
fn alloc_fail_index() -> u32 {
    NEXT_FAIL_INDEX.fetch_add(1, Ordering::SeqCst)
}

/// Per-descr dynasm-backend payload (Unified-Descr Port Epic, Session 2
/// scaffold).  PyPy parity model: `history.py:121 AbstractFailDescr._attrs_
/// = ('adr_jump_offset', 'rd_locs', 'rd_loop_token', 'rd_vector_info')`
/// keeps backend fields on the same descr object.  pyre's split currently
/// stores those backend fields on `DynasmFailDescr`
/// (`majit-backend-dynasm/src/guard.rs:25`); subsequent sessions of the
/// epic move them onto this typed payload so the `Arc<dyn FailDescr>` on
/// the IR op carries the backend state directly.  Empty in Session 2.
#[derive(Debug, Default)]
pub struct DynasmDescrPayload {}

/// Per-descr cranelift-backend payload (Unified-Descr Port Epic, Session 2
/// scaffold).  Mirror of `DynasmDescrPayload` for the cranelift backend's
/// counterpart fields on `CraneliftFailDescr`
/// (`majit-backend-cranelift/src/guard.rs:106`).  Empty in Session 2.
#[derive(Debug, Default)]
pub struct CraneliftDescrPayload {}

/// Per-guard backend FailDescr carrying a unique `fail_index`, the
/// runtime fail-arg `Type` vector, and a vectorization accumulator
/// chain.  Pyre-only adaptation: this is the bare backend descr used
/// when a guard does not yet carry resume data
/// (`ResumeGuardDescr` in this module is the resume-bearing variant).
///
/// **NOT** a port of RPython `history.py:156 BasicFailDescr`, which
/// is a test-only identifier descr.  The closest RPython analogue is
/// the abstract `AbstractFailDescr` (`history.py:131`) — pyre keeps
/// the `fail_arg_types` and `vector_info` slots that RPython would
/// otherwise host on `ResumeGuardDescr` so the bare and resume
/// variants share the same backend interface.
#[derive(Debug)]
struct MetaFailDescr {
    fail_index: u32,
    /// `compile.py:869 store_final_boxes` mutates the descr's types in
    /// place. Pyre uses `UnsafeCell` so the optimizer's
    /// `store_final_boxes_in_guard` can rewrite types after numbering
    /// without replacing the `Arc<dyn FailDescr>` (preserving identity
    /// and `fail_index`). Single-threaded JIT, no atomic needed.
    types: UnsafeCell<Vec<Type>>,
    /// schedule.py:654: vector accumulation info attached during vectorization.
    /// RPython history.py:127 rd_vector_info — no Mutex needed, single-threaded.
    vector_info: UnsafeCell<Option<Box<AccumInfo>>>,
    /// FINISH-vs-guard discriminator. Set by `make_finish_fail_descr_typed`
    /// when this descr stands in for an evicted FINISH op (terminal exit
    /// previous-token fallback) or for a test fallback that bypasses
    /// `MetaInterp::new`'s FINISH-singleton attachment. Default `false`
    /// matches the guard-exit case used by `make_fail_descr_typed`.
    is_finish: bool,
    /// Unified-Descr Port Epic Session 2 scaffold: empty placeholder
    /// for the dynasm-backend payload that future sessions migrate
    /// from `DynasmFailDescr` (`majit-backend-dynasm/src/guard.rs:25`)
    /// onto this descr (PyPy `history.py:121` single-object model).
    dynasm: DynasmDescrPayload,
    /// Mirror of `dynasm` for the cranelift backend.
    cranelift: CraneliftDescrPayload,
}

// Safety: JIT is single-threaded. UnsafeCell replaces Mutex for rd_vector_info.
unsafe impl Send for MetaFailDescr {}
unsafe impl Sync for MetaFailDescr {}

impl majit_ir::Descr for MetaFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn clone_descr(&self) -> Option<DescrRef> {
        // RPython: clone() preserves the concrete subtype.
        // MetaFailDescr.clone() → MetaFailDescr (same type, new fail_index).
        Some(Arc::new(MetaFailDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(unsafe { (&*self.types.get()).clone() }),
            vector_info: UnsafeCell::new(unsafe { (&*self.vector_info.get()).clone() }),
            is_finish: self.is_finish,
            dynasm: DynasmDescrPayload::default(),
            cranelift: CraneliftDescrPayload::default(),
        }))
    }
}

impl FailDescr for MetaFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        // Safety: single-threaded JIT, no concurrent writers.
        unsafe { &*self.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        // Safety: single-threaded JIT, no concurrent readers.
        unsafe { *self.types.get() = types }
    }
    fn is_finish(&self) -> bool {
        self.is_finish
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.vector_info.get() = build_vector_info_chain(chain) }
    }
}

/// Per-guard FailDescr that also carries resume data for deoptimization.
///
/// Mirrors RPython's ResumeGuardDescr with snapshot information.
/// When a guard fails, the backend uses the resume data to reconstruct
/// the interpreter state (virtual objects, frame variables, etc.).
#[derive(Debug)]
struct ResumeGuardDescr {
    fail_index: u32,
    /// `compile.py:869 store_final_boxes` mutates types in place; pyre
    /// uses `UnsafeCell` so identity is preserved across the optimizer.
    types: UnsafeCell<Vec<Type>>,
    /// Pyre keeps `resume_data` (the RPython-style ResumeValueSource
    /// payload used by `prepare_pendingfields` for the
    /// `PendingFieldInfo` path) on the descr alongside the RPython
    /// `_attrs_` rd_* slots — both representations co-exist while
    /// the runtime resume reader is being aligned with upstream.
    resume_data: ResumeData,
    /// `compile.py:855` `_attrs_ = ('rd_numb', 'rd_consts',
    /// 'rd_virtuals', 'rd_pendingfields', 'status')`.
    payload: RdPayload,
    /// RPython history.py:127 rd_vector_info — no Mutex needed, single-threaded.
    vector_info: UnsafeCell<Option<Box<AccumInfo>>>,
    /// `compile.py:186` `descr.rd_loop_token = clt` line-by-line port.
    ///
    /// The owning `Arc<CompiledLoopToken>` typed as `Arc<dyn Any +
    /// Send + Sync>` (the `token_handle_any` pattern — `majit-ir`
    /// cannot reference `majit-backend::CompiledLoopToken` without a
    /// dependency cycle).  Stamped by `record_loop_or_bridge`
    /// (`pyjitpl/mod.rs::record_loop_or_bridge`, compile.py:183-186
    /// counterpart) for every resume guard, alongside (and
    /// transitionally with) the same stamp on the backend descr.
    /// `pyjitpl.py:2897 resumedescr.rd_loop_token.loop_token_wref()`
    /// readers chain through this slot; `clt.loop_token_wref()`
    /// (compile.py:180-181) yields the owning `JitCellToken` weakref.
    rd_loop_token_clt: UnsafeCell<Option<std::sync::Arc<CompiledLoopToken>>>,
    /// Phase E.1 — backend-specific extension slot.
    ///
    /// In RPython the same descr object carries
    /// `llsupport/assembler.py:279 descr.rd_locs = positions` and
    /// `x86/assembler.py:857 tok.faildescr.adr_jump_offset = addr`
    /// alongside the metainterp-side `rd_*` payload via Python's
    /// dynamic-attribute model.  Rust direct translation: a
    /// type-erased `Box<dyn Any + Send + Sync>` filled by the backend
    /// at compile time (e.g. `DynasmGuardData`, `CraneliftGuardData`).
    /// `UnsafeCell` so identity is preserved across the write — the
    /// `Arc<dyn FailDescr>` address stamped on the IR op stays valid.
    backend_data: UnsafeCell<Option<Box<dyn std::any::Any + Send + Sync>>>,
    /// Unified-Descr Port Epic Session 2 scaffold: empty placeholder
    /// for the dynasm-backend payload that future sessions migrate
    /// from `DynasmFailDescr` (`majit-backend-dynasm/src/guard.rs:25`)
    /// onto this descr.  PyPy `history.py:121 AbstractFailDescr._attrs_`
    /// keeps `adr_jump_offset` / `rd_locs` / `rd_loop_token` on the
    /// same object as `rd_numb` / `rd_consts` etc.; this field is the
    /// pyre-side typed substitute for `backend_data` boxing.
    dynasm: DynasmDescrPayload,
    /// Mirror of `dynasm` for the cranelift backend.
    cranelift: CraneliftDescrPayload,
    /// Pyre-only: identifier of the compiled trace that owns this guard.
    ///
    /// RPython resolves descr identity directly by Python `id(descr)`
    /// (`history.py:125`).  Pyre's runtime exit path uses a
    /// `(trace_id, fail_index)` key when threading guard failures back
    /// from `runner.rs::find_descr` / `compiler.rs::find_descr_by_ptr`,
    /// so the owning trace_id is captured on the descr by
    /// `record_loop_or_bridge` (`compile.py:185-186` walker) and read
    /// directly via `FailDescr::trace_id`.
    trace_id: AtomicU64,
    /// Pyre-only: per-trace `fail_index` assigned by
    /// `build_guard_metadata`.  The struct's `fail_index` field above
    /// holds the global `alloc_fail_index()` value used for descr
    /// identity; this slot mirrors the per-trace key the optimizer
    /// hands the backend (`(trace_id, fail_index)` lookup parity).
    /// Stamped at `build_guard_metadata` time, where the local
    /// `fail_index` counter increments per guard.
    fail_index_per_trace: AtomicU32,
}

unsafe impl Send for ResumeGuardDescr {}
unsafe impl Sync for ResumeGuardDescr {}

impl majit_ir::Descr for ResumeGuardDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    /// compile.py:844-846: ResumeGuardDescr.clone()
    fn clone_descr(&self) -> Option<DescrRef> {
        Some(Arc::new(ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(unsafe { (&*self.types.get()).clone() }),
            resume_data: self.resume_data.clone(),
            payload: self.payload.deep_clone(),
            vector_info: UnsafeCell::new(unsafe { (&*self.vector_info.get()).clone() }),
            // Phase E.1: clone produces a fresh descr with no backend
            // data — RPython compile.py:844-846 `ResumeGuardDescr()`
            // mints a default-attributes object; backend-stored
            // extension fields are populated only when this fresh descr
            // reaches the backend codegen path
            // (`llsupport/assembler.py:279`).
            backend_data: UnsafeCell::new(None),
            rd_loop_token_clt: UnsafeCell::new(None),
            dynasm: DynasmDescrPayload::default(),
            cranelift: CraneliftDescrPayload::default(),
            // Cloned descrs reach the backend codegen path freshly, so
            // `record_loop_or_bridge` re-stamps `trace_id` for the
            // owning compiled trace.
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
        }))
    }
}

impl FailDescr for ResumeGuardDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn trace_id(&self) -> u64 {
        self.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.vector_info.get() = build_vector_info_chain(chain) }
    }

    fn rd_numb(&self) -> Option<&[u8]> {
        self.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        self.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<Arc<[u8]>>) {
        self.payload.set_rd_numb_arc(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        self.payload.rd_consts_arc()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<[Const]>>) {
        self.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.payload.set_rd_virtuals_arc(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.payload.set_rd_pendingfields_arc(value)
    }
    fn backend_data(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        // Safety: single-threaded JIT, no concurrent writers.
        unsafe { (*self.backend_data.get()).as_deref() }
    }
    fn set_backend_data(&self, data: Box<dyn std::any::Any + Send + Sync>) {
        // Safety: single-threaded JIT, no concurrent readers.
        unsafe { *self.backend_data.get() = Some(data) };
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        // compile.py:186 reader: return `&Arc<CompiledLoopToken>` typed
        // through `dyn Any` so `descr_owning_clt` can `downcast_ref::<Arc<
        // CompiledLoopToken>>()` consistently across backend + metainterp
        // descrs.  Safety: single-threaded JIT, no concurrent writers.
        let cell = unsafe { &*self.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        // compile.py:186 stamp.  Downcast at storage time so the cell
        // holds the strongly typed `Arc<CompiledLoopToken>` matching
        // backend descrs.  Safety: single-threaded JIT, no concurrent
        // readers.
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.rd_loop_token_clt.get() = Some(typed) };
    }
}

/// Create a FailDescr for `num_live` integer values with an auto-assigned
/// unique fail_index.
///
/// Each call produces a distinct fail_index so the backend can identify
/// which guard failed.
pub fn make_fail_descr(num_live: usize) -> DescrRef {
    Arc::new(MetaFailDescr {
        fail_index: alloc_fail_index(),
        types: UnsafeCell::new(vec![Type::Int; num_live]),
        vector_info: UnsafeCell::new(None),
        is_finish: false,
        dynasm: DynasmDescrPayload::default(),
        cranelift: CraneliftDescrPayload::default(),
    })
}

/// Create a FailDescr with an explicit fail_index. Tests only — see
/// `compile.rs::tests` for the invocation that needs a fixed fail_index
/// to align against a synthesised bridge descr.
#[cfg(test)]
pub fn make_fail_descr_with_index(fail_index: u32, num_live: usize) -> DescrRef {
    Arc::new(MetaFailDescr {
        fail_index,
        types: UnsafeCell::new(vec![Type::Int; num_live]),
        vector_info: UnsafeCell::new(None),
        is_finish: false,
        dynasm: DynasmDescrPayload::default(),
        cranelift: CraneliftDescrPayload::default(),
    })
}

/// Create a FailDescr with explicit types and auto-assigned fail_index.
pub fn make_fail_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(MetaFailDescr {
        fail_index: alloc_fail_index(),
        types: UnsafeCell::new(types),
        vector_info: UnsafeCell::new(None),
        is_finish: false,
        dynasm: DynasmDescrPayload::default(),
        cranelift: CraneliftDescrPayload::default(),
    })
}

/// FINISH-flavored variant of [`make_fail_descr_typed`]. Used by the
/// terminal-exit fallback in `merge_backend_terminal_exit_layouts` when
/// the FINISH op has been evicted, and by FINISH-singleton fallbacks in
/// tests that bypass `MetaInterp::new` (so the resulting descr's
/// `is_finish()` matches `compile.py:658-662 ExitFrameWithExceptionDescrRef`
/// / `pyjitpl.py:3198-3220 compile_done_with_this_frame` semantics).
pub fn make_finish_fail_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(MetaFailDescr {
        fail_index: alloc_fail_index(),
        types: UnsafeCell::new(types),
        vector_info: UnsafeCell::new(None),
        is_finish: true,
        dynasm: DynasmDescrPayload::default(),
        cranelift: CraneliftDescrPayload::default(),
    })
}

/// compile.py:840-843 `ResumeGuardDescr` parity: a fresh guard descr
/// carrying the post-numbering `fail_arg_types`. Used by
/// `store_final_boxes_in_guard` to replace the tracer-stamped
/// `MetaFailDescr` (whose `types` reflect the pre-numbering snapshot)
/// with a descr whose `fail_arg_types()` matches `op.fail_arg_types`
/// exactly.
///
/// `payload` is initialized empty here; `store_final_boxes_in_guard`
/// at optimizeopt/mod.rs:3508 fills `rd_numb / rd_consts / rd_virtuals
/// / rd_pendingfields` post-numbering through the descr-side
/// `set_rd_*` setters (compile.py:855 `_attrs_`).  The legacy
/// `ResumeData` field is kept only for tests that still mint synthetic
/// guards; production reads route through `payload`.
pub fn make_resume_guard_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeGuardDescr {
        fail_index: alloc_fail_index(),
        types: UnsafeCell::new(types),
        resume_data: ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: Vec::new(),
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        },
        payload: RdPayload::empty(),
        vector_info: UnsafeCell::new(None),
        backend_data: UnsafeCell::new(None),
        rd_loop_token_clt: UnsafeCell::new(None),
        dynasm: DynasmDescrPayload::default(),
        cranelift: CraneliftDescrPayload::default(),
        trace_id: AtomicU64::new(0),
        fail_index_per_trace: AtomicU32::new(0),
    })
}

/// compile.py:892: ResumeAtPositionDescr(ResumeGuardDescr) — subclass
/// with no additional fields or method overrides. Type tag only.
///
/// In RPython, ResumeAtPositionDescr inherits all of ResumeGuardDescr's
/// fields (rd_numb, rd_consts, rd_virtuals, rd_pendingfields) and its
/// clone() method (which calls copy_all_attributes_from). The only
/// difference is the type tag used by compile_trace to decide
/// inline_short_preamble.
///
/// We model this as a newtype wrapping ResumeGuardDescr so that
/// clone_descr() produces a plain ResumeGuardDescr with resume data
/// preserved — matching RPython's inherited clone() behavior exactly.
#[derive(Debug)]
pub struct ResumeAtPositionDescr {
    inner: ResumeGuardDescr,
}

// Safety: same as ResumeGuardDescr (single-threaded JIT).
unsafe impl Send for ResumeAtPositionDescr {}
unsafe impl Sync for ResumeAtPositionDescr {}

impl majit_ir::Descr for ResumeAtPositionDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_resume_at_position(&self) -> bool {
        true
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    // compile.py:878-881: inherited ResumeGuardDescr.clone() →
    // plain ResumeGuardDescr with copy_all_attributes_from(self).
    // Marker lost, resume data preserved.
    fn clone_descr(&self) -> Option<DescrRef> {
        self.inner.clone_descr()
    }
}

impl FailDescr for ResumeAtPositionDescr {
    fn fail_index(&self) -> u32 {
        self.inner.fail_index
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.inner.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.inner.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.inner.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.inner.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.inner.vector_info.get() = build_vector_info_chain(chain) }
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        self.inner.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<Arc<[u8]>>) {
        self.inner.payload.set_rd_numb_arc(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        self.inner.payload.rd_consts_arc()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<[Const]>>) {
        self.inner.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.inner.payload.set_rd_virtuals_arc(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.inner.payload.set_rd_pendingfields_arc(value)
    }
    fn backend_data(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        unsafe { (*self.inner.backend_data.get()).as_deref() }
    }
    fn set_backend_data(&self, data: Box<dyn std::any::Any + Send + Sync>) {
        unsafe { *self.inner.backend_data.get() = Some(data) };
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.inner.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.inner.rd_loop_token_clt.get() = Some(typed) };
    }
}

/// Create a ResumeAtPositionDescr with auto-assigned fail_index, the
/// supplied `types`, and empty resume data.
pub fn make_resume_at_position_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeAtPositionDescr {
        inner: ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(types),
            resume_data: ResumeData {
                vable_array: Vec::new(),
                vref_array: Vec::new(),
                frames: Vec::new(),
                virtuals: Vec::new(),
                pending_fields: Vec::new(),
            },
            payload: RdPayload::empty(),
            vector_info: UnsafeCell::new(None),
            backend_data: UnsafeCell::new(None),
            rd_loop_token_clt: UnsafeCell::new(None),
            dynasm: DynasmDescrPayload::default(),
            cranelift: CraneliftDescrPayload::default(),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
        },
    })
}

/// Create a ResumeAtPositionDescr with auto-assigned fail_index and
/// empty resume data + empty types. The optimizer's
/// `store_final_boxes_in_guard` mutates `types` in place via
/// `FailDescr::set_fail_arg_types` (preserving subtype + fail_index).
pub fn make_resume_at_position_descr() -> DescrRef {
    make_resume_at_position_descr_typed(Vec::new())
}

/// compile.py:945-948: ResumeGuardForcedDescr(ResumeGuardDescr) — subtype
/// minted by `invent_fail_descr_for_op` for `GUARD_NOT_FORCED` /
/// `GUARD_NOT_FORCED_2`. Upstream attaches `metainterp_sd` /
/// `jitdriver_sd` via `_init` (compile.py:946-948) so
/// `handle_async_forcing` (compile.py:986) can call back into resume
/// during a residual call.
///
/// PYRE-ADAPTATION: pyre's forced-guard handling currently routes
/// through opcode checks (`pyjitpl/mod.rs` GUARD_NOT_FORCED chain),
/// not the descr's `handle_fail`, so this subtype is tag-only.
/// `is_guard_forced()` returns true so descr-keyed dispatch can
/// migrate later without reshaping the optimizer call site.
#[derive(Debug)]
pub struct ResumeGuardForcedDescr {
    inner: ResumeGuardDescr,
}

unsafe impl Send for ResumeGuardForcedDescr {}
unsafe impl Sync for ResumeGuardForcedDescr {}

impl majit_ir::Descr for ResumeGuardForcedDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_guard_forced(&self) -> bool {
        true
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    /// compile.py:873-876 ResumeGuardDescr.clone() — `ResumeGuardForcedDescr`
    /// inherits the base implementation (no override at compile.py:939+),
    /// so cloning produces a plain `ResumeGuardDescr` with resume attributes
    /// copied over via `copy_all_attributes_from`. The Forced subtype tag
    /// is intentionally dropped.
    fn clone_descr(&self) -> Option<DescrRef> {
        self.inner.clone_descr()
    }
}

impl FailDescr for ResumeGuardForcedDescr {
    fn fail_index(&self) -> u32 {
        self.inner.fail_index
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.inner.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.inner.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.inner.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.inner.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.inner.vector_info.get() = build_vector_info_chain(chain) }
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        self.inner.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<Arc<[u8]>>) {
        self.inner.payload.set_rd_numb_arc(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        self.inner.payload.rd_consts_arc()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<[Const]>>) {
        self.inner.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.inner.payload.set_rd_virtuals_arc(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.inner.payload.set_rd_pendingfields_arc(value)
    }
    fn backend_data(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        unsafe { (*self.inner.backend_data.get()).as_deref() }
    }
    fn set_backend_data(&self, data: Box<dyn std::any::Any + Send + Sync>) {
        unsafe { *self.inner.backend_data.get() = Some(data) };
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.inner.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.inner.rd_loop_token_clt.get() = Some(typed) };
    }
}

/// Create a ResumeGuardForcedDescr with auto-assigned fail_index, the
/// supplied `types`, and empty resume data.
pub fn make_resume_guard_forced_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeGuardForcedDescr {
        inner: ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(types),
            resume_data: ResumeData {
                vable_array: Vec::new(),
                vref_array: Vec::new(),
                frames: Vec::new(),
                virtuals: Vec::new(),
                pending_fields: Vec::new(),
            },
            payload: RdPayload::empty(),
            vector_info: UnsafeCell::new(None),
            backend_data: UnsafeCell::new(None),
            rd_loop_token_clt: UnsafeCell::new(None),
            dynasm: DynasmDescrPayload::default(),
            cranelift: CraneliftDescrPayload::default(),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
        },
    })
}

/// compile.py:888-889: ResumeGuardExcDescr(ResumeGuardDescr) — subtype
/// minted by `invent_fail_descr_for_op` for `GUARD_EXCEPTION` /
/// `GUARD_NO_EXCEPTION`. Upstream uses `pass` to make it a tag-only
/// subclass; `handle_fail` routes the exception path off this tag.
#[derive(Debug)]
pub struct ResumeGuardExcDescr {
    inner: ResumeGuardDescr,
}

unsafe impl Send for ResumeGuardExcDescr {}
unsafe impl Sync for ResumeGuardExcDescr {}

impl majit_ir::Descr for ResumeGuardExcDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_guard_exc(&self) -> bool {
        true
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    /// compile.py:881-882 `class ResumeGuardExcDescr(ResumeGuardDescr): pass`
    /// — no clone() override, so inheriting compile.py:873-876
    /// `ResumeGuardDescr.clone()` produces a plain `ResumeGuardDescr` with
    /// resume attributes copied via `copy_all_attributes_from`. The Exc
    /// subtype tag is intentionally dropped.
    fn clone_descr(&self) -> Option<DescrRef> {
        self.inner.clone_descr()
    }
}

impl FailDescr for ResumeGuardExcDescr {
    fn fail_index(&self) -> u32 {
        self.inner.fail_index
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.inner.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.inner.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.inner.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.inner.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.inner.vector_info.get() = build_vector_info_chain(chain) }
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        self.inner.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<Arc<[u8]>>) {
        self.inner.payload.set_rd_numb_arc(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        self.inner.payload.rd_consts_arc()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<[Const]>>) {
        self.inner.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.inner.payload.set_rd_virtuals_arc(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.inner.payload.set_rd_pendingfields_arc(value)
    }
    fn backend_data(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        unsafe { (*self.inner.backend_data.get()).as_deref() }
    }
    fn set_backend_data(&self, data: Box<dyn std::any::Any + Send + Sync>) {
        unsafe { *self.inner.backend_data.get() = Some(data) };
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.inner.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.inner.rd_loop_token_clt.get() = Some(typed) };
    }
}

/// Create a ResumeGuardExcDescr with auto-assigned fail_index, the
/// supplied `types`, and empty resume data.
pub fn make_resume_guard_exc_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeGuardExcDescr {
        inner: ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(types),
            resume_data: ResumeData {
                vable_array: Vec::new(),
                vref_array: Vec::new(),
                frames: Vec::new(),
                virtuals: Vec::new(),
                pending_fields: Vec::new(),
            },
            payload: RdPayload::empty(),
            vector_info: UnsafeCell::new(None),
            backend_data: UnsafeCell::new(None),
            rd_loop_token_clt: UnsafeCell::new(None),
            dynasm: DynasmDescrPayload::default(),
            cranelift: CraneliftDescrPayload::default(),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
        },
    })
}

/// compile.py:832-851: `ResumeGuardCopiedDescr(prev)` —
/// shared-resume subtype minted by `invent_fail_descr_for_op` when
/// `_copy_resume_data_from` shares a donor guard's resume data.
/// `get_resumestorage()` (compile.py:849) returns the donor
/// `ResumeGuardDescr` so reads chase through to the original
/// `rd_numb` / `rd_consts` / `rd_virtuals` / `rd_pendingfields`.
///
/// Reads route through `prev_descr()` (compile.py:849
/// `get_resumestorage(): return prev`); every `rd_*` getter on
/// `FailDescr` chases the donor stored in `prev` so a copied descr
/// has no owned resume payload of its own.
#[derive(Debug)]
pub struct ResumeGuardCopiedDescr {
    fail_index: u32,
    /// compile.py:836: `assert isinstance(prev, ResumeGuardDescr)`.
    /// pyre keeps the donor as a `DescrRef` so chained sharing
    /// (`prev.prev` etc.) can be walked uniformly through
    /// `prev_descr()` until a non-copied descr is reached.
    ///
    /// `compile.py:840-842 ResumeGuardCopiedDescr.copy_all_attributes_from`
    /// mutates `self.prev = other.prev` in place, preserving the
    /// receiver's identity (`fail_index` / status).  Pyre wraps it in
    /// `UnsafeCell` so the optimizer-side helper can swap the donor
    /// pointer through `&self` without minting a new Arc — same
    /// single-threaded contract used for the `rd_*` cells.
    prev: UnsafeCell<DescrRef>,
    /// history.py:125 `_attrs_ = ('adr_jump_offset', 'rd_locs',
    /// 'rd_loop_token', 'rd_vector_info')` — `rd_vector_info` lives on
    /// `AbstractFailDescr` itself, not on the resume storage.  Copied
    /// descrs share their donor's resume payload via `prev`, but each
    /// guard owns its own vector-info chain (history.py:143
    /// `attach_vector_info` writes `self.rd_vector_info`).
    vector_info: UnsafeCell<Option<Box<AccumInfo>>>,
    /// Phase E.1 — backend-specific extension slot.  Copied descrs
    /// share their donor's *resume* payload via `prev`
    /// (`compile.py:849 get_resumestorage(): return prev`), but the
    /// backend-stored fields (`rd_locs`, `adr_jump_offset`) are written
    /// per-fail at codegen time (`llsupport/assembler.py:279`), so
    /// each copied descr owns its own slot — same scoping as
    /// `vector_info` above (history.py:143 writes `self.rd_vector_info`,
    /// not `prev.rd_vector_info`).
    backend_data: UnsafeCell<Option<Box<dyn std::any::Any + Send + Sync>>>,
    /// `compile.py:186` `descr.rd_loop_token = clt`.  Copied descrs are
    /// stamped per-guard by the same `record_loop_or_bridge` walker
    /// (`compile.py:185 isinstance(descr, ResumeDescr)` covers
    /// `ResumeGuardCopiedDescr` — `_attrs_` lists `rd_loop_token`
    /// directly on `AbstractFailDescr` at history.py:125, owned by
    /// the receiver, not chased through `prev`).
    rd_loop_token_clt: UnsafeCell<Option<std::sync::Arc<CompiledLoopToken>>>,
    /// Pyre-only owning-trace identifier — same role as on
    /// `ResumeGuardDescr`. Stamped by `record_loop_or_bridge`
    /// (`compile.py:185-186` walker) since copied descrs are equally
    /// owned by exactly one trace.
    trace_id: AtomicU64,
    /// Pyre-only per-trace fail-index — same role as on
    /// `ResumeGuardDescr`. Stamped by `build_guard_metadata`.
    fail_index_per_trace: AtomicU32,
}

unsafe impl Send for ResumeGuardCopiedDescr {}
unsafe impl Sync for ResumeGuardCopiedDescr {}

impl ResumeGuardCopiedDescr {
    /// Read the current `prev` Arc.
    fn prev(&self) -> &DescrRef {
        // Safety: single-threaded JIT, no concurrent writers.
        unsafe { &*self.prev.get() }
    }
    /// `compile.py:842 self.prev = other.prev` — overwrite the donor
    /// pointer in place. Identity (`fail_index` / subtype tag) stays.
    fn set_prev(&self, prev: DescrRef) {
        // Safety: single-threaded JIT, no concurrent readers.
        unsafe { *self.prev.get() = prev }
    }
}

impl majit_ir::Descr for ResumeGuardCopiedDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_resume_guard_copied(&self) -> bool {
        true
    }
    fn prev_descr(&self) -> Option<DescrRef> {
        Some(self.prev().clone())
    }
    fn set_prev_descr(&self, prev: DescrRef) {
        self.set_prev(prev);
    }
    /// compile.py:843-846: `clone()` constructs a fresh
    /// `ResumeGuardCopiedDescr(self.prev)` — identity on `prev` is
    /// preserved (Arc share), only `fail_index` is fresh.
    fn clone_descr(&self) -> Option<DescrRef> {
        // history.py:127 `rd_vector_info = None` is the class default;
        // `clone()` does not copy it (compile.py:843-846 only forwards
        // `prev`).  Mint a fresh empty chain.
        Some(Arc::new(ResumeGuardCopiedDescr {
            fail_index: alloc_fail_index(),
            prev: UnsafeCell::new(self.prev().clone()),
            vector_info: UnsafeCell::new(None),
            backend_data: UnsafeCell::new(None),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
        }))
    }
}

impl FailDescr for ResumeGuardCopiedDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn trace_id(&self) -> u64 {
        self.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    /// compile.py:849 `get_resumestorage(): return prev`: reads chase
    /// to the donor.  The `fail_arg_types` slot is shared too —
    /// upstream stores the type list on the donor `ResumeGuardDescr`,
    /// which `prev` references.
    fn fail_arg_types(&self) -> &[Type] {
        self.prev()
            .as_fail_descr()
            .map(|fd| fd.fail_arg_types())
            .unwrap_or(&[])
    }
    /// `_copy_resume_data_from` does not call
    /// `store_final_boxes_in_guard`, so the optimizer never invokes
    /// `set_fail_arg_types` on a copied descr.  Match RPython's
    /// implicit invariant by panicking — a setter that wrote
    /// through to `prev` would silently mutate the donor's type
    /// vector and is never the desired behavior.
    fn set_fail_arg_types(&self, _types: Vec<Type>) {
        panic!(
            "set_fail_arg_types invoked on a ResumeGuardCopiedDescr — \
             RPython optimizer.py:724 only allows ResumeGuardDescr; \
             copied descrs share their donor's type vector via prev"
        );
    }
    /// history.py:143 `AbstractFailDescr.attach_vector_info`: writes
    /// `self.rd_vector_info`, never `self.prev`.  `prev` is for resume
    /// storage only (compile.py:849 `get_resumestorage`); vector info
    /// lives on the copied descr itself.
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.vector_info.get() = build_vector_info_chain(chain) }
    }

    // compile.py:849 `get_resumestorage(): return prev` — every rd_*
    // read chases through to the donor descr.  Setters panic for the
    // same reason `set_fail_arg_types` does: `_copy_resume_data_from`
    // never finalizes a copied descr; mutation must go through the
    // donor's own `ResumeGuardDescr`.
    fn rd_numb(&self) -> Option<&[u8]> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_numb())
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_numb_arc())
    }
    fn set_rd_numb(&self, _value: Option<Vec<u8>>) {
        panic!(
            "set_rd_numb invoked on a ResumeGuardCopiedDescr — \
             upstream optimizer.py:728 only finalizes ResumeGuardDescr"
        );
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_consts())
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_consts_arc())
    }
    fn set_rd_consts(&self, _value: Option<Vec<Const>>) {
        panic!(
            "set_rd_consts invoked on a ResumeGuardCopiedDescr — \
             upstream optimizer.py:728 only finalizes ResumeGuardDescr"
        );
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_virtuals())
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_virtuals_arc())
    }
    fn set_rd_virtuals(&self, _value: Option<Vec<Rc<RdVirtualInfo>>>) {
        panic!(
            "set_rd_virtuals invoked on a ResumeGuardCopiedDescr — \
             upstream optimizer.py:728 only finalizes ResumeGuardDescr"
        );
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_pendingfields())
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_pendingfields_arc())
    }
    fn set_rd_pendingfields(&self, _value: Option<Vec<GuardPendingFieldEntry>>) {
        panic!(
            "set_rd_pendingfields invoked on a ResumeGuardCopiedDescr — \
             upstream optimizer.py:728 only finalizes ResumeGuardDescr"
        );
    }
    fn backend_data(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        unsafe { (*self.backend_data.get()).as_deref() }
    }
    fn set_backend_data(&self, data: Box<dyn std::any::Any + Send + Sync>) {
        unsafe { *self.backend_data.get() = Some(data) };
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.rd_loop_token_clt.get() = Some(typed) };
    }
}

/// compile.py:891-892: `class ResumeGuardCopiedExcDescr(ResumeGuardCopiedDescr): pass`
/// — exception variant of the shared-resume descr, minted by
/// `invent_fail_descr_for_op` for `GUARD_EXCEPTION` /
/// `GUARD_NO_EXCEPTION` on the sharing path.
#[derive(Debug)]
pub struct ResumeGuardCopiedExcDescr {
    inner: ResumeGuardCopiedDescr,
}

unsafe impl Send for ResumeGuardCopiedExcDescr {}
unsafe impl Sync for ResumeGuardCopiedExcDescr {}

impl majit_ir::Descr for ResumeGuardCopiedExcDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_resume_guard_copied(&self) -> bool {
        true
    }
    fn is_guard_exc(&self) -> bool {
        true
    }
    fn prev_descr(&self) -> Option<DescrRef> {
        Some(self.inner.prev().clone())
    }
    fn set_prev_descr(&self, prev: DescrRef) {
        self.inner.set_prev(prev);
    }
    fn clone_descr(&self) -> Option<DescrRef> {
        Some(Arc::new(ResumeGuardCopiedExcDescr {
            inner: ResumeGuardCopiedDescr {
                fail_index: alloc_fail_index(),
                prev: UnsafeCell::new(self.inner.prev().clone()),
                vector_info: UnsafeCell::new(None),
                backend_data: UnsafeCell::new(None),
                rd_loop_token_clt: UnsafeCell::new(None),
                trace_id: AtomicU64::new(0),
                fail_index_per_trace: AtomicU32::new(0),
            },
        }))
    }
}

impl FailDescr for ResumeGuardCopiedExcDescr {
    fn fail_index(&self) -> u32 {
        self.inner.fail_index
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        self.inner.fail_arg_types()
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        self.inner.set_fail_arg_types(types)
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        self.inner.attach_vector_info(info)
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        self.inner.vector_info()
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        self.inner.replace_vector_info(chain)
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        self.inner.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.set_rd_numb(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        self.inner.rd_consts_arc()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.set_rd_consts(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.set_rd_virtuals(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.set_rd_pendingfields(value)
    }
    fn backend_data(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        self.inner.backend_data()
    }
    fn set_backend_data(&self, data: Box<dyn std::any::Any + Send + Sync>) {
        self.inner.set_backend_data(data)
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        self.inner.rd_loop_token_clt()
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        self.inner.set_rd_loop_token_clt(clt)
    }
}

/// Mint a `ResumeGuardCopiedDescr` whose `get_resumestorage()` chases
/// back to `prev`.  `prev` must already carry the donor's
/// `fail_arg_types` (RPython invariant: copied descrs share the
/// donor's type vector via `get_resumestorage`).
///
/// compile.py:835-838 `ResumeGuardCopiedDescr.__init__`:
///   `assert isinstance(prev, ResumeGuardDescr)` —
/// the donor must be a head-of-chain ResumeGuardDescr (or its
/// subclasses), never another ResumeGuardCopiedDescr.  Two-hop
/// chasing would be silent in pyre (`prev.prev` returns the head
/// anyway) but masks real bugs in the optimizer's sharing path.
pub fn make_resume_guard_copied_descr(prev: DescrRef) -> DescrRef {
    // compile.py:837-838 ResumeGuardCopiedDescr.__init__:
    //   assert isinstance(prev, ResumeGuardDescr)
    // The donor must be a head-of-chain ResumeGuardDescr (or any
    // subclass thereof: ResumeAtPositionDescr / ResumeGuardForcedDescr /
    // ResumeGuardExcDescr / CompileLoopVersionDescr).  Reject siblings
    // (ResumeGuardCopiedDescr itself) and unrelated FailDescr subtypes
    // (MetaFailDescr) — the descr-side rd_* readers chase `prev` at
    // resume time and would observe garbage if prev cannot carry resume
    // data.
    assert!(
        prev.is_resume_guard(),
        "compile.py:838 assert isinstance(prev, ResumeGuardDescr): \
         donor must be a ResumeGuardDescr subclass (got descr_index={:?})",
        prev.index()
    );
    Arc::new(ResumeGuardCopiedDescr {
        fail_index: alloc_fail_index(),
        prev: UnsafeCell::new(prev),
        vector_info: UnsafeCell::new(None),
        backend_data: UnsafeCell::new(None),
        rd_loop_token_clt: UnsafeCell::new(None),
        trace_id: AtomicU64::new(0),
        fail_index_per_trace: AtomicU32::new(0),
    })
}

/// Mint a `ResumeGuardCopiedExcDescr` for the GUARD_EXCEPTION /
/// GUARD_NO_EXCEPTION sharing path.
///
/// compile.py:889-890 `ResumeGuardCopiedExcDescr` inherits
/// `ResumeGuardCopiedDescr.__init__`, so the same
/// `isinstance(prev, ResumeGuardDescr)` invariant applies.
pub fn make_resume_guard_copied_exc_descr(prev: DescrRef) -> DescrRef {
    // compile.py:889-890 `class ResumeGuardCopiedExcDescr(...)` inherits
    // `ResumeGuardCopiedDescr.__init__`; same `isinstance(prev,
    // ResumeGuardDescr)` invariant.
    assert!(
        prev.is_resume_guard(),
        "compile.py:838 assert isinstance(prev, ResumeGuardDescr): \
         ResumeGuardCopiedExcDescr donor must be a ResumeGuardDescr \
         subclass (got descr_index={:?})",
        prev.index()
    );
    Arc::new(ResumeGuardCopiedExcDescr {
        inner: ResumeGuardCopiedDescr {
            fail_index: alloc_fail_index(),
            prev: UnsafeCell::new(prev),
            vector_info: UnsafeCell::new(None),
            backend_data: UnsafeCell::new(None),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
        },
    })
}

/// `compile.py:861-867 ResumeGuardDescr.copy_all_attributes_from` +
/// `compile.py:840-842 ResumeGuardCopiedDescr.copy_all_attributes_from`
/// dispatched on the receiver's variant.  Mutates `my_descr` in place;
/// the receiver's identity (`fail_index` / status / subtype tag) is
/// always preserved.
///
/// Used by `optimizer.py:713-720 replace_guard_op` and
/// `guard.py:120-121 inhert_attributes` — both call
/// `new_descr.copy_all_attributes_from(old_descr)` on a guard whose
/// own descr is already in place.
///
/// `Plain` self (default `ResumeGuardDescr`, including subclasses
/// `ResumeGuardExcDescr` / `ResumeAtPositionDescr` / `CompileLoopVersionDescr`
/// / `ResumeGuardForcedDescr`): copy `rd_numb` / `rd_consts` / `rd_virtuals`
/// / `rd_pendingfields` / `rd_vector_info` from the donor onto self via
/// descr-side setters.  `donor.rd_*()` chases through
/// `ResumeGuardCopiedDescr.prev` automatically (`compile.py:861 other =
/// other.get_resumestorage()`).
///
/// `Copied` self (`ResumeGuardCopiedDescr` / `ResumeGuardCopiedExcDescr`):
/// `compile.py:840-842` overwrites `self.prev = other.prev` in place.
/// Pyre stores `prev` in `UnsafeCell<DescrRef>` and exposes
/// `set_prev_descr(&self, prev)` so the swap preserves the receiver's
/// `fail_index` and subtype tag — observable from guard failure tables
/// / status / bridge attachment.
///
/// Panics if either descr lacks a `FailDescr`, or if `Copied`-self path
/// is hit but the donor is not itself a `ResumeGuardCopiedDescr`
/// (matches RPython's `compile.py:841 assert isinstance(other,
/// ResumeGuardCopiedDescr)`).
pub fn copy_all_attributes_from(my_descr: &DescrRef, donor_descr: &DescrRef) {
    if my_descr.is_resume_guard_copied() {
        // compile.py:840-842 ResumeGuardCopiedDescr.copy_all_attributes_from:
        //     assert isinstance(other, ResumeGuardCopiedDescr)
        //     self.prev = other.prev
        let donor_prev = donor_descr
            .prev_descr()
            .expect("compile.py:841 other must be a ResumeGuardCopiedDescr with a prev");
        my_descr.set_prev_descr(donor_prev);
    } else {
        // compile.py:861-872 ResumeGuardDescr.copy_all_attributes_from:
        //     other = other.get_resumestorage()
        //     assert isinstance(other, ResumeGuardDescr)
        //     self.rd_consts = other.rd_consts
        //     self.rd_pendingfields = other.rd_pendingfields
        //     self.rd_virtuals = other.rd_virtuals
        //     self.rd_numb = other.rd_numb
        //     # we don't copy status
        //     if other.rd_vector_info:
        //         self.rd_vector_info = other.rd_vector_info.clone()
        // compile.py:862 `other = other.get_resumestorage()`: copied
        // donors route reads through `prev`. Resolve the chain so we
        // never deep-copy from a copied descr's empty payload.
        let resolved_donor = if donor_descr.is_resume_guard_copied() {
            donor_descr
                .prev_descr()
                .expect("compile.py:849 ResumeGuardCopiedDescr.get_resumestorage requires prev")
        } else {
            donor_descr.clone()
        };
        // compile.py:863 `assert isinstance(other, ResumeGuardDescr)` —
        // post-resolution donor must be a ResumeGuardDescr (or subclass).
        assert!(
            resolved_donor.is_resume_guard(),
            "compile.py:863 copy_all_attributes_from: \
             resolved donor must be a ResumeGuardDescr (got descr_index={:?})",
            resolved_donor.index()
        );
        let my_fd = my_descr
            .as_fail_descr()
            .expect("copy_all_attributes_from: my_descr must be a FailDescr");
        let donor_fd = resolved_donor
            .as_fail_descr()
            .expect("copy_all_attributes_from: donor must be a FailDescr after get_resumestorage");
        // RPython compile.py:864-867 does reference-share (`self.rd_consts
        // = other.rd_consts` etc.).  Pyre stores rd_* as `Arc<[T]>` so
        // the share is a single refcount bump per slot.  Reads still
        // get `&[T]` slices via the Deref impl; swap-replacement
        // through `set_rd_*` (Vec input) just builds a fresh Arc.
        my_fd.set_rd_numb_arc(donor_fd.rd_numb_arc());
        my_fd.set_rd_consts_arc(donor_fd.rd_consts_arc());
        my_fd.set_rd_virtuals_arc(donor_fd.rd_virtuals_arc());
        my_fd.set_rd_pendingfields_arc(donor_fd.rd_pendingfields_arc());
        // compile.py:869-870 — chain.clone() preserves the donor's
        // (already-flattened) accumulator chain on self, identity-stable.
        let donor_chain = donor_fd.vector_info();
        if !donor_chain.is_empty() {
            my_fd.replace_vector_info(donor_chain);
        }
    }
}

/// compile.py:895-908: CompileLoopVersionDescr(ResumeGuardDescr)
///
/// A guard descriptor for loop-version guards. These guards must never
/// fail at runtime — they exist only to mark where a specialized loop
/// version should be compiled and stitched.
#[derive(Debug)]
pub struct CompileLoopVersionDescr {
    fail_index: u32,
    types: UnsafeCell<Vec<Type>>,
    resume_data: ResumeData,
    payload: RdPayload,
    vector_info: UnsafeCell<Option<Box<AccumInfo>>>,
    /// Phase E.1 — backend-specific extension slot.  Same role as on
    /// `ResumeGuardDescr`; loop-version guards are a `ResumeGuardDescr`
    /// subclass per `compile.py:895` and reach the backend codegen
    /// path identically.
    backend_data: UnsafeCell<Option<Box<dyn std::any::Any + Send + Sync>>>,
    /// `compile.py:186` `descr.rd_loop_token = clt`.  Same role as on
    /// `ResumeGuardDescr`; loop-version descrs are a `ResumeGuardDescr`
    /// subclass per `compile.py:895` and reach `record_loop_or_bridge`
    /// identically (CompileLoopVersionDescr inherits the
    /// `_attrs_` slot list at history.py:125 via AbstractFailDescr).
    rd_loop_token_clt: UnsafeCell<Option<std::sync::Arc<CompiledLoopToken>>>,
    /// Pyre-only owning-trace identifier — same role as on
    /// `ResumeGuardDescr`. Stamped by `record_loop_or_bridge`.
    trace_id: AtomicU64,
    /// Pyre-only per-trace fail-index — same role as on
    /// `ResumeGuardDescr`. Stamped by `build_guard_metadata`.
    fail_index_per_trace: AtomicU32,
}

unsafe impl Send for CompileLoopVersionDescr {}
unsafe impl Sync for CompileLoopVersionDescr {}

impl majit_ir::Descr for CompileLoopVersionDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_loop_version(&self) -> bool {
        true
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    /// compile.py:905-908: CompileLoopVersionDescr.clone()
    fn clone_descr(&self) -> Option<DescrRef> {
        Some(Arc::new(CompileLoopVersionDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(unsafe { (&*self.types.get()).clone() }),
            resume_data: self.resume_data.clone(),
            payload: self.payload.deep_clone(),
            vector_info: UnsafeCell::new(unsafe { (&*self.vector_info.get()).clone() }),
            // Phase E.1: clone mints a fresh descr with no backend
            // extension yet — same scoping as ResumeGuardDescr.clone_descr.
            backend_data: UnsafeCell::new(None),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
        }))
    }
}

impl FailDescr for CompileLoopVersionDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn trace_id(&self) -> u64 {
        self.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.types.get() = types }
    }
    /// compile.py:899-900
    fn exits_early(&self) -> bool {
        true
    }
    /// compile.py:902-903
    fn loop_version(&self) -> bool {
        true
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.vector_info.get() = build_vector_info_chain(chain) }
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        self.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<Arc<[u8]>>) {
        self.payload.set_rd_numb_arc(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        self.payload.rd_consts_arc()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<[Const]>>) {
        self.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.payload.set_rd_virtuals_arc(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.payload.set_rd_pendingfields_arc(value)
    }
    fn backend_data(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        unsafe { (*self.backend_data.get()).as_deref() }
    }
    fn set_backend_data(&self, data: Box<dyn std::any::Any + Send + Sync>) {
        unsafe { *self.backend_data.get() = Some(data) };
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.rd_loop_token_clt.get() = Some(typed) };
    }
}

/// guard.py:89-91:
///   descr = CompileLoopVersionDescr()
///   descr.copy_all_attributes_from(self.op.getdescr())
///   descr.rd_vector_info = None
///
/// Creates a fresh CompileLoopVersionDescr.  rd_* (compile.py:855
/// `_attrs_`) are reference-shared from the source descr via the
/// `_arc` getters — same semantics as `copy_all_attributes_from`
/// (compile.py:861-867).  `rd_vector_info` is reset to None per
/// guard.py:91.  The descr also carries fail_arg types from the
/// source so the backend layout matches the donor guard.
///
/// Panics if source_op has no descr or the resolved donor (after
/// `get_resumestorage()`) is not a `ResumeGuardDescr` — matching
/// RPython's invariant at compile.py:861-863
/// (`other = other.get_resumestorage(); assert isinstance(other,
/// ResumeGuardDescr)`).
pub fn make_compile_loop_version_descr_from(source_op: &majit_ir::Op) -> DescrRef {
    let src_descr = source_op
        .descr
        .as_ref()
        .expect("guard.py:90: self.op.getdescr() must exist");
    // compile.py:862 `other = other.get_resumestorage()`: if the source
    // is a `ResumeGuardCopiedDescr`, resolve to its `prev` so we read
    // resume data from the canonical donor.  ResumeGuardDescr's
    // `get_resumestorage` returns self, so direct sources pass through
    // unchanged.
    let resolved_descr = if src_descr.is_resume_guard_copied() {
        src_descr
            .prev_descr()
            .expect("compile.py:849 ResumeGuardCopiedDescr.prev must be set")
    } else {
        src_descr.clone()
    };
    // compile.py:863 `assert isinstance(other, ResumeGuardDescr)`:
    // reject non-resume FailDescr (e.g. MetaFailDescr) that would
    // otherwise yield an empty rd_* payload on the loop-version descr.
    assert!(
        resolved_descr.is_resume_guard(),
        "compile.py:863 assert isinstance(other, ResumeGuardDescr): \
         loop-version donor descr_index={} is not a ResumeGuardDescr \
         subclass",
        resolved_descr.index()
    );
    let src_fd = resolved_descr
        .as_fail_descr()
        .expect("compile.py:863 ResumeGuardDescr is also a FailDescr");
    let types = src_fd.fail_arg_types().to_vec();
    // compile.py:861-872 copy_all_attributes_from copies rd_*; mirror
    // RPython's reference-share by reusing the donor's `Arc<[T]>`
    // slots — `Arc::clone` only bumps a refcount.
    let payload = RdPayload {
        rd_numb: UnsafeCell::new(src_fd.rd_numb_arc()),
        rd_consts: UnsafeCell::new(src_fd.rd_consts_arc()),
        rd_virtuals: UnsafeCell::new(src_fd.rd_virtuals_arc()),
        rd_pendingfields: UnsafeCell::new(src_fd.rd_pendingfields_arc()),
    };
    Arc::new(CompileLoopVersionDescr {
        fail_index: alloc_fail_index(),
        types: UnsafeCell::new(types),
        resume_data: ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: Vec::new(),
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        },
        payload,
        // guard.py:91: descr.rd_vector_info = None
        vector_info: UnsafeCell::new(None),
        backend_data: UnsafeCell::new(None),
        rd_loop_token_clt: UnsafeCell::new(None),
        trace_id: AtomicU64::new(0),
        fail_index_per_trace: AtomicU32::new(0),
    })
}

/// Resume data for a guard now lives on `StoredExitLayout.resume_layout`
/// (per-guard `ResumeLayoutSummary`) rather than a separate trace-side
/// `HashMap<u32, ResumeData>`.  See `pyjitpl/mod.rs CompiledTrace`.
/// `enrich_guard_resume_layouts_for_trace` and
/// `attach_resume_data_to_trace` are the two producers; readers go
/// through `exit_layout.resume_layout.{reconstruct_state,
/// materialize_virtuals, resolve_pending_field_writes}` in
/// `handle_guard_failure_in_trace_with_savedata` and the blackhole
/// fallback paths.  This collapses the prior two-store model
/// (`CompiledTrace.resume_data` + `StoredExitLayout.resume_layout`)
/// onto the descr-owned layer, mirroring RPython where
/// `ResumeGuardDescr` (`compile.py:855`) is the single guard-owned
/// resume container.

#[cfg(test)]
mod fail_descr_tests {
    use super::*;

    #[test]
    fn test_attach_vector_info_builds_prev_chain() {
        let descr = make_fail_descr(2);
        let fail_descr = descr.as_fail_descr().unwrap();
        fail_descr.attach_vector_info(AccumInfo {
            prev: None,
            failargs_pos: 0,
            variable: majit_ir::OpRef::int_op(10),
            location: majit_ir::OpRef::int_op(20),
            accum_operation: '+',
            scalar: majit_ir::OpRef::NONE,
        });
        fail_descr.attach_vector_info(AccumInfo {
            prev: None,
            failargs_pos: 1,
            variable: majit_ir::OpRef::int_op(11),
            location: majit_ir::OpRef::int_op(21),
            accum_operation: '*',
            scalar: majit_ir::OpRef::NONE,
        });

        let vector_info = fail_descr.vector_info();
        assert_eq!(vector_info.len(), 2);
        assert_eq!(vector_info[0].failargs_pos, 1);
        assert_eq!(vector_info[1].failargs_pos, 0);
        assert_eq!(
            vector_info[0].prev.as_ref().map(|info| info.failargs_pos),
            Some(0)
        );
        assert!(vector_info[1].prev.is_none());

        let cloned = descr.clone_descr().unwrap();
        let cloned_vector_info = cloned.as_fail_descr().unwrap().vector_info();
        assert_eq!(cloned_vector_info.len(), 2);
        assert_eq!(cloned_vector_info[0].failargs_pos, 1);
        assert_eq!(
            cloned_vector_info[0]
                .prev
                .as_ref()
                .map(|info| info.failargs_pos),
            Some(0)
        );
    }

    #[test]
    fn test_fail_descr_unique_indices() {
        reset_fail_index_counter();
        let d1 = make_fail_descr(2);
        let d2 = make_fail_descr(3);
        let d3 = make_fail_descr(1);

        let fi1 = d1.as_fail_descr().unwrap().fail_index();
        let fi2 = d2.as_fail_descr().unwrap().fail_index();
        let fi3 = d3.as_fail_descr().unwrap().fail_index();

        // All indices must be unique
        assert_ne!(fi1, fi2);
        assert_ne!(fi2, fi3);
        assert_ne!(fi1, fi3);
    }

    #[test]
    fn test_fail_descr_with_explicit_index() {
        let d = make_fail_descr_with_index(42, 3);
        assert_eq!(d.as_fail_descr().unwrap().fail_index(), 42);
        assert_eq!(d.as_fail_descr().unwrap().fail_arg_types().len(), 3);
    }

    #[test]
    fn test_fail_descr_typed() {
        let types = vec![Type::Int, Type::Ref, Type::Float];
        let d = make_fail_descr_typed(types.clone());
        assert_eq!(d.as_fail_descr().unwrap().fail_arg_types(), &types);
    }

    /// `compile.py:869 ResumeGuardDescr.store_final_boxes` parity:
    /// `store_final_boxes_in_guard` mutates types in place, preserving
    /// the descr's `Arc` identity, `fail_index`, and concrete subtype.
    #[test]
    fn test_set_fail_arg_types_preserves_identity_and_subtype() {
        // ResumeAtPositionDescr (unroll extra_guards path).
        let descr = make_resume_at_position_descr_typed(vec![Type::Int]);
        let original_fail_index = descr.as_fail_descr().unwrap().fail_index();
        let original_ptr = Arc::as_ptr(&descr);
        assert!(descr.is_resume_at_position());

        descr
            .as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Ref, Type::Float]);

        // Identity preserved: same Arc, same fail_index, same subtype tag.
        assert_eq!(Arc::as_ptr(&descr), original_ptr);
        assert_eq!(
            descr.as_fail_descr().unwrap().fail_index(),
            original_fail_index
        );
        assert!(descr.is_resume_at_position());
        // Types updated.
        assert_eq!(
            descr.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Ref, Type::Float]
        );

        // CompileLoopVersionDescr.
        let lv = Arc::new(CompileLoopVersionDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(vec![Type::Int]),
            resume_data: ResumeData {
                vable_array: Vec::new(),
                vref_array: Vec::new(),
                frames: Vec::new(),
                virtuals: Vec::new(),
                pending_fields: Vec::new(),
            },
            payload: RdPayload::empty(),
            vector_info: UnsafeCell::new(None),
            backend_data: UnsafeCell::new(None),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
        }) as DescrRef;
        let lv_fi = lv.as_fail_descr().unwrap().fail_index();
        assert!(lv.as_fail_descr().unwrap().loop_version());

        lv.as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Ref]);

        assert_eq!(lv.as_fail_descr().unwrap().fail_index(), lv_fi);
        assert!(lv.as_fail_descr().unwrap().loop_version());
        assert_eq!(lv.as_fail_descr().unwrap().fail_arg_types(), &[Type::Ref]);

        // MetaFailDescr / plain ResumeGuardDescr factory.
        let plain = make_resume_guard_descr_typed(vec![Type::Int, Type::Int]);
        let plain_fi = plain.as_fail_descr().unwrap().fail_index();
        assert!(!plain.is_resume_at_position());

        plain
            .as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Float]);

        assert_eq!(plain.as_fail_descr().unwrap().fail_index(), plain_fi);
        assert!(!plain.is_resume_at_position());
        assert!(!plain.is_guard_forced());
        assert!(!plain.is_guard_exc());
        assert_eq!(
            plain.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Float]
        );

        // ResumeGuardForcedDescr — `is_guard_forced()` survives
        // `set_fail_arg_types`, identity preserved.
        let forced = make_resume_guard_forced_descr_typed(vec![Type::Int]);
        let forced_fi = forced.as_fail_descr().unwrap().fail_index();
        let forced_ptr = Arc::as_ptr(&forced);
        assert!(forced.is_guard_forced());
        assert!(!forced.is_guard_exc());

        forced
            .as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Ref]);

        assert_eq!(Arc::as_ptr(&forced), forced_ptr);
        assert_eq!(forced.as_fail_descr().unwrap().fail_index(), forced_fi);
        assert!(forced.is_guard_forced());
        assert_eq!(
            forced.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Ref]
        );

        // ResumeGuardExcDescr — `is_guard_exc()` survives, identity preserved.
        let exc = make_resume_guard_exc_descr_typed(vec![Type::Ref, Type::Int]);
        let exc_fi = exc.as_fail_descr().unwrap().fail_index();
        let exc_ptr = Arc::as_ptr(&exc);
        assert!(exc.is_guard_exc());
        assert!(!exc.is_guard_forced());

        exc.as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Float]);

        assert_eq!(Arc::as_ptr(&exc), exc_ptr);
        assert_eq!(exc.as_fail_descr().unwrap().fail_index(), exc_fi);
        assert!(exc.is_guard_exc());
        assert_eq!(
            exc.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Float]
        );

        // compile.py:873-876 ResumeGuardDescr.clone() returns a plain
        // ResumeGuardDescr — both `ResumeGuardForcedDescr` (compile.py:939+,
        // no clone override) and `ResumeGuardExcDescr` (compile.py:881-882
        // `pass`) inherit this base implementation, so the subtype tag is
        // intentionally dropped on clone. Resume attributes / fail_arg_types
        // are copied; fail_index is fresh.
        let forced_clone = forced.clone_descr().unwrap();
        assert!(!forced_clone.is_guard_forced());
        assert!(!forced_clone.is_guard_exc());
        assert_ne!(
            forced_clone.as_fail_descr().unwrap().fail_index(),
            forced_fi
        );
        assert_eq!(
            forced_clone.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Ref]
        );

        let exc_clone = exc.clone_descr().unwrap();
        assert!(!exc_clone.is_guard_exc());
        assert!(!exc_clone.is_guard_forced());
        assert_ne!(exc_clone.as_fail_descr().unwrap().fail_index(), exc_fi);
        assert_eq!(
            exc_clone.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Float]
        );
    }

    /// compile.py:832-851 ResumeGuardCopiedDescr(prev) parity:
    /// `get_resumestorage()` chases to `prev`, `fail_arg_types`
    /// shares the donor's vector, `is_resume_guard_copied()` flags
    /// the subtype, and the exc variant additionally reports
    /// `is_guard_exc() = true` while tracking the same `prev`.
    #[test]
    fn test_resume_guard_copied_descr_delegates_to_prev() {
        // Plain copied descr over a ResumeGuardDescr donor.
        let donor = make_resume_guard_descr_typed(vec![Type::Int, Type::Ref]);
        let donor_fi = donor.as_fail_descr().unwrap().fail_index();
        let donor_ptr = Arc::as_ptr(&donor);

        let copied = make_resume_guard_copied_descr(donor.clone());
        assert!(copied.is_resume_guard_copied());
        assert!(!copied.is_guard_exc());
        assert!(!copied.is_guard_forced());
        assert_ne!(copied.as_fail_descr().unwrap().fail_index(), donor_fi);
        assert_eq!(
            copied.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Int, Type::Ref]
        );
        // get_resumestorage() chases to prev — same Arc ptr.
        let prev = copied.prev_descr().unwrap();
        assert_eq!(Arc::as_ptr(&prev), donor_ptr);

        // clone_descr() preserves prev identity, allocates fresh fail_index.
        let copied_clone = copied.clone_descr().unwrap();
        assert!(copied_clone.is_resume_guard_copied());
        assert_eq!(Arc::as_ptr(&copied_clone.prev_descr().unwrap()), donor_ptr);
        assert_ne!(
            copied_clone.as_fail_descr().unwrap().fail_index(),
            copied.as_fail_descr().unwrap().fail_index()
        );

        // Exc-copied subtype carries the same prev and additionally
        // reports is_guard_exc() = true.
        let exc_donor = make_resume_guard_exc_descr_typed(vec![Type::Float]);
        let exc_donor_ptr = Arc::as_ptr(&exc_donor);
        let copied_exc = make_resume_guard_copied_exc_descr(exc_donor);
        assert!(copied_exc.is_resume_guard_copied());
        assert!(copied_exc.is_guard_exc());
        assert!(!copied_exc.is_guard_forced());
        assert_eq!(
            copied_exc.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Float]
        );
        assert_eq!(
            Arc::as_ptr(&copied_exc.prev_descr().unwrap()),
            exc_donor_ptr
        );

        // set_fail_arg_types must NOT mutate the donor through a copied
        // descr — `_copy_resume_data_from` never calls
        // store_final_boxes_in_guard, so any invocation indicates an
        // upstream invariant violation.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            copied
                .as_fail_descr()
                .unwrap()
                .set_fail_arg_types(vec![Type::Float])
        }));
        assert!(
            result.is_err(),
            "set_fail_arg_types on ResumeGuardCopiedDescr must panic"
        );
        // Donor's types unchanged.
        assert_eq!(
            donor.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Int, Type::Ref]
        );
    }
}
